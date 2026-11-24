// moe_trace: teacher-forced MoE routing traces with llama.cpp (links libllama only).
//
// For every token and every MoE layer it records the top-k expert ids that the router selected, by capturing the
// `ffn_moe_topk-<layer>` tensors (llm_graph_context::build_moe_ffn, src/llama-graph.cpp) through the backend
// scheduler's eval callback. Every token of a batch is marked as an output so that the last layer is not pruned to
// the output rows (llama.cpp skips the last layer's FFN for non-output tokens).
//
//   moe_trace tokenize -m MODEL -l LIST
//       LIST lines: in.txt <TAB> out.tok <TAB> add_special(0|1) <TAB> parse_special(0|1)      (vocab only, no GPU)
//       out.tok = int32 token ids
//   moe_trace trace -m MODEL -l LIST -o OUTDIR [-c 32768] [-ub 512] [-t 4] [--validate DOC:P:N] [--min-avail-gib 12]
//       LIST lines: doc_id <TAB> tokens.tok
//       OUTDIR/<doc>.route.npy   uint16 [T, L_moe, k]   expert ids, router-score order (best first)
//       OUTDIR/<doc>.nll.npy     float32 [T]            -log p(token t+1 | ..t) under teacher forcing (last = NaN)
//       OUTDIR/<doc>.decode.route.npy (validation)      routing of tokens P..P+N-1 decoded one token per llama_decode
//       OUTDIR/trace_meta.json
//
// The tool aborts if MemAvailable falls under --min-avail-gib (checked before every llama_decode).

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ------------------------------------------------------------------------------------------------ small helpers
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double mem_available_gib() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 1e9;
    char line[256];
    double v = 1e9;
    while (fgets(line, sizeof(line), f)) {
        long kb;
        if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) { v = kb / 1048576.0; break; }
    }
    fclose(f);
    return v;
}

static std::vector<std::vector<std::string>> read_list(const std::string & path) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, '\t')) cols.push_back(c);
        rows.push_back(cols);
    }
    return rows;
}

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::vector<llama_token> read_tokens(const std::string & path) {
    std::string s = read_file(path);
    std::vector<llama_token> t(s.size() / sizeof(llama_token));
    memcpy(t.data(), s.data(), t.size() * sizeof(llama_token));
    return t;
}

static void write_npy(const std::string & path, const char * descr, const std::vector<size_t> & shape,
                      const void * data, size_t nbytes) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        sh += std::to_string(shape[i]);
        sh += (shape.size() == 1 || i + 1 < shape.size()) ? "," : "";
        if (i + 1 < shape.size()) sh += " ";
    }
    sh += ")";
    std::string hdr = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t total = 10 + hdr.size() + 1;
    hdr += std::string((64 - total % 64) % 64, ' ');
    hdr += "\n";
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    fwrite(magic, 1, 8, f);
    uint16_t hl = (uint16_t) hdr.size();
    fwrite(&hl, 2, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    fwrite(data, 1, nbytes, f);
    fclose(f);
}

static std::string meta_str(const llama_model * model, const std::string & key) {
    char buf[512];
    int n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n >= 0 ? std::string(buf) : std::string();
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char) c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN || getenv("MOE_TRACE_VERBOSE")) fputs(text, stderr);
}

// ------------------------------------------------------------------------------------------------ routing capture
struct capture {
    bool discovering = false;
    std::vector<int> seen;                 // layer ids in callback order during discovery
    int k = 0, n_expert = 0;
    std::vector<int> slot_of_layer;        // model layer -> MoE slot (-1 = no MoE)
    int n_slots = 0;

    uint16_t * dst = nullptr;              // [n_tok, n_slots, k] for the tokens of the current llama_decode
    int n_tok = 0;
    std::vector<int> written;              // per slot: tokens written in the current llama_decode
    std::vector<int32_t> buf;
    std::string error;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    static const char prefix[] = "ffn_moe_topk-";
    const char * name = ggml_get_name(t);
    if (strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return ask ? false : true;
    }
    if (ask) return true;
    auto * c = (capture *) ud;
    const int il = atoi(name + sizeof(prefix) - 1);
    const int k = (int) t->ne[0];
    const int n = (int) t->ne[1];
    if (t->type != GGML_TYPE_I32 || t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != 4) {
        c->error = std::string("unexpected tensor layout for ") + name;
        return true;
    }
    const size_t nbytes = ggml_nbytes(t);
    c->buf.resize((nbytes + 3) / 4);
    ggml_backend_tensor_get(t, c->buf.data(), 0, nbytes);
    if (c->discovering) {
        c->seen.push_back(il);
        c->k = k;
        c->n_expert = (int) (t->view_src ? t->view_src->ne[0] : t->ne[0]);
        return true;
    }
    const int slot = il < (int) c->slot_of_layer.size() ? c->slot_of_layer[il] : -1;
    if (slot < 0 || k != c->k) {
        c->error = std::string("unexpected MoE layer or k: ") + name;
        return true;
    }
    const int row0 = c->written[slot];
    if (row0 + n > c->n_tok) {
        c->error = std::string("more routed tokens than submitted: ") + name;
        return true;
    }
    const size_t stride = t->nb[1] / 4;
    for (int i = 0; i < n; ++i) {
        const int32_t * src = c->buf.data() + (size_t) i * stride;
        uint16_t * d = c->dst + ((size_t) (row0 + i) * c->n_slots + slot) * c->k;
        for (int j = 0; j < k; ++j) {
            if (src[j] < 0 || src[j] >= c->n_expert) { c->error = "expert id out of range"; return true; }
            d[j] = (uint16_t) src[j];
        }
    }
    c->written[slot] += n;
    return true;
}

// ------------------------------------------------------------------------------------------------ tokenize
static int cmd_tokenize(const std::string & model_path, const std::string & list) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load vocab\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    for (auto & row : read_list(list)) {
        if (row.size() < 4) { fprintf(stderr, "bad list row\n"); return 1; }
        const std::string text = read_file(row[0]);
        const bool add_special = row[2] == "1", parse_special = row[3] == "1";
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), add_special, parse_special);
        if (n < 0) {
            toks.resize(-n);
            n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), add_special, parse_special);
        }
        if (n < 0) { fprintf(stderr, "tokenize failed for %s\n", row[0].c_str()); return 1; }
        toks.resize(n);
        FILE * f = fopen(row[1].c_str(), "wb");
        fwrite(toks.data(), sizeof(llama_token), toks.size(), f);
        fclose(f);
        printf("%s\t%d\n", row[1].c_str(), n);
    }
    llama_model_free(model);
    return 0;
}

// ------------------------------------------------------------------------------------------------ trace
struct doc_result {
    std::string id;
    size_t n_tokens = 0;
    double seconds = 0, mean_nll = 0;
    size_t n_nll = 0;
};

// Decode tokens[p0, p1) in chunks of `chunk` tokens (all outputs), writing routing to route (row = absolute position)
// and NLL to nll (if not null). Returns false on error.
static bool run_range(llama_context * ctx, capture & cap, llama_batch & batch, const std::vector<llama_token> & toks,
                      size_t p0, size_t p1, int chunk, uint16_t * route, float * nll, int n_vocab, int n_threads,
                      double min_avail) {
    for (size_t s = p0; s < p1; s += chunk) {
        const int n = (int) std::min<size_t>(chunk, p1 - s);
        if (mem_available_gib() < min_avail) {
            fprintf(stderr, "MemAvailable %.1f GiB < %.1f GiB: aborting\n", mem_available_gib(), min_avail);
            return false;
        }
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = toks[s + i];
            batch.pos[i] = (llama_pos) (s + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        cap.dst = route + s * cap.n_slots * cap.k;
        cap.n_tok = n;
        std::fill(cap.written.begin(), cap.written.end(), 0);
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) { fprintf(stderr, "llama_decode failed: %d at %zu\n", rc, s); return false; }
        if (!cap.error.empty()) { fprintf(stderr, "capture error: %s\n", cap.error.c_str()); return false; }
        for (int sl = 0; sl < cap.n_slots; ++sl) {
            if (cap.written[sl] != n) {
                fprintf(stderr, "slot %d: routed %d of %d tokens\n", sl, cap.written[sl], n);
                return false;
            }
        }
        if (nll) {
            auto work = [&](int i0, int i1) {
                for (int i = i0; i < i1; ++i) {
                    const size_t pos = s + i;
                    if (pos + 1 >= toks.size()) { nll[pos] = NAN; continue; }
                    const float * lg = llama_get_logits_ith(ctx, i);
                    float mx = -INFINITY;
                    for (int v = 0; v < n_vocab; ++v) mx = std::max(mx, lg[v]);
                    double z = 0;
                    for (int v = 0; v < n_vocab; ++v) z += expf(lg[v] - mx);
                    nll[pos] = (float) (std::log(z) + mx - lg[toks[pos + 1]]);
                }
            };
            std::vector<std::thread> th;
            const int nt = std::max(1, std::min(2 * n_threads, n));
            for (int w = 0; w < nt; ++w) th.emplace_back(work, n * w / nt, n * (w + 1) / nt);
            for (auto & x : th) x.join();
        }
    }
    return true;
}

static int cmd_trace(const std::string & model_path, const std::string & list, const std::string & outdir,
                     int n_ctx, int n_ubatch, int n_threads, const std::string & validate, double min_avail) {
    auto docs = read_list(list);
    if (mem_available_gib() < min_avail + 8) {
        fprintf(stderr, "MemAvailable %.1f GiB too low to start\n", mem_available_gib());
        return 3;
    }
    capture cap;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mtp = false;
    const double t_load0 = now_s();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const double t_load = now_s() - t_load0;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ubatch;
    cp.n_ubatch = n_ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cp.no_perf = true;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);

    // discover the MoE layers with a one-token decode
    {
        cap.discovering = true;
        llama_token t0 = llama_vocab_bos(vocab);
        if (t0 == LLAMA_TOKEN_NULL) t0 = 0;
        batch.n_tokens = 1;
        batch.token[0] = t0; batch.pos[0] = 0; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0 || cap.seen.empty()) { fprintf(stderr, "discovery decode failed\n"); return 1; }
        cap.discovering = false;
        llama_memory_clear(mem, true);
        std::vector<int> layers = cap.seen;
        std::sort(layers.begin(), layers.end());
        if (std::adjacent_find(layers.begin(), layers.end()) != layers.end()) {
            fprintf(stderr, "a layer reported ffn_moe_topk twice\n");
            return 1;
        }
        cap.slot_of_layer.assign(layers.back() + 1, -1);
        for (size_t i = 0; i < layers.size(); ++i) cap.slot_of_layer[layers[i]] = (int) i;
        cap.n_slots = (int) layers.size();
        cap.written.assign(cap.n_slots, 0);
        cap.seen = layers;
        fprintf(stderr, "MoE layers: %d (first %d, last %d), n_expert %d, k %d\n", cap.n_slots, layers.front(),
                layers.back(), cap.n_expert, cap.k);
    }

    const size_t row = (size_t) cap.n_slots * cap.k;
    std::vector<doc_result> results;
    std::map<std::string, std::vector<uint16_t>> kept;   // routing of the validation doc
    std::string vdoc;
    size_t vP = 0, vN = 0;
    if (!validate.empty()) {
        size_t a = validate.find(':'), b = validate.rfind(':');
        vdoc = validate.substr(0, a);
        vP = std::stoul(validate.substr(a + 1, b - a - 1));
        vN = std::stoul(validate.substr(b + 1));
    }
    for (auto & d : docs) {
        if (d.size() < 2) continue;
        doc_result r;
        r.id = d[0];
        auto toks = read_tokens(d[1]);
        r.n_tokens = toks.size();
        if ((int) toks.size() > n_ctx) { fprintf(stderr, "%s: %zu tokens > n_ctx\n", r.id.c_str(), toks.size()); return 1; }
        std::vector<uint16_t> route(toks.size() * row, 0xFFFF);
        std::vector<float> nll(toks.size(), NAN);
        llama_memory_clear(mem, true);
        const double t0 = now_s();
        if (!run_range(ctx, cap, batch, toks, 0, toks.size(), n_ubatch, route.data(), nll.data(), n_vocab, n_threads, min_avail)) return 2;
        r.seconds = now_s() - t0;
        double sum = 0; size_t cnt = 0;
        for (float x : nll) if (std::isfinite(x)) { sum += x; ++cnt; }
        r.mean_nll = cnt ? sum / cnt : NAN;
        r.n_nll = cnt;
        write_npy(outdir + "/" + r.id + ".route.npy", "<u2", {toks.size(), (size_t) cap.n_slots, (size_t) cap.k}, route.data(), route.size() * 2);
        write_npy(outdir + "/" + r.id + ".nll.npy", "<f4", {toks.size()}, nll.data(), nll.size() * 4);
        fprintf(stderr, "%-24s %6zu tokens %7.1f s %7.0f tok/s  mean NLL %.3f  MemAvailable %.1f GiB\n", r.id.c_str(),
                toks.size(), r.seconds, toks.size() / r.seconds, r.mean_nll, mem_available_gib());
        if (r.id == vdoc) kept[r.id] = route;
        results.push_back(r);
    }

    // validation: the same tokens, prefix batched, then one token per llama_decode (what generation does)
    std::string vjson = "null";
    if (!vdoc.empty() && kept.count(vdoc)) {
        std::string path;
        for (auto & d : docs) if (d[0] == vdoc) path = d[1];
        auto toks = read_tokens(path);
        vN = std::min(vN, toks.size() - vP);
        std::vector<uint16_t> route(toks.size() * row, 0xFFFF);
        llama_memory_clear(mem, true);
        const double t0 = now_s();
        if (!run_range(ctx, cap, batch, toks, 0, vP, n_ubatch, route.data(), nullptr, n_vocab, n_threads, min_avail)) return 2;
        if (!run_range(ctx, cap, batch, toks, vP, vP + vN, 1, route.data(), nullptr, n_vocab, n_threads, min_avail)) return 2;
        const double secs = now_s() - t0;
        const auto & ref = kept[vdoc];
        size_t same_set = 0, inter = 0, top1 = 0, n_tl = 0;
        std::vector<double> per_layer(cap.n_slots, 0);
        for (size_t t = vP; t < vP + vN; ++t) {
            for (int l = 0; l < cap.n_slots; ++l) {
                const uint16_t * a = &route[(t * cap.n_slots + l) * cap.k];
                const uint16_t * b = &ref[(t * cap.n_slots + l) * cap.k];
                int c = 0;
                for (int i = 0; i < cap.k; ++i) for (int j = 0; j < cap.k; ++j) c += a[i] == b[j];
                inter += c; same_set += c == cap.k; top1 += a[0] == b[0]; per_layer[l] += (double) c / cap.k; ++n_tl;
            }
        }
        write_npy(outdir + "/" + vdoc + ".decode.route.npy", "<u2", {vN, (size_t) cap.n_slots, (size_t) cap.k},
                  route.data() + vP * row, vN * row * 2);
        std::string pl = "[";
        for (int l = 0; l < cap.n_slots; ++l) {
            char b[32]; snprintf(b, sizeof(b), "%s%.5f", l ? ", " : "", per_layer[l] / vN); pl += b;
        }
        pl += "]";
        char buf[1024];
        snprintf(buf, sizeof(buf), "{\"doc\": \"%s\", \"prefix_tokens\": %zu, \"decoded_tokens\": %zu, \"seconds\": %.2f, "
                 "\"set_identical\": %.6f, \"mean_overlap\": %.6f, \"top1_identical\": %.6f, \"per_layer_overlap\": ",
                 vdoc.c_str(), vP, vN, secs, (double) same_set / n_tl, (double) inter / (n_tl * cap.k), (double) top1 / n_tl);
        vjson = std::string(buf) + pl + "}";
        fprintf(stderr, "validation %s: %zu decoded tokens: identical sets %.4f, mean overlap %.4f, top-1 identical %.4f\n",
                vdoc.c_str(), vN, (double) same_set / n_tl, (double) inter / (n_tl * cap.k), (double) top1 / n_tl);
    }

    // metadata
    const std::string arch = meta_str(model, "general.architecture");
    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));
    std::string layers = "[";
    for (size_t i = 0; i < cap.seen.size(); ++i) layers += (i ? ", " : "") + std::to_string(cap.seen[i]);
    layers += "]";
    FILE * f = fopen((outdir + "/trace_meta.json").c_str(), "w");
    fprintf(f, "{\n \"model_path\": \"%s\",\n \"arch\": \"%s\",\n \"name\": \"%s\",\n \"desc\": \"%s\",\n \"file_type\": \"%s\",\n"
               " \"model_bytes\": %llu,\n \"n_layer\": %d,\n \"n_expert\": %d,\n \"k\": %d,\n \"moe_layers\": %s,\n"
               " \"expert_shared_count\": \"%s\",\n \"n_ctx\": %d,\n \"n_ubatch\": %d,\n \"load_seconds\": %.1f,\n"
               " \"llama_cpp\": \"%s\",\n \"validation\": %s,\n \"docs\": [\n",
            json_escape(model_path).c_str(), arch.c_str(), json_escape(meta_str(model, "general.name")).c_str(), json_escape(desc).c_str(),
            meta_str(model, "general.file_type").c_str(), (unsigned long long) llama_model_size(model), llama_model_n_layer(model),
            cap.n_expert, cap.k, layers.c_str(), meta_str(model, arch + ".expert_shared_count").c_str(), n_ctx, n_ubatch, t_load,
            LLAMA_COMMIT, vjson.c_str());
    for (size_t i = 0; i < results.size(); ++i) {
        auto & r = results[i];
        fprintf(f, "  {\"id\": \"%s\", \"n_tokens\": %zu, \"seconds\": %.2f, \"mean_nll\": %.5f}%s\n", r.id.c_str(), r.n_tokens,
                r.seconds, r.mean_nll, i + 1 < results.size() ? "," : "");
    }
    fprintf(f, " ]\n}\n");
    fclose(f);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: moe_trace tokenize|trace -m MODEL -l LIST [-o OUTDIR] [-c N] [-ub N] [-t N] [--validate DOC:P:N]\n");
        return 1;
    }
    const std::string cmd = argv[1];
    std::string model, list, outdir = ".", validate;
    int n_ctx = 32768, n_ubatch = 512, n_threads = 4;
    double min_avail = 12.0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "-m") model = next();
        else if (a == "-l") list = next();
        else if (a == "-o") outdir = next();
        else if (a == "-c") n_ctx = std::stoi(next());
        else if (a == "-ub") n_ubatch = std::stoi(next());
        else if (a == "-t") n_threads = std::stoi(next());
        else if (a == "--validate") validate = next();
        else if (a == "--min-avail-gib") min_avail = std::stod(next());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    int rc = 1;
    if (cmd == "tokenize") rc = cmd_tokenize(model, list);
    else if (cmd == "trace") rc = cmd_trace(model, list, outdir, n_ctx, n_ubatch, n_threads, validate, min_avail);
    else fprintf(stderr, "unknown command %s\n", cmd.c_str());
    llama_backend_free();
    return rc;
}
