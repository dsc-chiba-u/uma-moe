// replay_io: replay a routing trace against a GGUF's routed experts on NVMe, reading every cache miss with O_DIRECT.
// No model compute: the result is the I/O time per token a DRAM expert cache of a given size would need.
//
//   replay_io <map> <tokens.i32> <warmup:measure> <cache_experts> <policy> <mode> <qd> [ranking.i64]
//     warmup:measure  the first `warmup` tokens update the cache without I/O, the next `measure` tokens are timed
//     map          from analysis/expert_map.py (byte ranges of each (layer, expert) slice)
//     tokens.i32   int32 [T][48][10] routed expert ids, token order
//     policy       lru | belady | static (needs ranking.i64: keys layer*n_expert+expert, best first) | none
//     mode         layer: each layer's misses are read when the layer needs them (no lookahead)
//                  token: all misses of a token are read together (perfect lookahead within the token)
//     qd           I/O threads (queue depth)
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Range { int32_t fi; int64_t off; int64_t len; };
struct Req { int fd; int64_t off; int64_t len; };

struct Pool {
    std::vector<std::thread> th; std::mutex m; std::condition_variable cv, done_cv;
    std::vector<Req> jobs; size_t next = 0, done = 0; bool quit = false; uint64_t gen = 0;
    explicit Pool(int n) {
        for (int i = 0; i < n; ++i) th.emplace_back([this] { work(); });
    }
    void work() {
        void* buf = nullptr; size_t cap = 0; uint64_t seen = 0;
        for (;;) {
            Req r;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return quit || next < jobs.size(); });
                if (quit) break;
                r = jobs[next++];
            }
            if ((size_t) r.len > cap) { free(buf); cap = (size_t) r.len; if (posix_memalign(&buf, 4096, cap)) abort(); }
            if (pread(r.fd, buf, (size_t) r.len, (off_t) r.off) != r.len) { perror("pread"); abort(); }
            { std::lock_guard<std::mutex> lk(m); if (++done == jobs.size()) done_cv.notify_all(); }
            (void) seen;
        }
        free(buf);
    }
    void run(const std::vector<Req>& batch) {
        if (batch.empty()) return;
        std::unique_lock<std::mutex> lk(m);
        jobs = batch; next = 0; done = 0;
        cv.notify_all();
        done_cv.wait(lk, [&] { return done == jobs.size(); });
    }
    ~Pool() { { std::lock_guard<std::mutex> lk(m); quit = true; } cv.notify_all(); for (auto& t : th) t.join(); }
};

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "see the header of replay_io.cpp\n"); return 2; }
    const char* map_path = argv[1]; const char* tok_path = argv[2];
    long warm = 0, max_tok = 0; { std::string wm = argv[3]; size_t c = wm.find(':'); if (c == std::string::npos) max_tok = atol(argv[3]); else { warm = atol(wm.substr(0, c).c_str()); max_tok = atol(wm.substr(c + 1).c_str()); } } long cap = atol(argv[4]); std::string policy = argv[5], mode = argv[6]; int qd = atoi(argv[7]);
    // ---- map
    FILE* f = fopen(map_path, "rb"); int32_t hdr[4]; if (fread(hdr, 4, 4, f) != 4) return 1;
    int n_files = hdr[0], n_layer = hdr[1], n_expert = hdr[2], n_mat = hdr[3];
    std::vector<int> fds;
    for (int i = 0; i < n_files; ++i) {
        std::string p; int c; while ((c = fgetc(f)) > 0) p.push_back((char) c);
        int fd = open(p.c_str(), O_RDONLY | O_DIRECT); if (fd < 0) { perror(p.c_str()); return 1; } fds.push_back(fd);
    }
    std::vector<Range> ranges((size_t) n_layer * n_expert * n_mat);
    for (auto& r : ranges) { if (fread(&r.fi, 4, 1, f) != 1 || fread(&r.off, 8, 1, f) != 1 || fread(&r.len, 8, 1, f) != 1) return 1; }
    fclose(f);
    // ---- trace
    FILE* tf = fopen(tok_path, "rb"); fseek(tf, 0, SEEK_END); long nbytes = ftell(tf); fseek(tf, 0, SEEK_SET);
    const int K = 10; long T = nbytes / (4L * n_layer * K); if (max_tok > 0) T = std::min(T, warm + max_tok);
    std::vector<int32_t> tok((size_t) T * n_layer * K); if (fread(tok.data(), 4, tok.size(), tf) != tok.size()) return 1; fclose(tf);
    const size_t n_acc = tok.size();
    auto key = [&](size_t i) { size_t l = (i / K) % n_layer; return (int64_t) l * n_expert + tok[i]; };
    // ---- policy state
    std::vector<uint8_t> resident((size_t) n_layer * n_expert, 0);
    std::list<int64_t> lru; std::unordered_map<int64_t, std::list<int64_t>::iterator> where;
    std::vector<size_t> nxt;
    std::priority_queue<std::pair<size_t, int64_t>> heap;   // (next use, key) max-heap for Belady
    std::vector<size_t> cur_next((size_t) n_layer * n_expert, 0);
    long n_res = 0;
    if (policy == "static") {
        FILE* rf = fopen(argv[8], "rb"); std::vector<int64_t> rank((size_t) n_layer * n_expert);
        size_t got = fread(rank.data(), 8, rank.size(), rf); fclose(rf);
        for (long i = 0; i < cap && i < (long) got; ++i) resident[rank[i]] = 1;
    }
    if (policy == "belady") {
        nxt.assign(n_acc, SIZE_MAX); std::vector<size_t> last((size_t) n_layer * n_expert, SIZE_MAX);
        for (size_t i = n_acc; i-- > 0;) { int64_t k = key(i); nxt[i] = last[k]; last[k] = i; }
    }
    auto access = [&](size_t i) -> bool {   // true = hit; updates the cache
        int64_t k = key(i);
        if (policy == "none") return false;
        if (policy == "static") return resident[k];
        if (policy == "lru") {
            auto it = where.find(k);
            if (it != where.end()) { lru.splice(lru.end(), lru, it->second); return true; }
            if (cap <= 0) return false;
            lru.push_back(k); where[k] = std::prev(lru.end());
            if ((long) lru.size() > cap) { where.erase(lru.front()); lru.pop_front(); }
            return false;
        }
        // belady
        bool hit = resident[k];
        cur_next[k] = nxt[i];
        if (!hit) { if (cap <= 0) return false; resident[k] = 1; ++n_res; }
        heap.push({nxt[i], k});
        while (n_res > cap) {
            auto top = heap.top(); heap.pop();
            if (resident[top.second] && cur_next[top.second] == top.first) { resident[top.second] = 0; --n_res; }
        }
        return hit;
    };
    // ---- replay
    Pool pool(qd);
    std::vector<Req> batch; std::vector<double> per_tok; long misses = 0; double bytes = 0;
    auto add = [&](int64_t k) {
        for (int m = 0; m < n_mat; ++m) {
            const Range& r = ranges[(size_t) k * n_mat + m];
            int64_t a = r.off & ~4095LL, b = (r.off + r.len + 4095) & ~4095LL;
            batch.push_back({fds[r.fi], a, b - a}); bytes += (double) (b - a);
        }
        ++misses;
    };
    for (long t = 0; t < warm && t < T; ++t)      // warm the cache: no I/O, no timing
        for (int l = 0; l < n_layer; ++l) for (int j = 0; j < K; ++j) access(((size_t) t * n_layer + l) * K + j);
    const double warm_misses = 0;
    auto t0 = Clock::now();
    for (long t = warm; t < T; ++t) {
        auto a = Clock::now();
        for (int l = 0; l < n_layer; ++l) {
            for (int j = 0; j < K; ++j) {
                size_t i = ((size_t) t * n_layer + l) * K + j;
                if (!access(i)) add(key(i));
            }
            if (mode == "layer") { pool.run(batch); batch.clear(); }
        }
        if (mode == "token") { pool.run(batch); batch.clear(); }
        per_tok.push_back(secs(a, Clock::now()) * 1e3);
    }
    double el = secs(t0, Clock::now());
    (void) warm_misses;
    const long Tm = T - warm; const size_t n_meas = (size_t) Tm * n_layer * K;
    std::sort(per_tok.begin(), per_tok.end());
    printf("replay policy=%s cap=%ld mode=%s qd=%d tokens=%ld : miss_rate=%.4f misses/token=%.1f MB/token=%.1f "
           "io_ms/token mean=%.2f p50=%.2f p99=%.2f -> %.1f tok/s (I/O only), %.2f GB/s\n",
           policy.c_str(), cap, mode.c_str(), qd, Tm, (double) misses / n_meas, (double) misses / Tm, bytes / Tm / 1e6,
           el * 1e3 / Tm, per_tok[per_tok.size() / 2], per_tok[(size_t) (per_tok.size() * 0.99)], Tm / el, bytes / el / 1e9);
    return 0;
}
