// locsim: expert-cache simulation and locality statistics for MoE routing traces (fast path for the Python analysis).
//
//   locsim [options] trace1.route.npy [trace2.route.npy ...]
//     The traces (uint16 [T, L, k]) are concatenated in order into one access stream: token by token, layer by
//     layer, the k experts of a layer in router order. A cache key is a (layer, expert) pair.
//   options:
//     --caps F1,F2,..      cache capacities as fractions of all L*E pairs (global cache; per-layer: of E per layer)
//     --policies P1,..     lru, lfu, arc, belady (MIN with bypass), belady_nb (MIN, always admit), la<h> (lookahead-h)
//     --scope global|layer|both
//     --window W           also report misses per W-token window (positional analysis)
//     --stats              reuse distance in tokens, consecutive-token overlap, union (working set) growth
//     --stack              LRU stack-distance histogram of the global stream (hit-rate curve for every capacity)
//     --threads N          worker threads (default 4)
//
// Policies (all count a miss per access whose key is not resident; misses include compulsory first uses):
//   lru         least recently used, always admits
//   lfu         perfect LFU (frequency over the whole history, including evicted keys), ties -> least recent
//   arc         adaptive replacement cache (Megiddo & Modha), same algorithm as analysis/policies_extra.py
//   belady      Belady's MIN with bypass: on a miss the key used furthest in the future (possibly the accessed key
//               itself) is not kept -- the offline optimum, same as analysis/traces.py sim_belady
//   belady_nb   MIN without bypass (the accessed key is always admitted): the limit of la<h> for h -> infinity
//   dlfu<tau>   decayed LFU: frequency with exponential decay, half-life tau tokens (tau -> 0: LRU, -> inf: LFU)
//   la<h>       lookahead-h: exact knowledge of the keys used by the rest of the current token and the next h tokens;
//               evict the least recently used key that is not used within that horizon; if every resident key is
//               used within the horizon, evict the one used furthest (MIN within the horizon). Always admits.
//               (analysis/policies_extra.py falls back to plain LRU there and bounds the scan to 512 entries;
//               the difference only matters when nearly every resident key is inside the horizon.)

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static const int32_t NEVER = INT32_MAX;

// ------------------------------------------------------------------------------------------------ npy input
struct trace {
    int64_t T = 0;
    int L = 0, k = 0;
    std::vector<uint16_t> ids;   // [T][L][k]
};

static void load_npy_u16(const char * path, trace & tr) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    unsigned char magic[10];
    if (fread(magic, 1, 10, f) != 10 || memcmp(magic + 1, "NUMPY", 5) != 0) { fprintf(stderr, "%s: not npy\n", path); exit(1); }
    uint32_t hl;
    if (magic[6] == 1) hl = magic[8] | (magic[9] << 8);
    else { unsigned char b[2]; if (fread(b, 1, 2, f) != 2) exit(1); hl = magic[8] | (magic[9] << 8) | (b[0] << 16) | (b[1] << 24); }
    std::string hdr(hl, ' ');
    if (fread(&hdr[0], 1, hl, f) != hl) exit(1);
    if (hdr.find("'<u2'") == std::string::npos || hdr.find("False") == std::string::npos) { fprintf(stderr, "%s: need <u2 C-order\n", path); exit(1); }
    size_t a = hdr.find('('), b = hdr.find(')');
    long long T, L, k;
    if (sscanf(hdr.substr(a + 1, b - a - 1).c_str(), "%lld, %lld, %lld", &T, &L, &k) != 3) { fprintf(stderr, "%s: bad shape\n", path); exit(1); }
    if (tr.L == 0) { tr.L = (int) L; tr.k = (int) k; }
    if (tr.L != L || tr.k != k) { fprintf(stderr, "%s: shape mismatch\n", path); exit(1); }
    size_t old = tr.ids.size();
    tr.ids.resize(old + (size_t) T * L * k);
    if (fread(tr.ids.data() + old, 2, (size_t) T * L * k, f) != (size_t) T * L * k) { fprintf(stderr, "%s: short read\n", path); exit(1); }
    tr.T += T;
    fclose(f);
}

// ------------------------------------------------------------------------------------------------ data structures
struct dlist {   // intrusive doubly linked list over keys 0..n-1; head = least recent, tail = most recent
    std::vector<int32_t> prv, nxt;
    std::vector<uint8_t> in;
    int32_t head = -1, tail = -1;
    int size = 0;
    void init(int n) { prv.assign(n, -1); nxt.assign(n, -1); in.assign(n, 0); head = tail = -1; size = 0; }
    void push_back(int x) { prv[x] = tail; nxt[x] = -1; if (tail >= 0) nxt[tail] = x; else head = x; tail = x; in[x] = 1; ++size; }
    void remove(int x) {
        if (prv[x] >= 0) nxt[prv[x]] = nxt[x]; else head = nxt[x];
        if (nxt[x] >= 0) prv[nxt[x]] = prv[x]; else tail = prv[x];
        in[x] = 0; --size;
    }
    int pop_front() { int x = head; remove(x); return x; }
    void to_back(int x) { remove(x); push_back(x); }
};

template <class Before> struct iheap {   // indexed binary heap; before(a, b) = a belongs above b
    std::vector<int32_t> h, pos;
    Before before;
    explicit iheap(Before b) : before(b) {}
    void init(int n) { pos.assign(n, -1); h.clear(); }
    bool has(int x) const { return pos[x] >= 0; }
    int size() const { return (int) h.size(); }
    int top() const { return h[0]; }
    void sw(int i, int j) { std::swap(h[i], h[j]); pos[h[i]] = i; pos[h[j]] = j; }
    void up(int i) { while (i > 0) { int p = (i - 1) / 2; if (before(h[i], h[p])) { sw(i, p); i = p; } else break; } }
    void down(int i) {
        int n = (int) h.size();
        for (;;) {
            int l = 2 * i + 1, r = l + 1, b = i;
            if (l < n && before(h[l], h[b])) b = l;
            if (r < n && before(h[r], h[b])) b = r;
            if (b == i) break;
            sw(i, b); i = b;
        }
    }
    void push(int x) { pos[x] = (int) h.size(); h.push_back(x); up((int) h.size() - 1); }
    void remove(int x) {
        int i = pos[x], last = h.back();
        h.pop_back(); pos[x] = -1;
        if (i < (int) h.size()) { h[i] = last; pos[last] = i; up(i); down(pos[last]); }
    }
    void update(int x) { int i = pos[x]; up(i); down(pos[x]); }
};

// ------------------------------------------------------------------------------------------------ policies
// keys: access stream, n accesses, nk distinct key ids, cap slots, per_token accesses per token.
// miss_tok[t] += misses at token t (t = i / per_token + tok0).

static uint64_t sim_lru(const int32_t * keys, size_t n, int nk, int cap, int per_token, uint32_t * miss_tok) {
    dlist c; c.init(nk);
    uint64_t miss = 0;
    for (size_t i = 0; i < n; ++i) {
        int x = keys[i];
        if (c.in[x]) { c.to_back(x); continue; }
        ++miss; miss_tok[i / per_token]++;
        c.push_back(x);
        if (c.size > cap) c.pop_front();
    }
    return miss;
}

static uint64_t sim_lfu(const int32_t * keys, size_t n, int nk, int cap, int per_token, uint32_t * miss_tok) {
    std::vector<int64_t> cnt(nk, 0), last(nk, -1);
    auto before = [&](int a, int b) { return cnt[a] < cnt[b] || (cnt[a] == cnt[b] && last[a] < last[b]); };
    iheap<decltype(before)> hp(before); hp.init(nk);
    uint64_t miss = 0;
    for (size_t i = 0; i < n; ++i) {
        int x = keys[i];
        cnt[x]++; last[x] = (int64_t) i;
        if (hp.has(x)) { hp.update(x); continue; }
        ++miss; miss_tok[i / per_token]++;
        if (hp.size() >= cap) hp.remove(hp.top());
        hp.push(x);
    }
    return miss;
}

static uint64_t sim_arc(const int32_t * keys, size_t n, int nk, int cap, int per_token, uint32_t * miss_tok) {
    dlist T1, T2, B1, B2;
    T1.init(nk); T2.init(nk); B1.init(nk); B2.init(nk);
    int p = 0;
    uint64_t miss = 0;
    auto replace = [&](int x) {
        if (T1.size > 0 && ((B2.in[x] && T1.size == p) || T1.size > p)) { int y = T1.pop_front(); B1.push_back(y); }
        else if (T2.size > 0) { int y = T2.pop_front(); B2.push_back(y); }
        else if (T1.size > 0) { int y = T1.pop_front(); B1.push_back(y); }
    };
    for (size_t i = 0; i < n; ++i) {
        int x = keys[i];
        if (T1.in[x]) { T1.remove(x); T2.push_back(x); continue; }
        if (T2.in[x]) { T2.to_back(x); continue; }
        ++miss; miss_tok[i / per_token]++;
        if (B1.in[x]) {
            p = std::min(cap, p + std::max(1, B2.size / std::max(1, B1.size)));
            replace(x); B1.remove(x); T2.push_back(x);
        } else if (B2.in[x]) {
            p = std::max(0, p - std::max(1, B1.size / std::max(1, B2.size)));
            replace(x); B2.remove(x); T2.push_back(x);
        } else {
            const int l1 = T1.size + B1.size, tot = l1 + T2.size + B2.size;
            if (l1 == cap) {
                if (T1.size < cap) { B1.pop_front(); replace(x); }
                else T1.pop_front();
            } else if (l1 < cap && tot >= cap) {
                if (tot == 2 * cap) B2.pop_front();
                replace(x);
            }
            T1.push_back(x);
        }
    }
    return miss;
}

static std::vector<int32_t> next_use(const int32_t * keys, size_t n, int nk) {
    std::vector<int32_t> nxt(n), last(nk, NEVER);
    for (size_t i = n; i-- > 0;) { nxt[i] = last[keys[i]]; last[keys[i]] = (int32_t) i; }
    return nxt;
}

static uint64_t sim_belady(const int32_t * keys, const int32_t * nxt, size_t n, int nk, int cap, int per_token, bool bypass,
                           uint32_t * miss_tok) {
    std::vector<int32_t> nu(nk, NEVER);
    auto before = [&](int a, int b) { return nu[a] > nu[b] || (nu[a] == nu[b] && a < b); };   // max-heap on next use
    iheap<decltype(before)> hp(before); hp.init(nk);
    uint64_t miss = 0;
    for (size_t i = 0; i < n; ++i) {
        int x = keys[i];
        if (hp.has(x)) { nu[x] = nxt[i]; hp.update(x); continue; }
        ++miss; miss_tok[i / per_token]++;
        if (hp.size() >= cap) {
            int y = hp.top();
            if (bypass && nxt[i] >= nu[y]) continue;   // the accessed key is reused last: do not keep it
            hp.remove(y);
        }
        nu[x] = nxt[i];
        hp.push(x);
    }
    return miss;
}

static uint64_t sim_lookahead(const int32_t * keys, const int32_t * nxt, size_t n, int nk, int cap, int per_token, int h,
                              uint32_t * miss_tok) {
    // far: resident keys whose next use lies beyond the horizon at the time they were last accessed, in recency
    // order (keys enter it only when accessed, so appending keeps it ordered); keys in `far` whose next use has
    // since come within the horizon are moved to `near` lazily when they reach the LRU end.
    std::vector<int32_t> nu(nk, NEVER);
    std::vector<uint8_t> res(nk, 0);
    dlist far; far.init(nk);
    auto before = [&](int a, int b) { return nu[a] > nu[b] || (nu[a] == nu[b] && a < b); };
    iheap<decltype(before)> near(before); near.init(nk);
    int size = 0;
    uint64_t miss = 0;
    for (size_t i = 0; i < n; ++i) {
        const int x = keys[i];
        const int64_t t = (int64_t) (i / per_token);
        const int64_t limit = (t + 1 + h) * (int64_t) per_token;   // accesses < limit are inside the horizon
        auto place = [&](int key, int32_t next) {
            nu[key] = next;
            if ((int64_t) next < limit) near.push(key); else far.push_back(key);
        };
        if (res[x]) {
            if (far.in[x]) far.remove(x); else near.remove(x);
            place(x, nxt[i]);
            continue;
        }
        ++miss; miss_tok[t]++;
        if (size >= cap) {
            int victim = -1;
            while (far.size > 0) {
                int y = far.head;
                far.remove(y);
                if ((int64_t) nu[y] >= limit) { victim = y; break; }
                near.push(y);   // came within the horizon since its last access
            }
            if (victim < 0) { victim = near.top(); near.remove(victim); }
            res[victim] = 0; --size;
        }
        res[x] = 1; ++size;
        place(x, nxt[i]);
    }
    return miss;
}

static uint64_t sim_dlfu(const int32_t * keys, size_t n, int nk, int cap, int per_token, double tau, uint32_t * miss_tok) {
    // decayed LFU: score = sum over past accesses of 2^(-(t_now - t_access) / tau), t in tokens; evict the lowest score.
    // Ordering by score is time-invariant in the log domain: Ls = log2(score at time t) + t / tau.
    std::vector<double> Ls(nk, -INFINITY);
    auto before = [&](int a, int b) { return Ls[a] < Ls[b] || (Ls[a] == Ls[b] && a < b); };
    iheap<decltype(before)> hp(before); hp.init(nk);
    uint64_t miss = 0;
    for (size_t i = 0; i < n; ++i) {
        const int x = keys[i];
        const double tt = (double) (i / per_token) / tau;
        const double a = Ls[x];
        Ls[x] = std::isinf(a) ? tt : std::max(a, tt) + std::log2(1.0 + std::exp2(-std::fabs(a - tt)));
        if (hp.has(x)) { hp.update(x); continue; }
        ++miss; miss_tok[i / per_token]++;
        if (hp.size() >= cap) hp.remove(hp.top());
        hp.push(x);
    }
    return miss;
}

// ------------------------------------------------------------------------------------------------ statistics
struct fenwick {
    std::vector<int32_t> t;
    explicit fenwick(size_t n) : t(n + 1, 0) {}
    void add(size_t i, int v) { for (++i; i < t.size(); i += i & (~i + 1)) t[i] += v; }
    int64_t sum(size_t i) const { int64_t s = 0; for (++i; i > 0; i -= i & (~i + 1)) s += t[i]; return s; }   // [0, i]
};

int main(int argc, char ** argv) {
    std::vector<double> caps;
    std::vector<std::string> policies;
    std::string scope = "both";
    int window = 0, threads = 4;
    bool stats = false, stack = false;
    trace tr;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--caps") { std::string s = next(); size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); caps.push_back(atof(s.substr(p, q - p).c_str())); p = q + 1; } }
        else if (a == "--policies") { std::string s = next(); size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); policies.push_back(s.substr(p, q - p)); p = q + 1; } }
        else if (a == "--scope") scope = next();
        else if (a == "--window") window = atoi(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--stats") stats = true;
        else if (a == "--stack") stack = true;
        else load_npy_u16(argv[i], tr);
    }
    if (tr.T == 0) { fprintf(stderr, "no trace\n"); return 1; }
    const int64_t T = tr.T;
    const int L = tr.L, k = tr.k;
    int E = 0;
    for (uint16_t v : tr.ids) E = std::max(E, (int) v + 1);
    // the expert count is not stored in the trace: take it from the environment if given (ids may not reach E-1)
    if (getenv("LOCSIM_E")) E = std::max(E, atoi(getenv("LOCSIM_E")));
    const int nk = L * E;
    const size_t N = (size_t) T * L * k;

    // global stream and next uses
    std::vector<int32_t> gkeys(N);
    for (int64_t t = 0; t < T; ++t)
        for (int l = 0; l < L; ++l)
            for (int j = 0; j < k; ++j) {
                size_t i = ((size_t) t * L + l) * k + j;
                gkeys[i] = l * E + tr.ids[i];
            }
    const bool want_global = scope == "global" || scope == "both";
    const bool want_layer = scope == "layer" || scope == "both";
    bool need_next = false;
    for (auto & p : policies) need_next |= p == "belady" || p == "belady_nb" || p.rfind("la", 0) == 0;
    std::vector<int32_t> gnext;
    if (want_global && need_next) gnext = next_use(gkeys.data(), N, nk);
    // per-layer streams
    std::vector<std::vector<int32_t>> lkeys, lnext;
    if (want_layer && !policies.empty()) {
        lkeys.resize(L); lnext.resize(L);
        for (int l = 0; l < L; ++l) {
            lkeys[l].resize((size_t) T * k);
            for (int64_t t = 0; t < T; ++t)
                for (int j = 0; j < k; ++j) lkeys[l][(size_t) t * k + j] = tr.ids[((size_t) t * L + l) * k + j];
            if (need_next) lnext[l] = next_use(lkeys[l].data(), lkeys[l].size(), E);
        }
    }

    struct task { bool global; std::string pol; int ci; int cap; uint64_t miss = 0; std::vector<uint32_t> mt; };
    std::vector<task> tasks;
    for (int g = 0; g < 2; ++g) {
        if ((g == 0 && !want_global) || (g == 1 && !want_layer)) continue;
        for (auto & p : policies)
            for (size_t ci = 0; ci < caps.size(); ++ci) {
                task tk; tk.global = g == 0; tk.pol = p; tk.ci = (int) ci;
                tk.cap = g == 0 ? std::max(1, (int) std::lround(caps[ci] * nk)) : std::max(1, (int) std::lround(caps[ci] * E));
                tasks.push_back(tk);
            }
    }
    auto run_one = [&](const int32_t * keys, const int32_t * nxt, size_t n, int nkeys, int cap, int per_token,
                       const std::string & p, uint32_t * mt) -> uint64_t {
        if (p == "lru") return sim_lru(keys, n, nkeys, cap, per_token, mt);
        if (p == "lfu") return sim_lfu(keys, n, nkeys, cap, per_token, mt);
        if (p == "arc") return sim_arc(keys, n, nkeys, cap, per_token, mt);
        if (p == "belady") return sim_belady(keys, nxt, n, nkeys, cap, per_token, true, mt);
        if (p == "belady_nb") return sim_belady(keys, nxt, n, nkeys, cap, per_token, false, mt);
        if (p.rfind("dlfu", 0) == 0) return sim_dlfu(keys, n, nkeys, cap, per_token, atof(p.c_str() + 4), mt);
        if (p.rfind("la", 0) == 0) return sim_lookahead(keys, nxt, n, nkeys, cap, per_token, atoi(p.c_str() + 2), mt);
        fprintf(stderr, "unknown policy %s\n", p.c_str()); exit(1);
    };
    std::atomic<size_t> next_task{0};
    auto worker = [&]() {
        for (;;) {
            size_t ti = next_task++;
            if (ti >= tasks.size()) break;
            task & tk = tasks[ti];
            tk.mt.assign(T, 0);
            if (tk.global) {
                tk.miss = run_one(gkeys.data(), gnext.empty() ? nullptr : gnext.data(), N, nk, tk.cap, L * k, tk.pol, tk.mt.data());
            } else {
                for (int l = 0; l < L; ++l)
                    tk.miss += run_one(lkeys[l].data(), lnext.empty() || lnext[l].empty() ? nullptr : lnext[l].data(),
                                       lkeys[l].size(), E, tk.cap, k, tk.pol, tk.mt.data());
            }
        }
    };
    std::vector<std::thread> th;
    for (int w = 0; w < std::max(1, threads); ++w) th.emplace_back(worker);
    for (auto & x : th) x.join();

    // ------------------------------------------------------------------ output
    printf("{\"T\": %lld, \"L\": %d, \"E\": %d, \"k\": %d, \"accesses\": %zu", (long long) T, L, E, k, N);
    {
        std::vector<uint8_t> seen(nk, 0);
        int64_t d = 0;
        for (size_t i = 0; i < N; ++i) if (!seen[gkeys[i]]) { seen[gkeys[i]] = 1; ++d; }
        printf(", \"distinct_pairs\": %lld", (long long) d);
    }
    printf(", \"caps\": [");
    for (size_t i = 0; i < caps.size(); ++i) printf("%s%g", i ? ", " : "", caps[i]);
    printf("]");
    for (int g = 0; g < 2; ++g) {
        if ((g == 0 && !want_global) || (g == 1 && !want_layer) || policies.empty()) continue;
        printf(", \"%s\": {\"slots\": [", g == 0 ? "global" : "per_layer");
        bool first = true;
        for (auto & tk : tasks) if (tk.global == (g == 0) && tk.pol == policies[0]) { printf("%s%d", first ? "" : ", ", tk.cap); first = false; }
        printf("], \"misses\": {");
        for (size_t pi = 0; pi < policies.size(); ++pi) {
            printf("%s\"%s\": [", pi ? ", " : "", policies[pi].c_str());
            first = true;
            for (auto & tk : tasks) if (tk.global == (g == 0) && tk.pol == policies[pi]) { printf("%s%llu", first ? "" : ", ", (unsigned long long) tk.miss); first = false; }
            printf("]");
        }
        printf("}");
        if (window > 0) {
            printf(", \"window\": %d, \"window_misses\": {", window);
            for (size_t pi = 0; pi < policies.size(); ++pi) {
                printf("%s\"%s\": [", pi ? ", " : "", policies[pi].c_str());
                first = true;
                for (auto & tk : tasks) {
                    if (tk.global != (g == 0) || tk.pol != policies[pi]) continue;
                    printf("%s[", first ? "" : ", "); first = false;
                    for (int64_t w0 = 0; w0 < T; w0 += window) {
                        uint64_t s = 0;
                        for (int64_t t = w0; t < std::min<int64_t>(T, w0 + window); ++t) s += tk.mt[t];
                        printf("%s%llu", w0 ? ", " : "", (unsigned long long) s);
                    }
                    printf("]");
                }
                printf("]");
            }
            printf("}");
        }
        printf("}");
    }
    if (stats) {
        // reuse distance in tokens per (layer, expert) pair: bins d = 1, 2, 3-4, 5-8, ..., and first uses
        std::vector<int64_t> last(nk, -1);
        std::vector<uint64_t> bins(40, 0);
        uint64_t first_use = 0;
        for (int64_t t = 0; t < T; ++t)
            for (int l = 0; l < L; ++l)
                for (int j = 0; j < k; ++j) {
                    int key = gkeys[((size_t) t * L + l) * k + j];
                    if (last[key] < 0) ++first_use;
                    else {
                        int64_t d = t - last[key];
                        int b = 0;
                        while ((1LL << b) < d) ++b;   // bin b holds 2^(b-1) < d <= 2^b
                        bins[b]++;
                    }
                    last[key] = t;
                }
        printf(", \"reuse_tokens\": {\"first_use\": %llu, \"le_pow2\": [", (unsigned long long) first_use);
        for (int b = 0; b < 40; ++b) printf("%s%llu", b ? ", " : "", (unsigned long long) bins[b]);
        printf("]}");
        // consecutive-token overlap per layer, and mean distinct experts per layer in w-token windows
        std::vector<double> ov(L, 0);
        std::vector<uint8_t> prev((size_t) L * E, 0), cur((size_t) L * E, 0);
        for (int64_t t = 0; t < T; ++t) {
            std::fill(cur.begin(), cur.end(), 0);
            for (int l = 0; l < L; ++l)
                for (int j = 0; j < k; ++j) {
                    int e = tr.ids[((size_t) t * L + l) * k + j];
                    cur[(size_t) l * E + e] = 1;
                    if (t > 0 && prev[(size_t) l * E + e]) ov[l] += 1;
                }
            std::swap(prev, cur);
        }
        printf(", \"overlap_per_layer\": [");
        for (int l = 0; l < L; ++l) printf("%s%.5f", l ? ", " : "", T > 1 ? ov[l] / ((T - 1) * (double) k) : 0.0);
        printf("], \"union_window\": {");
        const int ws[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
        bool firstw = true;
        for (int w : ws) {
            if (w > T) break;
            std::vector<int64_t> stamp((size_t) L * E, -1);
            double tot = 0; int64_t nwin = 0;
            for (int64_t w0 = 0; w0 + w <= T; w0 += w, ++nwin) {
                int64_t distinct = 0;
                for (int64_t t = w0; t < w0 + w; ++t)
                    for (int l = 0; l < L; ++l)
                        for (int j = 0; j < k; ++j) {
                            size_t key = (size_t) l * E + tr.ids[((size_t) t * L + l) * k + j];
                            if (stamp[key] != w0) { stamp[key] = w0; ++distinct; }
                        }
                tot += (double) distinct / L;
            }
            printf("%s\"%d\": %.4f", firstw ? "" : ", ", w, nwin ? tot / nwin : 0.0);
            firstw = false;
        }
        printf("}");
    }
    if (stack) {
        // LRU stack distance (1 = most recent key) of every global access; misses at capacity c = accesses with
        // distance > c (plus first uses).
        fenwick fw(N);
        std::vector<int64_t> lastpos(nk, -1);
        std::vector<uint64_t> hist(nk + 1, 0);
        uint64_t cold = 0;
        for (size_t i = 0; i < N; ++i) {
            int x = gkeys[i];
            if (lastpos[x] < 0) ++cold;
            else {
                int64_t d = fw.sum(i) - fw.sum(lastpos[x]) + 1;   // keys whose last access is after x's, plus x
                hist[std::min<int64_t>(d, nk)]++;
                fw.add(lastpos[x], -1);
            }
            fw.add(i, 1);
            lastpos[x] = (int64_t) i;
        }
        printf(", \"stack\": {\"cold\": %llu, \"hist\": [", (unsigned long long) cold);
        for (int d = 0; d <= nk; ++d) printf("%s%llu", d ? ", " : "", (unsigned long long) hist[d]);
        printf("]}");
    }
    printf("}\n");
    return 0;
}
