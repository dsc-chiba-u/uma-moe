// uma_bw: memory-path microbenchmarks for a unified-memory (UMA) CPU+GPU machine (DGX Spark / GB10).
//
//   uma_bw gpu-seq    <alloc> <GiB>                         GPU streaming read bandwidth
//   uma_bw gpu-gather <alloc> <GiB> <expert_KiB> <per_step> <steps>
//                                                           GPU reads <per_step> random "experts" per kernel
//   uma_bw cpu-seq    <threads> <GiB> <cpus>                CPU streaming read bandwidth (cpus: all|big|little)
//   uma_bw concurrent <alloc> <GiB> <cpu_threads> <cpus> <seconds>
//                                                           GPU and CPU readers at the same time
//
// <alloc>: dev | pinned | registered | pageable | pageable-thp | managed | file:<path>
//   dev        cudaMalloc
//   pinned     cudaMallocHost (GPU reads it zero-copy through its device pointer)
//   registered malloc + cudaHostRegister (what Strata does for its expert arena)
//   pageable   plain malloc, touched by the CPU; the GPU reads it through the host page tables (ATS/HMM)
//   pageable-thp  same, with madvise(MADV_HUGEPAGE)
//   managed    cudaMallocManaged
//   file:PATH  read-only mmap of an existing file (page cache), touched by the CPU first; GPU reads it directly
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

__global__ void read_seq(const int4* __restrict__ p, size_t n, int* out) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        int4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) out[0] = 1;
}

// each block-row reads one expert: experts are `ex_n` int4 long, offsets[] picks which
__global__ void read_gather(const int4* __restrict__ base, const int* __restrict__ ids, int n_ids, size_t ex_n, int* out) {
    int4 acc = make_int4(0, 0, 0, 0);
    const int e = blockIdx.y;
    if (e < n_ids) {
        const int4* p = base + (size_t) ids[e] * ex_n;
        for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < ex_n; i += (size_t) gridDim.x * blockDim.x) {
            int4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) out[0] = 1;
}

struct Buf { void* p = nullptr; size_t bytes = 0; std::string kind; int fd = -1; size_t map_bytes = 0; };

static long smaps_kb(const char* field) {   // a field of /proc/self/smaps_rollup, in kB
    FILE* f = fopen("/proc/self/smaps_rollup", "r"); char line[256]; long v = -1;
    size_t n = strlen(field);
    while (f && fgets(line, sizeof line, f)) if (!strncmp(line, field, n)) { v = atol(line + n + 1); break; }
    if (f) fclose(f); return v;
}

static void cpu_touch(const void* p, size_t bytes) {
    volatile uint64_t s = 0; const uint64_t* q = (const uint64_t*) p;
    for (size_t i = 0; i < bytes / 8; i += 512) s += q[i];   // one read per 4 KiB page
    (void) s;
}

static Buf alloc_buf(const std::string& kind, size_t bytes) {
    Buf b; b.bytes = bytes; b.kind = kind;
    if (kind == "dev") { CK(cudaMalloc(&b.p, bytes)); CK(cudaMemset(b.p, 1, bytes)); }
    else if (kind == "pinned") { CK(cudaMallocHost(&b.p, bytes)); memset(b.p, 1, bytes); }
    else if (kind == "registered") {
        b.p = aligned_alloc(1 << 21, bytes); memset(b.p, 1, bytes);
        CK(cudaHostRegister(b.p, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
    } else if (kind == "pageable" || kind == "pageable-thp") {
        b.p = aligned_alloc(1 << 21, bytes);
        if (kind == "pageable-thp") madvise(b.p, bytes, MADV_HUGEPAGE);
        memset(b.p, 1, bytes);
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
        if (kind == "pageable-thp") {
            int rc = madvise(b.p, bytes, MADV_COLLAPSE);
            fprintf(stderr, "[thp] MADV_COLLAPSE rc=%d, AnonHugePages %ld kB of %zu kB\n", rc, smaps_kb("AnonHugePages:"), bytes / 1024);
        }
    } else if (kind == "managed") { CK(cudaMallocManaged(&b.p, bytes)); memset(b.p, 1, bytes); }
    else if (kind.rfind("file:", 0) == 0 || kind.rfind("filethp:", 0) == 0) {
        bool thp = kind.rfind("filethp:", 0) == 0;
        std::string path = kind.substr(thp ? 8 : 5);
        b.fd = open(path.c_str(), O_RDONLY); if (b.fd < 0) { perror(path.c_str()); exit(1); }
        off_t sz = lseek(b.fd, 0, SEEK_END); if ((size_t) sz < bytes) { fprintf(stderr, "file too small\n"); exit(1); }
        b.map_bytes = bytes;
        b.p = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, b.fd, 0); if (b.p == MAP_FAILED) { perror("mmap"); exit(1); }
        cpu_touch(b.p, bytes);   // make it page-cache resident
        if (thp) {
            madvise(b.p, bytes, MADV_HUGEPAGE);
            int rc = madvise(b.p, bytes, MADV_COLLAPSE);
            fprintf(stderr, "[thp] file MADV_COLLAPSE rc=%d (%s), FilePmdMapped %ld kB of %zu kB\n", rc, rc ? strerror(errno) : "ok",
                    smaps_kb("FilePmdMapped:"), bytes / 1024);
        }
    } else { fprintf(stderr, "unknown alloc %s\n", kind.c_str()); exit(1); }
    return b;
}

static const void* gpu_ptr(const Buf& b) {
    if (b.kind == "pinned" || b.kind == "registered") { void* d; CK(cudaHostGetDevicePointer(&d, b.p, 0)); return d; }
    return b.p;
}

static void free_buf(Buf& b) {
    if (b.kind == "dev") cudaFree(b.p);
    else if (b.kind == "pinned") cudaFreeHost(b.p);
    else if (b.kind == "registered") { cudaHostUnregister(b.p); free(b.p); }
    else if (b.kind == "managed") cudaFree(b.p);
    else if (b.fd >= 0) { munmap(b.p, b.map_bytes); close(b.fd); }
    else free(b.p);
}

static int n_sms() { int s; CK(cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, 0)); return s; }

static void gpu_seq(const std::string& kind, double gib) {
    size_t bytes = (size_t) (gib * (1ull << 30)); bytes &= ~(size_t) 4095;
    Buf b = alloc_buf(kind, bytes);
    const int4* p = (const int4*) gpu_ptr(b); size_t n = bytes / 16; int* o; CK(cudaMalloc(&o, 4));
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e); float ms;
    cudaEventRecord(s); read_seq<<<n_sms() * 8, 512>>>(p, n, o); cudaEventRecord(e); CK(cudaEventSynchronize(e));
    cudaEventElapsedTime(&ms, s, e);
    printf("gpu-seq alloc=%s GiB=%.1f first_pass=%.1f GB/s", kind.c_str(), gib, bytes / (ms / 1e3) / 1e9);
    std::vector<double> r;
    for (int it = 0; it < 7; ++it) {
        cudaEventRecord(s); read_seq<<<n_sms() * 8, 512>>>(p, n, o); cudaEventRecord(e); CK(cudaEventSynchronize(e));
        cudaEventElapsedTime(&ms, s, e); r.push_back(bytes / (ms / 1e3) / 1e9);
    }
    std::sort(r.begin(), r.end());
    printf(" median=%.1f GB/s max=%.1f GB/s\n", r[r.size() / 2], r.back());
    cudaFree(o); free_buf(b);
}

static void gpu_gather(const std::string& kind, double gib, int ex_kib, int per_step, int steps) {
    size_t ex_bytes = (size_t) ex_kib * 1024; ex_bytes &= ~(size_t) 15;
    size_t bytes = (size_t) (gib * (1ull << 30)); int n_ex = (int) (bytes / ex_bytes); bytes = (size_t) n_ex * ex_bytes;
    Buf b = alloc_buf(kind, bytes);
    const int4* p = (const int4*) gpu_ptr(b); int* o; CK(cudaMalloc(&o, 4));
    std::mt19937 rng(42);
    std::vector<int> ids((size_t) per_step * steps);
    for (int s = 0; s < steps; ++s) {           // distinct experts inside a step, like one layer's top-k
        std::vector<int> pick; while ((int) pick.size() < per_step) { int x = rng() % n_ex; if (std::find(pick.begin(), pick.end(), x) == pick.end()) pick.push_back(x); }
        std::copy(pick.begin(), pick.end(), ids.begin() + (size_t) s * per_step);
    }
    int* d_ids; CK(cudaMalloc(&d_ids, ids.size() * sizeof(int))); CK(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    size_t ex_n = ex_bytes / 16;
    int bx = std::max(1, (n_sms() * 8) / per_step);
    dim3 grid(bx, per_step);
    for (int s = 0; s < std::min(steps, 16); ++s) read_gather<<<grid, 256>>>(p, d_ids + (size_t) s * per_step, per_step, ex_n, o);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s0, e0; cudaEventCreate(&s0); cudaEventCreate(&e0); float ms;
    cudaEventRecord(s0);
    for (int s = 0; s < steps; ++s) read_gather<<<grid, 256>>>(p, d_ids + (size_t) s * per_step, per_step, ex_n, o);
    cudaEventRecord(e0); CK(cudaEventSynchronize(e0)); cudaEventElapsedTime(&ms, s0, e0);
    double moved = (double) ex_bytes * per_step * steps;
    printf("gpu-gather alloc=%s pool=%.1fGiB expert=%dKiB per_step=%d steps=%d -> %.1f GB/s, %.1f us/step\n",
           kind.c_str(), gib, ex_kib, per_step, steps, moved / (ms / 1e3) / 1e9, ms * 1e3 / steps);
    cudaFree(d_ids); cudaFree(o); free_buf(b);
}

static std::vector<int> cpu_list(const std::string& which) {
    std::vector<int> big = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19}, little = {0, 1, 2, 3, 4, 10, 11, 12, 13, 14};
    if (which == "big") return big;
    if (which == "little") return little;
    std::vector<int> all = big; all.insert(all.end(), little.begin(), little.end()); return all;
}

static void pin_to(int cpu) { cpu_set_t c; CPU_ZERO(&c); CPU_SET(cpu, &c); pthread_setaffinity_np(pthread_self(), sizeof c, &c); }

// each CPU thread streams over its own slice repeatedly until stop; returns bytes read
static void cpu_reader(const uint64_t* p, size_t n, int cpu, std::atomic<bool>* stop, std::atomic<uint64_t>* total, int passes) {
    pin_to(cpu);
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, bytes = 0;
    for (int pass = 0; passes < 0 ? !stop->load(std::memory_order_relaxed) : pass < passes; ++pass) {
        for (size_t i = 0; i + 8 <= n; i += 8) { a0 ^= p[i] + p[i + 4]; a1 ^= p[i + 1] + p[i + 5]; a2 ^= p[i + 2] + p[i + 6]; a3 ^= p[i + 3] + p[i + 7]; }
        bytes += n * 8;
    }
    if ((a0 ^ a1 ^ a2 ^ a3) == 0x1234567) printf("!");
    total->fetch_add(bytes);
}

static void cpu_seq(int threads, double gib, const std::string& which) {
    size_t bytes = (size_t) (gib * (1ull << 30));
    uint64_t* p = (uint64_t*) aligned_alloc(1 << 21, bytes); memset(p, 1, bytes);
    auto cpus = cpu_list(which);
    std::vector<double> r;
    for (int it = 0; it < 5; ++it) {
        std::atomic<bool> stop{false}; std::atomic<uint64_t> total{0};
        std::vector<std::thread> th; size_t per = bytes / 8 / threads;
        auto a = Clock::now();
        for (int t = 0; t < threads; ++t) th.emplace_back(cpu_reader, p + per * t, per, cpus[t % cpus.size()], &stop, &total, 1);
        for (auto& x : th) x.join();
        r.push_back(total.load() / secs(a, Clock::now()) / 1e9);
    }
    std::sort(r.begin(), r.end());
    printf("cpu-seq threads=%d cpus=%s GiB=%.1f median=%.1f GB/s max=%.1f GB/s\n", threads, which.c_str(), gib, r[r.size() / 2], r.back());
    free(p);
}

static void concurrent(const std::string& kind, double gib, int threads, const std::string& which, double seconds) {
    size_t bytes = (size_t) (gib * (1ull << 30)); bytes &= ~(size_t) 4095;
    Buf b; const int4* gp = nullptr;
    if (kind != "none") { b = alloc_buf(kind, bytes); gp = (const int4*) gpu_ptr(b); }
    size_t n = bytes / 16; int* o; CK(cudaMalloc(&o, 4));
    size_t cbytes = (size_t) 8 << 30; uint64_t* cp = (uint64_t*) aligned_alloc(1 << 21, cbytes); memset(cp, 1, cbytes);
    auto cpus = cpu_list(which);
    std::atomic<bool> stop{false}; std::atomic<uint64_t> total{0};
    std::vector<std::thread> th; size_t per = cbytes / 8 / std::max(1, threads);
    auto a = Clock::now();
    for (int t = 0; t < threads; ++t) th.emplace_back(cpu_reader, cp + per * t, per, cpus[t % cpus.size()], &stop, &total, -1);
    uint64_t gbytes = 0;
    if (kind != "none") {
        while (secs(a, Clock::now()) < seconds) { read_seq<<<n_sms() * 8, 512>>>(gp, n, o); CK(cudaDeviceSynchronize()); gbytes += bytes; }
    } else {
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    }
    double el = secs(a, Clock::now());
    stop = true; for (auto& x : th) x.join();
    double cel = secs(a, Clock::now());
    printf("concurrent gpu_alloc=%s cpu_threads=%d cpus=%s : GPU %.1f GB/s, CPU %.1f GB/s, sum %.1f GB/s\n", kind.c_str(), threads,
           which.c_str(), gbytes / el / 1e9, total.load() / cel / 1e9, gbytes / el / 1e9 + total.load() / cel / 1e9);
    free(cp); cudaFree(o); if (kind != "none") free_buf(b);
}

__global__ void empty_kernel() {}

// GPU rings a mapped host flag, the host answers on another mapped flag, n times (Strata's per-layer doorbell)
__global__ void pingpong(volatile unsigned* to_host, volatile unsigned* to_dev, int n) {
    for (int i = 1; i <= n; ++i) {
        *to_host = i; __threadfence_system();
        while (*to_dev != (unsigned) i) {}
    }
}

static void launch_lat() {
    cudaStream_t st; CK(cudaStreamCreate(&st));
    empty_kernel<<<1, 1, 0, st>>>(); CK(cudaStreamSynchronize(st));
    const int N = 20000;
    auto a = Clock::now();
    for (int i = 0; i < N; ++i) { empty_kernel<<<1, 1, 0, st>>>(); CK(cudaStreamSynchronize(st)); }
    double sync_us = secs(a, Clock::now()) / N * 1e6;
    a = Clock::now();
    for (int i = 0; i < N; ++i) empty_kernel<<<1, 1, 0, st>>>();
    CK(cudaStreamSynchronize(st));
    double async_us = secs(a, Clock::now()) / N * 1e6;
    // a captured graph of 100 empty kernels, replayed
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < 100; ++i) empty_kernel<<<1, 1, 0, st>>>();
    CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    a = Clock::now();
    for (int i = 0; i < 200; ++i) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    double graph_us = secs(a, Clock::now()) / (200 * 100) * 1e6;
    // GPU<->host doorbell round trip through mapped pinned memory
    unsigned *h_to_host, *h_to_dev; CK(cudaHostAlloc(&h_to_host, 64, cudaHostAllocMapped)); CK(cudaHostAlloc(&h_to_dev, 64, cudaHostAllocMapped));
    *h_to_host = 0; *h_to_dev = 0;
    unsigned *d_to_host, *d_to_dev; CK(cudaHostGetDevicePointer(&d_to_host, h_to_host, 0)); CK(cudaHostGetDevicePointer(&d_to_dev, h_to_dev, 0));
    const int R = 20000;
    a = Clock::now();
    pingpong<<<1, 1, 0, st>>>(d_to_host, d_to_dev, R);
    for (int i = 1; i <= R; ++i) {
        while (*(volatile unsigned*) h_to_host != (unsigned) i) {}
        *(volatile unsigned*) h_to_dev = i;
    }
    CK(cudaStreamSynchronize(st));
    double rt_us = secs(a, Clock::now()) / R * 1e6;
    printf("launch: launch+sync %.2f us, async launch %.2f us/kernel, graph replay %.3f us/kernel, GPU->host->GPU doorbell %.2f us\n",
           sync_us, async_us, graph_us, rt_us);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "see the header of uma_bw.cu\n"); return 2; }
    std::string m = argv[1];
    if (m == "gpu-seq" && argc >= 4) gpu_seq(argv[2], atof(argv[3]));
    else if (m == "gpu-gather" && argc >= 7) gpu_gather(argv[2], atof(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    else if (m == "cpu-seq" && argc >= 5) cpu_seq(atoi(argv[2]), atof(argv[3]), argv[4]);
    else if (m == "launch") launch_lat();
    else if (m == "concurrent" && argc >= 7) concurrent(argv[2], atof(argv[3]), atoi(argv[4]), argv[5], atof(argv[6]));
    else { fprintf(stderr, "bad arguments\n"); return 2; }
    return 0;
}
