// nvme_bench: read-path microbenchmarks for streaming MoE experts from NVMe.
//
//   nvme_bench seq   <file> <block_KiB> <threads> <GiB>          O_DIRECT sequential reads, one region per thread
//   nvme_bench rand  <file> <block_KiB> <threads> <reads>        O_DIRECT random aligned reads; threads = queue depth
//   nvme_bench mmap  <file> <block_KiB> <threads> <reads> <adv>  page-fault reads of random blocks through mmap
//                                                                 adv: none (MADV_RANDOM: 4 KiB faults, no readahead)
//                                                                      willneed (MADV_RANDOM + MADV_WILLNEED per block)
//                                                                      normal (kernel default readahead)
//                                                                      seq (MADV_SEQUENTIAL + POSIX_FADV_SEQUENTIAL, as llama.cpp)
//   nvme_bench evict <file>                                      drop the file from the page cache (fadvise DONTNEED)
//
// The file is only read. Every mode except `evict` first evicts the file so reads come from the device.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static size_t file_size(int fd) { struct stat st; fstat(fd, &st); return (size_t) st.st_size; }

static void evict(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}

static void report(const char* mode, const char* path, size_t block, int threads, double bytes, double el, std::vector<double>& lat) {
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double p) { return lat.empty() ? 0.0 : lat[std::min(lat.size() - 1, (size_t) (p * lat.size()))]; };
    printf("%s block=%zuKiB threads=%d : %.2f GB/s, %.0f IOPS, lat p50=%.0fus p90=%.0fus p99=%.0fus  (%s)\n", mode, block / 1024,
           threads, bytes / el / 1e9, lat.size() / el, pct(0.5), pct(0.9), pct(0.99), path);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "see the header of nvme_bench.cpp\n"); return 2; }
    std::string mode = argv[1]; const char* path = argv[2];
    if (mode == "evict") { evict(path); return 0; }
    if (argc < 6) { fprintf(stderr, "bad arguments\n"); return 2; }
    size_t block = (size_t) atoi(argv[3]) * 1024; int threads = atoi(argv[4]);
    evict(path);
    if (mode == "seq") {
        double gib = atof(argv[5]); size_t total = (size_t) (gib * (1ull << 30));
        int fd = open(path, O_RDONLY | O_DIRECT); size_t fsz = file_size(fd);
        total = std::min(total, fsz / block * block); size_t per = total / threads / block * block;
        std::vector<std::vector<double>> lats(threads);
        auto a = Clock::now();
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t) th.emplace_back([&, t] {
            void* buf; posix_memalign(&buf, 4096, block);
            for (size_t off = 0; off < per; off += block) {
                auto s = Clock::now();
                if (pread(fd, buf, block, (off_t) (per * t + off)) != (ssize_t) block) { perror("pread"); exit(1); }
                lats[t].push_back(secs(s, Clock::now()) * 1e6);
            }
            free(buf);
        });
        for (auto& x : th) x.join();
        double el = secs(a, Clock::now());
        std::vector<double> lat; for (auto& v : lats) lat.insert(lat.end(), v.begin(), v.end());
        report("seq", path, block, threads, (double) per * threads, el, lat);
        close(fd);
    } else if (mode == "rand") {
        long reads = atol(argv[5]);
        int fd = open(path, O_RDONLY | O_DIRECT); size_t fsz = file_size(fd); size_t nblk = fsz / block;
        std::atomic<long> next{0};
        std::vector<std::vector<double>> lats(threads);
        auto a = Clock::now();
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t) th.emplace_back([&, t] {
            void* buf; posix_memalign(&buf, 4096, block);
            std::mt19937_64 rng(1234 + t);
            while (next.fetch_add(1) < reads) {
                size_t off = (rng() % nblk) * block;
                auto s = Clock::now();
                if (pread(fd, buf, block, (off_t) off) != (ssize_t) block) { perror("pread"); exit(1); }
                lats[t].push_back(secs(s, Clock::now()) * 1e6);
            }
            free(buf);
        });
        for (auto& x : th) x.join();
        double el = secs(a, Clock::now());
        std::vector<double> lat; for (auto& v : lats) lat.insert(lat.end(), v.begin(), v.end());
        report("rand-odirect", path, block, threads, (double) block * reads, el, lat);
        close(fd);
    } else if (mode == "mmap") {
        long reads = atol(argv[5]); std::string adv = argc > 6 ? argv[6] : "none";
        int fd = open(path, O_RDONLY); size_t fsz = file_size(fd); size_t nblk = fsz / block;
        const unsigned char* p = (const unsigned char*) mmap(nullptr, fsz, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { perror("mmap"); return 1; }
        if (adv == "none" || adv == "willneed") madvise((void*) p, fsz, MADV_RANDOM);   // 4 KiB faults, no readahead
        if (adv == "seq") { madvise((void*) p, fsz, MADV_SEQUENTIAL); posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL); }
        std::atomic<long> next{0};
        std::vector<std::vector<double>> lats(threads);
        std::atomic<unsigned long> sink{0};
        auto a = Clock::now();
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t) th.emplace_back([&, t] {
            std::mt19937_64 rng(1234 + t); unsigned long s0 = 0;
            while (next.fetch_add(1) < reads) {
                size_t off = (rng() % nblk) * block;
                auto s = Clock::now();
                if (adv == "willneed") madvise((void*) (p + off), block, MADV_WILLNEED);
                for (size_t i = 0; i < block; i += 4096) s0 += p[off + i];
                lats[t].push_back(secs(s, Clock::now()) * 1e6);
            }
            sink += s0;
        });
        for (auto& x : th) x.join();
        double el = secs(a, Clock::now());
        std::vector<double> lat; for (auto& v : lats) lat.insert(lat.end(), v.begin(), v.end());
        std::string m = "mmap-fault(" + adv + ")";
        report(m.c_str(), path, block, threads, (double) block * reads, el, lat);
        munmap((void*) p, fsz); close(fd);
    } else { fprintf(stderr, "unknown mode\n"); return 2; }
    return 0;
}
