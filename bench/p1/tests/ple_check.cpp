// The PLE table reader on the Q8_0 GGUF: rows through PleTable (Direct and Mmap modes) printed for comparison with gguf-py.
#include "strata/kernels/ngram.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
using namespace strata::kernels;
int main(int argc, char** argv) {
    for (int mode = 0; mode < 2; ++mode) {
        PleTable t;
        PleIoOptions io;
        io.mode = mode == 0 ? PleIo::Direct : PleIo::Mmap;
        std::string err;
        if (!t.open(argv[1], err, io)) { std::printf("open: %s\n", err.c_str()); return 1; }
        std::printf("mode %s rows %llu\n", mode == 0 ? "direct" : "mmap", (unsigned long long) t.rows());
        const uint32_t rows[4] = {0, 1, 123456789u, 320001535u};
        for (uint32_t r : rows) {
            float out[160];
            t.read_row(r, out);
            std::printf("row %u:", r);
            for (int i = 0; i < 6; ++i) std::printf(" %.6f", out[i]);
            std::printf(" ... %.6f %.6f\n", out[158], out[159]);
        }
        // a batch gather (the verify window's path): rows 1 and 123456789 among 16
        uint32_t rows16[16];
        for (int h = 0; h < 16; ++h) rows16[h] = h % 2 ? 1u : 123456789u;
        static float out2560[2560];
        if (!t.gather_batch(rows16, 1, out2560, err)) { std::printf("gather_batch: %s\n", err.c_str()); return 1; }
        std::printf("batch h0 %.6f h1 %.6f\n", out2560[0], out2560[160]);
    }
    return 0;
}
