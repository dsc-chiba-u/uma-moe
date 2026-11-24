// native_mmvq on a real Q8_0 matrix of the merged GGUF (offset/rows/cols given) against a float reference, 1 and 4 columns.
//   q8_dense_check <gguf> <abs offset> <n_in> <n_out>
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace strata::kernels;
static float h2f(uint16_t h) { __half_raw r; r.x = h; return __half2float(__half(r)); }
int main(int argc, char** argv) {
    const long long off = std::atoll(argv[2]); const int n_in = std::atoi(argv[3]), n_out = std::atoi(argv[4]);
    const size_t rb = (size_t) n_in / 32 * 34, bytes = rb * n_out;
    std::vector<uint8_t> w(bytes);
    FILE* f = std::fopen(argv[1], "rb"); fseeko(f, off, SEEK_SET);
    if (std::fread(w.data(), 1, bytes, f) != bytes) { std::printf("short\n"); return 1; } std::fclose(f);
    const int NC = 4;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) n_in * NC); for (auto& v : x) v = nd(rng);
    std::vector<float> ref((size_t) n_out * NC);
    for (int r = 0; r < n_out; ++r)
        for (int c = 0; c < NC; ++c) {
            double s = 0;
            for (int b = 0; b < n_in / 32; ++b) {
                uint16_t d; std::memcpy(&d, &w[r * rb + b * 34], 2);
                const int8_t* q = (const int8_t*) &w[r * rb + b * 34 + 2];
                for (int j = 0; j < 32; ++j) s += (double) h2f(d) * q[j] * x[(size_t) c * n_in + b * 32 + j];
            }
            ref[(size_t) c * n_out + r] = (float) s;
        }
    void *dw, *dxq; float *dx, *dy;
    cudaMalloc(&dw, bytes); cudaMemcpy(dw, w.data(), bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dx, x.size() * 4); cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dxq, native_q8_1_bytes(n_in, NC)); cudaMalloc(&dy, ref.size() * 4);
    cudaStream_t st; cudaStreamCreate(&st);
    for (int nc : {1, NC}) {
        cudaMemset(dy, 0, ref.size() * 4);
        native_quantize_q8_1(dx, dxq, n_in, nc, st);
        native_mmvq(8, dw, dxq, dy, n_in, n_out, nc, st); cudaStreamSynchronize(st);
        std::vector<float> y((size_t) n_out * nc);
        cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
        double num = 0, den = 0;
        for (size_t i = 0; i < y.size(); ++i) { num += std::fabs(y[i] - ref[i]); den += std::fabs(ref[i]); }
        std::printf("ncols %d: rel L1 err %.3e  y[0..2] %.4f %.4f %.4f  ref %.4f %.4f %.4f  last col y %.4f ref %.4f\n", nc,
                    num / den, y[0], y[1], y[2], ref[0], ref[1], ref[2], y[(size_t) (nc - 1) * n_out], ref[(size_t) (nc - 1) * n_out]);
    }
    std::printf("cuda: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    return 0;
}
