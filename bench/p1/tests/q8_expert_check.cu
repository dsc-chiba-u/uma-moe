// Q8_0 expert on the GPU (native_expert_grouped) against a float reference on the CPU, on a real blob of
// packs/q8_0/experts.bin; plus the Q8_0 embedding row dequantizer against the CPU.
//   q8_expert_check <experts.bin> <layer> <expert>
#include "strata/kernels/iq_kernels.hpp"
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
// one Q8_0 row (n values) to floats
static void deq_row(const uint8_t* row, int n, float* out) {
    for (int b = 0; b < n / 32; ++b) {
        uint16_t d; std::memcpy(&d, row + b * 34, 2);
        const int8_t* q = (const int8_t*) (row + b * 34 + 2);
        for (int j = 0; j < 32; ++j) out[b * 32 + j] = h2f(d) * q[j];
    }
}
int main(int argc, char** argv) {
    const int64_t H = 2560, FF = 640;
    const NativeExpertLayout L = native_expert_layout(8, 8, H, FF);
    std::printf("layout: gu_row %zu d_row %zu up_off %zu down_off %zu bytes %zu\n", L.gu_row, L.d_row, L.up_off, L.down_off, L.bytes);
    const int layer = argc > 2 ? std::atoi(argv[2]) : 0, expert = argc > 3 ? std::atoi(argv[3]) : 0;
    FILE* f = std::fopen(argv[1], "rb");
    std::vector<uint8_t> blob(L.bytes);
    const uint64_t off = ((uint64_t) layer * 512 + expert) * L.bytes;
    fseeko(f, (off_t) off, SEEK_SET);
    if (std::fread(blob.data(), 1, L.bytes, f) != L.bytes) { std::printf("short read\n"); return 1; }
    std::fclose(f);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(H);
    for (auto& v : x) v = nd(rng);
    // CPU float reference
    std::vector<float> gate(FF), up(FF), hbuf(FF), ref(H), row(H > FF ? H : FF);
    for (int r = 0; r < FF; ++r) {
        deq_row(blob.data() + (size_t) r * L.gu_row, H, row.data());
        double s = 0; for (int i = 0; i < H; ++i) s += (double) row[i] * x[i]; gate[r] = (float) s;
        deq_row(blob.data() + L.up_off + (size_t) r * L.gu_row, H, row.data());
        s = 0; for (int i = 0; i < H; ++i) s += (double) row[i] * x[i]; up[r] = (float) s;
        hbuf[r] = gate[r] / (1.f + std::exp(-gate[r])) * up[r];
    }
    for (int r = 0; r < H; ++r) {
        deq_row(blob.data() + L.down_off + (size_t) r * L.d_row, FF, row.data());
        double s = 0; for (int i = 0; i < FF; ++i) s += (double) row[i] * hbuf[i]; ref[r] = (float) s;
    }
    // GPU
    uint8_t* d_blob; float *d_x, *d_out; void *d_xq, *d_scr;
    unsigned long long* d_ptr; int32_t *d_start, *d_ng, *d_dst, *d_tok;
    cudaMalloc(&d_blob, L.bytes); cudaMemcpy(d_blob, blob.data(), L.bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_x, H * 4); cudaMemcpy(d_x, x.data(), H * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_xq, H / 32 * 36); cudaMalloc(&d_out, H * 4); cudaMemset(d_out, 0, H * 4);
    cudaMalloc(&d_scr, native_expert_scratch_bytes(1, FF));
    unsigned long long p = (unsigned long long) d_blob; int32_t st[2] = {0, 1}, ng = 1, z = 0;
    cudaMalloc(&d_ptr, 8); cudaMemcpy(d_ptr, &p, 8, cudaMemcpyHostToDevice);
    cudaMalloc(&d_start, 8); cudaMemcpy(d_start, st, 8, cudaMemcpyHostToDevice);
    cudaMalloc(&d_ng, 4); cudaMemcpy(d_ng, &ng, 4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_dst, 4); cudaMemcpy(d_dst, &z, 4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_tok, 4); cudaMemcpy(d_tok, &z, 4, cudaMemcpyHostToDevice);
    quantize_q8_1_rows(d_x, 1, H, d_xq, 0);
    native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, 1, 1, d_xq, d_scr, d_out, 0);
    std::vector<float> out(H);
    cudaMemcpy(out.data(), d_out, H * 4, cudaMemcpyDeviceToHost);
    double num = 0, den = 0, dot = 0, na = 0, nb = 0;
    for (int i = 0; i < H; ++i) { num += std::fabs(out[i] - ref[i]); den += std::fabs(ref[i]); dot += out[i] * ref[i]; na += out[i] * out[i]; nb += ref[i] * ref[i]; }
    std::printf("expert L%d E%d: rel L1 err %.3e  cosine %.6f  ref[0..3] %.4f %.4f %.4f  gpu %.4f %.4f %.4f\n", layer, expert,
                num / den, dot / std::sqrt(na * nb), ref[0], ref[1], ref[2], out[0], out[1], out[2]);
    // embedding-style dequant of one row through iq_dequant_f32 (the embed path's dq_q8_0)
    std::vector<float> rf(H), gf(H);
    deq_row(blob.data(), H, rf.data());   // gate row 0 is a 2560-wide Q8_0 row
    float* d_f; cudaMalloc(&d_f, H * 4);
    iq_dequant_f32(8, d_blob, H, d_f, 0);
    cudaMemcpy(gf.data(), d_f, H * 4, cudaMemcpyDeviceToHost);
    double md = 0; for (int i = 0; i < H; ++i) md = std::fmax(md, std::fabs(gf[i] - rf[i]));
    std::printf("dequant row: max abs diff %.3e (rf[0..2] %.4f %.4f %.4f gpu %.4f %.4f %.4f)\n", md, rf[0], rf[1], rf[2], gf[0], gf[1], gf[2]);
    std::printf("cuda: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    return 0;
}
