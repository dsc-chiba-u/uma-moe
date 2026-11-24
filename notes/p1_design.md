# P1 design: single-copy expert tier over NVMe on a unified-memory GPU (Strata)

Status: implemented in `~/src/Strata` (uncommitted working tree; patches saved under `env/`), not yet measured.

## Goal

Run Qwen3.8-Flash-Next Q8_0 (188 GB: experts 128.35 GB, PLE table 54.4 GB, dense 4.79 GB) on a 128 GB DGX Spark
(GB10, ~121 GiB usable) at a decode speed close to what the performance model predicts for full DRAM residency
(22.9 tok/s without drafts, 37.5 tok/s with MTP), and characterise the gap.

## Starting point (P0)

On GB10 the GPU reads a pinned host blob in place about as fast as a cudaMalloc copy of it (end-to-end, the
verify window is within noise of full replication). After the P0 fix the verify window sends every expert that
is not in the VRAM tier to the GPU, read in place from the pinned arena (`--pcie-mode direct`, `--pcie-frac 1`), and
the CPU computes none. P1 replaces the all-experts arena (which cannot hold 128 GB) with a tier.

## Components

1. **experts.bin** (offline, `tools/iq_pack.py --experts-bin`): the GGUF's expert slices regrouped per
   (layer, expert) as one contiguous blob `[gate rows | up rows | down rows]`, layer-major. Q8_0 blob = 5,222,400 B
   (= 1,275 pages of 4 KiB, so every blob is page-aligned and is read with one O_DIRECT request); Q2_0 blob =
   1,382,400 B (half the blobs start mid-page; read with the preceding page, one page of slack per slot).
2. **TieredExpertSource** (`src/core/tiered_source.cpp`): `S` slots of pinned DRAM (one cudaHostRegister'd mmap
   region, device alias via cudaHostGetDevicePointer); residency map (layer, expert) -> slot; exact LRU (intrusive
   list); unbuffered reads through the engine's `DirectFile` thread pool (16 threads, pread O_DIRECT), each blob split
   into 1 MiB requests so one miss is served by several threads.
3. **VRAM tier** (unchanged Strata ExpertCache, cudaMalloc): a small static set of the hottest experts by the routing
   profile, filled at start; adaptive swaps disabled with the tier. The DRAM tier never admits an expert the VRAM
   tier holds: **one copy of every cached expert across both tiers**.
4. **Verify-window protocol** (per MoE layer, host side, in the pool callback that the GPU's doorbell wakes):
   - `begin_layer(ids)`: touch the routed experts that are resident (LRU), then admit the missing ones: evict the
     least recently used slot not touched by this layer, queue its reads.
   - plan: VRAM-tier and resident DRAM-tier experts go to the first list (computed by the GPU as soon as flag A is
     raised); experts still loading go to the second list with their (future) slot address.
   - publish (flag A) -> `wait_loads()` (host blocks on the SSD completions) -> flag B -> the GPU computes the second
     list. The GPU thus computes the resident experts while the SSD delivers the misses (intra-layer overlap).
   - No CPU expert computation.
5. **Eviction safety**: the host serves layer L only after the GPU rang layer L, i.e. after every expert read of
   layers < L finished (single stream). Only layer L's slots can be in use; they are touched first (MRU) and the
   eviction scan skips anything touched by the current layer.
6. **Prefill** (prompt chunks): per layer, the experts routed by the chunk are admitted in one batch
   (`begin_layer` + `wait_loads`, all reads in flight), then the existing staging path DMAs them from their slots.
   The all-layers streaming mode (chunks >= 2048) is disabled with the tier (it hands out every layer's blobs up
   front, which do not fit). Admission during prefill churns the decode working set (measured, see below).
7. **Q8_0 support added to Strata**: Q8_0 grouped expert kernels (llama.cpp's vec_dot_q8_0_q8_1), Q8_0 dequantizer
   for the token embedding and the prefill FP16 path, Q8_0 rows (170 B) in the PLE table reader (was IQ4_NL-only).
8. **PLE table**: stays on the SSD (direct 4 KiB reads, 16 rows/token, bounded row cache), as in Strata.

## What is not new (prior work) and what we claim

- Expert caching with LRU and SSD offload: many systems (see report section 7). Pre-gating / next-layer
  prefetch: Pre-gated MoE, ProMoE, AdapMoE, HOBBIT... (not used in v1).
- Our angle: on UMA the "GPU cache" and the "host cache" are the same DRAM; one pinned copy read in place removes
  both the duplicate and the staging copy; the remaining tier is NVMe, and its overlap structure interacts with MTP
  verification (the window's union of experts).

## Measurement plan

- Q2_0 (fits): tier 100% / 50% / 25% vs the arena -> overhead of the tier and output identity (must be bit-exact:
  the GPU computes every expert either way).
- Q8_0: tier 25/45/60/75% of the experts (+ small VRAM tier) x MTP on/off; decode tok/s, NVMe bytes and wait time,
  CPU/GPU utilisation, memory; per-window GPU stage profile (waitB = SSD wait).
- Baselines: llama.cpp master with `-lm mmap` (E5), Strata 0.1.27 `--mmap-experts` (page-cache tier, CPU misses),
  the cost model (clearly marked as a model, not a measurement).
- Ablations: no intra-layer overlap (wait before publishing), no VRAM tier, MTP off, prefill bypass vs admit.
