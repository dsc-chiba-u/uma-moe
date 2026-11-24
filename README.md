# Expert Tiering on Unified Memory: code, benchmarks and data

Artifacts of the paper *Expert Tiering on Unified Memory: Running a 188 GB Mixture-of-Experts Model on a 128 GB GPU
Workstation* (Hiromu Nishiuchi and Tazro Ohta; submitted to IEEE Transactions on Parallel and Distributed Systems; preprint:
arXiv:XXXX.XXXXX, to be filled in). This software release is archived on Zenodo: DOI
[10.5281/zenodo.24486238](https://doi.org/10.5281/zenodo.24486238) (v1.0.0, this release) and
[10.5281/zenodo.24486237](https://doi.org/10.5281/zenodo.24486237) (all versions).

The paper runs the 188 GB Q8_0 quantization of Qwen3.8-Flash-Next on a 128 GB NVIDIA DGX Spark (GB10) with a
single-copy expert tier: a 1,024-expert device tier, a DRAM tier of pinned slots that the GPU reads in place, and the
remaining experts read from NVMe with unbuffered I/O while the GPU computes. The tier is a patch to the Strata
inference engine (0.1.27). This repository holds the engine patches, the benchmark harness and microbenchmarks, the
analysis scripts that produce the paper's tables and figures, and the logs of every run. Routing traces, the
locality study's per-model traces and the logits of the quality comparison are in a separate data archive
(Zenodo DOI: 10.5281/zenodo.XXXXXXX, to be filled in).

## Contents

| path | what |
|---|---|
| `patches/` | Strata engine patches (one per engine build used in the paper, see *Engines*), and the aarch64 build changes for upstream Strata 0.2.2 and 0.2.6 |
| `bench/` | benchmark harness (`harness.py`: server launcher, CPU/GPU/memory/NVMe sampler, memory watchdog), prompts, microbenchmarks (`uma_bw.cu`, `nvme_bench.cpp`, `replay_io.cpp`), and the experiment scripts by phase (`p0/`, `p1/`, `p2/`; `exp_*.py` for the initial characterization) |
| `analysis/` | log parsers, cache simulation on routing traces, the decode cost model, and the scripts that write the paper's tables (`paper_tables.py`) and figures (`paper_figs.py`) |
| `locality/` | the cross-model routing-locality study (trace collector against libllama, cache simulator, corpus builder); see `locality/README.md` and `locality/RESULTS.md` |
| `results/` | one directory per run: `engine.log`, `server.log`, `requests.jsonl` (generated texts and client timings), `samples.csv` (CPU, GPU, memory and NVMe at 4 Hz), `run.json`; summaries such as `final_report.json`. Runs prefixed `x_` are set aside from the tables: three runs set aside for slow requests, two of them while other jobs ran on the machine (repeated under their normal names), three Strata 0.2.6 runs with a configuration difference (`x_f_up026_b66_exact_*_convcache`: conversation cache on; `x_f_up026_b90_default_mtp_watchdog`: stopped by the memory watchdog), and the deliberate CPU-contention tests (`x_burn_*`); `results/locality/analysis/` holds the locality summaries |
| `notes/p1_design.md` | design notes of the expert tier |
| `ENVIRONMENT.md` | the measurement machine |
| `download_models.sh`, `models.sha256` | the Q8_0 and MTP downloads, and the checksums of every model file used |
| `THIRD_PARTY.md`, `LICENSES/` | third-party components, their licenses, and what this release includes of them |

## Hardware and software

* NVIDIA DGX Spark: GB10 (Blackwell GPU, compute capability 12.1; 10 Cortex-X925 + 10 Cortex-A725 cores), 128 GB
  LPDDR5x unified memory (121.7 GiB visible to Linux), 4 TB NVMe (Samsung MZALC4T0HBL1). The expert tier relies on a
  GPU that reads pinned host memory in place; results on other machines will differ.
* Ubuntu 24.04.4, kernel 6.17.0-1014-nvidia, NVIDIA driver 580.126.09, CUDA 13.0 (V13.0.88), GCC 13.3, CMake 3.28
  or newer, Ninja, Python 3.12.
* Kernel settings. Recorded at measurement time: transparent huge pages `enabled` = `madvise` (`ENVIRONMENT.md`,
  2026-10-21) and `vm.compaction_proactiveness=20` (saved by the compaction A/B runner before each series, 2026-10-23;
  the series set it to 0 and back). Not recorded then, kernel defaults assumed unchanged (no persistent override on the
  machine; the values read on 2026-11-22): transparent huge pages `defrag` = `madvise`, `vm.watermark_boost_factor=15000`.
* Disk: about 700 GB for models and packs (Q8_0 shards 176 GiB and the merged file 176 GiB, both used; Q8_0 pack
  121 GiB; Q2_0 GGUF 62 GiB, its pack 1.4 GiB and tier file 34 GiB; MTP layers 13 GiB; locality models 93 GiB).

Keep the machine otherwise idle while measuring. The engine's host thread spins on the first CPU it may use (CPU 0):
another process on that core slows MTP decoding by about 30%. The harness refuses to start a server when
`MemAvailable` is too low and kills it below 6 GiB, because running out of unified memory freezes the machine.

### Known limitation: decoding slows while kswapd reclaims memory

Twice on 2026-10-23, soon after other jobs had built software and written archives on the machine, the main
configuration decoded 10-20 times slower than in the paper's runs; minutes later it decoded normally. Only the second
episode had the reclaim sampler running: there kswapd reclaimed pages during each of the four slow requests, in bursts
(the counter rose in 15-82 of the 66-158 half-second intervals of each request). The first episode has no reclaim
counter. Filling the page cache with 44 GiB of model files did not reproduce the slowdown. Whether the reclaim caused
it was not established: the investigation was stopped so as not to delay the paper, which reports the episodes as a
limitation.
`results/pagecache/README.md` has the observations and a list of what to check and record if a run is slow. Run
`bench/p2/reclaim_check.sh` before measuring and again once the engine has loaded; if kswapd is reclaiming, wait until
it stops or drop the page cache (`sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`).

With the DRAM tier in `cudaHostAlloc` memory (an ablation), proactive compaction lengthens the GPU's part of a window
by up to 44% when free memory is low and fragmented; `vm.compaction_proactiveness=0` removes that, and the main
configuration is not affected (`results/compaction/`).

## Directory layout

The scripts expect this layout (paths are relative to the home directory):

| path | what |
|---|---|
| `~/work/uma-moe` | this repository (with a Python venv in `.venv`) |
| `~/src/Strata` | Strata 0.1.27 (tag `v0.1.27`) set up by its `setup.py`; engine builds in `engine/` |
| `~/src/Strata-data` | Strata's data folder: the Q2_0 GGUF, the packs, the MTP layer |
| `~/src/Strata-upstream` | Strata 0.2.2 (`v0.2.2`, commit `011c9e60`) with `build-gb10/` |
| `~/src/Strata-upstream-026` | Strata 0.2.6 (`v0.2.6`, commit `4b9e05a2`) with `build-gb10/` |
| `~/src/llama.cpp-master` | llama.cpp commit `339f324` built with CUDA |
| `~/models` | the Q8_0 GGUF (`unsloth-Qwen3.8-Flash-Next-GGUF/`) and the locality models (`locality/`) |

GPU jobs take `flock ~/work/uma-moe/.gpu.lock`, so two scripts never share the GPU.

## Setup

**1. This repository.**

```bash
git clone https://github.com/dsc-chiba-u/uma-moe ~/work/uma-moe
cd ~/work/uma-moe && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Without the GitHub repository, start from the release archive instead: `tar xzf uma-moe-release-<date>.tar.gz` unpacks
the same tree as `repo/`; move it to `~/work/uma-moe` and create the virtual environment as above.

**2. Strata 0.1.27, its data, and the base engine.** Our copy of Strata was cloned at commit `a790805`; the
repository's history has since been rewritten, and the same source tree (git tree `9aac603b`) is now tag `v0.1.27`,
commit `99b40973`. The patches apply to it unchanged. Strata's `setup.py` downloads the Q2_0 GGUF of ISTA-DASLab
(66 GB) and the MTP layer of the original checkpoint (about 5 GB), builds the Q2_0 pack and the MTP runtime files in
`~/src/Strata-data`, fetches llama.cpp `3cf0325` into `third_party/llama.cpp`, and compiles the engine (10 to 20
minutes). With the aarch64 patch applied, the result is the base engine (`strata-base`, Strata 0.1.27 for GB10).

```bash
git clone https://github.com/Niko1221/Strata ~/src/Strata && cd ~/src/Strata && git checkout v0.1.27
git apply ~/work/uma-moe/patches/strata-0.1.27-base-aarch64.diff
tar xzf ~/work/uma-moe/patches/strata-0.1.27-base-aarch64-newfiles.tgz
python3 setup.py --family qwen --model Q2_0 --context 32768 --kv int8 --vision no \
  --experimental-speed-projection off --build --no-start --yes
cp engine/strata engine/strata-base
```

**3. The other engine builds.** Each one is a clean `v0.1.27` tree with one patch (see *Engines*), built like
`setup.py` builds:

```bash
build_engine() {   # $1 = engine name, $2 = patch, $3 = new-file archive or "", $4 = patch applied on top or ""
  local P=~/work/uma-moe/patches W=~/src/Strata-build/$1
  rm -rf "$W" && git -C ~/src/Strata worktree add --detach "$W" v0.1.27 && cd "$W" || return 1
  git apply "$P/$2" && { [ -z "$3" ] || tar xzf "$P/$3"; } && { [ -z "$4" ] || git apply "$P/$4"; } || return 1
  ln -s ~/src/Strata/third_party/llama.cpp third_party/llama.cpp
  cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DSTRATA_GGML_DIR="$W/third_party/llama.cpp" && cmake --build build --target strata &&
  cp build/strata ~/src/Strata/engine/strata-$1
}
build_engine p1d strata-0.1.27-p1d.diff strata-0.1.27-p1d-newfiles.tgz ""     # the paper's Q8_0 engine
build_engine p0  strata-0.1.27-p0.diff  strata-0.1.27-base-aarch64-newfiles.tgz ""
build_engine p0a strata-0.1.27-base-aarch64.diff strata-0.1.27-base-aarch64-newfiles.tgz strata-0.1.27-p0a-diag.diff
build_engine p1b strata-0.1.27-p1b.diff strata-0.1.27-p1b-newfiles.tgz ""
```

Run `export PATH=~/src/Strata/.venv/bin:$PATH` first if `cmake` or `ninja` is missing: `setup.py` installs both there.
The engines are started by `~/src/Strata/serve/server.py` with `~/src/Strata/.venv/bin/python`; no patch changes the
server.

**4. llama.cpp master (baseline, locality traces, and the shard merge of step 5).**

```bash
git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp-master && cd ~/src/llama.cpp-master && git checkout 339f324
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=121 && cmake --build build
```

**5. The Q8_0 model and its pack.** Downloads 176 GiB of Q8_0 shards and the MTP GGUF, then merges the shards and
writes the Q8_0 pack (`experts.bin`, 128 GB, the file the expert tier reads) and a Q2_0 `experts.bin` for tier tests
(about 15 minutes after the downloads; `prepare.sh` builds `llama-gguf-split` in the llama.cpp tree of step 4).
`prepare.sh` packs with the `tools/iq_pack.py` of the p1d tree of step 3 (`~/src/Strata-build/p1d`; set `P1D` if it
is elsewhere), not with Strata's own packer: the unsloth Q8_0 file stores `blk.1.ple_conv1d.weight` as F32, the engine
reads that tensor as F16, and only the p1d packer converts it (the `F16_TENSORS` rule in `patches/strata-0.1.27-p1d.diff`);
a pack made with the v0.1.27 packer keeps it as F32 (index kind 2), which the engine would read as F16 halves of F32
words. The script ends by printing that tensor's line of `~/src/Strata-data/packs/q8_0/index.txt`, which must read
`blk.1.ple_conv1d.weight 0 5 ... 81920 ...` (kind 5 = F16; 81,920 bytes for its 4 x 10,240 values). The pack
regenerated this way on 2026-11-24 from the released packer was byte-identical to the pack of the paper's runs
(entry S-6 of the authors' research validation record, which is kept with the manuscript and is not part of this
release).

```bash
mkdir -p ~/models && ~/work/uma-moe/download_models.sh
~/work/uma-moe/bench/p1/prepare.sh
(cd ~ && grep -v ' models/locality/' ~/work/uma-moe/models.sha256 | sha256sum -c)   # the files of steps 2 and 5; the merged Q8_0 file is not listed
```

**6. Upstream Strata 0.2.2 (baseline).** Built for aarch64 + CUDA with the build changes in
`patches/strata-0.2.2-gb10.diff`, which arranges the build as Strata's own Metal build does on Arm: the x86-intrinsic
CPU kernels are left out, the CPU computes its experts with ggml-cpu's NEON code, and the host's spin loops use an
aarch64 pause and fence. It also needs its own pack of the Q8_0 file:

```bash
git clone https://github.com/Niko1221/Strata ~/src/Strata-upstream && cd ~/src/Strata-upstream && git checkout v0.2.2
git apply ~/work/uma-moe/patches/strata-0.2.2-gb10.diff && tar xzf ~/work/uma-moe/patches/strata-0.2.2-gb10-newfiles.tgz
cmake -G Ninja -S . -B build-gb10 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && cmake --build build-gb10 --target strata
STRATA_GGUF_PY=$PWD/build-gb10/_deps/strata_llamacpp-src/gguf-py ~/src/Strata/.venv/bin/python tools/iq_pack.py \
  --gguf ~/models/unsloth-Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
  --out ~/src/Strata-data/packs-upstream/q8_0 --compat-bf16
```

**7. Upstream Strata 0.2.6 (released 2026-10-25, the newest release when we measured; 0.2.7 followed on 2026-10-27).** The same arrangement as 0.2.2, plus stubs for three
AVX-512 entry points added since (`patches/strata-0.2.6-gb10.diff`). Its Qwen engine is unchanged since 0.2.5 and
reads 0.2.2's pack, MTP layer and expert profile (their formats did not change):

```bash
git -C ~/src/Strata-upstream fetch --tags && git -C ~/src/Strata-upstream worktree add --detach ~/src/Strata-upstream-026 v0.2.6
cd ~/src/Strata-upstream-026 && git apply ~/work/uma-moe/patches/strata-0.2.6-gb10.diff && tar xzf ~/work/uma-moe/patches/strata-0.2.6-gb10-newfiles.tgz
cmake -G Ninja -S . -B build-gb10 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && cmake --build build-gb10 --target strata
```

**8. The locality study** (only for the cross-model table). Its tools link a snapshot of the llama.cpp build of step 4
in `locality/llama-bin/`, so that a rebuild of that tree cannot change the libraries under a running job:

```bash
cp -r ~/src/llama.cpp-master/build/bin ~/work/uma-moe/locality/llama-bin && ~/work/uma-moe/locality/build.sh
~/work/uma-moe/locality/dl_all.sh gpt-oss-20b qwen3-30b-a3b glm-4.7-flash qwen3.6-35b-a3b mixtral-8x7b   # to ~/models/locality/
(cd ~ && grep ' models/locality/' ~/work/uma-moe/models.sha256 | sha256sum -c)   # the five locality models
```

## Engines

| engine | patch (against Strata `v0.1.27`) | used for |
|---|---|---|
| `strata-base` | `strata-0.1.27-base-aarch64.diff` + `…-newfiles.tgz` (aarch64 port only) | the 0.1.27 rows of `tab:p0-window` and `tab:p0`, the initial characterization (`results/e2_*`, `e4_*`), the routing traces |
| `strata-p0a` | base + `strata-0.1.27-p0a-diag.diff` | the "direct reads, uncapped" row of `tab:p0-window` (a diagnostic build whose source was not archived; this patch restates its two changes) |
| `strata-p0` | `strata-0.1.27-p0.diff` + base new files | the fixed verification window (`tab:p0`, `results/p0f_fix_*`, `p0_sweep_*`) |
| `strata-p1` | `strata-0.1.27-p1-wip.diff` | the "without" columns of `tab:pred`. Its new files were not archived, so it cannot be rebuilt exactly; `strata-p1d` with `STRATA_TIER_PREFETCH=0` runs the later tier without prediction |
| `strata-p1b` | `strata-0.1.27-p1b.diff` + `…-newfiles.tgz` | the "with" columns of `tab:pred` |
| `strata-p1d` | `strata-0.1.27-p1d.diff` + `…-newfiles.tgz` | every Q8_0 result of this work in `tab:q8`, `tab:abl`, `fig:breakdown` and `fig:residency`, the prompt-processing and tuning runs |
| `strata-p1e` | p1d + `strata-0.1.27-p1e-arena-thp.diff` | only the attempted 2 MB-page slots (`STRATA_ARENA_THP=1`), which did not start in time |
| upstream 0.2.2 | `strata-0.2.2-gb10.diff` + `…-newfiles.tgz` against `v0.2.2` | the Strata 0.2.2 rows of `tab:q8` |
| upstream 0.2.6 | `strata-0.2.6-gb10.diff` + `…-newfiles.tgz` against `v0.2.6` | the Strata 0.2.6 rows of `tab:q8` and the comparison with release 0.2.6 in the evaluation |
| llama.cpp | none, commit `339f324` | the llama.cpp rows of `tab:q8`, the locality traces |

Tables and figures are named by their labels in the manuscript's LaTeX source.

## Reproducing the paper

All commands run from `~/work/uma-moe`; run the Python scripts with `.venv/bin/python`. Times are those of our runs,
without model downloads.

| paper item | commands | results | time |
|---|---|---|---|
| Bytes per token (`tab:bytes`) | `.venv/bin/python analysis/gguf_bytes.py <shards>` for the Q2_0 and the Q8_0 files | stdout | < 1 min |
| GB10 memory and storage paths (`tab:membw`) | build `bench/uma_bw.cu` and `bench/nvme_bench.cpp` (commands below); the E1 blocks of `bench/pipeline.sh` and `bench/post.sh` | `results/e1_initial_microbench.txt`, `results/pipeline/e1_*.log`, `results/pipeline/post_mmap.log` | 20 min |
| Where an MTP window's time went (`tab:p0-window`) | `bench/p0/diag.py p0_diag_c1024_mtp_pcie1`; `bench/p0/ab.py p0_ab_c1024_mtp_pcie1_direct base 1024 mtp --pcie-frac 1.0 --pcie-mode direct`; the same with name `…_direct_p0a` and engine `strata-p0a`; `bench/p0/diag.py p0_diag_c24576_mtp`; print with `bench/p0/show.sh RUN` | `results/p0_diag_*`, `results/p0_ab_*` | 10 min |
| The verification-window fix on Q2_0 (`tab:p0`) | `.venv/bin/python bench/p0/final.py`, then `analysis/p0_report.py` | `results/p0f_*`, `results/p0_report.json` | 75 min |
| Q8_0 decoding, all engines (`tab:q8`), window breakdown (`fig:breakdown`) | `bench/p2/final_runs.sh` and `bench/p2/extra_runs4.sh` (Strata 0.2.2 at 90 GiB exact and at 66.8 GiB with its defaults); `bench/p2/extra_runs3.sh` repeats two runs that other work disturbed and runs the CPU-0 contention test; then `analysis/final_report.py`, `analysis/paper_tables.py`, `analysis/paper_figs.py` | `results/f_*`, `results/x_burn_*`, `results/final_report.json`, `paper/tables/`, `paper/figs/` | 3 h |
| Strata 0.2.6 (the newest release when we measured) on the Q8_0 file (`tab:q8`, text of the comparison) | `bench/p2/upstream026_runs.sh`: the five configurations of the 0.2.2 rows and two runs with its unbuffered file tier forced on at our memory, all with its conversation cache off (`--conversation-cache-mib 0`; the option exists in 0.2.2 with the default 0, and 0.2.6 sizes the cache from free RAM by default; it keeps each request's prompt state for an agent's next turn, which these requests never reuse, and with it the 90 GiB budget ran out of memory); then `analysis/final_report.py`, `analysis/paper_tables.py` | `results/f_up026_*` (the first two runs, with the cache on, and the run stopped by the memory watchdog are kept as `results/x_f_up026_*`), `results/final_report.json` | 2 h |
| Ablations at 60% (`tab:abl`) | part of `final_runs.sh`; `bench/p2/ple_placement.sh` (n-gram table via the page cache); the `cudaHostAlloc` run of `bench/p2/extra_runs5.sh` (`extra_runs.sh` holds the first attempt and the 2 MB-page attempt, which rebuilds `strata-p1e` in place: use `build_engine p1e strata-0.1.27-p1d.diff strata-0.1.27-p1d-newfiles.tgz strata-0.1.27-p1e-arena-thp.diff` instead) | `results/f_ours_t60_*` | 40 min |
| Prediction on/off by residency (`tab:pred`) | `bench/p1/q8_sweep.sh` (engine `strata-p1`), `bench/p1/q8_sweep_b.sh` (`strata-p1b`); `analysis/p1_summary.py 'p1_q8_t*' 'p1b_q8_t*'` | `results/p1_q8_t*`, `results/p1b_q8_t*` | 25 min |
| Model against measurement (`fig:residency`) | `analysis/cost_model.py --expert-mb 5.222 --dense-gb 4.79 --w-hit 165 --w-nvme 10.5 --t-fix 8.54 --t-tok 2.82 --t-draft 2.0 --residency 1.0,0.75,0.6,0.45 --caps 0,3 --overlap --out results/cost_model_q8_overlap_wh165.json`, the same with `--w-hit 223` and `…_wh223.json` (both read the routing traces of the data archive); then `analysis/paper_figs.py`, which needs only the JSON files in `results/` | `results/cost_model_q8_overlap_wh*.json` | 2 min |
| Miss rates on the decode trace (`tab:locality`) | `bench/exp_traces.py nodraft`, `bench/exp_traces.py mtp` (engine `strata-base`); `analysis/analyze_traces.py`; `analysis/policies_extra.py`; or extract the routing-trace archive instead of tracing | `results/trace_*`, `results/trace_analysis.json`, `results/policies_extra.json` | 20 min + 10 min |
| Locality across MoE models (`tab:locality-models`) | `locality/README.md` (*Reproduce*); or extract the locality archive and run `locality/analyze.py`, `locality/report.py` | `results/locality/` | about 15 min per model + analysis |
| Prompt processing and tuning knobs (text of the evaluation and ablation sections) | `bench/p1/prefill_test.sh`, `bench/p1/tune.sh`, `bench/p2/extra_runs2.sh` (Strata 0.2.2 on the 5.4K-token prompt); `bench/p1/pf_variants.sh` used an intermediate build `strata-p1c` whose source was not archived (`strata-p1d` reads the same `STRATA_TIER_PREFETCH_K` and `_DEPTH` settings) | `results/p1d_*`, `results/p1c_*`, `results/f_up_b66_exact_prefill` | 30 min |
| Kernel disturbances (text of the evaluation setup) | `bench/p2/compaction_ab.sh` and `compaction_ab2.sh`, with `bench/p2/compaction_root.sh` run as root; `bench/p2/pagecache_ab.sh`; `analysis/compaction_ab.py run1` or `run2`, `analysis/pagecache_ab.py` | `results/compaction/`, `results/pagecache/`, `results/k_*`, `results/k2_*`, `results/pc_*` | 70 min per compaction series, 6 min for the page-cache test |

The initial characterization behind the paper's background and characterization sections (engine `strata-base`
and llama.cpp; Q2_0 unless noted):

| step | script | results |
|---|---|---|
| E1 memory and storage microbenchmarks | `bench/uma_bw.cu`, `bench/nvme_bench.cpp`; `bench/pipeline.sh` | `results/e1_initial_microbench.txt`, `results/pipeline/e1_*.log` |
| E2 Strata and llama.cpp, MTP off and on | `bench/exp_reproduce.py` | `results/e2_*`, `results/summary_e2*.json` |
| E3 routing traces | `bench/exp_traces.py` | `results/trace_*` (data archive) |
| E4 Strata expert-cache sweep and miss path | `bench/exp_strata_cache.py` | `results/e4_*` |
| E4b llama.cpp `--cpu-moe` with an expert cache | `bench/exp_llama_cache.py` | `results/e4b_*` |
| E5 llama.cpp on the Q8_0 file through the page cache | `bench/exp_overflow.py` | `results/e5_*` |
| E6 perplexity and KL divergence of Q2_0 against Q8_0 | `bench/exp_quality.sh` | `results/e6_quality/` (data archive) |
| E8 I/O replay of a routing trace against the Q8_0 file | `bench/replay_io.cpp`, `analysis/expert_map.py`; `bench/pipeline.sh` | `results/pipeline/e8_*.log` |

Microbenchmark builds:

```bash
nvcc -O3 -arch=sm_121 -std=c++17 -Xcompiler -O3,-mcpu=native,-pthread bench/uma_bw.cu -o bench/uma_bw
g++ -O2 -std=c++17 -pthread bench/nvme_bench.cpp -o bench/nvme_bench
cp bench/nvme_bench bench/nvme_bench_v2     # the name bench/post.sh calls (built from the same source)
g++ -O2 -std=c++17 -pthread bench/replay_io.cpp -o bench/replay_io
```

`paper_tables.py` and `paper_figs.py` write to `paper/tables/` and `paper/figs/` (the manuscript's source is not part
of this repository). Every Strata run writes the window breakdown and the expert-tier counters to `engine.log` when
`STRATA_DECODE_TIMING=1`; `analysis/final_report.py` parses them. The generated texts of the runs are deterministic:
all configurations of this work produce the same 12 texts (`requests.jsonl`), which the analysis checks.

## Data archive

The Zenodo record holds three archives; extract them in this directory (they unpack into `results/`):

| archive | contents | license |
|---|---|---|
| `routing-traces.tar.zst` | Strata's routing traces of Qwen3.8-Flash-Next on the six short and two long prompts, drafts off and MTP (`results/trace_*`), and the inputs of the I/O replay (`results/maps`; the byte maps of the GGUF files are regenerated with `analysis/expert_map.py`) | CC BY 4.0 |
| `locality.tar.zst` | per model: generated answers, token ids, routing traces and per-token NLL; the text corpora (`results/locality/`) | CC BY-SA 4.0 (Wikipedia text); see its `ATTRIBUTION.md` |
| `quality-logits.tar.zst` | Q8_0 logits on WikiText-2 for the KL divergence of Q2_0 (`results/e6_quality/`) | CC BY-SA 4.0 (WikiText) |

## Licenses

Code (`bench/`, `analysis/`, `locality/`, `patches/`, the scripts): MIT, see `LICENSE`. The patches modify Strata,
which is MIT-licensed (`LICENSES/third_party/Strata-MIT.txt`). Results, logs and documentation (`results/`,
`notes/`, this README): CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`). `bench/data/longdoc.txt` is a copy of three Strata
source files (MIT, with this study's aarch64 changes) used as long prompts. Third-party components and data: `THIRD_PARTY.md`. The models are not
included; they keep their own licenses.

## How this work was produced

This work was carried out with substantial use of an AI system. Claude (Anthropic), used through the Claude Code
agent with the Claude Opus 5.5 model, wrote the engine patches and the benchmark and analysis code in this repository,
ran the experiments, analyzed the results, produced the paper's tables and figures, wrote the automated check of the
paper's numbers against these data, and drafted the paper and this documentation. The human authors defined the
research objectives, directed the investigation, and take full responsibility for the accuracy, originality,
experimental validity, and conclusions of this work. Claude checked the paper's numbers, code and references against
the raw data and the primary sources. A second AI agent, ChatGPT (OpenAI) with the Astra 6.2 model, independently
checked the manuscript and selected results before submission, as the authors' research validation record documents
(the record is kept with the manuscript and is not part of this release). [PENDING: TO BE CONFIRMED WHEN THE AUTHORS' VALIDATION RECORD IS
COMPLETE] The authors reviewed the manuscript, the source code, the experimental records and the references to the
extent that record documents.

## Citation

See `CITATION.cff`. Please cite the paper (arXiv:XXXX.XXXXX; the identifier will be filled in when the preprint is posted) and this software record (Zenodo DOI 10.5281/zenodo.24486237 for all versions, 10.5281/zenodo.24486238 for v1.0.0) and, for the data, the Zenodo dataset record (not yet uploaded as of 2026-11-24; its DOI will be added here).
