# Slow runs while kswapd reclaims memory, and a page-cache test

Twice on 2026-10-23, the main configuration (60% residency, registered slots) decoded 10-20 times slower than in the
paper's runs. Only the second episode had the reclaim sampler running: there, the kernel's kswapd reclaimed pages during
all four slow requests, at request-average rates of about 10,000-61,000 pages/s. The first episode has no reclaim
counter at those times. Both times, other jobs had just built software and written archives on the machine. A test that
filled the page cache with 44 GiB of model files did not reproduce the slowdown; its cause was not established.

## The two slow episodes (`cleanroom_smoke/`)

Each run: `code_en` and `essay_ja`, 64 tokens, `STRATA_DECODE_TIMING=1`, the configuration of `f_ours_t60_*`.

Rates in the two tables are the client's (tokens over the time from the first to the last streamed token); the
paper quotes the server's own count (64 tokens over its generation time), which on the ten requests of the five
completed runs below is 0.26-3.36% lower (`run.json` `server_stats` against `decode_tok_s_client` in `requests.jsonl`).

| run | engine | started | tokens/s | first token after | GPU's part of a window, ms |
|---|---|---|---|---|---|
| `smoke_t60_nodraft` | clean-room build | 16:21 | 1.35, 0.82 | 66 s, 22 s | 656, 1058 |
| `smoke_t60_mtp` | clean-room build | 16:26 | 1.67, 3.07 | 57 s, 17 s | 1526, 534 (MTP windows) |
| `smoke2_cleanroom_t60_nodraft` | clean-room build | 17:57 | 1.44, 0.81 | 65 s, 22 s | 573, 1042 |
| `smoke2_paper_t60_nodraft` | `strata-p1d` (the paper's) | 18:02 | 1.42, 1.93 | 65 s, 20 s | 665, 357 |
| `smoke2_paper_t60_mtp` | `strata-p1d` (the paper's) | 18:06 | 31.51, 23.38 | 1 s, 1 s | 49, 40 (MTP windows) |

In the paper's drafts-off runs the GPU's part of a window took 36-42 ms; in the page-cache test below the first token
came after 1.1 s.

- The first episode was the release's clean-room smoke test (`~/work/uma-moe-release/cleanroom`), right after the clean
  build, a fresh venv, and the end of the first compaction series. It was not logged beyond the harness's samples:
  the GPU reported 96% utilization at 15 W, against 29 W in the paper's run.
- The second episode, 12-14 minutes after the release's data archives had been compressed and the Strata 0.2.2 runs of
  `upstream/` had ended, and a few minutes after a clean-room run of the locality simulator, was logged
  (`sampler_smoke2.csv`, every 0.5 s): during the slow requests kswapd used 8-40% of a core and reclaimed
  10,000-61,000 pages per second (request averages; the counter rose in 15-82 of the 66-158 half-second intervals of
  each request), and 0.67-0.82 GiB were free when each request's first token came. At 18:06, the
  paper's engine decoded normally with no reclaim and 1.7-1.9 GiB free.
- `smoke2_cleanroom_t60_mtp` was interrupted on purpose (to reach the paper's engine sooner) and has no requests.
- Both engines were equally slow, and their startup logs match line for line, so the clean-room build is not the cause.
  The clean-room build was configured like the paper's (Release, CUDA architecture 121); only the generator differs.

## Page-cache test (`bench/p2/pagecache_ab.sh`, 18:09-18:14)

Before each run, the page cache of every file of 16 MB or more under `~/models`, `~/src/Strata-data`,
`~/work/uma-moe-release` and `~/work/uma-moe/results` was dropped (`bench/p2/evict_cache.py`, `posix_fadvise`). For
F, 45 GB of other model files (the Mixtral-8x7B and GLM-4.7-Flash GGUF files) were then read twice. Runs
`pc_<E1|F1|E2|F2>_t60_nodraft` with `strata-p1d`; `refaults.csv` adds the refault and major-fault counters.

| run | at start: MemFree / Cached GiB | tokens/s | first token | GPU's part, ms | free during decode, GiB | kswapd reclaim |
|---|---|---|---|---|---|---|
| E1 | 115.8 / 0.7 | 17.3, 16.3 | 1.1 s | 39.1, 37.4 | 27.4 | none |
| F1 | 72.2 / 44.3 | 17.3, 16.2 | 1.1 s | 38.7, 38.3 | 1.8-1.9 | none |
| E2 | 115.9 / 0.6 | 17.2, 16.1 | 1.1 s | 39.1, 38.2 | 27.5 | none |
| F2 | 72.3 / 44.2 | 17.2, 16.5 | 1.1 s | 39.3, 37.4 | 1.3-1.4 | none |

`python analysis/pagecache_ab.py` prints this table and writes `summary.json`.

## What was measured

- Four slow runs in two episodes: 0.8-3.1 tokens/s (0.8-3.0 by the server's count, which the paper uses), the 34- and
  35-token prompts read in 17-65 s (the engine's read times; 66 s was a time to the first token), and the GPU's part of a
  drafts-off window at 357-1058 ms instead of 36-42 ms. Both the clean-room build and the paper's engine were slow.
- In the logged (second) episode, kswapd reclaimed pages during all four slow requests, at request-average rates of
  10,000-61,000 pages/s, and the next run, minutes later, decoded normally without reclaim. The first episode has no
  reclaim counter at those times. Whether the reclaim caused the slowdown was not established.
- A page cache of 44 GiB of model files read twice, which leaves the engine 1.3-1.9 GiB free, did not slow decoding,
  and kswapd did not reclaim during decoding. Neither did an almost empty page cache.

## What is not known

- What state of the kernel starts the reclaim. Both episodes followed builds and archive writes; a page cache of the
  same size made of two large files did not. Candidates we did not test: page cache made of many small files,
  fragmented free memory with the watermark boost (`vm.watermark_boost_factor=15000`) raising kswapd's target, and
  the multi-generational LRU (enabled) walking the engine's page tables.
- How reclaim slows the GPU. This was not measured. (The DRAM tier's registered slots are read by the GPU through the
  host's page tables, which reclaim walks and changes; whether that is the link was not tested.)

The investigation was stopped on 2026-10-26, so as not to delay the paper: the cause is unknown, and the paper reports
the episodes as a known limitation (Section 9 of the paper, Section 10 before the 12-page cut of 2026-11-23) with the check below.

## Before measuring

`bench/p2/reclaim_check.sh` reports whether kswapd is reclaiming (it exits 1 if so). Run it before a measurement and
once the engine has loaded. If it is reclaiming, wait until it stops, or drop the page cache
(`sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`).

## If a run is slow: what to check and record

The signs, for the main configuration (`f_ours_t60_*`): decoding below about 5 tokens/s, the first token of a short
prompt after more than 10 s, the GPU near full utilization at about half the power of a normal run (15 W against
29 W in `samples.csv`), or, with `STRATA_DECODE_TIMING=1`, the GPU's part of a drafts-off window above 100 ms (normal:
36-42 ms).

1. Before starting the engine, run `bench/p2/reclaim_check.sh` and note `MemFree` and `Cached` from `/proc/meminfo`,
   and what else ran on the machine in the last half hour (builds, archive writes, large copies).
2. Run the kernel sampler next to the measurement: `python -I bench/p2/compaction_sampler.py OUT.csv &` (every 0.5 s:
   kswapd's and kcompactd's CPU time, `pgscan_kswapd`, `pgsteal_kswapd`, compaction counters, free memory and the
   fragmentation score). Stop it after the run.
3. Once the engine has loaded, run `bench/p2/reclaim_check.sh` again.
4. If the signs appear and the sampler shows kswapd reclaiming during decoding, keep the run under an `x_` name, wait
   until `reclaim_check.sh` reports kswapd idle (or drop the page cache), and repeat the run. Report the slow run and
   its sampler file separately; do not average it with the normal runs.
5. If the signs appear without reclaim, the cause is something else (e.g. a process on CPU 0, see the README, or
   kcompactd, see `results/compaction/`).
