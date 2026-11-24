# vm.compaction_proactiveness: A/B test

Does turning off the kernel's proactive memory compaction (`vm.compaction_proactiveness=0`, default 20) remove the
slowdown we saw while kcompactd was running (paper, Section 7.1)? Two series on 2026-10-23, each with five blocks at
20, 0, 20, 0, 20 (A1 B1 A2 B2 A3). The original value (20, `proactiveness_before.txt`) was restored after each B block
and at the end of each series.

Each block runs three configurations of `strata-p1d` (6 prompts, 256 tokens, `STRATA_DECODE_TIMING=1`):

| configuration | DRAM tier | slots | drafts | requests |
|---|---|---|---|---|
| `t60_hostalloc_nodraft` | 66.8 GiB (60%) | `cudaHostAlloc` (`STRATA_ARENA_HOSTALLOC=1`) | off | 24 |
| `t60_mtp` | 66.8 GiB (60%) | registered (main configuration) | MTP | 12 |
| `t75_mtp` | 84.7 GiB (75%) | registered (main configuration) | MTP | 12 |

- `run1/`: runs `k_<block>_<configuration>` (15:10-16:16). Only A1 reached the slow state (below).
- `run2/`: runs `k2_<block>_<configuration>` (16:30-17:37). Before every configuration the runner reads a 28 GB file
  (a Mixtral-8x7B GGUF) so that the page cache is full when the engine loads.

Files in each run folder: `runner.log` and `runner_stdout.log` (runner), `sampler.csv` (every 0.5 s: setting,
kcompactd0 and kswapd0 CPU ticks, `/proc/vmstat` compaction counters, free memory, 2 MB fragmentation score),
`summary.json` (per request and per block), `proactiveness_before.txt` (the saved value) and the empty handshake files
`armed`, `ready_B<n>` and `done_B<n>`. The score is
the share of the Normal zone's free memory outside free blocks of order 9 or more (2 MB), the quantity the kernel
compares with its threshold (90 at the default 20; `mm/compaction.c`, Linux 6.17). The kernel weights it over zones;
the DMA zone holds 1.3% of the pages and is left out here.

## Results

Second series, `cudaHostAlloc` slots. "Busy" is a request during which kcompactd ran for more than a tenth of the decode.

Rates in this README are the client's (tokens over the time from the first to the last streamed token, as
`analysis/compaction_ab.py` computes them), about 0.5% above the server's own count. The paper quotes the server's
rates of the same requests: 17.5 against 16.6 tokens/s, with a coefficient of variation of 2.1% against 7.0%.

| block | setting | tokens/s | CV | GPU's part of a window, ms: mean (max) | busy | kcompactd, s |
|---|---|---|---|---|---|---|
| A1 | 20 | 16.7 | 6.9% | 41.7 (49.4) | 13/24 | 101.9 |
| B1 | 0 | 17.7 | 2.1% | 38.5 (40.9) | 1/24 | 3.4 |
| A2 | 20 | 16.1 | 9.0% | 43.5 (54.8) | 14/24 | 142.8 |
| B2 | 0 | 17.5 | 2.1% | 38.8 (42.1) | 1/24 | 4.0 |
| A3 | 20 | 17.3 | 2.7% | 39.7 (42.4) | 11/24 | 31.8 |
| all at 20 | | 16.7 (13.3-18.1) | 7.1% | 41.6 +- 4.1 | 38/72 | 276.5 |
| all at 0 | | 17.6 (16.6-18.2) | 2.1% | 38.7 +- 0.7 | 2/48 | 7.4 |

During every `cudaHostAlloc` decode of the second series, 1.1-1.9 GiB were free and the score was 97-100, at both
settings. Main configuration, both series: 30.1 and 31.7 tokens/s (60% and 75%) at 20 against 30.1 and 31.7 at 0
in the first series, 29.9 and 31.7 against 30.0 and 31.6 in the second; kcompactd ran at most 1.2 s per run.

First series: in A1, 1.2-1.5 GiB were free (score 98-99), and kcompactd ran 294 s in 19 of 24 requests with no
on-demand wake-ups, so it was compacting proactively; the page cache was about 16 GiB larger at the start of A1 than
at the start of later blocks. In B1-A3, 17 GiB were free (score 1-4) and kcompactd did not run at either setting, so
those blocks do not compare the settings. That is why the second series fills the page cache. During B1 of the first
series (15:33-15:37), the release assembly read and wrote about 0.6 GB on the same machine; kcompactd did not run in
B1, and its GPU times (38.6 ms per window on average, at most 39.0) match those of the calm requests of other blocks.

## What was measured

- At 0, kcompactd stopped compacting proactively. In the second series it was busy in 2 of 48 `cudaHostAlloc` requests
  (against 38 of 72 at 20), each the first request of a block: kswapd had woken it repeatedly while the engine loaded
  (`compact_daemon_wake`), the last time within half a second of the first token.
- At 0, the GPU's part of a window and the decode rate stayed stable: 38.7 +- 0.7 ms and 17.6 tokens/s (CV 2.1%)
  against 41.6 +- 4.1 ms and 16.7 tokens/s (CV 7.1%) at 20; the slowest request decoded at 16.6 against 13.3 tokens/s.
- Over all 240 `cudaHostAlloc` requests of both series, the GPU's part of a window rose linearly with the share of the
  decode during which kcompactd ran: 38.5 + 15.3 x share ms (r = 0.990, residual s.d. 0.61 ms). The 172 requests
  without kcompactd took 37.7-39.2 ms. The two busy requests at 0 lie on the same line (residuals -0.9 and -0.4 ms).
- The main configuration (registered slots) was unaffected by the setting.
- The setting was 20 again after each B block and after both series (checked at 17:38).

## What was not measured

- How kcompactd's work slows the GPU. kcompactd0 may run on all 20 cores; in the earlier runs with thread logging
  (`results/pipeline/threadmon.log`, a sample every 2 s) it was never sampled on the core of the engine's host thread.
  We did not log cores in these two series. The slots it slowed are the cudaHostAlloc ones, which the GPU does not
  reach through the host's page tables (paper, Section 4); with the registered slots, which it does, kcompactd hardly
  ran (next item), so the two read paths were not compared under the same load. (An earlier version of this note
  guessed at migrations invalidating host page-table translations; that guess does not fit the slots that slowed.)
- Why kcompactd hardly runs with registered slots, even in blocks whose score exceeded 90 (second series, A2: 86-97,
  at most 0.1 s). It may find little to migrate and defer; we did not test it.
- What the setting costs other workloads. Compaction on demand (kswapd's wake-ups, direct compaction) still runs at 0.
- Whether the slow state arises in other memory states: the first series shows that it depends on how much page cache
  is present when the engine loads.

## Reproducing

`bench/p2/compaction_ab.sh` (first series) and `bench/p2/compaction_ab2.sh` (second series) run the blocks and wait
for the setting to change; `bench/p2/compaction_root.sh` (as root) saves the setting, writes 0 for the B blocks and
restores the saved value after each block and on any exit. `python analysis/compaction_ab.py run1|run2` writes
`summary.json` and prints the tables.
