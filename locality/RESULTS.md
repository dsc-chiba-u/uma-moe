# Expert locality across MoE models, tasks and long contexts

Side study of the UMA × huge-MoE project, dgx-spark (DGX Spark GB10), 2026-10-22. Question: is the expert-reuse
structure found in the Qwen3.8-Flash-Next decode traces (Strata) general across MoE models, tasks and long
contexts, and is expert reuse short-range or does it carry long-range context dependence?

**Status.** Four models are traced and analyzed: gpt-oss-20b (32 experts, top-4), GLM-4.7-Flash (64, top-4 + 1
shared), Qwen3-30B-A3B-Instruct-2507 (128, top-8), Qwen3.8-Flash-Next (512, top-10 + 1 shared). **Pending**
(queued behind the main session's GPU reservation; `results/locality/analysis/tables.md` is refreshed automatically when they
finish, the text below is not): Qwen3.6-35B-A3B (256 experts, top-8 + 1 shared; answers already generated,
trace pending), Mixtral-8x7B-Instruct (8 experts, top-2; not downloaded yet), and a re-trace of GLM-4.7-Flash with
its `[gMASK]<sop>` prefix (see 1.3). All tables: `results/locality/analysis/tables.md`; per-model JSON:
`results/locality/analysis/<model>.json`; how to reproduce: `locality/README.md`.

## Summary

Measured (global cache unless noted; "x tok" = capacity in units of the per-token working set L·k):

* **LRU misses ≈ 2 × Belady misses for every model, task and context length.** Corpus `all` (≈157K tokens per
  model, 10 corpora concatenated): LRU/Belady = 1.77–1.92 at 10 % of the (layer, expert) pairs (gpt-oss: 2.53, where
  LRU thrashes), 2.16–2.33 at 25 %, 2.58–2.69 at 50 %. Over the 10 corpora separately at 25 %: 2.07–2.58. Per-layer
  caches: 1.55–1.82 / 1.92–2.17 / 2.30–2.51 at 10/25/50 %. ARC is within 3.1 points of LRU; perfect LFU has
  1.5–3.3 × LRU's misses at 25 % (1.3–8.6 × over 10–50 %, except where LRU thrashes).
* **Reuse is short-range.** Of all reuses of a (layer, expert) pair, 36–50 % happen at the next token, 74–93 %
  within 16 tokens, 89–99 % within 64, ≥ 96.8 % within 256, and ≤ 0.7 % beyond 1,024 tokens. Consecutive-token
  overlap is 36–50 % (k/E = 2–12.5 %; token-shuffled stream 5–21 %). Shuffling the token order (same popularity)
  raises LRU misses at 25 % from 13–34 % to 57–71 % and Belady misses from 6–16 % to 27–35 %.
* **The lookahead needed to reach the offline optimum grows with the cache size in tokens.** Share of the LRU→MIN
  gap closed by exact knowledge of the next h tokens, 25 % capacity: gpt-oss (2 tok) h=4 79 %, h=16 100 %; GLM and
  Qwen3-30B (4 tok) h=4 40–45 %, h=16 83–93 %, h=64 100 %; Qwen3.8 (12.8 tok) h=4 11 %, h=16 32 %, h=64 75 %,
  h=256 100 %. At 50 %: Qwen3.8 h=256 74 %, Qwen3-30B h=256 98 %. Nothing beyond 256 tokens is needed except for
  the largest caches of the most fine-grained model.
* **History beyond a few dozen tokens does not help an online policy.** Decayed LFU is best at a half-life of 16
  tokens for all four models at 25 % (4 % (Qwen3.8) to 20 % (gpt-oss) fewer misses than LRU); longer half-lives lose
  the gain (τ = 1,024: more misses than LRU for GLM and both Qwen models, 3 % fewer for gpt-oss), and perfect LFU is
  far worse.
* **Popularity is domain-specific; static profiles transfer badly.** A static cache filled from the other five
  domains hits 27–41 % of accesses at 25 % capacity (mean over 6 target domains), the target's own profile
  58–75 %, LRU 67–88 % (gpt-oss: LRU 67.5 vs own profile 62.5). Within one text domain (code, prose, ja, math, the
  novel) the top 10 % of a layer's experts take 50–66 % of the accesses (Qwen3.8), in the mixed stream 34 %.
* **Long context: locality is stationary within a document, but routing depends on distant context.** Over 32K-token
  single-domain documents the 1,024-token-window LRU miss rate moves by ≤ 4 points from the first to the last
  windows, and a static profile from the first 4K tokens keeps its hit rate to the end (Qwen3.8 book 85.9→85.1 %,
  Qwen3-30B code 73.4→74.9 %; whole-document oracle 87–89 % / 70–77 %). Yet the same last 2,048 tokens routed with
  only 0 or 1,024 tokens of preceding context instead of ~30K share only 66–94 % of their expert choices with the
  full-context routing (Qwen3.8 code 66/69 %, book 83/87 %; Qwen3-30B code 81/84 %, book 93/94 %), against
  96.5–97.6 % for the decode-vs-batched numerical noise.
* **The Qwen3.8 numbers reproduce across engines and modes.** llama.cpp teacher-forced routing of the model's own
  chat answers (`chat_gen`) vs the Strata decode trace: LRU 33.0/12.9/3.6 % vs 33.3/13.4/3.9 %, Belady 17.2/5.9/1.9 %
  vs 17.4/6.4/2.0 % at 10/25/50 %. Batched (512-token ubatches) vs one-token decode of the same 512 tokens: 95.8–97.6 %
  of the expert choices identical, cache miss rates within 0.33 points.

Interpretation:

* The Qwen3.8 picture generalizes in shape: ~2× LRU-vs-Belady gap, ARC/LFU no better than LRU, strong
  next-token reuse, domain-dependent popularity. What changes across models is the scale: the gap is fixed at
  ~2× while the horizon needed to close it scales with how many tokens' worth of experts the cache holds, which at a
  fixed memory fraction grows with E/k (fine-grained models). "Short-horizon prediction cannot close the gap" is
  therefore a statement about fine-grained MoE with large caches (Qwen3.8 at ≥ 25 %); for 32–128-expert models at
  ≤ 25 % a 4–16-token oracle closes most of it.
* Reuse as seen by a cache is short-range (tens to a few hundred tokens), but *which* experts are used is shaped by
  the whole document: a slowly varying, document-specific working set (stable over 32K tokens) on top of fast
  token-to-token reuse. A useful predictor would combine a document-level prior with the last few dozen tokens,
  not a long frequency memory.

Open questions: whether an imperfect predictor (MTP drafts, a learned router lookahead) can realize the 64–256-token
horizon that fine-grained models need; whether Qwen3.6 (256e) and Mixtral (8e, top-2) fall on the same curves
(pending); how much quantization (Q2_0 for Qwen3.8) changes routing (not measured).

## 1. What was measured

### 1.1 Models

| model | arch (llama.cpp) | quant | GiB | MoE layers L | experts E | top-k | shared | k/E | L·E pairs | L·k per token | tokens traced |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | gpt-oss | MXFP4 (native) | 11.3 | 24 | 32 | 4 | 0 | 12.5 % | 768 | 96 | 180,127 |
| GLM-4.7-Flash | deepseek2 (MLA, sigmoid + bias gating) | Q4_K_M | 17.0 | 46 | 64 | 4 | 1 | 6.2 % | 2,944 | 184 | 179,003 |
| Qwen3-30B-A3B-Instruct-2507 | qwen3moe | Q4_K_M | 17.3 | 48 | 128 | 8 | 0 | 6.2 % | 6,144 | 384 | 178,792 |
| Qwen3.8-Flash-Next | qwen4exp (hybrid linear/full attention) | Q2_0 (GSQ-RCO) | 61.9 | 48 | 512 | 10 | 1 | 2.0 % | 24,576 | 480 | 178,449 |
| *pending:* Qwen3.6-35B-A3B | qwen35moe | UD-Q4_K_M | 20.6 | 40 | 256 | 8 | 1 | 3.1 % | 10,240 | 320 | – |
| *pending:* Mixtral-8x7B-Instruct-v0.1 | llama | Q4_K_M | 26.5 | 32 | 8 | 2 | 0 | 25 % | 256 | 64 | – |

Token counts include the context probes (≈22K tokens); every analysis except the probes uses the ≈157K-token
`all` stream. Shared experts are dense FFNs used by every token and are not part of the traces.

### 1.2 Corpora (same text for every model; budgets in each model's own tokens)

| corpus | documents | tokens |
|---|---|---|
| code | llama.cpp `conversion/base.py` (Python), `src/llama-graph.cpp` (C++) | 2 × 6,144 |
| prose | Wikipedia "Photosynthesis" + "Roman Empire"; Doyle, *A Scandal in Bohemia* | 2 × 6,144 |
| ja | ja.wikipedia 光合成 + 江戸時代; 夏目漱石『こころ』 (Aozora Bunko) | 2 × 6,144 |
| math | GSM8K test problems with worked solutions; MATH-500 problems with LaTeX solutions | 2 × 6,144 |
| chat | the 6 short prompts of the main study with the model's own greedy answer (≤ 1,024 tokens, thinking off; gpt-oss: reasoning effort low, analysis channel included), teacher-forced as template prompt + answer | 5.5–6.3K |
| dialog | 10-turn conversation (trip planning, a Python script and its revision, Japanese phrases, a question in Japanese, arithmetic, summary, haiku, etiquette); turns 1–9 answered by the model (≤ 384 tokens), the final turn's full context + answer teacher-forced | 3.2–3.9K |
| long_mixed | 8 blocks × 4,096 tokens from other sources (en Wikipedia, Python, ja Wikipedia, MATH, Dickens, CUDA, 宮沢賢治, en Wikipedia): a topic shift every 4,096 tokens | 32,768 |
| long_book | Austen, *Pride and Prejudice* | 32,768 |
| long_code | Strata `setup.py` + `docs/DETAILS.md` + `generate.cpp` (the long prompt of the main study) | 32,768 |
| probes | last 2,048 tokens of each long document again, with 0 / 128 / 1,024 tokens of preceding context | 9 sequences |

`chat_gen` = only the answer tokens of the chat transcripts (the decode-like part, 5.3–5.9K tokens). `all` =
every corpus concatenated (code, prose, ja, math, chat, dialog, long_mixed, long_book, long_code).

### 1.3 Traces

* `locality/moe_trace` (C++, libllama of llama.cpp master 339f324): teacher-forced prompt processing, one sequence
  per document, n_ctx 32,768, 512-token ubatches, all layers on the GPU, flash attention auto. The top-k expert ids
  of every token and MoE layer are read from the `ffn_moe_topk-<layer>` tensor (`build_moe_ffn`,
  `src/llama-graph.cpp`; a view of the router argsort, so ids are in score order) via the scheduler eval callback.
  Every token is an output (otherwise llama.cpp prunes the last layer to the output rows). Stored as uint16
  `[tokens, L, k]` + metadata JSON. Speed with the callback: 660–830 tok/s (Qwen3.8 Q2_0), 860–3,100 tok/s for the
  others (32K-token documents at the low end).
* Validation (first chat transcript; prompt in ubatches, then 512 answer tokens one per `llama_decode`, i.e. the way
  generation computes them): identical expert *sets* in 75–88 % of (token, layer) cells, 95.8–97.6 % of the
  individual expert choices identical, top-1 identical 94–97 %. Mismatches are near-ties at the bottom of the top-k
  (Qwen3.8: 56 % of the differing experts are rank 10, 75 % rank 9–10) and grow with depth (Qwen3.8: 98–99 %
  agreement in the first layers, 94–96 % in the last). LRU/Belady miss rates on the decode routing and on the batched routing differ by
  ≤ 0.33 points.
* Format caveats. gpt-oss cannot model raw text (teacher-forced NLL rises with position to 4–12 nats/token on prose,
  math and the book, with or without wrapping the document as the final-channel answer of a harmony assistant
  turn; NLL is normal on its own answers, 0.5–1.0, and on code, 0.7–1.5). The tables use the harmony-wrapped traces;
  the raw-text traces (kept as `gpt-oss-20b-rawtext`) give the same overlap and LRU/Belady miss rates within 0.3
  points (static-profile and LFU numbers within a few points). GLM-4.7-Flash
  was traced without the `[gMASK]<sop>` prefix that every GLM prompt starts with (its tokenizer adds no BOS); its
  domain documents have normal NLL (0.7–2.4), but two of its nine context probes are unstable (NLL 5.3 and 9.5) and
  the last block of `long_mixed` degrades (NLL 2.95 vs 1.76 for Qwen3-30B); a re-trace with the prefix is queued.
  Qwen models need no BOS. Teacher-forced NLL of every document is in `tables.md`.

### 1.4 Cache model and metrics

* Access stream: token by token, layer by layer, the k experts of a layer; a key is a (layer, expert) pair; the
  documents of a corpus are concatenated and the cache stays warm across them; compulsory misses are counted (they
  are < 0.03 % of the accesses of `all`). Miss rate = misses / accesses (= the mean fraction of a token's expert
  loads that miss; misses per token = miss rate × L·k).
* Capacity: 5/10/25/50/75 % of the L·E pairs. Global cache: one pool over all layers. Per-layer caches:
  round(f·E) slots per layer. Because L·k/L·E = k/E, a fraction f holds f·E/k tokens' worth of experts: 10 % is
  0.8 tok for gpt-oss, 1.6 tok for GLM and Qwen3-30B, 5.1 tok for Qwen3.8. Below 1 tok LRU thrashes (a cyclic scan
  over the layers evicts every key before its reuse): gpt-oss, GLM and Qwen3-30B at 5 %, gpt-oss at 10 %.
* Policies: LRU; LFU (perfect, ties → least recent); ARC; Belady = MIN with bypass (the offline optimum, as in
  `analysis/traces.py`); MIN without bypass; lookahead-h (h = 4/16/64/256 tokens: LRU that knows exactly which keys
  the rest of the current token and the next h tokens use and never evicts those while another victim exists; MIN
  among them otherwise); decayed LFU with half-life τ = 4…1,024 tokens. Null model: the same stream with the token
  order shuffled. Simulator: `locality/locsim` (C++), which reproduces `analysis/traces.py` and
  `analysis/policies_extra.py` on the Strata trace (LRU, LFU, ARC, Belady to 4 decimals).

## 2. Results

### 2.1 Popularity and temporal locality (corpus `all`)

| model | top-10 % experts' share (per layer) | consecutive overlap | overlap, shuffled | reuse ≤ 1 token | ≤ 4 | ≤ 16 | ≤ 64 | ≤ 256 | ≤ 1,024 | > 1,024 | ≤ 16, shuffled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | 28.2 | 49.1 | 21.2 | 49.1 | 74.9 | 92.8 | 98.6 | 99.8 | 100.0 | 0.0 | 88.5 |
| GLM-4.7-Flash | 23.3 | 36.2 | 8.6 | 36.2 | 56.8 | 80.9 | 95.6 | 99.4 | 99.9 | 0.1 | 68.7 |
| Qwen3-30B-A3B | 28.5 | 50.4 | 10.4 | 50.4 | 70.4 | 88.4 | 97.0 | 99.3 | 99.9 | 0.1 | 72.8 |
| Qwen3.8-Flash-Next | 34.2 | 39.6 | 4.6 | 39.6 | 56.6 | 74.5 | 89.3 | 96.8 | 99.3 | 0.7 | 42.2 |
| *Qwen3.8, Strata decode trace (main study)* | 34.1 | 37.3 | – | 37.6 | 55.1 | 74.6 | 89.7 | 97.1 | 99.4 | 0.6 | – |

Reuse shares are % of all reuses (first uses excluded). Per corpus, consecutive overlap ranges over 41–59 %
(gpt-oss), 29–47 % (GLM), 41–59 % (Qwen3-30B), 34–48 % (Qwen3.8); it is highest for Japanese and the novel and
lowest for math and code. The per-layer top-10 % share is much higher inside one domain than in the mixed stream
(Qwen3.8: code 59 %, book 66 %, chat 36 %, `all` 34 %; Qwen3-30B: code 61 %, book 63 %, `all` 29 %). Working set
(distinct experts per layer in a 1,024-token window): gpt-oss 28 of 32, GLM 61 of 64, Qwen3-30B 96 of 128, Qwen3.8
361 of 512.

### 2.2 LRU vs Belady across models and capacities (corpus `all`, miss rate %)

| model | 10 %: tok | LRU | Belady | LRU/Belady | 25 %: tok | LRU | Belady | LRU/Belady | 50 %: tok | LRU | Belady | LRU/Belady |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | 0.8 | 100.0 | 39.6 | 2.53 | 2 | 32.1 | 14.2 | 2.26 | 4 | 7.05 | 2.70 | 2.61 |
| GLM-4.7-Flash | 1.6 | 60.6 | 34.3 | 1.77 | 4 | 33.9 | 15.7 | 2.16 | 8 | 12.1 | 4.51 | 2.69 |
| Qwen3-30B-A3B | 1.6 | 42.3 | 22.6 | 1.88 | 4 | 15.5 | 6.65 | 2.33 | 8 | 2.50 | 0.97 | 2.58 |
| Qwen3.8-Flash-Next | 5.1 | 31.4 | 16.4 | 1.92 | 12.8 | 12.9 | 5.75 | 2.25 | 25.6 | 3.35 | 1.30 | 2.58 |
| *Qwen3.8, Strata decode trace* | 5.1 | 33.3 | 17.4 | 1.91 | 12.8 | 13.4 | 6.37 | 2.11 | 25.6 | 3.89 | 1.98 | 1.96 |

Miss rates at a capacity of m × L·k slots, i.e. m tokens' worth of experts (corpus `all`):

| model | m = 1: LRU / Belady | m = 2 | m = 4 | m = 8 | m = 16 |
|---|---|---|---|---|---|
| gpt-oss-20b | 60.3 / 32.7 | 32.1 / 14.2 | 7.0 / 2.7 | – | – |
| GLM-4.7-Flash | 71.8 / 44.4 | 53.1 / 29.6 | 33.9 / 15.7 | 12.1 / 4.5 | – |
| Qwen3-30B-A3B | 62.1 / 33.5 | 35.2 / 18.0 | 15.5 / 6.6 | 2.5 / 1.0 | – |
| Qwen3.8-Flash-Next | 71.3 / 43.1 | 50.5 / 30.9 | 36.5 / 19.9 | 22.0 / 10.6 | 9.3 / 3.9 |

The ratio grows with m: 1.6–1.9 at m = 1, 1.6–2.3 at m = 2, 1.8–2.6 at m = 4, 2.1–2.7 at m = 8. Per tokens'
worth, gpt-oss and Qwen3-30B miss least and GLM and Qwen3.8 most (GLM's sigmoid gating with a load-balancing bias
gives the flattest popularity, top-10 % share 23 %; Qwen3.8 spreads a token's 10 experts over 512). Qwen3.8 at m = 16
(31 % of the pairs, ≈ the 8,192 slots of the main study) gives LRU 9.3 % vs Belady 3.9 % (main study: 8.7 % vs
4.0 %).

### 2.3 Across tasks (LRU/Belady, global, 25 % capacity)

| model | code | prose | ja | math | chat | chat_gen | dialog | long_mixed | long_book | long_code | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | 2.25 | 2.33 | 2.28 | 2.23 | 2.20 | 2.21 | 2.13 | 2.29 | 2.36 | 2.19 | 2.26 |
| GLM-4.7-Flash | 2.15 | 2.22 | 2.20 | 2.12 | 2.12 | 2.12 | 2.14 | 2.17 | 2.30 | 2.07 | 2.16 |
| Qwen3-30B-A3B | 2.31 | 2.47 | 2.40 | 2.24 | 2.25 | 2.27 | 2.25 | 2.38 | 2.58 | 2.12 | 2.33 |
| Qwen3.8-Flash-Next | 2.19 | 2.32 | 2.22 | 2.15 | 2.17 | 2.18 | 2.08 | 2.25 | 2.34 | 2.19 | 2.25 |

Same at 10 %: 1.74–2.03 (gpt-oss 2.25–2.97 with LRU thrashing); at 50 %: 1.87–2.82; per-layer caches at 25 %:
1.85–2.30. Absolute miss rates differ by task (LRU at 25 %: Japanese 7–24 % and the novel 12–29 % are the easiest;
math 16–41 %, dialog 14–37 % and, except for Qwen3-30B, code the hardest), the ratio barely does.

### 2.4 How far ahead and how far back

Lookahead-h, global, corpus `all`: miss rate % (share of the LRU → MIN-without-bypass gap closed).

| model | capacity | LRU | h = 4 | h = 16 | h = 64 | h = 256 | MIN | Belady |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | 25 % (2 tok) | 32.1 | 18.0 (79 %) | 14.3 (100 %) | 14.3 | 14.3 | 14.3 | 14.2 |
| gpt-oss-20b | 50 % (4 tok) | 7.05 | 5.36 (39 %) | 3.62 (79 %) | 2.72 (100 %) | 2.71 | 2.71 | 2.70 |
| GLM-4.7-Flash | 25 % (4 tok) | 33.9 | 25.8 (45 %) | 17.0 (93 %) | 15.7 (100 %) | 15.7 | 15.7 | 15.7 |
| GLM-4.7-Flash | 50 % (8 tok) | 12.1 | 10.3 (24 %) | 7.73 (58 %) | 4.70 (97 %) | 4.51 | 4.51 | 4.51 |
| Qwen3-30B-A3B | 25 % (4 tok) | 15.5 | 11.9 (40 %) | 8.13 (83 %) | 6.65 (100 %) | 6.65 | 6.65 | 6.65 |
| Qwen3-30B-A3B | 50 % (8 tok) | 2.50 | 2.28 (15 %) | 1.95 (36 %) | 1.36 (75 %) | 1.00 (98 %) | 0.97 | 0.97 |
| Qwen3.8-Flash-Next | 25 % (12.8 tok) | 12.9 | 12.1 (11 %) | 10.6 (32 %) | 7.56 (75 %) | 5.76 (100 %) | 5.75 | 5.75 |
| Qwen3.8-Flash-Next | 50 % (25.6 tok) | 3.35 | 3.27 (4 %) | 3.11 (12 %) | 2.66 (34 %) | 1.82 (74 %) | 1.30 | 1.30 |

Closing ~75 % of the gap takes a horizon of about 2–10 × the cache capacity in tokens, closing all of it 8–32 ×.
Bypass (not caching a key whose next use is furthest) is worth ≤ 0.3 points. Decayed LFU, miss rate % at 25 %:

| model | LRU | τ = 4 | τ = 16 | τ = 64 | τ = 256 | τ = 1,024 | LFU |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b | 32.1 | 27.7 | **25.7** | 26.3 | 28.0 | 31.0 | 47.5 |
| GLM-4.7-Flash | 33.9 | 32.1 | **30.3** | 31.0 | 33.2 | 37.0 | 56.3 |
| Qwen3-30B-A3B | 15.5 | 14.8 | **14.2** | 15.4 | 18.0 | 22.7 | 47.9 |
| Qwen3.8-Flash-Next | 12.9 | 12.8 | **12.4** | 12.4 | 14.9 | 19.5 | 42.3 |

Same pattern at 10 % (best τ = 16 for all) and 50 % (τ = 16; τ = 64 for GLM and Qwen3.8).

### 2.5 Domain shift (static frequency cache, hit rate %, global, mean over the 6 target domains)

| model | 10 %: own profile / other 5 domains / LRU | 25 % | 50 % |
|---|---|---|---|
| gpt-oss-20b | 35.4 / 21.6 / 0.0 (thrash) | 62.5 / 40.5 / 67.5 | 86.4 / 65.2 / 92.4 |
| GLM-4.7-Flash | 35.0 / 12.4 / 40.6 | 57.5 / 28.4 / 66.6 | 81.0 / 52.7 / 87.9 |
| Qwen3-30B-A3B | 42.8 / 10.8 / 55.9 | 69.0 / 26.6 / 83.0 | 90.3 / 52.3 / 96.9 |
| Qwen3.8-Flash-Next | 50.6 / 16.4 / 69.0 | 74.7 / 35.2 / 87.7 | 92.7 / 64.1 / 96.6 |

Code is the extreme case: the profile of the other domains hits 6–20 % of code accesses at 25 % (own profile 62–86 %).
The single best other corpus reaches 31–80 % (code ← long_code, prose ← long_book, ja ← long_mixed, chat ↔
dialog). In the mixed long document
a block's own profile hits 65–86 %, another block of the same domain 39–61 %, a block of another domain 27–49 %.

### 2.6 Long context

Per-1,024-token window, global cache, 10 % capacity (windows 2–4 → last 4 of 32):

| model | document | LRU miss % | Belady miss % | experts per layer per window | static cache from the first 4K tokens, hit @ 25 % (windows 5–8 → last 4) | whole-document profile |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | long_book | 39.0 → 37.6 | 21.3 → 20.6 | 88 → 79 | 87.5 → 87.2 | 86.6 → 88.6 |
| Qwen3-30B-A3B | long_code | 47.7 → 45.5 | 24.9 → 23.8 | 106 → 103 | 73.4 → 74.9 | 70.4 → 77.0 |
| Qwen3.8-Flash-Next | long_book | 30.1 → 31.5 | 15.1 → 15.7 | 340 → 347 | 85.9 → 85.1 | 87.4 → 87.1 |
| Qwen3.8-Flash-Next | long_code | 35.7 → 39.2 | 19.3 → 21.7 | 380 → 398 | 68.9 → 69.3 | 70.2 → 74.5 |
| GLM-4.7-Flash | long_book | 54.7 → 56.4 | 31.0 → 31.9 | 57 → 57 | 74.9 → 73.4 | 74.7 → 74.1 |
| gpt-oss-20b | long_code | thrash | 45.1 → 44.4 | 31 → 31 | 56.1 → 56.3 | 53.9 → 57.5 |

In `long_mixed` the profile of the first 4K tokens (block 0, en Wikipedia) hits 9–25 % in block 1 (Python) and
39–64 % in the last block (en Wikipedia again). Context probes, expert-choice overlap (%) of the last 2,048 tokens
routed with c tokens of preceding context vs with the full ~30K tokens:

| model | document | c = 0 | c = 128 | c = 1,024 | numerical noise floor | NLL c = 0 / 1,024 / full |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | long_book | 93.0 | 93.4 | 94.3 | 97.6 | 1.55 / 1.57 / 1.32 |
| Qwen3-30B-A3B | long_code | 81.3 | 81.8 | 83.6 | 97.6 | 2.20 / 2.01 / 1.39 |
| Qwen3-30B-A3B | long_mixed | 92.6 | 92.8 | 94.1 | 97.6 | 2.15 / 2.13 / 2.17 |
| Qwen3.8-Flash-Next | long_book | 83.0 | 84.8 | 87.4 | 96.7 | 0.25 / 0.15 / 0.09 |
| Qwen3.8-Flash-Next | long_code | 65.5 | 66.5 | 69.4 | 96.7 | 1.54 / 1.27 / 0.84 |
| Qwen3.8-Flash-Next | long_mixed | 89.2 | 90.6 | 92.2 | 96.7 | 0.60 / 0.53 / 0.62 |
| GLM-4.7-Flash | long_book | 90.9 | 91.5 | 93.1 | 96.5 | 1.26 / 1.23 / 1.28 |

(GLM `long_code`/`long_mixed` probes and all gpt-oss probes are omitted here: unstable or out-of-distribution, see
1.3; they are in `tables.md`.) The difference is not a start-of-sequence effect: excluding the first 64 tokens of
the segment changes the overlap by < 1 point, and in Qwen3.8 `long_code` it is 53 % in the first 256 tokens and still
66–76 % in the later 256-token stretches. It is largest in deep layers (Qwen3.8 `long_code`, c = 0: 93 % in layer 0,
54–64 % in layers ≥ 16). Two occurrences of the same token id ≥ 256 tokens apart share 38–63 % of their experts
(random positions: 5–21 %), most in the first third of the layers.

### 2.7 Per-layer vs global caches

Global pooling helps Belady (9–33 % fewer misses than per-layer caches at 25–50 %) and LRU at 50 % (gpt-oss 7.0 vs
9.2 %, Qwen3.8 3.3 vs 3.9 %); at 25 % LRU is within 1.1 points either way. Per-layer caches avoid LRU's cyclic
thrashing below 1 tok (gpt-oss 10 %: 100 → 83.9 %). The LRU/Belady ratio is slightly smaller per layer (1.92–2.17 at
25 %).

## 3. Conclusions

Measured facts:

1. In all four models LRU misses are 1.8–2.7 × Belady misses at 10–50 % capacity, on every task and in 32K-token
   documents; ARC is within 3.1 points of LRU; LFU has 1.5–3.3 × LRU's misses at 25 %.
2. 74–93 % of expert reuses occur within 16 tokens and ≥ 96.8 % within 256 tokens; consecutive-token overlap is
   36–50 %, 2–9 × the token-shuffled value; removing temporal order raises Belady's misses 2.2–4.7 × at 25 %.
3. Exact lookahead of 16 tokens closes 79–100 % of the LRU→MIN gap when the cache holds ≤ 4 tokens' worth of
   experts; Qwen3.8 at 12.8 tokens' worth needs 64 tokens for 75 % and 256 for 100 %.
4. Frequency memory beyond ~16–64 tokens makes an online policy worse; static profiles from other domains reach
   39–65 % of the hit rate of the domain's own profile, and at ≥ 25 % LRU beats even the own-profile oracle on
   average over the domains.
5. Windowed miss rates, working sets and a first-4K-token profile are stable over 32K-token single-domain documents;
   topic shifts reset the profile. Routing of the same tokens changes in 6–31 % of expert choices when the context
   beyond the last 1,024 tokens is removed (numerical noise: 2.4–3.5 %).

Interpretations:

* The Qwen3.8 findings (2× gap, ARC no better than LRU, short lookahead insufficient, static profiles fragile)
  generalize qualitatively. The quantitative "a 4–16-token lookahead barely helps" is specific to caches that hold
  many tokens' worth of experts, which is what a fixed memory fraction means for a model with large E/k; for the
  coarser models the same fraction is only 2–4 tokens' worth and short lookahead suffices.
* Expert reuse relevant to caching is short- to medium-range (16–256 tokens). Long-range context enters through
  *which* experts a document uses (a stable per-document working set and context-dependent routing), not through
  long reuse distances; exploiting it means predicting the document's expert distribution, not remembering
  frequencies over thousands of tokens.
* Since teacher-forced routing reproduces decode routing (≥ 95.8 % of choices, cache metrics within 0.33 points)
  and the Strata trace, batched teacher forcing is an adequate, much cheaper way to collect routing traces.

## 4. Limitations

* Quantization: Q2_0 for Qwen3.8, Q4_K_M/MXFP4 for the others; router decisions near ties depend on it (the
  decode-vs-batched noise alone flips 2.4–4.2 % of choices). Not compared against BF16 routing.
* Teacher forcing over fixed text: apart from `chat`/`dialog` the text was not generated by the model; gpt-oss does
  not model plain prose/math/novel text (NLL 4–12), so its numbers on those corpora describe routing of text the model
  does not predict (its per-corpus statistics follow the same pattern as the other models', e.g. LRU/Belady
  2.13–2.36 at 25 % across all its corpora, in- and out-of-distribution).
  GLM was traced without its BOS-like prefix (see 1.3).
* One sequence at a time (batch size 1): no cross-request sharing of a cache between concurrent users.
* Corpora are small (≈12K tokens per domain, three 32K documents) and partly memorized by the models (llama.cpp
  code, *Pride and Prejudice*: Qwen3.8 NLL 0.12).
* The cache model counts expert loads only (equal size per expert, no prefetch overlap, no batching of misses);
  the shared expert and attention weights are assumed resident. Per-layer caches use equal capacities per layer.
* Lookahead-h is an oracle; realistic predictors (MTP drafts, router predictors) are not evaluated here.
* Pending: Qwen3.6-35B-A3B (256 experts) and Mixtral-8x7B (8 experts, top-2), GLM with prefix.

## 5. Files

`locality/README.md` (pipeline and reproduction), `results/locality/<model>/{gen,tokens,trace}/` (answers, token
ids, routing traces, NLL, metadata), `results/locality/analysis/<model>.json`, `results/locality/analysis/tables.md`
(all tables, refreshed by `locality/watch_refresh.sh`), `results/locality/analysis/strata_decode_ref.json` (the main
study's Strata trace through the same simulator). The GLM-4.7-Flash numbers above come from
`results/locality/glm-4.7-flash-nohead/` (analysis key `glm-4.7-flash-nohead`); the queued re-trace with the
`[gMASK]<sop>` prefix writes `results/locality/glm-4.7-flash/trace/`. gpt-oss-20b: `results/locality/gpt-oss-20b/`
(harmony-wrapped documents, used above) and `results/locality/gpt-oss-20b-rawtext/` (raw text).
