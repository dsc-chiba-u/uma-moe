# Expert-locality study across MoE models (llama.cpp routing traces)

Side study of the UMA × huge-MoE project: is the expert-reuse structure measured on Qwen3.8-Flash-Next decode traces
(Strata engine) general across MoE models, tasks and long contexts? Results and discussion: `RESULTS.md`.

Code lives here (`~/work/uma-moe/locality/`), data in `~/work/uma-moe/results/locality/`, models in
`~/models/locality/` (the Qwen3.8 Q2_0 GGUF is the one in `~/src/Strata-data/models/Q2_0/`).

## Pieces

| file | what |
|---|---|
| `moe_trace.cpp` | trace collector linked against libllama (llama.cpp master 339f324). `tokenize`: vocab-only tokenization. `trace`: teacher-forced prompt processing in 512-token ubatches (one sequence, n_ctx 32,768, flash attention auto, all layers on the GPU) that records, for every token and MoE layer, the top-k expert ids from the `ffn_moe_topk-<layer>` tensors (`build_moe_ffn`, `src/llama-graph.cpp`) through the scheduler eval callback; every token is marked as an output so the last layer is not pruned; also per-token NLL; `--validate DOC:P:N` re-runs a document with tokens P..P+N-1 decoded one per `llama_decode` (what generation does) and reports the agreement with the batched routing |
| `locsim.cpp` | cache simulator and locality statistics (LRU, LFU, ARC, Belady/MIN with and without bypass, lookahead-h, decayed LFU; global and per-layer caches; reuse distance in tokens; consecutive overlap; working-set growth; LRU stack distances). Reproduces `analysis/traces.py` and `analysis/policies_extra.py` on the Strata trace (LRU, LFU, ARC, Belady and lookahead-1 identical to 4 decimals, lookahead-4 within 2e-4; see the note on lookahead-h below) |
| `models.py` | the models (HF repo/file, quant, routing configuration from the GGUF header, chat-template kwargs) |
| `gguf_head.py` | reads GGUF key/values from the first bytes of a local or `hf://` file (used to pick the models) |
| `download.py`, `dl_all.sh` | downloads, each under the shared GPU lock |
| `corpora.py` | builds the model-independent text corpora (`results/locality/corpora/`) |
| `run_model.py` | per model: `gen` (llama-server, greedy answers to the 6 short prompts of `bench/prompts.py` and a 10-turn dialog), `tok` (tokenize and cut the corpora to the token budgets), `trace` (moe_trace) |
| `run_all.sh` | runs gen → tok → trace for a list of models, each GPU stage through `locked_job.sh`; a stage that finds too little free memory exits 75 and that model is retried in a later round |
| `locked_job.sh` | runs one command under `flock ~/work/uma-moe/.gpu.lock`, but never while `~/work/uma-moe/.main-reserved` exists (polls every 60 s, re-checks after taking the lock) |
| `chain_rest.sh`, `watch_refresh.sh` | follow-up queue (GLM re-trace with `[gMASK]<sop>`, Mixtral download) and a CPU-only watcher that re-runs `analyze.py` for every new trace and regenerates `results/locality/analysis/tables.md` |
| `strata_ref.py` | the main study's Strata decode trace (`results/trace_nodraft/routing.bin`) through the same simulator → `results/locality/analysis/strata_decode_ref.json` |
| `analyze.py` | runs locsim over every corpus of every model → `results/locality/analysis/<model>.json` |
| `report.py` | markdown tables from the analysis JSON (`results/locality/analysis/tables.md`) |
| `llama-bin/` | snapshot of `~/src/llama.cpp-master/build/bin` (339f324) so that a rebuild of the main tree cannot change the libraries under a running job; run binaries with `LD_LIBRARY_PATH=llama-bin` |

## Document formats per model

Raw-text documents start with the BOS token(s) the tokenizer adds (Mixtral `<s>`; none for Qwen, gpt-oss, GLM),
except: gpt-oss-20b documents are wrapped as the final-channel answer of a harmony assistant turn (system prompt of
the chat template + `<|start|>user<|message|>Write a document.<|end|><|start|>assistant<|channel|>final<|message|>`,
74 tokens), because gpt-oss does not model raw text (`models.py: doc_wrap`; the raw-text traces are kept in
`results/locality/gpt-oss-20b-rawtext/`); GLM-4.7-Flash documents start with `[gMASK]<sop>` (`doc_prefix`; the first
trace without it is kept in `results/locality/glm-4.7-flash-nohead/`). Chat and dialog transcripts are the chat
template's rendering (llama-server `/apply-template` + `/tokenize`) followed by the generated token ids
(`/completion` with `return_tokens`), so they are exactly the sequence the model saw while generating.

## Trace format

`results/locality/<model>/trace/<doc>.route.npy`: uint16 `[n_tokens, n_moe_layers, k]`, expert ids in router-score
order (best first); `<doc>.nll.npy`: float32 `[n_tokens]` teacher-forced NLL of the next token;
`trace_meta.json`: model path, arch, quant (`desc`, `file_type`), bytes, n_layer, n_expert, k, MoE layer ids,
shared-expert count, n_ctx, n_ubatch, validation result, per-document token counts and mean NLL.
`results/locality/<model>/tokens/docs.json`: corpus of each document, token counts, prompt/answer split of the chat
and dialog transcripts, block boundaries of the mixed long document; `*.tok` are the int32 token ids.
`results/locality/<model>/gen/gen.json`: the generated answers (text), server command, timings.

## Reproduce

```bash
cd ~/work/uma-moe
locality/build.sh                                   # moe_trace (links llama-bin/) and locsim
.venv/bin/python -I locality/corpora.py             # text corpora (downloads Wikipedia/Gutenberg/Aozora/GSM8K/MATH-500 once)
locality/dl_all.sh gpt-oss-20b qwen3-30b-a3b glm-4.7-flash qwen3.6-35b-a3b mixtral-8x7b
locality/run_all.sh gpt-oss-20b qwen3-30b-a3b glm-4.7-flash qwen3.6-35b-a3b mixtral-8x7b qwen3.8-flash-next
.venv/bin/python -I locality/analyze.py             # 1-3 min per model, CPU only (nice 19, 4 threads, < 2 GB)
.venv/bin/python -I locality/strata_ref.py          # the Strata decode trace through the same analysis
.venv/bin/python -I locality/report.py > results/locality/analysis/tables.md
```

Generation of the chat answers takes 2-6 min per model (llama-server, greedy, 6 × ≤ 1,024 + 10 × ≤ 384 tokens), a
trace 3-5 min (≈ 179K tokens), so each locked job stays well under 10 minutes.

Every GPU stage refuses to start unless MemAvailable ≥ model size + 26 GiB and is killed if MemAvailable falls
under 12 GiB (watchdog in `run_model.py`, and `moe_trace` checks before every `llama_decode`).

## Notes

* lookahead-h here evicts the least recently used key that is not used in the rest of the current token and the
  next h tokens, and, if every resident key is used within that horizon, the one used furthest (MIN within the
  horizon). `analysis/policies_extra.py` fell back to plain LRU in that case and scanned only the 512 least recent
  keys; the two agree except when nearly all resident keys are inside the horizon (small caches, long horizons):
  on the Strata trace at 8,192 slots la4/la16 give 8.38 %/7.65 % misses here vs 8.38 %/7.65 % there, at 2,048 slots
  la16 gives 22.9 % here vs 30.0 % there.
* Belady = MIN with bypass (a key whose next use is furthest, possibly the one just used, is not kept), as in
  `analysis/traces.py`; `belady_nb` always admits the accessed key and is the h → ∞ limit of lookahead-h.
