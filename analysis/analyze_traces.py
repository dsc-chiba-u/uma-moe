"""E3 analysis: expert locality and cache-policy hit rates from the routing traces.

    python analysis/analyze_traces.py            -> results/trace_analysis.json and a printed summary
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import traces as T

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PROFILE = Path.home() / "src/Strata/data/expert-profile.bin"
CAPS = [1024, 2048, 4096, 6144, 8192, 12288, 16384, 20480]   # cache slots out of 24,576 (layer, expert) pairs


def split_requests(tok, run_json):
    """Cut the token stream into requests using the server's per-request counts: a short prompt (<= 64 tokens) is
    read through decode windows and therefore traced, a long prompt is prefilled in batches and is not; every
    generated token is traced. A few tokens per request are unaccounted for; they go to the last request."""
    stats = json.load(open(run_json))["server_stats"]
    bounds, pos = [], 0
    for r in stats:
        n = (r["prompt_tokens"] if r["prompt_tokens"] <= 64 else 0) + r["gen_tokens"]
        bounds.append((r["prompt_id"], pos, pos + n)); pos += n
    pid, a, _ = bounds[-1]; bounds[-1] = (pid, a, len(tok))
    return [(pid, tok[a:b]) for pid, a, b in bounds]


def main():
    out = {}
    nod = T.load_windows(RES / "trace_nodraft/routing.bin")
    mtp = T.load_windows(RES / "trace_mtp/routing.bin")
    tok = T.tokens(nod)
    stream = T.keys(tok)
    out["nodraft"] = {"windows": len(nod), "tokens": int(len(tok)), "accesses": int(len(stream)),
                      "window_sizes": np.bincount([len(w) for w in nod]).tolist()}
    # ---- popularity skew
    freq = np.bincount(stream, minlength=T.N_LAYER * T.N_EXPERT).reshape(T.N_LAYER, T.N_EXPERT)
    srt = -np.sort(-freq, axis=1)
    cov = np.cumsum(srt, axis=1) / srt.sum(axis=1, keepdims=True)
    out["popularity"] = {
        "never_used_pairs": int((freq == 0).sum()),
        "share_of_accesses_top10pct_experts": float(cov[:, 51 - 1].mean()),
        "share_top25pct": float(cov[:, 128 - 1].mean()),
        "share_top50pct": float(cov[:, 256 - 1].mean()),
        "experts_for_80pct_mean": float((cov < 0.8).sum(axis=1).mean() + 1),
        "experts_for_90pct_mean": float((cov < 0.9).sum(axis=1).mean() + 1),
    }
    # ---- temporal locality
    out["locality"] = {"consecutive_overlap": T.consecutive_overlap(tok)}
    rd = T.reuse_distance_tokens(tok)
    rdv = rd[rd > 0]
    out["locality"].update({f"reuse_within_{d}": float((rdv <= d).mean()) for d in (1, 2, 4, 8, 16, 64, 256)})
    out["locality"]["first_use_share"] = float((rd < 0).mean())
    out["union_growth_nodraft"] = T.union_growth(tok, 16)
    # ---- per-request popularity similarity (domain dependence)
    reqs = split_requests(tok, RES / "trace_nodraft/run.json")
    names = [p for p, _ in reqs]
    rq_tok = [t for _, t in reqs]
    out["requests_tokens"] = [int(len(r)) for r in rq_tok]
    # profile from all other requests -> static hit rate on this request, vs oracle on itself
    cross = {}
    for i, rt in enumerate(rq_tok):
        s_i = T.keys(rt)
        others = np.concatenate([T.keys(x) for j, x in enumerate(rq_tok) if j != i])
        rank_o = T.static_oracle_ranking(others); rank_self = T.static_oracle_ranking(s_i)
        cross[names[i] if i < len(names) else f"req{i}"] = {
            str(c): {"self_oracle": float(T.sim_static(s_i, rank_self, c)), "profile_from_others": float(T.sim_static(s_i, rank_o, c))}
            for c in (4096, 8192, 12288)}
    out["domain_shift"] = cross
    # ---- cache policies over the whole stream
    prof = T.read_profile(PROFILE)
    rank_oracle = T.static_oracle_ranking(stream)
    nxt = T.next_use(stream)
    pol = {}
    for c in CAPS:
        t0 = time.time()
        pol[c] = {"static_strata_profile": float(T.sim_static(stream, prof, c)),
                  "static_oracle": float(T.sim_static(stream, rank_oracle, c)),
                  "lru": T.sim_lru(stream, c), "slru": T.sim_slru(stream, c), "lfu": T.sim_lfu(stream, c),
                  "belady": T.sim_belady(stream, c, nxt)}
        print(c, {k: round(v, 4) for k, v in pol[c].items()}, f"{time.time() - t0:.0f}s", flush=True)
    out["policies"] = {str(k): v for k, v in pol.items()}
    # ---- MTP verify windows
    ws = [w for w in mtp if len(w) <= 6]   # windows > 6 tokens read prompts
    sizes = np.bincount([len(w) for w in ws], minlength=7)
    uni = {}
    for n in range(1, 7):
        sel = [w for w in ws if len(w) == n]
        if sel:
            u = np.mean([np.mean([len(np.unique(w[:, l, :])) for l in range(T.N_LAYER)]) for w in sel])
            uni[n] = float(u)
    tot_tok = sum(len(w) for w in ws)
    out["mtp"] = {"verify_windows": len(ws), "window_size_hist": sizes.tolist(), "mean_window": float(tot_tok / len(ws)),
                  "union_per_layer_by_window_size": uni,
                  "mean_union_per_layer": float(np.mean([np.mean([len(np.unique(w[:, l, :])) for l in range(T.N_LAYER)]) for w in ws]))}
    json.dump(out, open(RES / "trace_analysis.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("policies", "domain_shift")}, indent=1))


if __name__ == "__main__":
    main()
