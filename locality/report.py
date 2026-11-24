"""Markdown tables from results/locality/analysis/<model>.json (used to write locality/RESULTS.md).

    python -I locality/report.py [KEY ...] > results/locality/analysis/tables.md
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import MODELS  # noqa: E402

RES = Path(os.environ.get("LOC_RES", HERE.parent / "results/locality"))
CAPS = [0.05, 0.10, 0.25, 0.50, 0.75]
ORDER = ["mixtral-8x7b", "gpt-oss-20b", "glm-4.7-flash", "qwen3-30b-a3b", "qwen3.6-35b-a3b", "qwen3.8-flash-next"]
VARIANTS = {"gpt-oss-20b-rawtext": "gpt-oss-20b (raw text, no harmony wrapper)",
            "glm-4.7-flash-nohead": "GLM-4.7-Flash (no [gMASK]<sop> prefix)"}
CORPORA = ["code", "prose", "ja", "math", "chat", "chat_gen", "dialog", "long_mixed", "long_book", "long_code", "all"]


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def load(keys):
    res = {}
    for k in keys:
        p = RES / "analysis" / f"{k}.json"
        if p.exists():
            res[k] = json.load(open(p))
    return res


def label(k):
    return MODELS[k]["label"] if k in MODELS else VARIANTS.get(k, k)


def main():
    keys = sys.argv[1:] or [k for k in ORDER + list(VARIANTS) if (RES / "analysis" / f"{k}.json").exists()]
    R = load(keys)
    ci = {c: i for i, c in enumerate(CAPS)}
    print("## Models and traces\n")
    rows = []
    for k, r in R.items():
        m, meta = MODELS.get(k, {}), r["meta"]
        L, E, kk = len(meta["moe_layers"]), meta["n_expert"], meta["k"]
        ntok = sum(d["n_tokens"] for d in r["docs"].values())
        v = meta.get("validation") or {}
        m = m or MODELS.get(k.rsplit("-", 1)[0], {})
        rows.append([label(k), meta.get("arch", ""), m.get("quant", ""), f"{meta.get('model_bytes', 0) / 2**30:.1f}",
                     L, E, kk, m.get("shared", ""), f"{100 * kk / E:.1f}", L * E, L * kk, ntok,
                     f"{pct(v.get('set_identical'))} / {pct(v.get('mean_overlap'))}" if v else "–"])
    print(table(["model", "arch", "quant", "GiB", "MoE layers", "experts", "top-k", "shared", "k/E %", "pairs L·E",
                 "accesses/token L·k", "tokens traced", "decode vs batched: identical sets % / expert overlap %"], rows))

    print("\n## Teacher-forced NLL per document (sanity check, nats/token)\n")
    docs = ["code_py", "code_cpp", "en_wiki", "en_fiction", "ja_wiki", "ja_lit", "math_gsm8k", "math_comp", "chat_code_en",
            "dialog", "long_mixed", "long_book", "long_code"]
    print(table(["model"] + docs, [[label(k)] + [f2(r["nll"].get(d)) for d in docs] for k, r in R.items()]))

    print("\n## Tokens per corpus\n")
    print(table(["model"] + CORPORA, [[label(k)] + [r["corpora"][c]["T"] for c in CORPORA] for k, r in R.items()]))

    for corpus in ("chat_gen", "all"):
        print(f"\n## Miss rate (% of expert accesses), global cache, corpus `{corpus}`\n")
        rows = []
        for k, r in R.items():
            g = r["corpora"][corpus]["global"]["miss_rate"]
            for pol in ("lru", "arc", "lfu", "belady"):
                rows.append([label(k), pol] + [pct(x, 2) for x in g[pol]])
        print(table(["model", "policy"] + [f"{int(c * 100)}%" for c in CAPS], rows))

    for scope in ("global", "per_layer"):
        print(f"\n## LRU / Belady miss ratio by corpus, {scope} cache\n")
        for c in (0.10, 0.25, 0.50):
            rows = []
            for k, r in R.items():
                rows.append([label(k)] + [f2(r["corpora"][cp][scope]["lru_over_belady"][ci[c]]) for cp in CORPORA])
            print(f"capacity {int(c * 100)}%\n")
            print(table(["model"] + CORPORA, rows))

    print("\n## Lookahead-h: miss rate (%) and share of the LRU -> MIN(no bypass) gap closed, global cache, corpus `all`\n")
    for c in (0.10, 0.25, 0.50):
        rows = []
        for k, r in R.items():
            g = r["corpora"]["all"]["global"]["miss_rate"]
            lru, mn = g["lru"][ci[c]], g["belady_nb"][ci[c]]
            cells = [pct(lru, 2)]
            for h in (4, 16, 64, 256):
                x = g[f"la{h}"][ci[c]]
                cells.append(f"{pct(x, 2)} ({100 * (lru - x) / (lru - mn):.0f}%)" if lru > mn else pct(x, 2))
            cells += [pct(mn, 2), pct(g["belady"][ci[c]], 2)]
            rows.append([label(k)] + cells)
        print(f"capacity {int(c * 100)}%\n")
        print(table(["model", "LRU", "la4", "la16", "la64", "la256", "MIN (admit)", "Belady (bypass)"], rows))

    print("\n## Decayed LFU (half-life in tokens) vs LRU and LFU: miss rate (%), global cache, corpus `all`\n")
    for c in (0.10, 0.25, 0.50):
        rows = []
        for k, r in R.items():
            g = r["corpora"]["all"]["global"]["miss_rate"]
            cells = [pct(g["lru"][ci[c]], 2)] + [pct(g[f"dlfu{t}"][ci[c]], 2) for t in (4, 16, 64, 256, 1024)] + [pct(g["lfu"][ci[c]], 2)]
            rows.append([label(k)] + cells)
        print(f"capacity {int(c * 100)}%\n")
        print(table(["model", "LRU", "τ=4", "τ=16", "τ=64", "τ=256", "τ=1024", "LFU"], rows))

    print("\n## Temporal locality vs popularity: corpus `all`\n")
    rows = []
    for k, r in R.items():
        a = r["corpora"]["all"]
        meta = r["meta"]
        kE = meta["k"] / meta["n_expert"]
        rw = a["reuse_share_within"]
        rows.append([label(k), pct(a["popularity"]["top10pct_share_per_layer"]), pct(a["overlap"]), pct(a["shuffled"]["overlap"]),
                     pct(kE)] + [pct(rw[str(d)]) for d in (1, 4, 16, 64, 256, 1024)] + [pct(1 - rw["1024"]), pct(a["first_use_share"], 2)])
    print(table(["model", "top-10% experts' share", "consecutive overlap", "overlap, shuffled", "k/E",
                 "reuse ≤1 tok", "≤4", "≤16", "≤64", "≤256", "≤1024", ">1024", "first uses"], rows))

    print("\n## Real vs token-shuffled stream: miss rate (%), global cache, corpus `all`\n")
    rows = []
    for k, r in R.items():
        a = r["corpora"]["all"]
        for pol in ("lru", "lfu", "belady"):
            rows.append([label(k), pol] + [f"{pct(a['global']['miss_rate'][pol][i], 1)} / {pct(a['shuffled']['global']['miss_rate'][pol][i], 1)}"
                                           for i in range(len(CAPS))])
    print(table(["model", "policy (real / shuffled)"] + [f"{int(c * 100)}%" for c in CAPS], rows))

    print("\n## Per-layer vs global cache (LRU and Belady miss rate %, corpus `all`)\n")
    rows = []
    for k, r in R.items():
        a = r["corpora"]["all"]
        for pol in ("lru", "belady"):
            rows.append([label(k), pol] + [f"{pct(a['global']['miss_rate'][pol][i], 1)} / {pct(a['per_layer']['miss_rate'][pol][i], 1)}"
                                           for i in range(len(CAPS))])
    print(table(["model", "policy (global / per-layer)"] + [f"{int(c * 100)}%" for c in CAPS], rows))

    print("\n## Domain shift: static frequency cache, hit rate (%) at 25 % capacity, global\n")
    rows = []
    for k, r in R.items():
        ds = r["domain_shift"]
        for tgt, row in ds.items():
            fr = row["from"]["global"]
            best_other = max(v[ci[0.25]] for s, v in fr.items() if s not in ("other_domains",))
            worst_other = min(v[ci[0.25]] for s, v in fr.items() if s not in ("other_domains",))
            rows.append([label(k), tgt, pct(row["self_oracle"]["global"][ci[0.25]]), pct(fr["other_domains"][ci[0.25]]),
                         pct(worst_other), pct(best_other), pct(row["lru"]["global"][ci[0.25]])])
    print(table(["model", "target corpus", "own profile (oracle)", "profile of the other 5 domains", "worst single other",
                 "best single other", "LRU"], rows))

    print("\n## Long documents: per-1,024-token-window LRU miss rate (%) at 10 % capacity: first 4 windows / last 4 windows\n")
    rows = []
    for k, r in R.items():
        for name in ("long_book", "long_code", "long_mixed"):
            lw = r["long"][name]
            lru = lw["miss_rate"]["lru"]["0.1"]
            bel = lw["miss_rate"]["belady"]["0.1"]
            rows.append([label(k), name, f"{pct(np.mean(lru[1:4]), 1)} / {pct(np.mean(lru[-4:]), 1)}",
                         f"{pct(np.mean(bel[1:4]), 1)} / {pct(np.mean(bel[-4:]), 1)}",
                         f"{pct(np.mean(lw['overlap'][:4]))} / {pct(np.mean(lw['overlap'][-4:]))}",
                         f"{np.mean(lw['union_per_layer'][:4]):.1f} / {np.mean(lw['union_per_layer'][-4:]):.1f}",
                         f"{pct(np.mean(lw['static_head4096_hit25'][4:8]))} / {pct(np.mean(lw['static_head4096_hit25'][-4:]))}",
                         f"{pct(np.mean(lw['static_doc_hit25'][:4]))} / {pct(np.mean(lw['static_doc_hit25'][-4:]))}"])
    print(table(["model", "document", "LRU miss (windows 2-4 / last 4)", "Belady miss", "consecutive overlap",
                 "experts per layer per window", "static cache from first 4K tokens, hit @25% (windows 5-8 / last 4)",
                 "static cache from whole doc, hit @25%"], rows))

    print("\n## Mixed document: static profile of block i applied to block j (hit rate % at 25 %), diagonal vs off-diagonal\n")
    rows = []
    for k, r in R.items():
        m = np.array(r["long"]["long_mixed"]["block_profile_hit25"])
        diag = np.diag(m).mean()
        doms = r["long"]["long_mixed"]["block_domains"]
        same = [m[i, j] for i in range(len(m)) for j in range(len(m)) if i != j and doms[i] == doms[j]]
        diff = [m[i, j] for i in range(len(m)) for j in range(len(m)) if doms[i] != doms[j]]
        rows.append([label(k), pct(diag), pct(np.mean(same)) if same else "–", pct(np.mean(diff))])
    print(table(["model", "same block (oracle)", "other block, same domain", "other block, other domain"], rows))
    extra(R)


def tokcaps(R):
    print("\n## Capacity in units of the per-token working set (m × L·k slots = m·k/E of the pairs), corpus `all`: miss rate (%)\n")
    rows = []
    for k, r in R.items():
        t = r.get("all_tokcaps")
        if not t:
            continue
        for i, m in enumerate(t["tokens_worth"]):
            g, pl = t["global"], t["per_layer"]
            rows.append([label(k), m, pct(t["fraction"][i]), pct(g["lru"][i], 1), pct(g["arc"][i], 1), pct(g["lfu"][i], 1),
                         pct(g["la16"][i], 1), pct(g["la64"][i], 1), pct(g["belady"][i], 1),
                         f2(g["lru"][i] / g["belady"][i]) if g["belady"][i] > 0 else "–",
                         pct(pl["lru"][i], 1), pct(pl["belady"][i], 1)])
    print(table(["model", "m", "% of pairs", "LRU", "ARC", "LFU", "la16", "la64", "Belady", "LRU/Belady",
                 "per-layer LRU", "per-layer Belady"], rows))


def extra(R):
    tokcaps(R)
    print("\n## Context dependence: routing of the last 2,048 tokens of a long document with c tokens of context vs the full context\n")
    print("Expert-set overlap (%) with the full-context routing; in parentheses: tokens 65..2048 of the segment only. "
          "Noise floor: decode vs batched routing of the same tokens.\n")
    rows = []
    for k, r in R.items():
        v = r["meta"].get("validation") or {}
        for src in ("long_mixed", "long_book", "long_code"):
            cells = []
            for c in (0, 128, 1024):
                pr = r.get("context_probes", {}).get(f"probe_{src}_c{c}")
                cells.append(f"{pct(pr['overlap'])} ({pct(pr['overlap_after64'])})" if pr else "–")
            nl = [r.get("context_probes", {}).get(f"probe_{src}_c{c}", {}).get("nll_probe") for c in (0, 128, 1024)]
            nf = r.get("context_probes", {}).get(f"probe_{src}_c0", {}).get("nll_full_context")
            rows.append([label(k), src] + cells + [pct(v.get("mean_overlap")), " / ".join(f2(x) for x in nl + [nf])])
    print(table(["model", "document", "c = 0", "c = 128", "c = 1024", "noise floor", "NLL c=0 / 128 / 1024 / full"], rows))
    print("\n## Token identity: expert overlap (%) between two occurrences of the same token (>= 256 tokens apart) vs random positions\n")
    rows = []
    for k, r in R.items():
        t = r.get("token_identity")
        if not t:
            continue
        s_, r_ = np.array(t["same_token_overlap_per_layer"]), np.array(t["random_overlap_per_layer"])
        L = len(s_)
        thirds = [slice(0, L // 3), slice(L // 3, 2 * L // 3), slice(2 * L // 3, L)]
        rows.append([label(k), t["pairs"]] + [f"{pct(s_[sl].mean())} / {pct(r_[sl].mean())}" for sl in thirds] +
                    [f"{pct(s_.mean())} / {pct(r_.mean())}"])
    print(table(["model", "pairs", "first third of layers (same / random)", "middle third", "last third", "all layers"], rows))
    print("\n## Teacher forcing vs decode on the validation segment: miss rate (%) at 10/25/50 % (global), batched / decode\n")
    rows = []
    for k, r in R.items():
        vc = r.get("validation_cache")
        if not vc:
            continue
        for pol in ("lru", "belady"):
            rows.append([label(k), pol] + [f"{pct(vc['batched']['miss_rate'][pol][i], 2)} / {pct(vc['decode']['miss_rate'][pol][i], 2)}"
                                           for i in (1, 2, 3)])
    print(table(["model", "policy", "10%", "25%", "50%"], rows))


if __name__ == "__main__":
    main()
