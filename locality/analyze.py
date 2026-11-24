"""Locality analysis of the llama.cpp routing traces.

    python -I locality/analyze.py [KEY ...]      -> results/locality/analysis/<KEY>.json  (default: every traced model)

Per model and corpus (documents of a corpus concatenated into one access stream, cache warm across documents):
  * cache simulation (locality/locsim): LRU, LFU, ARC, Belady (MIN with bypass), MIN without bypass, lookahead-h
    (h = 4/16/64/256 tokens), decayed LFU (half-life 4..1024 tokens); global cache over all (layer, expert) pairs and
    per-layer caches; capacities 5/10/25/50/75 % of the pairs (per layer: of the experts of a layer)
  * the same for a token-shuffled copy of the stream (keeps popularity, destroys temporal order): LRU, LFU, Belady
  * popularity skew, consecutive-token overlap, reuse distance in tokens, working-set (union) growth
  * static frequency caches: profile from the corpus itself (oracle) and from every other corpus (domain shift)
Long documents: misses per 1,024-token window (warm cache), per-window overlap / working set / drift of the expert
distribution, a static cache built from the first 4,096 tokens applied to later windows, block-to-block profiles of
the mixed document.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("LOC_RES", HERE.parent / "results/locality"))
OUT = RES / "analysis"
LOCSIM = HERE / "locsim"
CAPS = [0.05, 0.10, 0.25, 0.50, 0.75]
POLICIES = ["lru", "lfu", "arc", "belady", "belady_nb", "la4", "la16", "la64", "la256",
            "dlfu4", "dlfu16", "dlfu64", "dlfu256", "dlfu1024"]
SHUF_POLICIES = ["lru", "lfu", "belady"]
DOMAINS = ["code", "prose", "ja", "math", "chat", "dialog"]
WINDOW = 1024
THREADS = int(os.environ.get("LOCSIM_THREADS", "4"))


def run_locsim(paths, E, policies, caps, scope="both", window=0, stats=False, stack=False):
    cmd = ["nice", "-n", "19", str(LOCSIM), "--caps", ",".join(map(str, caps)), "--policies", ",".join(policies),
           "--scope", scope, "--threads", str(THREADS)]
    if window:
        cmd += ["--window", str(window)]
    if stats:
        cmd.append("--stats")
    if stack:
        cmd.append("--stack")
    cmd += [str(p) for p in paths]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, LOCSIM_E=str(E)), check=True)
    return json.loads(r.stdout)


def corpus_docs(docs):
    c = {d: [] for d in DOMAINS}
    for name, info in docs.items():
        if info["corpus"] in c:
            c[info["corpus"]].append(name)
    for name in ("long_mixed", "long_book", "long_code"):
        c[name] = [name]
    return c


def lorenz_share(counts, frac):
    """Share of accesses taken by the most used `frac` of the experts of each layer (interpolated), mean over layers."""
    srt = -np.sort(-counts, axis=1).astype(np.float64)
    cov = np.cumsum(srt, axis=1) / np.maximum(srt.sum(axis=1, keepdims=True), 1)
    cov = np.concatenate([np.zeros((len(cov), 1)), cov], axis=1)
    x = frac * counts.shape[1]
    i = int(np.floor(x))
    s = cov[:, i] + (x - i) * (cov[:, min(i + 1, counts.shape[1])] - cov[:, i])
    return float(s.mean())


def popularity(route, E):
    T, L, k = route.shape
    counts = np.zeros((L, E), dtype=np.int64)
    for l in range(L):
        counts[l] = np.bincount(route[:, l, :].ravel(), minlength=E)
    flat = np.sort(counts.ravel())[::-1].astype(np.float64)
    cf = np.cumsum(flat) / flat.sum()
    n = len(flat)
    return counts, {
        "top10pct_share_per_layer": lorenz_share(counts, 0.10),
        "top25pct_share_per_layer": lorenz_share(counts, 0.25),
        "top50pct_share_per_layer": lorenz_share(counts, 0.50),
        "top10pct_share_global": float(cf[max(0, int(round(0.10 * n)) - 1)]),
        "unused_pairs": int((counts == 0).sum()),
        "experts_for_80pct_per_layer": float(np.mean([(np.cumsum(-np.sort(-c)) / c.sum() < 0.8).sum() + 1 for c in counts])),
    }


def summarize_sim(r, T):
    N = r["accesses"]
    out = {"T": T, "accesses": N, "distinct_pairs": r["distinct_pairs"]}
    for scope in ("global", "per_layer"):
        if scope not in r:
            continue
        mr = {p: [m / N for m in v] for p, v in r[scope]["misses"].items()}
        out[scope] = {"slots": r[scope]["slots"], "miss_rate": mr}
        if "lru" in mr and "belady" in mr:
            out[scope]["lru_over_belady"] = [a / b if b > 0 else None for a, b in zip(mr["lru"], mr["belady"])]
    if "reuse_tokens" in r:
        b = np.array(r["reuse_tokens"]["le_pow2"], dtype=np.float64)
        reuses = b.sum()
        cum = np.cumsum(b) / max(reuses, 1)
        out["reuse_share_within"] = {str(2 ** i): float(cum[i]) for i in range(0, 15)}
        out["first_use_share"] = r["reuse_tokens"]["first_use"] / N
        out["overlap"] = float(np.mean(r["overlap_per_layer"]))
        out["overlap_per_layer"] = r["overlap_per_layer"]
        out["union_window"] = r["union_window"]
    if "stack" in r:
        h = np.array(r["stack"]["hist"], dtype=np.float64)
        out["lru_curve"] = {"slots": list(range(0, len(h), max(1, len(h) // 256))),
                            "miss_rate": [float(1 - h[:c + 1].sum() / N) for c in range(0, len(h), max(1, len(h) // 256))]}
    return out


def static_hits(freq_profile, target_counts, caps, per_layer):
    """Hit rate on `target_counts` (L, E) of a static cache holding the most frequent pairs of `freq_profile`."""
    L, E = target_counts.shape
    tot = target_counts.sum()
    out = []
    for c in caps:
        if per_layer:
            n = max(1, int(round(c * E)))
            idx = np.argsort(-freq_profile, axis=1, kind="stable")[:, :n]
            out.append(float(np.take_along_axis(target_counts, idx, axis=1).sum() / tot))
        else:
            n = max(1, int(round(c * L * E)))
            idx = np.argsort(-freq_profile.ravel(), kind="stable")[:n]
            out.append(float(target_counts.ravel()[idx].sum() / tot))
    return out


def window_stats(route, E, w):
    T, L, k = route.shape
    nw = T // w
    allc = np.zeros((L, E), dtype=np.float64)
    for l in range(L):
        allc[l] = np.bincount(route[:, l, :].ravel(), minlength=E)
    p_all = allc / allc.sum(axis=1, keepdims=True)
    head = min(4096, T // 2)
    headc = np.zeros((L, E))
    for l in range(L):
        headc[l] = np.bincount(route[:head, l, :].ravel(), minlength=E)
    res = {"window": w, "overlap": [], "union_per_layer": [], "js_vs_doc": [], "static_head4096_hit25": [],
           "static_doc_hit25": []}
    for i in range(nw):
        seg = route[i * w:(i + 1) * w]
        ov, un = [], []
        cnt = np.zeros((L, E))
        for l in range(L):
            oh = np.zeros((w, E), dtype=bool)
            oh[np.arange(w)[:, None], seg[:, l, :]] = True
            ov.append((oh[1:] & oh[:-1]).sum() / ((w - 1) * k))
            un.append(oh.any(axis=0).sum())
            cnt[l] = oh.sum(axis=0)
        p = cnt / cnt.sum(axis=1, keepdims=True)
        m = 0.5 * (p + p_all)
        with np.errstate(divide="ignore", invalid="ignore"):
            js = 0.5 * np.nansum(np.where(p > 0, p * np.log2(p / m), 0), axis=1) + \
                 0.5 * np.nansum(np.where(p_all > 0, p_all * np.log2(p_all / m), 0), axis=1)
        res["overlap"].append(float(np.mean(ov)))
        res["union_per_layer"].append(float(np.mean(un)))
        res["js_vs_doc"].append(float(np.mean(js)))
        res["static_head4096_hit25"].append(static_hits(headc, cnt.astype(np.int64), [0.25], False)[0])
        res["static_doc_hit25"].append(static_hits(allc, cnt.astype(np.int64), [0.25], False)[0])
    return res


def analyze_model(key):
    tdir = RES / key / "trace"
    meta = json.load(open(tdir / "trace_meta.json"))
    docs = json.load(open(RES / key / "tokens/docs.json"))["docs"]
    E, k, L = meta["n_expert"], meta["k"], len(meta["moe_layers"])
    routes = {d: np.load(tdir / f"{d}.route.npy") for d in docs}
    out = {"model": key, "meta": {kk: v for kk, v in meta.items() if kk != "docs"}, "docs": docs,
           "nll": {d["id"]: d["mean_nll"] for d in meta["docs"]}, "corpora": {}}
    corpora = corpus_docs(docs)
    tmp = Path(tempfile.mkdtemp(prefix=f"loc_{key}_", dir=os.environ.get("LOC_TMP", "/tmp")))
    rng = np.random.default_rng(0)
    counts = {}

    def sim_corpus(name, arrs):
        paths = []
        for i, a in enumerate(arrs):
            p = tmp / f"{name}_{i}.npy"; np.save(p, np.ascontiguousarray(a, dtype=np.uint16)); paths.append(p)
        T = sum(len(a) for a in arrs)
        r = run_locsim(paths, E, POLICIES, CAPS, "both", stats=True, stack=True)
        s = summarize_sim(r, T)
        # token-shuffled null model (same popularity, no temporal order)
        cat = np.concatenate(arrs, axis=0)
        sh = cat[rng.permutation(len(cat))]
        ps = tmp / f"{name}_shuf.npy"; np.save(ps, sh)
        rs = run_locsim([ps], E, SHUF_POLICIES, CAPS, "both", stats=True)
        s["shuffled"] = summarize_sim(rs, T)
        s["shuffled"].pop("overlap_per_layer", None)
        c, pop = popularity(cat, E)
        s["popularity"] = pop
        counts[name] = c
        for f in paths + [ps]:
            f.unlink()
        return s

    for name, ds in corpora.items():
        out["corpora"][name] = sim_corpus(name, [routes[d] for d in ds])
        print(f"[{key}] {name}: {out['corpora'][name]['T']} tokens", flush=True)
    # decode-like: only the generated part of the chat transcripts
    gen = [routes[d][docs[d]["prompt_tokens"]:] for d in corpora["chat"]]
    out["corpora"]["chat_gen"] = sim_corpus("chat_gen", gen)
    # everything in manifest order (domain shifts between documents)
    main_docs = [d for d in docs if docs[d]["corpus"] != "probe"]
    out["corpora"]["all"] = sim_corpus("all", [routes[d] for d in main_docs])
    # capacities in units of the per-token working set (L*k pairs): m tokens' worth = m*k/E of all pairs
    ms = [m for m in (1, 2, 4, 8, 16) if m * k / E < 0.95]
    pa = tmp / "all_tokcaps.npy"; np.save(pa, np.concatenate([routes[d] for d in main_docs]))
    pols = ["lru", "lfu", "arc", "belady", "belady_nb", "la4", "la16", "la64", "la256", "dlfu16"]
    rt = run_locsim([pa], E, pols, [m * k / E for m in ms], "both")
    pa.unlink()
    out["all_tokcaps"] = {"tokens_worth": ms, "fraction": [m * k / E for m in ms],
                          "global": {p: [x / rt["accesses"] for x in v] for p, v in rt["global"]["misses"].items()},
                          "per_layer": {p: [x / rt["accesses"] for x in v] for p, v in rt["per_layer"]["misses"].items()}}
    # static profiles and domain shift
    names = DOMAINS + ["long_mixed", "long_book", "long_code"]
    ds = {}
    for tgt in DOMAINS:
        row = {"self_oracle": {}, "lru": {}, "from": {}}
        for scope in ("global", "per_layer"):
            pl = scope == "per_layer"
            row["self_oracle"][scope] = static_hits(counts[tgt], counts[tgt], CAPS, pl)
            row["lru"][scope] = [1 - m for m in out["corpora"][tgt][scope]["miss_rate"]["lru"]]
            row["from"][scope] = {src: static_hits(counts[src], counts[tgt], CAPS, pl) for src in names if src != tgt}
            loo = sum(counts[s] for s in DOMAINS if s != tgt)
            row["from"][scope]["other_domains"] = static_hits(loo, counts[tgt], CAPS, pl)
        ds[tgt] = row
    out["domain_shift"] = ds
    # long documents: positional analysis
    longd = {}
    for name in ("long_mixed", "long_book", "long_code"):
        p = tmp / f"{name}_w.npy"; np.save(p, routes[name])
        rw = run_locsim([p], E, ["lru", "belady", "dlfu16"], [0.10, 0.25], "global", window=WINDOW)
        per_tok = WINDOW * L * k
        nwin = len(routes[name]) // WINDOW
        longd[name] = {"window": WINDOW,
                       "miss_rate": {pol: {str(c): [m / per_tok for m in series[:nwin]]
                                           for c, series in zip([0.10, 0.25], v)}
                                     for pol, v in rw["global"]["window_misses"].items()},
                       **window_stats(routes[name], E, WINDOW)}
        p.unlink()
        if name == "long_mixed":
            blocks = docs[name]["blocks"]
            bc = []
            for a, b in blocks:
                c = np.zeros((L, E), dtype=np.int64)
                for l in range(L):
                    c[l] = np.bincount(routes[name][a:b, l, :].ravel(), minlength=E)
                bc.append(c)
            longd[name]["block_domains"] = docs[name]["block_domains"]
            longd[name]["block_profile_hit25"] = [[static_hits(bc[i], bc[j], [0.25], False)[0] for j in range(len(bc))]
                                                  for i in range(len(bc))]
    out["long"] = longd
    # context dependence: routing of the last tokens of a long document with the full context vs with c tokens of context
    probes = {}
    for d, info in docs.items():
        if info["corpus"] != "probe":
            continue
        src, a, off, n = info["source"], info["target_start_in_source"], info["target_offset"], info["target_len"]
        full = routes[src][a:a + n]
        part = routes[d][off:off + n]
        inter = (np.sort(full, axis=2)[..., :, None] == np.sort(part, axis=2)[..., None, :]).sum(axis=(2, 3))  # (n, L)
        per_layer = inter.mean(axis=0) / k
        nll_full = np.load(tdir / f"{src}.nll.npy")[a:a + n]
        nll_part = np.load(tdir / f"{d}.nll.npy")[off:off + n]
        probes[d] = {"source": src, "context": int(d.rsplit("_c", 1)[1]), "overlap": float(per_layer.mean()),
                     "nll_full_context": float(np.nanmean(nll_full)), "nll_probe": float(np.nanmean(nll_part)),
                     "overlap_by_256": [float(inter[i:i + 256].mean() / k) for i in range(0, n, 256)],
                     "identical_sets": float((inter == k).mean()), "overlap_per_layer": per_layer.tolist(),
                     "overlap_first64": float(inter[:64].mean() / k), "overlap_after64": float(inter[64:].mean() / k)}
    out["context_probes"] = probes
    # token identity: expert overlap between two occurrences of the same token id (>= 256 tokens apart or in other
    # documents) vs two random positions, per layer
    toks = np.concatenate([np.fromfile(RES / key / "tokens" / f"{d}.tok", dtype=np.int32) for d in main_docs])
    allr = np.concatenate([routes[d] for d in main_docs])
    order = np.argsort(toks, kind="stable")
    st = toks[order]
    bounds = np.flatnonzero(np.diff(st)) + 1
    groups = [g for g in np.split(order, bounds) if len(g) >= 2]
    pairs = []
    for g in rng.permutation(len(groups))[:20000]:
        pp = groups[g]
        for _ in range(4):
            i, j = rng.choice(len(pp), 2, replace=False)
            if abs(int(pp[i]) - int(pp[j])) >= 256:
                pairs.append((pp[i], pp[j])); break
    pairs = np.array(pairs[:20000])
    rnd = rng.integers(0, len(toks), size=(len(pairs), 2))

    def pair_overlap(pr):
        A, B = np.sort(allr[pr[:, 0]], axis=2), np.sort(allr[pr[:, 1]], axis=2)
        return ((A[..., :, None] == B[..., None, :]).sum(axis=(2, 3)) / k).mean(axis=0)

    out["token_identity"] = {"pairs": int(len(pairs)), "same_token_overlap_per_layer": pair_overlap(pairs).tolist(),
                             "random_overlap_per_layer": pair_overlap(rnd).tolist()}
    # teacher forcing vs decode: the same answer tokens, routed in 512-token batches vs one token per llama_decode
    v = meta.get("validation")
    if v and (tdir / f"{v['doc']}.decode.route.npy").exists():
        dec = np.load(tdir / f"{v['doc']}.decode.route.npy")
        bat = routes[v["doc"]][v["prefix_tokens"]:v["prefix_tokens"] + v["decoded_tokens"]]
        vc = {}
        for name, arr in (("decode", dec), ("batched", bat)):
            p = tmp / f"val_{name}.npy"; np.save(p, np.ascontiguousarray(arr))
            r = run_locsim([p], E, ["lru", "belady"], CAPS, "global", stats=True)
            vc[name] = {"miss_rate": {pol: [m / r["accesses"] for m in ms] for pol, ms in r["global"]["misses"].items()},
                        "overlap": float(np.mean(r["overlap_per_layer"]))}
            p.unlink()
        out["validation_cache"] = vc
    tmp.rmdir()
    return out


def main():
    keys = sys.argv[1:] or sorted(p.parent.parent.name for p in RES.glob("*/trace/trace_meta.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    for key in keys:
        res = analyze_model(key)
        json.dump(res, open(OUT / f"{key}.json", "w"), indent=1)
        print(f"[{key}] written {OUT / (key + '.json')}", flush=True)


if __name__ == "__main__":
    main()
