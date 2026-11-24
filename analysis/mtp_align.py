"""Which tokens of each MTP verify window were accepted, recovered by aligning the MTP routing trace with the
drafts-off trace of the same prompts (greedy outputs are identical for the six short prompts).

A window holds [current token, draft 1..m]. Token 0 and accepted drafts are true tokens, whose routing equals the
drafts-off routing at the same position; the first draft whose routing does not match is rejected, and so is
everything after it. The next window starts at the next true position.

Outputs results/mtp_align.json: accepted/rejected token shares, expert loads caused only by rejected drafts,
per-position acceptance, and LRU hit rates with and without admitting rejected-draft experts.

    python analysis/mtp_align.py
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import traces as T

RES = Path(__file__).resolve().parent.parent / "results"
SHORT = ["code_en", "story_en", "explain_en", "math_en", "essay_ja", "tech_ja"]


def sim(rows, k):
    return float(np.mean([len(np.intersect1d(a, b)) / k for a, b in zip(rows[0], rows[1])]))


def similarity(a, b):
    """mean over layers of |a_l ∩ b_l| / 10 for two (48, 10) routings"""
    return float(np.mean([len(np.intersect1d(a[l], b[l], assume_unique=True)) for l in range(T.N_LAYER)]) / T.K)


def main(threshold=0.97):
    """Accepted tokens reproduce the drafts-off routing exactly (similarity 1.00); a rejected draft can still look
    similar (0.6-0.9), so the match must be near-exact, and the next window's first token confirms the count."""
    nod = T.tokens(T.load_windows(RES / "trace_nodraft/routing.bin"))
    mtp = T.load_windows(RES / "trace_mtp/routing.bin")
    run = json.load(open(RES / "trace_mtp/run.json"))["server_stats"]
    # only the six short prompts have identical outputs; they come first in both traces
    n_short_tokens = sum(r["prompt_tokens"] + r["gen_tokens"] for r in run if r["prompt_id"] in SHORT)
    match = lambda w_tok, q: q < len(nod) and similarity(w_tok, nod[q]) >= threshold
    p, wins, mismatch_t0, resync = 0, [], 0, 0
    for wi, w in enumerate(mtp):
        if p >= n_short_tokens - 1:
            break
        n = len(w)
        if not match(w[0], p):
            mismatch_t0 += 1
        a = 0
        for j in range(1, n):
            if match(w[j], p + j):
                a += 1
            else:
                break
        nxt = mtp[wi + 1] if wi + 1 < len(mtp) else None
        if nxt is not None and not match(nxt[0], p + 1 + a):
            # pick the acceptance count that makes the next window start on a true token
            cands = [b for b in range(0, n) if match(nxt[0], p + 1 + b)]
            if cands:
                a = cands[0]; resync += 1
        wins.append({"n": n, "acc": a, "p": p})
        p += 1 + a
    # ---- statistics over decode windows (n <= 6; prompt-reading windows are accepted wholesale and excluded)
    dec = [(w, x) for w, x in zip(mtp, wins) if x["n"] <= 6 and x["n"] > 1]
    true_slots = sum(1 + x["acc"] for _, x in dec)
    rej_slots = sum(x["n"] - 1 - x["acc"] for _, x in dec)
    union_all, union_true, rej_only = [], [], []
    for w, x in dec:
        t = 1 + x["acc"]
        for l in range(T.N_LAYER):
            ua = np.unique(w[:, l, :]); ut = np.unique(w[:t, l, :])
            union_all.append(len(ua)); union_true.append(len(ut)); rej_only.append(len(ua) - len(ut))
    proposed = np.bincount([x["n"] - 1 for _, x in dec], minlength=7)
    acc_at = np.zeros(7); prop_at = np.zeros(7)
    for _, x in dec:
        for j in range(1, x["n"]):
            prop_at[j] += 1
            if j <= x["acc"]:
                acc_at[j] += 1
    out = {
        "windows_aligned": len(wins), "token0_mismatches": mismatch_t0, "resyncs": resync, "decode_windows": len(dec),
        "true_token_slots": true_slots, "rejected_draft_slots": rej_slots,
        "rejected_share_of_slots": rej_slots / (true_slots + rej_slots),
        "mean_union_per_layer_all": float(np.mean(union_all)), "mean_union_per_layer_true_only": float(np.mean(union_true)),
        "share_of_window_union_only_needed_by_rejected": float(np.sum(rej_only) / np.sum(union_all)),
        "drafts_proposed_hist": proposed.tolist(),
        "accept_prob_by_position": {j: float(acc_at[j] / prop_at[j]) for j in range(1, 7) if prop_at[j]},
        "tokens_per_window": float(true_slots / len(dec)),
    }
    # ---- cache pollution: LRU over the window access stream, admitting rejected-draft experts or not
    def lru_run(cap, admit_rejected):
        c, hits, acc_n, miss_true = OrderedDict(), 0, 0, 0
        for w, x in dec:
            t = 1 + x["acc"]
            for l in range(T.N_LAYER):
                need = np.unique(w[:, l, :]); need_true = set(np.unique(w[:t, l, :]).tolist())
                for e in need.tolist():
                    key = l * T.N_EXPERT + e; acc_n += 1
                    if key in c:
                        hits += 1
                        if admit_rejected or e in need_true:
                            c.move_to_end(key)
                    else:
                        if e in need_true:
                            miss_true += 1
                        if admit_rejected or e in need_true:
                            c[key] = True
                            if len(c) > cap:
                                c.popitem(last=False)
        return hits / acc_n, acc_n, (acc_n - hits)
    pol = {}
    for cap in (2048, 4096, 8192, 12288):
        h1, n1, m1 = lru_run(cap, True); h2, n2, m2 = lru_run(cap, False)
        pol[cap] = {"lru_admit_all_hit": h1, "lru_deferred_admission_hit": h2,
                    "misses_per_true_token_admit_all": m1 / true_slots, "misses_per_true_token_deferred": m2 / true_slots}
        print(cap, pol[cap], flush=True)
    out["window_lru"] = pol
    json.dump(out, open(RES / "mtp_align.json", "w"), indent=1)
    json.dump([{"n": x["n"], "acc": x["acc"], "p": x["p"]} for x in wins], open(RES / "mtp_windows.json", "w"))
    print(json.dumps({k: v for k, v in out.items() if k != "window_lru"}, indent=1))


if __name__ == "__main__":
    main()
