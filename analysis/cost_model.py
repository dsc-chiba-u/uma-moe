"""Trace-driven decode cost model: MTP draft cap L x expert residency (DRAM cache fraction) x expert bytes.

Windows are rebuilt from the aligned MTP trace (results/mtp_windows.json) for a draft cap L: a window keeps the
current token, min(accepted, L) accepted drafts and, if the original rejection falls inside the cap, the real
rejected drafts; true tokens beyond the cap are re-chunked into further windows, assuming capping does not change
which tokens would have been accepted. L = 0 is plain one-token decoding. Each window's unique experts per layer
go through an LRU cache of the given size; misses are read from NVMe.

Step time = t_fix + t_tok * n_window + t_draft * drafts + dense / W_dense + hit_bytes / W_hit + miss_bytes / W_nvme
(t_* calibrated on the DRAM-resident Q2_0 measurements; W_* from the microbenchmarks.)

    python analysis/cost_model.py [--expert-mb 1.382] [--dense-gb 3.37] [--w-nvme 6.0] ...
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import traces as T

RES = Path(__file__).resolve().parent.parent / "results"
SHORT = ["code_en", "story_en", "explain_en", "math_en", "essay_ja", "tech_ja"]


def load():
    nod = T.tokens(T.load_windows(RES / "trace_nodraft/routing.bin"))
    mtp = T.load_windows(RES / "trace_mtp/routing.bin")
    al = json.load(open(RES / "mtp_windows.json"))
    run = json.load(open(RES / "trace_mtp/run.json"))["server_stats"]
    # prompt regions (read through windows, not drafted) in true-token positions
    prompt_ranges, pos = [], 0
    for r in run:
        if r["prompt_id"] not in SHORT:
            break
        prompt_ranges.append((pos, pos + r["prompt_tokens"])); pos += r["prompt_tokens"] + r["gen_tokens"]
    in_prompt = lambda p: any(a <= p < b for a, b in prompt_ranges)
    dec = [(mtp[i], a) for i, a in enumerate(al) if not in_prompt(a["p"]) and a["n"] <= 6]
    return nod, dec


def windows_for_cap(nod, dec, L):
    """List of (routing array (n, 48, 10), drafts proposed, true tokens) for draft cap L."""
    out = []
    for w, a in dec:
        p, prop, acc = a["p"], a["n"] - 1, a["acc"]
        if L == 0:
            for q in range(p, p + acc + 1):
                out.append((nod[q:q + 1], 0, 1))
            continue
        true_left, q = acc + 1, p
        while true_left > 0:
            take = min(true_left, L + 1)            # current token + up to L accepted drafts
            last = take == true_left                # this window contains the original rejection point
            rows = [nod[q:q + take]]
            drafts = take - 1
            if last and prop > acc:
                n_rej = min(prop, L) - acc if prop > acc else 0
                n_rej = max(0, min(n_rej, L - (take - 1)))
                if n_rej > 0:
                    rows.append(w[acc + 1:acc + 1 + n_rej]); drafts += n_rej
            out.append((np.concatenate(rows, axis=0), drafts, take))
            q += take; true_left -= take
    return out


def simulate(wins, cap):
    """Per window: unique experts (summed over layers) and LRU misses; cap = cache slots (None = everything resident)."""
    c = OrderedDict(); uniq = np.zeros(len(wins)); miss = np.zeros(len(wins))
    for i, (r, _, _) in enumerate(wins):
        for l in range(T.N_LAYER):
            for e in np.unique(r[:, l, :]).tolist():
                uniq[i] += 1
                if cap is None:
                    continue
                key = l * T.N_EXPERT + e
                if key in c:
                    c.move_to_end(key)
                else:
                    miss[i] += 1; c[key] = True
                    if len(c) > cap:
                        c.popitem(last=False)
    return uniq, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-mb", type=float, default=1.382, help="bytes of one (layer, expert) slice, MB")
    ap.add_argument("--dense-gb", type=float, default=3.37, help="non-routed bytes read per step, GB")
    ap.add_argument("--w-dense", type=float, default=228.0, help="GB/s for dense weights")
    ap.add_argument("--w-hit", type=float, default=223.0, help="GB/s for resident experts (155 if read through host page tables)")
    ap.add_argument("--w-nvme", type=float, default=6.0, help="GB/s for missed experts")
    ap.add_argument("--t-fix", type=float, default=9.0, help="ms per step independent of window size")
    ap.add_argument("--t-tok", type=float, default=2.0, help="ms per token in the verify window")
    ap.add_argument("--t-draft", type=float, default=1.5, help="ms per draft token generated")
    ap.add_argument("--residency", default="1.0,0.6,0.45,0.33,0.25", help="fractions of the 24,576 expert slices cached")
    ap.add_argument("--caps", default="0,1,2,3,5")
    ap.add_argument("--overlap", action="store_true", help="NVMe reads overlap the step's DRAM-bound work (max instead of sum)")
    ap.add_argument("--out", default=str(RES / "cost_model.json"))
    a = ap.parse_args()
    nod, dec = load()
    e, res = a.expert_mb * 1e6, [float(x) for x in a.residency.split(",")]
    table = {}
    for L in [int(x) for x in a.caps.split(",")]:
        wins = windows_for_cap(nod, dec, L)
        true_tok = sum(t for _, _, t in wins); drafts = sum(d for _, d, _ in wins)
        sizes = np.array([len(r) for r, _, _ in wins]); ndraft = np.array([d for _, d, _ in wins])
        for r in res:
            cap = None if r >= 1.0 else int(r * T.N_LAYER * T.N_EXPERT)
            uniq, miss = simulate(wins, cap)
            hit = uniq - miss
            dram_ms = a.dense_gb * 1e9 / (a.w_dense * 1e9) * 1e3 + (hit + miss) * e / (a.w_hit * 1e9) * 1e3
            nvme_ms = miss * e / (a.w_nvme * 1e9) * 1e3
            base = a.t_fix + a.t_tok * sizes + a.t_draft * ndraft
            t = base + (np.maximum(dram_ms + 0 * nvme_ms, nvme_ms) if a.overlap else dram_ms + nvme_ms)
            row = {"tok_s": true_tok / (t.sum() / 1e3), "tokens_per_step": true_tok / len(wins),
                   "unique_experts_per_true_token": float(uniq.sum() / true_tok),
                   "nvme_mb_per_true_token": float(miss.sum() * e / true_tok / 1e6),
                   "dram_gb_per_true_token": float((a.dense_gb * 1e9 * len(wins) + hit.sum() * e) / true_tok / 1e9),
                   "ms_per_step": float(t.mean())}
            table.setdefault(str(r), {})[str(L)] = row
            print(f"residency {r:4.2f} L={L}: {row['tok_s']:6.1f} tok/s  tok/step {row['tokens_per_step']:.2f}  "
                  f"experts/token {row['unique_experts_per_true_token']:6.1f}  NVMe {row['nvme_mb_per_true_token']:7.1f} MB/token  "
                  f"DRAM {row['dram_gb_per_true_token']:.2f} GB/token", flush=True)
    # best fixed draft cap per residency, vs the default cap (3 drafts) and vs no drafting
    best = {}
    for r, row in table.items():
        L_best = max(row, key=lambda L: row[L]["tok_s"])
        best[r] = {"best_L": int(L_best), "best_tok_s": row[L_best]["tok_s"], "L3_tok_s": row.get("3", {}).get("tok_s"),
                   "L0_tok_s": row.get("0", {}).get("tok_s"),
                   "gain_vs_L3": row[L_best]["tok_s"] / row["3"]["tok_s"] if "3" in row else None,
                   "mtp_gain_at_L3": row["3"]["tok_s"] / row["0"]["tok_s"] if "3" in row and "0" in row else None}
        print(f"residency {r}: best L={L_best} {best[r]['best_tok_s']:.1f} tok/s; L=3 {best[r]['L3_tok_s']:.1f}; L=0 {best[r]['L0_tok_s']:.1f}"
              f"; MTP(L=3)/noMTP {best[r]['mtp_gain_at_L3']:.2f}x; best/L3 {best[r]['gain_vs_L3']:.2f}x")
    json.dump({"params": vars(a), "table": table, "best": best}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
