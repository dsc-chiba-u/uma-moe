"""Figures for the paper from results/final_report.json and the cost model (results/cost_model_q8_overlap.json).

    python analysis/paper_figs.py   -> paper/figs/decode_vs_residency.pdf, paper/figs/window_breakdown.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
# the dataviz reference palette, light mode, fixed slot order: ours MTP, ours drafts off, Strata, llama.cpp
C = {"mtp": "#2a78d6", "nodraft": "#eb6834", "upstream": "#1baf7a", "llama": "#eda100", "model": "#77756f",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.linewidth": 0.6, "pdf.fonttype": 42})


def rows():
    return {r["run"]: r for r in json.load(open(RES / "final_report.json"))}


def decode_vs_residency(R):
    # the performance model (each window's NVMe reads overlap its DRAM transfers and the longer of the two counts), for
    # expert reads at 223 GB/s (driver memory, the plan's assumption) and at 165 GB/s (host page tables, what the DRAM
    # tier's registered slots get)
    m_fast = json.load(open(RES / "cost_model_q8_overlap_wh223.json"))["table"]
    m_slow = json.load(open(RES / "cost_model_q8_overlap_wh165.json"))["table"]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    xs = [45, 60, 75]
    for mode, lab in (("mtp", "this work, MTP"), ("nodraft", "this work, drafts off")):
        ys = [R.get(f"f_ours_t{x}_{mode}", {}).get("mean") for x in xs]
        pts = [(x, y) for x, y in zip(xs, ys) if y]
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=4, lw=2, color=C[mode], label=lab)
            ax.annotate(f"{pts[-1][1]:.1f}", pts[-1], xytext=(4, 0), textcoords="offset points", va="center",
                        color=C["ink"], fontsize=7)
    key = {45: "0.45", 60: "0.6", 75: "0.75"}
    for cap in ("3", "0"):
        hi = [m_fast[key[x]][cap]["tok_s"] for x in xs]
        lo = [m_slow[key[x]][cap]["tok_s"] for x in xs]
        ax.fill_between(xs, lo, hi, color=C["model"], alpha=0.22, lw=0,
                        label="model, overlapping I/O (165–223 GB/s)" if cap == "3" else None)
    # baselines: they hold no DRAM tier, so they have no place on the x axis; each is a horizontal reference line with
    # its marker in a separate band at the right (Strata 0.2.6: the fastest exact configuration at a 66.8 GiB budget,
    # unbuffered file tier, 86 GiB taken; llama.cpp: every expert on the CPU)
    for run, col, lab, mk, x in (("f_up026_b66_exact_unbuf_mtp", "upstream", "Strata 0.2.6, MTP", "s", 81.5),
                                 ("f_up026_b66_exact_unbuf_nodraft", "upstream", "Strata 0.2.6, drafts off", "D", 83.0),
                                 ("f_llama_cpumoeall_mtp", "llama", "llama.cpp, MTP", "^", 84.5),
                                 ("f_llama_cpumoeall_nomtp", "llama", "llama.cpp, no MTP", "v", 86.0)):
        y = R.get(run, {}).get("mean")
        if y:
            ax.axhline(y, color=C[col], ls=":", lw=0.7, zorder=0)
            ax.plot([x], [y], marker=mk, ms=5, ls="none", color=C[col], label=lab)
    ax.axvline(79.5, color=C["grid"], lw=0.8)
    ax.text(83.8, 14.0, "baselines\n(no tier)", ha="center", va="bottom", fontsize=6, color=C["ink2"])
    ax.set_xlabel("experts in DRAM (% of 24,576)")
    ax.set_ylabel("decode tokens/s")
    ax.set_xticks([45, 60, 75])
    ax.set_xlim(42, 88)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.subplots_adjust(left=0.13, right=0.9, top=0.97, bottom=0.2)   # the legend goes below; the tight box keeps it
    fig.savefig(OUT / "decode_vs_residency.pdf", bbox_inches="tight", pad_inches=0.02)


def window_breakdown(R):
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    labels, comp, ssd, other = [], [], [], []
    for x in (45, 60, 75):
        for mode in ("nodraft", "mtp"):
            w = (R.get(f"f_ours_t{x}_{mode}") or {}).get("win")
            if not w:
                continue
            per_tok = 1.0 / w["tpw"]
            labels.append(f"{x}%\n{'MTP' if mode == 'mtp' else 'off'}")
            ssd.append(w["ssd_wait"] * per_tok)
            oth = (w["draft"] + w["stage"] + w["commit"]) * per_tok
            other.append(oth)
            comp.append(w["mspw"] * per_tok - ssd[-1] - oth)
    if not labels:
        return
    idx = range(len(labels))
    ax.bar(idx, comp, color=C["mtp"], width=0.6, label="computing")
    ax.bar(idx, ssd, bottom=comp, color=C["nodraft"], width=0.6, label="waiting for the SSD")
    ax.bar(idx, other, bottom=[a + b for a, b in zip(comp, ssd)], color=C["model"], width=0.6,
           label="drafting, staging, commit")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("ms per generated token")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Tight spacing keeps the three-column legend clear of the y-axis label at this figure height.
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "window_breakdown.pdf")


if __name__ == "__main__":
    R = rows()
    decode_vs_residency(R)
    window_breakdown(R)
    print("wrote", OUT / "decode_vs_residency.pdf", OUT / "window_breakdown.pdf")
