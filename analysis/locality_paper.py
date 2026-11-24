"""The numbers of the paper's routing-locality comparison (Table tab:locality-models and the two paragraphs after it),
recomputed from results/locality/analysis/tables.md (written by locality/report.py).

    python analysis/locality_paper.py
"""
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["Mixtral-8x7B-Instruct-v0.1", "gpt-oss-20b", "GLM-4.7-Flash", "Qwen3-30B-A3B-Instruct-2507", "Qwen3.6-35B-A3B",
          "Qwen3.8-Flash-Next"]


def sections():
    out, name, sub = {}, None, None
    for line in (ROOT / "results/locality/analysis/tables.md").read_text().splitlines():
        if line.startswith("## "):
            name, sub = line[3:], None
        elif line.startswith("capacity "):
            sub = line.split()[1]
        elif line.startswith("|") and not line.startswith("|---") and name:
            cells = [c.strip() for c in line.strip("|").split("|")]
            out.setdefault((name, sub), []).append(cells)
    return out


def table(S, prefix, sub=None):
    key = next(k for k in S if k[0].startswith(prefix) and k[1] == sub)
    head, *rows = S[key]
    return head, [r for r in rows if r[0] in MODELS]


def num(x):
    return float(re.match(r"[\d.]+", x.strip())[0])


def main():
    S = sections()
    _, models = table(S, "Models and traces")
    ek = {r[0]: (int(r[5]), int(r[6])) for r in models}
    overlap = [num(r[12].split("/")[1]) for r in models]
    ratio = {cap: dict((r[0], [num(c) for c in r[1:]]) for r in table(S, "LRU / Belady miss ratio by corpus, global", cap)[1])
             for cap in ("10%", "25%", "50%")}
    head, la = table(S, "Lookahead-h", "25%")
    la = {r[0]: r for r in la}
    head_t, temp = table(S, "Temporal locality")
    temp = {r[0]: r for r in temp}
    print("Table: model, E / k, tokens held (0.25 E/k), LRU/Belady (25%, all), h=16 closes, reuse <=16 tokens")
    for m in MODELS:
        e, k = ek[m]
        closes = re.search(r"\((\d+)%\)", la[m][head.index("la16")])[1]
        print(f"  {m:30s} {e:3d} / {k:2d}  {0.25 * e / k:5.1f}  {ratio['25%'][m][-1]:.2f}  {closes}%  "
              f"{temp[m][head_t.index('≤16')]}%")
    whole = [ratio[c][m][-1] for c in ratio for m in MODELS]
    per = [x for m in MODELS for x in ratio["25%"][m][:-1]]
    print(f"LRU/Belady on the whole mix at 10-50%: {min(whole):.2f}-{max(whole):.2f}; per corpus at 25%: "
          f"{min(per):.2f}-{max(per):.2f}")
    print(f"expert choices identical, batched vs decode: {min(overlap):.1f}-{max(overlap):.1f}%")
    _, rows = table(S, "Miss rate (% of expert accesses), global cache, corpus `all`")
    miss = {}
    for r in rows:
        miss.setdefault(r[0], {})[r[1]] = [float(c) for c in r[2:]]
    caps = ["5%", "10%", "25%", "50%", "75%"]
    diffs = [(miss[m]["arc"][i] - miss[m]["lru"][i], m, caps[i]) for m in MODELS for i in range(5)]
    print(f"ARC - LRU (points): best {min(diffs)[0]:.2f} ({min(diffs)[1]}, {min(diffs)[2]}), "
          f"worst {max(diffs)[0]:.2f} ({max(diffs)[1]}, {max(diffs)[2]})")
    lfu = [miss[m]["lfu"][2] / miss[m]["lru"][2] for m in MODELS]
    print(f"LFU / LRU at 25%: {min(lfu):.2f}-{max(lfu):.2f}")
    r16 = [num(temp[m][head_t.index("≤16")]) for m in MODELS]
    r256 = [num(temp[m][head_t.index("≤256")]) for m in MODELS]
    print(f"reuse within 16 tokens: {min(r16):.1f}-{max(r16):.1f}%; within 256: >= {min(r256):.1f}%")
    _, dom = table(S, "Domain shift")
    by = {}
    for r in dom:
        by.setdefault(r[0], []).append([float(c) for c in r[2:]])
    print("domain shift at 25% (mean over the six target domains): own profile / other 5 domains / LRU")
    for m in MODELS:
        v = by[m]
        print(f"  {m:30s} {st.mean(x[0] for x in v):5.1f} {st.mean(x[1] for x in v):5.1f} {st.mean(x[4] for x in v):5.1f}")
    _, val = table(S, "Teacher forcing vs decode")
    d = [abs(float(a) - float(b)) for r in val for a, b in (c.split(" / ") for c in r[2:])]
    print(f"batched vs decode miss rates: max difference {max(d):.2f} points")


if __name__ == "__main__":
    main()
