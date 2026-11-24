#!/bin/bash
# one line per request: tok/s, windows, avg T, ms/window, CPU wait, staging, CPU/VRAM/PCIe experts per layer-window
cd ~/work/uma-moe/results
for r in "$@"; do
  echo "=== $r"
  python3 - "$r" <<'PY'
import re, sys
r = sys.argv[1]
log = open(f"{r}/engine.log").read().splitlines()
cur = {}
for l in log:
    m = re.search(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window.*CPU experts ([\d.]+) \(([\d.]+) entries\), VRAM hits ([\d.]+), PCIe ([\d.]+)", l)
    if m: cur = dict(win=m[1], T=m[2], tpw=m[3], mspw=m[4], cpu=m[5], vram=m[7], pcie=m[8])
    m = re.search(r"GPU stages \(ms/window\):(.*)", l)
    if m:
        g = m[1]
        def tot(key):
            return sum(float(x) for x in re.findall(key + r" ([\d.]+)", g))
        cur.update(waitB=f"{tot('waitB'):.2f}", waitCPU=f"{tot('waitCPU'):.2f}", vramk=f"{tot('VRAM hits'):.2f}", pciek=f"{tot('PCIe grp'):.2f}",
                   total=re.search(r"total ([\d.]+)", g)[1])
    m = re.search(r"generated in \d+ ms \(([\d.]+) tok/s\)", l)
    if m:
        print(f"  {m[1]:>5} tok/s | T {cur.get('T')} tok/win {cur.get('tpw')} ms/win {cur.get('mspw')} gpu {cur.get('total')} | waitB {cur.get('waitB')} waitCPU {cur.get('waitCPU')} vramK {cur.get('vramk')} pcieK {cur.get('pciek')} | per layer-win CPU {cur.get('cpu')} VRAM {cur.get('vram')} PCIe {cur.get('pcie')}")
        cur = {}
PY
  python3 -c "
import csv,sys
rows=list(csv.DictReader(open('$r/samples.csv')))
k=[c for c in rows[0].keys()]
print('  samples cols:', ','.join(k)) if False else None
" 2>/dev/null
done
