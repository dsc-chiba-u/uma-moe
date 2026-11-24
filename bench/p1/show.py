"""One line per request of a Strata run: tok/s, window shape, GPU stage totals and where each layer-window's experts
came from (VRAM tier, DRAM tier, SSD, CPU), the SSD wait, and the per-request tier summary line."""
import re, sys
from pathlib import Path
RES = Path.home() / "work/uma-moe/results"
for r in sys.argv[1:]:
    print("===", r)
    cur = {}
    for l in open(RES / r / "engine.log", errors="replace"):
        m = re.search(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify ([\d.]+).*stage ([\d.]+)\) \+ commit/emit ([\d.]+) \+ draft ([\d.]+); per layer-window: CPU experts ([\d.]+) \(([\d.]+) entries\), VRAM hits ([\d.]+), PCIe ([\d.]+)(?:; DRAM tier ([\d.]+), SSD ([\d.]+), SSD wait ([\d.]+))?", l)
        if m:
            cur = dict(T=m[2], tpw=m[3], mspw=m[4], stage=m[6], draft=m[8], cpu=m[9], vram=m[11], pcie=m[12],
                       dram=m[13] or "-", ssd=m[14] or "-", ssdw=m[15] or "-")
        m = re.search(r"GPU stages \(ms/window\):(.*)", l)
        if m:
            g = m[1]
            tot = lambda k: sum(float(x) for x in re.findall(k + r" ([\d.]+)", g))
            cur.update(waitA=f"{tot('waitA'):.2f}", waitB=f"{tot('waitB'):.2f}", waitCPU=f"{tot('waitCPU'):.2f}",
                       vramk=f"{tot('VRAM hits'):.2f}", pciek=f"{tot('PCIe grp'):.2f}", total=re.search(r"total ([\d.]+)", g)[1])
        m = re.search(r"generated in \d+ ms \(([\d.]+) tok/s\)", l)
        if m:
            c = cur
            print(f"  {m[1]:>5} tok/s | T {c.get('T')} tok/win {c.get('tpw')} ms/win {c.get('mspw')} gpu {c.get('total')} "
                  f"stage {c.get('stage')} draft {c.get('draft')} | waitA {c.get('waitA')} waitB {c.get('waitB')} "
                  f"waitCPU {c.get('waitCPU')} k1 {c.get('vramk')} k2 {c.get('pciek')} | per layer-win VRAM {c.get('vram')} "
                  f"DRAM {c.get('dram')} SSD {c.get('ssd')} PCIe {c.get('pcie')} CPU {c.get('cpu')} | SSD wait {c.get('ssdw')} ms/win")
            cur = {}
        if "decode expert tier:" in l or "expert tier:" in l and "warm" in l:
            print("   ", l.strip().replace("strata serve: ", "").replace("strata generate: ", ""))
