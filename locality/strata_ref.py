"""Reference: the main study's Strata decode trace of Qwen3.8-Flash-Next (results/trace_nodraft/routing.bin, 5,986
tokens, drafts off) through the same simulator and summaries as the llama.cpp traces.

    python -I locality/strata_ref.py      -> results/locality/analysis/strata_decode_ref.json
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analysis"))
import analyze as A  # noqa: E402
import traces as T  # noqa: E402  (read-only import of the main study's trace loader)


def main():
    tok = T.tokens(T.load_windows(HERE.parent / "results/trace_nodraft/routing.bin")).astype(np.uint16)
    tmp = Path(tempfile.mkdtemp(prefix="strata_ref_"))
    p = tmp / "strata.npy"; np.save(p, tok)
    r = A.run_locsim([p], 512, A.POLICIES, A.CAPS, "both", stats=True, stack=True)
    s = A.summarize_sim(r, len(tok))
    rng = np.random.default_rng(0)
    ps = tmp / "shuf.npy"; np.save(ps, tok[rng.permutation(len(tok))])
    rs = A.run_locsim([ps], 512, A.SHUF_POLICIES, A.CAPS, "both", stats=True)
    s["shuffled"] = A.summarize_sim(rs, len(tok)); s["shuffled"].pop("overlap_per_layer", None)
    _, s["popularity"] = A.popularity(tok, 512)
    for f in (p, ps):
        f.unlink()
    tmp.rmdir()
    A.OUT.mkdir(parents=True, exist_ok=True)
    json.dump(s, open(A.OUT / "strata_decode_ref.json", "w"), indent=1)
    g = s["global"]["miss_rate"]
    print({pol: [round(100 * x, 2) for x in g[pol]] for pol in ("lru", "belady", "la16", "la64", "la256")})


if __name__ == "__main__":
    main()
