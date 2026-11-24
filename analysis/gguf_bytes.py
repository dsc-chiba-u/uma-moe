"""Bytes read per decoded token, by tensor class, for a (split) GGUF of Qwen3.8-Flash-Next.

    python analysis/gguf_bytes.py SHARD1.gguf [SHARD2 ...]

Classes: routed experts (*_exps: only 10 of 512 per layer are read per token), token embedding (one row per
token), the n-gram/PLE table (the single tensor of the Q2_0 pack's second shard, or the largest non-layer
tensor; a few rows per token), and everything else (dense: read in full every step).
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.home() / "src/Strata/third_party/llama.cpp/gguf-py"))
sys.path.insert(0, str(Path.home() / "src/Strata/build/_deps/strata_llamacpp-src/gguf-py"))
import gguf  # noqa: E402


def main():
    cls = defaultdict(int); types = defaultdict(set); ple_name = None
    tensors = []
    for p in sys.argv[1:]:
        r = gguf.GGUFReader(p)
        tensors += [(t.name, int(t.n_bytes), t.tensor_type.name) for t in r.tensors]
    non_layer = [(n, b) for n, b, _ in tensors if not n.startswith("blk.") and n not in ("token_embd.weight", "output.weight")]
    if non_layer:
        cand = max(non_layer, key=lambda x: x[1])
        if cand[1] > 5e9:
            ple_name = cand[0]
    for n, b, ty in tensors:
        if re.search(r"_exps\.weight$", n):
            c = "experts"
        elif n == "token_embd.weight":
            c = "token_embd"
        elif n == ple_name:
            c = "ngram_table"
        else:
            c = "dense"
        cls[c] += b; types[c].add(ty)
    total = sum(cls.values())
    print(f"total {total / 1e9:.2f} GB")
    for c in ("dense", "experts", "ngram_table", "token_embd"):
        print(f"  {c:12s} {cls[c] / 1e9:8.2f} GB  types {sorted(types[c])}" + (f"  ({ple_name})" if c == "ngram_table" and ple_name else ""))
    per_tok_exp = cls["experts"] * 10 / 512
    print(f"per token: dense {cls['dense'] / 1e9:.2f} GB + routed experts {per_tok_exp / 1e9:.2f} GB = {(cls['dense'] + per_tok_exp) / 1e9:.2f} GB"
          f" (dense share {cls['dense'] / (cls['dense'] + per_tok_exp):.0%}); one (layer, expert) slice {cls['experts'] / 48 / 512 / 1e6:.3f} MB")


if __name__ == "__main__":
    main()
