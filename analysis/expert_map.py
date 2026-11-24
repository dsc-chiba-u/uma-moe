"""Byte ranges of every (layer, expert) slice of the routed-expert tensors in a (possibly split) GGUF.

    python analysis/expert_map.py OUT.bin SHARD1.gguf [SHARD2.gguf ...]

Writes a binary map for bench/replay_io: header int32 n_files, n_layer, n_expert, n_mat; then n_files
null-terminated paths; then for each layer, expert, matrix: int32 file index, int64 offset, int64 length.
Matrices are ffn_gate_exps, ffn_up_exps, ffn_down_exps (or a merged ffn_gate_up_exps, then down).
"""
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "src/Strata/third_party/llama.cpp/gguf-py"))
sys.path.insert(0, str(Path.home() / "src/Strata/build/_deps/strata_llamacpp-src/gguf-py"))
import gguf  # noqa: E402


def main():
    out, shards = sys.argv[1], sys.argv[2:]
    found = {}   # (layer, name) -> (file index, data offset, n_bytes, n_expert)
    for fi, path in enumerate(shards):
        r = gguf.GGUFReader(path)
        for t in r.tensors:
            m = re.match(r"blk\.(\d+)\.(ffn_gate_exps|ffn_up_exps|ffn_down_exps|ffn_gate_up_exps)\.weight$", t.name)
            if not m:
                continue
            n_expert = int(t.shape[-1])
            found[(int(m.group(1)), m.group(2))] = (fi, int(t.data_offset), int(t.n_bytes), n_expert, t.tensor_type.name)
    layers = sorted({l for l, _ in found})
    names = ["ffn_gate_up_exps", "ffn_down_exps"] if (layers[0], "ffn_gate_up_exps") in found else ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
    n_expert = found[(layers[0], names[0])][3]
    with open(out, "wb") as f:
        f.write(struct.pack("<4i", len(shards), len(layers), n_expert, len(names)))
        for p in shards:
            f.write(str(Path(p).resolve()).encode() + b"\0")
        total = 0
        for l in layers:
            for e in range(n_expert):
                for nm in names:
                    fi, off, nb, ne, _ = found[(l, nm)]
                    per = nb // ne
                    f.write(struct.pack("<iqq", fi, off + e * per, per))
                    total += per
    types = sorted({v[4] for v in found.values()})
    print(f"{out}: {len(layers)} layers x {n_expert} experts x {len(names)} matrices, {total / 1e9:.2f} GB of routed experts, "
          f"{total / len(layers) / n_expert / 1e6:.3f} MB per expert-layer, types {types}")


if __name__ == "__main__":
    main()
