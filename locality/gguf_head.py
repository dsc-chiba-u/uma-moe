"""Read the key/value metadata of a GGUF file from its first bytes (local path or hf://repo/file via HfFileSystem).

    python gguf_head.py hf://unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-F16.gguf  [more ...]
Prints architecture, block count, expert counts and context length; long arrays are summarized.
"""
import struct
import sys

TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Buf:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, n):
        if self.o + n > len(self.b):
            raise EOFError
        v = self.b[self.o:self.o + n]; self.o += n; return v

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def s(self): return self.take(self.u64()).decode("utf-8", "replace")

    def val(self, t):
        if t == 8:
            return self.s()
        if t == 9:
            et, n = self.u32(), self.u64()
            if et == 8:
                out = [self.s() for _ in range(n)]
                return out if n <= 16 else f"[{n} strings]"
            fmt = TYPES[et]; sz = struct.calcsize(fmt)
            arr = [struct.unpack(fmt, self.take(sz))[0] for _ in range(n)]
            return arr if n <= 64 else f"[{n} x type{et}] first={arr[:8]}"
        fmt = TYPES[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]


def read_kv(blob, tensors=False):
    b = Buf(blob)
    assert b.take(4) == b"GGUF", "not a GGUF file"
    _ver = b.u32(); nt = b.u64(); nkv = b.u64()
    kv = {}
    try:
        for _ in range(nkv):
            k = b.s(); t = b.u32(); kv[k] = b.val(t)
        if tensors:
            names = []
            for _ in range(nt):
                name = b.s(); nd = b.u32(); dims = [b.u64() for _ in range(nd)]; _typ = b.u32(); _off = b.u64()
                names.append((name, dims))
            kv["_tensors"] = names
    except EOFError:
        kv["_truncated"] = True
    return kv


def head_bytes(path, n=24 << 20):
    if path.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        with HfFileSystem().open(path[5:], "rb", block_size=n) as f:
            return f.read(n)
    with open(path, "rb") as f:
        return f.read(n)


def summary(kv):
    arch = kv.get("general.architecture", "?")
    keys = ["block_count", "expert_count", "expert_used_count", "expert_shared_count", "leading_dense_block_count",
            "context_length", "expert_feed_forward_length", "embedding_length", "interleave_moe_layer_step",
            "expert_group_count", "expert_group_used_count", "expert_gating_func", "moe_every_n_layers", "full_attention_interval"]
    out = {"arch": arch, "name": kv.get("general.name"), "file_type": kv.get("general.file_type")}
    for k in keys:
        if f"{arch}.{k}" in kv:
            out[k] = kv[f"{arch}.{k}"]
    out["has_chat_template"] = "tokenizer.chat_template" in kv
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            kv = read_kv(head_bytes(p), tensors=True)
            print(p, summary(kv))
            ts = kv.get("_tensors", [])
            print("   tensors:", len(ts), [t for t in ts if t[0].startswith("blk.1.")][:12])
        except Exception as e:
            print(p, "ERROR", type(e).__name__, e)
