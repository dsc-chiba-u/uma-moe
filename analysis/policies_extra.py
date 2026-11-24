"""How much of the LRU -> Belady gap do known online policies close, and how much would h tokens of exact
lookahead (what a perfect h-token expert predictor, e.g. one driven by MTP drafts, could give) close?

Policies on the drafts-off access stream (token -> layer -> 10 experts):
  lru, lru2 (LRU-K, K=2), arc (Megiddo & Modha), lookahead-h (LRU, but never evict a key used in the next h
  tokens while another victim exists), belady.

    python analysis/policies_extra.py   -> results/policies_extra.json
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import traces as T

RES = Path(__file__).resolve().parent.parent / "results"
PER_TOKEN = T.N_LAYER * T.K


def sim_lru2(stream, cap):
    """LRU-2: evict the key whose second most recent access is oldest (keys seen once rank oldest)."""
    import heapq
    last, prev, resident, heap, hits = {}, {}, set(), [], 0
    for t, x in enumerate(stream.tolist()):
        if x in resident:
            hits += 1
        prev[x] = last.get(x, -1); last[x] = t
        resident.add(x); heapq.heappush(heap, (prev[x], last[x], x))
        held = []
        while len(resident) > cap:
            p, l, y = heapq.heappop(heap)
            if y not in resident or last[y] != l:
                continue                      # stale entry
            if y == x:
                held.append((p, l, y)); continue
            resident.discard(y)
        for e in held:
            heapq.heappush(heap, e)
    return hits / len(stream)


def sim_arc(stream, cap):
    """ARC with ghost lists B1/B2 and adaptive target p."""
    T1, T2, B1, B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p, hits = 0, 0

    def replace(x):
        nonlocal p
        if T1 and ((x in B2 and len(T1) == p) or len(T1) > p):
            y, _ = T1.popitem(last=False); B1[y] = True
        else:
            y, _ = T2.popitem(last=False); B2[y] = True

    for x in stream.tolist():
        if x in T1:
            hits += 1; del T1[x]; T2[x] = True
        elif x in T2:
            hits += 1; T2.move_to_end(x)
        elif x in B1:
            p = min(cap, p + max(1, len(B2) // max(1, len(B1))))
            replace(x); del B1[x]; T2[x] = True
        elif x in B2:
            p = max(0, p - max(1, len(B1) // max(1, len(B2))))
            replace(x); del B2[x]; T2[x] = True
        else:
            if len(T1) + len(B1) == cap:
                if len(T1) < cap:
                    B1.popitem(last=False); replace(x)
                else:
                    T1.popitem(last=False)
            elif len(T1) + len(B1) < cap and len(T1) + len(T2) + len(B1) + len(B2) >= cap:
                if len(T1) + len(T2) + len(B1) + len(B2) == 2 * cap:
                    B2.popitem(last=False)
                replace(x)
            T1[x] = True
    return hits / len(stream)


def sim_lookahead(stream, cap, h_tokens, nxt):
    """LRU that knows exactly which keys the next h tokens will use, and evicts the least recent key outside that set."""
    c, hits = OrderedDict(), 0
    horizon = h_tokens * PER_TOKEN
    s, n = stream.tolist(), nxt.tolist()
    nextuse = {}
    for i, x in enumerate(s):
        nextuse[x] = n[i]
        if x in c:
            hits += 1; c.move_to_end(x)
            continue
        c[x] = True
        if len(c) > cap:
            limit = (i // PER_TOKEN + 1) * PER_TOKEN + horizon   # end of the lookahead window
            victim = None
            for j, y in enumerate(c):            # least recent first; bounded scan
                if j >= 512:
                    break
                if y != x and nextuse[y] >= limit:
                    victim = y; break
            if victim is None:
                victim = next(iter(c))
            del c[victim]
    return hits / len(stream)


def main():
    tok = T.tokens(T.load_windows(RES / "trace_nodraft/routing.bin"))
    stream = T.keys(tok); nxt = T.next_use(stream)
    out = {}
    for cap in (2048, 4096, 6144, 8192, 12288):
        row = {"lru": T.sim_lru(stream, cap), "lru2": sim_lru2(stream, cap), "arc": sim_arc(stream, cap)}
        for h in (1, 4, 16):
            row[f"lookahead_{h}"] = sim_lookahead(stream, cap, h, nxt)
        row["belady"] = T.sim_belady(stream, cap, nxt)
        out[cap] = row
        print(cap, {k: round(v, 4) for k, v in row.items()}, flush=True)
    json.dump({str(k): v for k, v in out.items()}, open(RES / "policies_extra.json", "w"), indent=1)


if __name__ == "__main__":
    main()
