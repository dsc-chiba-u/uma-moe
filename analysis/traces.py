"""Routing-trace loading and cache simulation for the UMA MoE study.

Trace format (Strata --dump-routing, serve path): records of int32 layer, int32 k, k int32 ids, k float32 weights.
In a verify window of n tokens the engine writes, for layer 0..47 in order, n records each (one per token).
Prompt tokens of short prompts (<= 64) are read through windows too; batched prefill is not traced.
"""
import heapq
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np

N_LAYER, N_EXPERT, K = 48, 512, 10
REC = 8 + 8 * K   # bytes per record for k = 10


def load_windows(path):
    """List of int32 arrays shaped (n_tokens, 48, 10), one per verify window, in order."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size % (REC // 4):
        raise ValueError(f"{path}: size is not a multiple of the record size")
    rec = raw.reshape(-1, REC // 4)
    layer, k, ids = rec[:, 0], rec[:, 1], rec[:, 2:2 + K]
    assert (k == K).all(), "unexpected k"
    windows, i, n_rec = [], 0, len(rec)
    while i < n_rec:
        if layer[i] != 0:
            raise ValueError(f"record {i}: window does not start at layer 0 (layer {layer[i]})")
        n = 1
        while i + n < n_rec and layer[i + n] == 0:
            n += 1
        block = ids[i:i + n * N_LAYER]
        if len(block) < n * N_LAYER or not (layer[i:i + n * N_LAYER].reshape(N_LAYER, n) == np.arange(N_LAYER)[:, None]).all():
            raise ValueError(f"record {i}: malformed window of {n} tokens")
        windows.append(block.reshape(N_LAYER, n, K).transpose(1, 0, 2).copy())
        i += n * N_LAYER
    return windows


def tokens(windows):
    """All traced token positions in order: int32 (T, 48, 10)."""
    return np.concatenate(windows, axis=0)


def keys(tok):
    """Flat access stream of cache keys layer*512+expert, ordered token by token, layer by layer: (T*48*10,)."""
    return (np.arange(N_LAYER, dtype=np.int64)[None, :, None] * N_EXPERT + tok.astype(np.int64)).reshape(-1)


def read_profile(path):
    blob = Path(path).read_bytes()
    assert blob[:4] == b"STRP"
    _ver, nl, ne, _slots, n = struct.unpack_from("<5I", blob, 4)
    pairs = np.frombuffer(blob, dtype=np.uint16, count=2 * n, offset=24).reshape(n, 2)
    return pairs[:, 0].astype(np.int64) * N_EXPERT + pairs[:, 1]


# ------------------------------------------------------------------ cache policies (hits counted per access)
def sim_static(stream, ranking, cap):
    resident = np.zeros(N_LAYER * N_EXPERT, dtype=bool)
    resident[np.asarray(ranking[:cap])] = True
    return resident[stream].mean()


def static_oracle_ranking(stream):
    freq = np.bincount(stream, minlength=N_LAYER * N_EXPERT)
    return np.argsort(-freq, kind="stable")


def sim_lru(stream, cap):
    c, hits = OrderedDict(), 0
    for x in stream.tolist():
        if x in c:
            hits += 1; c.move_to_end(x)
        else:
            c[x] = True
            if len(c) > cap:
                c.popitem(last=False)
    return hits / len(stream)


def sim_slru(stream, cap, protected_frac=0.8):
    """Segmented LRU: new keys enter probation; a hit promotes to protected; protected overflow demotes to probation."""
    prot_cap = max(1, int(cap * protected_frac))
    prob, prot, hits = OrderedDict(), OrderedDict(), 0
    for x in stream.tolist():
        if x in prot:
            hits += 1; prot.move_to_end(x)
        elif x in prob:
            hits += 1; del prob[x]; prot[x] = True
            if len(prot) > prot_cap:
                y, _ = prot.popitem(last=False); prob[y] = True
        else:
            prob[x] = True
        while len(prob) + len(prot) > cap:
            if prob:
                prob.popitem(last=False)
            else:
                prot.popitem(last=False)
    return hits / len(stream)


def sim_lfu(stream, cap):
    """Perfect LFU (counts over the whole history), ties broken by recency."""
    cnt, resident, heap, hits = {}, set(), [], 0
    for t, x in enumerate(stream.tolist()):
        cnt[x] = cnt.get(x, 0) + 1
        if x in resident:
            hits += 1
            heapq.heappush(heap, (cnt[x], t, x))
            continue
        resident.add(x); heapq.heappush(heap, (cnt[x], t, x))
        held = []
        while len(resident) > cap:
            c, tt, y = heapq.heappop(heap)
            if y not in resident or c != cnt[y]:
                continue                      # stale entry
            if y == x:
                held.append((c, tt, y)); continue   # always admit the key being accessed
            resident.discard(y)
        for e in held:
            heapq.heappush(heap, e)
    return hits / len(stream)


def next_use(stream):
    nxt = np.full(len(stream), np.iinfo(np.int64).max, dtype=np.int64)
    last = {}
    for i in range(len(stream) - 1, -1, -1):
        x = int(stream[i]); nxt[i] = last.get(x, np.iinfo(np.int64).max); last[x] = i
    return nxt


def sim_belady(stream, cap, nxt=None):
    """Belady's MIN: on a miss with a full cache evict the key reused furthest in the future (offline optimum)."""
    if nxt is None:
        nxt = next_use(stream)
    resident, heap, hits = {}, [], 0   # resident: key -> its next use
    s, n = stream.tolist(), nxt.tolist()
    for i, x in enumerate(s):
        if x in resident:
            hits += 1
        resident[x] = n[i]
        heapq.heappush(heap, (-n[i], x))
        while len(resident) > cap:
            nu, y = heapq.heappop(heap)
            if y in resident and resident[y] == -nu:
                del resident[y]
    return hits / len(stream)


# ------------------------------------------------------------------ locality statistics
def union_growth(tok, max_k=16):
    """Mean over positions and layers of |union of the experts of k consecutive tokens|, for k = 1..max_k."""
    T = len(tok); out = []
    onehot = np.zeros((T, N_LAYER, N_EXPERT), dtype=np.uint8)
    t_idx = np.arange(T)[:, None, None]; l_idx = np.arange(N_LAYER)[None, :, None]
    onehot[t_idx, l_idx, tok] = 1
    csum = np.concatenate([np.zeros((1, N_LAYER, N_EXPERT), dtype=np.int32), np.cumsum(onehot, axis=0, dtype=np.int32)])
    for k in range(1, max_k + 1):
        u = ((csum[k:] - csum[:-k]) > 0).sum(axis=2)   # (T-k+1, 48)
        out.append(float(u.mean()))
    return out


def consecutive_overlap(tok):
    """Mean fraction of a token's 10 experts (per layer) that the previous token also used."""
    a = np.sort(tok[1:], axis=2); b = np.sort(tok[:-1], axis=2)
    inter = np.array([[len(np.intersect1d(a[t, l], b[t, l], assume_unique=True)) for l in range(N_LAYER)] for t in range(len(a))])
    return float(inter.mean() / K)


def reuse_distance_tokens(tok):
    """For every (token, layer, expert) access, the distance in tokens to the previous use of that pair (inf if first)."""
    T = len(tok); last = np.full(N_LAYER * N_EXPERT, -1, dtype=np.int64); out = np.empty(T * N_LAYER * K, dtype=np.int64)
    ks = keys(tok).reshape(T, N_LAYER * K)
    for t in range(T):
        row = ks[t]; prev = last[row]
        out[t * N_LAYER * K:(t + 1) * N_LAYER * K] = np.where(prev >= 0, t - prev, -1)
        last[row] = t
    return out
