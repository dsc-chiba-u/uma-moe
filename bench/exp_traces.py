"""E3: routing traces from Strata's serve path (--dump-routing). Record = int32 layer, int32 k, k int32 ids, k float weights.
nodraft: every verify window is one token, so records are exactly per generated token.
mtp:     default MTP drafting; consecutive records of one layer form a verify window (drafts included)."""
import sys, harness as H
which = sys.argv[1]
prompts = list(H.P.SHORT) + ["long5k", "long19k"]
name = f"trace_{which}"
out = H.RESULTS / name; out.mkdir(parents=True, exist_ok=True)
args = H.strata_args("auto", mtp=(which == "mtp"), extra=["--dump-routing", str(out / "routing.bin")])
s = H.run(name, "strata", args, prompts, repeats=1, need_gib=85, max_tokens=1024, meta={"purpose": "routing trace"})
print({k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_min_gib")})
