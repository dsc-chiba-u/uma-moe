"""Pairing a run's server statistics with its requests (shared by bench/harness.py and the analysis scripts).

The server prints one statistics line per request it finished (Strata: `N generated in M ms`; llama.cpp: `eval time`).
The harness logs every request it sent in requests.jsonl, in order, with the usage block the server returns at the end
of a successful stream (`prompt_tokens`, `completion_tokens`). A request that failed has no statistics line: one that
produced no tokens (a server error, a restart) has `completion_tokens` 0; one that broke off after some tokens (the
memory watchdog, a lost connection) has the streamed count in `completion_tokens` but no usage block, so its
`prompt_tokens` is None. Neither leaves a line, and a plain zip of the two lists shifts every later label by one
(x_f_up026_b66_exact_nodraft_convcache, whose first request produced 0 tokens, had its labels off by one until
2026-11-22).

pair_server_stats() returns, for each statistics row, the index of the request it belongs to, or None, and a status:
  "complete"       every request has a row (no failure);
  "finished-only"  the rows are the requests that finished (usage received), in order; the prompt token counts agree;
  "aligned"        lengths differ beyond that, but exactly one increasing assignment of rows to finished requests
                   matches the prompt token counts;
  "undetermined"   none of the above (the counts disagree or several assignments match): every index is None, and the
                   callers report the server statistics of that run as missing rather than guess.
Prompt token counts: the harness's `prompt_tokens` (from the usage block) against Strata's `prompt_tokens` or, for
llama.cpp, `prompt_read` (its `prompt eval time ... / N tokens`, the prompt tokens when nothing was reused from a cache).
A count missing on either side matches anything.
"""
from itertools import combinations

MAX_ASSIGNMENTS = 20000   # bound on the assignments enumerated for "aligned"


def finished(req):
    """True if the request received its usage block (the server finished it and said how many tokens)."""
    return (req.get("completion_tokens") or 0) > 0 and req.get("prompt_tokens") is not None


def stat_prompt_tokens(engine, s):
    return s.get("prompt_tokens") if engine == "strata" else s.get("prompt_read")


def _agree(engine, stats, reqs):
    for s, r in zip(stats, reqs):
        a, b = stat_prompt_tokens(engine, s), r.get("prompt_tokens")
        if a is not None and b is not None and int(a) != int(b):
            return False
    return True


def pair_server_stats(engine, reqs, stats):
    """-> (indices, status): indices[i] is the index in `reqs` of the request behind stats[i], or None."""
    if not stats:
        return [], "complete" if not reqs else "finished-only"
    fin = [i for i, r in enumerate(reqs) if finished(r)]
    if len(stats) == len(fin) and _agree(engine, stats, [reqs[i] for i in fin]):
        return fin, "complete" if len(fin) == len(reqs) else "finished-only"
    if len(stats) == len(reqs) and _agree(engine, stats, reqs):
        return list(range(len(reqs))), "complete"
    # lengths differ: a request that finished for the client but left no statistics line (or the reverse). Accept the
    # assignment only if the prompt token counts single it out.
    cands = fin if len(fin) >= len(stats) else list(range(len(reqs)))
    if len(cands) >= len(stats):
        n_comb = 1
        for j in range(len(cands) - len(stats)):
            n_comb = n_comb * (len(cands) - j) // (j + 1)
        if n_comb <= MAX_ASSIGNMENTS:
            found = [c for c in combinations(cands, len(stats)) if _agree(engine, stats, [reqs[i] for i in c])]
            if len(found) == 1:
                return list(found[0]), "aligned"
    return [None] * len(stats), "undetermined"


def label_stats(engine, reqs, stats):
    """Writes prompt_id/repeat into each statistics row whose request is known; returns the status."""
    idx, status = pair_server_stats(engine, reqs, stats)
    for s, i in zip(stats, idx):
        if i is None:
            s.pop("prompt_id", None); s.pop("repeat", None)
        else:
            s["prompt_id"], s["repeat"] = reqs[i]["prompt_id"], reqs[i]["repeat"]
    return status
