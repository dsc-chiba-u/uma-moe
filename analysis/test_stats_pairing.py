"""Tests of stats_pairing: `python3 analysis/test_stats_pairing.py` from the repository root. The last test reads the
run x_f_up026_b66_exact_nodraft_convcache from results/ when present."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_pairing as SP  # noqa: E402
import summarize_runs as S  # noqa: E402

PROMPTS = [("code_en", 34), ("story_en", 34), ("explain_en", 38), ("math_en", 70), ("essay_ja", 35), ("tech_ja", 42)]


def requests(repeats=2, fail_zero=(), fail_partial=()):
    """requests.jsonl rows; fail_zero: positions that produced no tokens; fail_partial: positions that broke off."""
    out = []
    for rep in range(repeats):
        for pid, pt in PROMPTS:
            i = len(out)
            if i in fail_zero:
                out.append({"prompt_id": pid, "repeat": rep, "completion_tokens": 0, "prompt_tokens": None})
            elif i in fail_partial:
                out.append({"prompt_id": pid, "repeat": rep, "completion_tokens": 100, "prompt_tokens": None})
            else:
                out.append({"prompt_id": pid, "repeat": rep, "completion_tokens": 256, "prompt_tokens": pt})
    return out


def stats_for(reqs, skip=(), tokens=True):
    """One Strata statistics row per request not in `skip`, carrying the request's index as decode_tok_s."""
    return [{"prompt_tokens": PROMPTS[i % len(PROMPTS)][1] if tokens else None, "gen_tokens": 256, "gen_ms": 1000,
             "decode_tok_s": float(i)} for i in range(len(reqs)) if i not in skip]


class Pairing(unittest.TestCase):
    def test_no_failure(self):
        reqs = requests()
        idx, status = SP.pair_server_stats("strata", reqs, stats_for(reqs))
        self.assertEqual((idx, status), (list(range(12)), "complete"))

    def test_first_request_zero_tokens(self):
        reqs = requests(fail_zero={0})
        idx, status = SP.pair_server_stats("strata", reqs, stats_for(reqs, skip={0}))
        self.assertEqual(status, "finished-only")
        self.assertEqual(idx, list(range(1, 12)))

    def test_middle_request_zero_tokens(self):
        reqs = requests(fail_zero={5})
        idx, status = SP.pair_server_stats("strata", reqs, stats_for(reqs, skip={5}))
        self.assertEqual(status, "finished-only")
        self.assertEqual(idx, [i for i in range(12) if i != 5])

    def test_partial_output_without_statistics(self):
        # the request streamed 100 tokens, then the server died: no usage block, no statistics line
        reqs = requests(fail_partial={6})
        idx, status = SP.pair_server_stats("strata", reqs, stats_for(reqs, skip={6}))
        self.assertEqual(status, "finished-only")
        self.assertEqual(idx, [i for i in range(12) if i != 6])

    def test_partial_output_then_later_requests_succeed(self):
        reqs = requests(fail_partial={1})
        idx, status = SP.pair_server_stats("strata", reqs, stats_for(reqs, skip={1}))
        self.assertEqual(status, "finished-only")
        self.assertEqual(idx, [0] + list(range(2, 12)))

    def test_finished_request_without_statistics_resolved_by_tokens(self):
        # the client got the usage block but the server's line was not parsed: lengths differ, the token counts decide
        reqs = requests()
        idx, status = SP.pair_server_stats("strata", reqs, stats_for(reqs, skip={3}))   # math_en, 70 tokens, unique
        self.assertEqual(status, "aligned")
        self.assertEqual(idx, [i for i in range(12) if i != 3])

    def test_ambiguous_is_undetermined(self):
        reqs = requests()
        idx, status = SP.pair_server_stats("strata", reqs, stats_for(reqs, skip={0}))   # code_en/0: story_en also 34
        self.assertEqual(status, "undetermined")
        self.assertEqual(idx, [None] * 11)

    def test_token_disagreement_is_undetermined(self):
        reqs = requests(fail_zero={0})
        st = stats_for(reqs, skip={0})
        st[0]["prompt_tokens"] = 999
        idx, status = SP.pair_server_stats("strata", reqs, st)
        self.assertEqual(status, "undetermined")

    def test_legacy_all_requests(self):
        # a row for every request although one has completion_tokens 0 (a server that printed a line anyway)
        reqs = requests(fail_zero={2})
        idx, status = SP.pair_server_stats("strata", reqs, stats_for(reqs))
        self.assertEqual((idx, status), (list(range(12)), "complete"))

    def test_llama_uses_prompt_read(self):
        reqs = requests(fail_zero={0})
        st = [{"prompt_read": PROMPTS[i % 6][1], "gen_tokens": 128, "gen_ms": 1000} for i in range(1, 12)]
        idx, status = SP.pair_server_stats("llama", reqs, st)
        self.assertEqual(status, "finished-only")
        self.assertEqual(idx, list(range(1, 12)))

    def test_label_stats(self):
        reqs = requests(fail_zero={0})
        st = stats_for(reqs, skip={0})
        self.assertEqual(SP.label_stats("strata", reqs, st), "finished-only")
        self.assertEqual((st[0]["prompt_id"], st[0]["repeat"]), ("story_en", 0))
        self.assertEqual((st[5]["prompt_id"], st[5]["repeat"]), ("code_en", 1))

    def test_summarize_with_failed_first_request(self):
        reqs = requests(fail_zero={0})
        st = stats_for(reqs, skip={0})
        run = {"name": "t", "engine": "strata", "server_stats": st}
        out = S.summarize_records(run, reqs, [])
        self.assertEqual(out["server_stats_pairing"], "finished-only")
        self.assertEqual(out["prompts"]["code_en"]["n"], 1)
        self.assertEqual(out["prompts"]["code_en"]["decode_tok_s"], 6.0)          # the row of request 6 (code_en/1)
        self.assertEqual(out["prompts"]["story_en"]["decode_tok_s"], (1.0 + 7.0) / 2)

    def test_summarize_undetermined_reports_missing(self):
        reqs = requests()
        st = stats_for(reqs, skip={0})
        run = {"name": "t", "engine": "strata", "server_stats": st}
        out = S.summarize_records(run, reqs, [])
        self.assertEqual(out["server_stats_pairing"], "undetermined")
        for p in out["prompts"].values():
            self.assertIsNone(p["decode_tok_s"]); self.assertEqual(p["n"], 0)


class RecordedRun(unittest.TestCase):
    RUN = Path(__file__).resolve().parent.parent / "results" / "x_f_up026_b66_exact_nodraft_convcache"

    @unittest.skipUnless(RUN.exists(), "run not present")
    def test_convcache_run(self):
        out = S.summarize(self.RUN)
        self.assertEqual(out["server_stats_pairing"], "finished-only")
        p = out["prompts"]
        self.assertEqual(p["code_en"]["n"], 1)
        self.assertAlmostEqual(p["code_en"]["decode_tok_s"], 5.0)
        self.assertAlmostEqual(p["story_en"]["decode_tok_s"], 5.65)
        self.assertAlmostEqual(p["tech_ja"]["decode_tok_s"], (5.3 + 2.5) / 2)
        run = json.load(open(self.RUN / "run.json"))
        self.assertEqual(len(run["server_stats"]), 11)


if __name__ == "__main__":
    unittest.main()
