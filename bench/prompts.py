"""The fixed prompt set. Short prompts measure decode; long prompts measure prefill and decode at depth.

Long prompts are prefixes of data/longdoc.txt (setup.py + docs/DETAILS.md + src/program/generate.cpp of Strata,
concatenated). The character counts reproduce the token counts of the 2026-09-30 runs: 16,000 chars -> 5,395
tokens and 64,000 chars -> 18,900 tokens with this model's tokenizer.
"""
from pathlib import Path

DOC = (Path(__file__).parent / "data" / "longdoc.txt").read_text()

SHORT = {
    "code_en": "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "story_en": "Write a 600-word short story about a lighthouse keeper who finds a message in a bottle.",
    "explain_en": "Explain to a high-school student how a transformer language model turns a prompt into an answer. "
                  "Use concrete examples and no equations.",
    "math_en": "A tank holds 1,200 liters. Pipe A fills it in 6 hours and pipe B drains it in 10 hours. Starting "
               "empty with both open, how long until it is full? Then generalize to fill time a and drain time b. "
               "Show each step.",
    "essay_ja": "日本の四季について、それぞれの季節の行事や食べ物を交えながら800字程度で説明してください。",
    "tech_ja": "Pythonのリスト内包表記とジェネレータ式の違いを、メモリ使用量と実行速度の観点から具体例を使って説明してください。",
}

LONG = {   # name -> characters of DOC
    "long5k": 16000,
    "long19k": 64000,
    "long26k": 88000,
}


def messages(pid: str, nonce: str = ""):
    """Chat messages for a prompt id. `nonce` is prepended to long prompts so repeats cannot reuse a cached prefix."""
    if pid in SHORT:
        return [{"role": "user", "content": SHORT[pid]}], 512
    if pid in LONG:
        head = f"[run {nonce}] " if nonce else ""
        return [{"role": "user", "content": head + "Summarize what this script does in 5 bullet points.\n\n" + DOC[: LONG[pid]]}], 200
    raise KeyError(pid)
