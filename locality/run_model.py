"""Per-model pipeline of the locality study.

    python -I locality/run_model.py KEY gen     # llama-server: chat answers + a 10-turn dialog, greedy (GPU, needs lock)
    python -I locality/run_model.py KEY tok     # tokenize the text corpora (vocab only; no GPU)
    python -I locality/run_model.py KEY trace   # moe_trace: teacher-forced routing of every document (GPU, needs lock)

Outputs under results/locality/KEY/: gen/ (server log, transcripts), tokens/ (*.tok int32, manifest.tsv, docs.json),
trace/ (*.route.npy, *.nll.npy, trace_meta.json, trace.log).
Exit code 75 = not enough free memory right now (retry later, without holding the lock).
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bench"))
from models import MODELS, file_bytes, gguf_path  # noqa: E402
import prompts as P  # noqa: E402  (read-only import of the main study's prompt set)

ROOT = HERE.parent
RES = ROOT / "results/locality"
CORP = RES / "corpora"
BIN = HERE / "llama-bin"
ENV = dict(os.environ, LD_LIBRARY_PATH=str(BIN))
PORT = 18431
MIN_AVAIL_GIB = 12.0      # kill the job below this
MARGIN_GIB = 20.0         # keep at least this much free while running

DOMAIN_DOCS = {"code": ["code_py", "code_cpp"], "prose": ["en_wiki", "en_fiction"], "ja": ["ja_wiki", "ja_lit"],
               "math": ["math_gsm8k", "math_comp"]}
LONG_DOCS = ["long_mixed", "long_book", "long_code"]
DOMAIN_BUDGET, BLOCK_BUDGET, LONG_BUDGET = 6144, 4096, 32768
CHAT_MAX, TURN_MAX = 1024, 384
PROBE_LEN, PROBE_CTX = 2048, (0, 128, 1024)

DIALOG = [
    "I'm planning a 4-day trip to Kyoto in mid-November. What should I prioritize?",
    "Which of those places are best for autumn leaves, and how early should I arrive to avoid the crowds?",
    "Write a short Python script that splits a travel budget of 120,000 yen across lodging, food, transport and "
    "entrance fees using given percentages, and prints a table.",
    "Change the script so the percentages come from the command line and are checked to sum to 100.",
    "How do I say 'Could you recommend a good local restaurant?' and 'Do you have an English menu?' in Japanese? "
    "Give the Japanese and the romaji.",
    "京都の紅葉の名所を3つ、それぞれ一文で紹介してください。",
    "A temple opens at 8:30. If 60 people are already waiting at opening, 25 people are admitted every 10 minutes, "
    "and 15 new people join the queue every 10 minutes, how long does someone who arrives at 9:00 wait? Show the steps.",
    "Summarize our conversation so far as a short checklist for the trip.",
    "Write a haiku about Kyoto in autumn and explain its structure.",
    "Finally, what etiquette mistakes do tourists commonly make at temples and shrines?",
]


def mem_avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    return 0.0


def need_gib(key):
    return file_bytes(key) / 2**30 + 6.0 + MARGIN_GIB


def check_memory(key):
    need, have = need_gib(key), mem_avail()
    print(f"[{key}] MemAvailable {have:.1f} GiB, need {need:.1f} GiB", flush=True)
    if have < need:
        sys.exit(75)


class Watchdog(threading.Thread):
    """Kill the process group of `proc` if MemAvailable falls under MIN_AVAIL_GIB; record the minimum seen."""

    def __init__(self, proc):
        super().__init__(daemon=True)
        self.proc, self.stop, self.min_avail, self.fired = proc, threading.Event(), 1e9, False

    def run(self):
        while not self.stop.is_set() and self.proc.poll() is None:
            a = mem_avail()
            self.min_avail = min(self.min_avail, a)
            if a < MIN_AVAIL_GIB:
                self.fired = True
                print(f"WATCHDOG: MemAvailable {a:.1f} GiB, killing pid {self.proc.pid}", flush=True)
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
            time.sleep(0.5)


def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def write_tok(path, toks):
    import numpy as np
    np.asarray(toks, dtype=np.int32).tofile(path)


# ------------------------------------------------------------------------------------------------ stage gen
def stage_gen(key):
    m = MODELS[key]
    out = RES / key / "gen"
    tokdir = RES / key / "tokens"
    out.mkdir(parents=True, exist_ok=True); tokdir.mkdir(parents=True, exist_ok=True)
    check_memory(key)
    if health():
        sys.exit(f"port {PORT} already in use")
    cmd = [str(BIN / "llama-server"), "-m", str(gguf_path(key)), "-ngl", "999", "-fa", "on", "-c", "32768", "--jinja",
           "--host", "127.0.0.1", "--port", str(PORT), "-np", "1", "-t", "4", "--no-warmup", "-cram", "0"]
    log = open(out / "server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
    wd = Watchdog(proc); wd.start()
    t0 = time.time()
    res = {"model": key, "cmd": cmd, "chat_template_kwargs": m["kwargs"], "chat": {}, "dialog": {}}
    try:
        while not health():
            if proc.poll() is not None or time.time() - t0 > 1200:
                raise RuntimeError("server did not start")
            time.sleep(2)
        res["load_s"] = time.time() - t0
        kw = m["kwargs"]

        def render_tokens(msgs):
            prompt = post("/apply-template", {"messages": msgs, "chat_template_kwargs": kw})["prompt"]
            toks = post("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
            dup = len(toks) > 1 and toks[0] == toks[1] and toks[0] in BOS_LIKE
            if dup:
                toks = toks[1:]
            return prompt, toks, dup

        def complete(toks, n):
            r = post("/completion", {"prompt": toks, "n_predict": n, "temperature": 0.0, "return_tokens": True,
                                     "cache_prompt": False, "stream": False})
            return r

        # BOS id (if the model adds one) to detect a doubled BOS from template + add_special
        BOS_LIKE = set(post("/tokenize", {"content": "", "add_special": True, "parse_special": True})["tokens"])
        for pid, text in P.SHORT.items():
            msgs = [{"role": "user", "content": text}]
            prompt, ptoks, dup = render_tokens(msgs)
            r = complete(ptoks, CHAT_MAX)
            gen = r["tokens"]
            write_tok(tokdir / f"chat_{pid}.tok", ptoks + gen)
            res["chat"][pid] = {"prompt_tokens": len(ptoks), "gen_tokens": len(gen), "stop_type": r.get("stop_type"),
                                "double_bos_removed": dup, "prompt": prompt, "content": r["content"],
                                "timings": r.get("timings")}
            print(f"[{key}] chat {pid}: prompt {len(ptoks)} + gen {len(gen)} ({r.get('stop_type')})", flush=True)
        msgs, turns = [], []
        for i, u in enumerate(DIALOG):
            msgs.append({"role": "user", "content": u})
            if i < len(DIALOG) - 1:
                body = {"messages": msgs, "max_tokens": TURN_MAX, "temperature": 0.0, "chat_template_kwargs": kw,
                        "cache_prompt": True, "stream": False}
                r = post("/v1/chat/completions", body)
                msg = r["choices"][0]["message"]
                content = msg.get("content") or ""
                turns.append({"user": u, "content": content, "reasoning": msg.get("reasoning_content"),
                              "usage": r.get("usage"), "finish": r["choices"][0].get("finish_reason")})
                msgs.append({"role": "assistant", "content": content})
                print(f"[{key}] dialog turn {i + 1}: {r.get('usage', {}).get('completion_tokens')} tokens", flush=True)
            else:
                prompt, ptoks, dup = render_tokens(msgs)
                r = complete(ptoks, TURN_MAX)
                gen = r["tokens"]
                write_tok(tokdir / "dialog.tok", ptoks + gen)
                turns.append({"user": u, "content": r["content"], "prompt_tokens": len(ptoks), "gen_tokens": len(gen),
                              "stop_type": r.get("stop_type"), "double_bos_removed": dup})
                res["dialog"] = {"turns": turns, "prompt_tokens": len(ptoks), "gen_tokens": len(gen), "final_prompt": prompt}
                print(f"[{key}] dialog final: prompt {len(ptoks)} + gen {len(gen)}", flush=True)
        res["seconds"] = time.time() - t0
        (out / "gen.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    finally:
        wd.stop.set()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
        res["mem_avail_min_gib"] = wd.min_avail
        res["watchdog_fired"] = wd.fired
        json.dump(res, open(out / "gen.json", "w"), indent=1, ensure_ascii=False)
    if wd.fired:
        sys.exit(2)


# ------------------------------------------------------------------------------------------------ stage tok
def stage_tok(key):
    import numpy as np
    tokdir = RES / key / "tokens"
    raw = tokdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    docs = CORP / "docs"
    names = [n for d in DOMAIN_DOCS.values() for n in d] + ["long_book", "long_code"]
    blocks = sorted(docs.glob("long_mixed_*.txt"))
    empty = raw / "empty.txt"; empty.write_text("")
    gen = json.load(open(RES / key / "gen/gen.json"))
    # every raw-text document starts with `head`: the BOS token(s) the model adds, or, for a model that cannot model
    # raw text (gpt-oss: NLL rises to 4-8 nats/token), its chat template up to the start of an assistant message, so
    # the document is teacher-forced as the model's own answer
    wrap = MODELS[key].get("doc_wrap")
    prefix = MODELS[key].get("doc_prefix")   # fixed special-token prefix, e.g. GLM's [gMASK]<sop>
    rows = [(empty, raw / "bos.tok", 1, 0)]
    if wrap:
        sys_part = gen["chat"]["code_en"]["prompt"].split(wrap["split"])[0]
        (raw / "wrap.txt").write_text(sys_part + wrap["append"])
        rows.append((raw / "wrap.txt", raw / "wrap.tok", 0, 1))
    elif prefix:
        (raw / "wrap.txt").write_text(prefix)
        rows.append((raw / "wrap.txt", raw / "wrap.tok", 0, 1))
        wrap = True
    rows += [(docs / f"{n}.txt", raw / f"{n}.tok", 0, 0) for n in names]
    rows += [(b, raw / f"{b.stem}.tok", 0, 0) for b in blocks]
    lst = raw / "tokenize.tsv"
    lst.write_text("".join(f"{a}\t{b}\t{s}\t{ps}\n" for a, b, s, ps in rows))
    # vocab only: hide the GPU so that tokenizing never creates a CUDA context
    subprocess.run([str(HERE / "moe_trace"), "tokenize", "-m", str(gguf_path(key)), "-l", str(lst)],
                   env=dict(ENV, CUDA_VISIBLE_DEVICES=""), check=True, stdout=subprocess.DEVNULL)
    load = lambda p: np.fromfile(p, dtype=np.int32)  # noqa: E731
    bos = load(raw / "bos.tok")
    head = load(raw / "wrap.tok") if wrap else bos
    info = {"bos": bos.tolist(), "head": head.tolist(), "wrapped": bool(wrap), "docs": {}}
    manifest = []

    def emit(doc_id, corpus, toks, extra=None):
        p = tokdir / f"{doc_id}.tok"
        toks = np.asarray(toks, dtype=np.int32); toks.tofile(p)
        info["docs"][doc_id] = {"corpus": corpus, "n_tokens": int(len(toks)), **(extra or {})}
        manifest.append((doc_id, p))

    for corpus, ds in DOMAIN_DOCS.items():
        for n in ds:
            t = load(raw / f"{n}.tok")
            if len(t) < DOMAIN_BUDGET:
                print(f"WARNING {n}: only {len(t)} tokens", flush=True)
            emit(n, corpus, np.concatenate([head, t[:DOMAIN_BUDGET]]), {"source_tokens": int(len(t)), "head_tokens": int(len(head))})
    for pid, c in gen["chat"].items():
        t = load(tokdir / f"chat_{pid}.tok")
        info["docs"][f"chat_{pid}"] = {"corpus": "chat", "n_tokens": int(len(t)), "prompt_tokens": c["prompt_tokens"],
                                      "gen_tokens": c["gen_tokens"]}
        manifest.append((f"chat_{pid}", tokdir / f"chat_{pid}.tok"))
    t = load(tokdir / "dialog.tok")
    info["docs"]["dialog"] = {"corpus": "dialog", "n_tokens": int(len(t)), "prompt_tokens": gen["dialog"]["prompt_tokens"],
                              "gen_tokens": gen["dialog"]["gen_tokens"]}
    manifest.append(("dialog", tokdir / "dialog.tok"))
    parts, bounds = [head], []
    pos = len(head)
    for b in blocks:
        t = load(raw / f"{b.stem}.tok")[:BLOCK_BUDGET]
        if len(t) < BLOCK_BUDGET:
            print(f"WARNING {b.stem}: only {len(t)} tokens", flush=True)
        bounds.append([pos, pos + len(t)]); pos += len(t); parts.append(t)
    mixed = np.concatenate(parts)[:LONG_BUDGET]
    bounds[-1][1] = min(bounds[-1][1], LONG_BUDGET)
    emit("long_mixed", "long", mixed, {"blocks": bounds,
                                       "block_domains": [d["domain"] for d in json.load(open(CORP / "corpora.json"))["long_mixed_blocks"]]})
    longs = {"long_mixed": mixed}
    for n in ("long_book", "long_code"):
        src = load(raw / f"{n}.tok")
        t = np.concatenate([head, src])[:LONG_BUDGET]
        emit(n, "long", t, {"source_tokens": int(len(src)), "head_tokens": int(len(head))})
        longs[n] = t
    # context probes: the last PROBE_LEN tokens of each long document again, with only c tokens of preceding context
    # (c = 0: standalone), to compare their routing with the routing in the full ~30K-token context
    for n, t in longs.items():
        a = len(t) - PROBE_LEN
        for c in PROBE_CTX:
            emit(f"probe_{n}_c{c}", "probe", np.concatenate([head, t[a - c:]]),
                 {"source": n, "target_start_in_source": int(a), "target_offset": int(len(head) + c), "target_len": PROBE_LEN})
    (tokdir / "manifest.tsv").write_text("".join(f"{d}\t{p}\n" for d, p in manifest))
    json.dump(info, open(tokdir / "docs.json", "w"), indent=1)
    tot = sum(d["n_tokens"] for d in info["docs"].values())
    print(f"[{key}] {len(manifest)} documents, {tot} tokens", flush=True)


# ------------------------------------------------------------------------------------------------ stage trace
def stage_trace(key):
    tokdir = RES / key / "tokens"
    out = RES / key / "trace"
    out.mkdir(parents=True, exist_ok=True)
    check_memory(key)
    docs = json.load(open(tokdir / "docs.json"))["docs"]
    v = docs["chat_code_en"]
    validate = f"chat_code_en:{v['prompt_tokens']}:{min(512, v['gen_tokens'])}"
    cmd = [str(HERE / "moe_trace"), "trace", "-m", str(gguf_path(key)), "-l", str(tokdir / "manifest.tsv"), "-o", str(out),
           "-c", "32768", "-ub", "512", "-t", "4", "--validate", validate, "--min-avail-gib", str(MIN_AVAIL_GIB)]
    t0 = time.time()
    with open(out / "trace.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
        wd = Watchdog(proc); wd.start()
        rc = proc.wait()
        wd.stop.set()
    print(open(out / "trace.log").read()[-3000:], flush=True)
    json.dump({"cmd": cmd, "rc": rc, "seconds": time.time() - t0, "mem_avail_min_gib": wd.min_avail,
               "watchdog_fired": wd.fired}, open(out / "run.json", "w"), indent=1)
    if rc != 0:
        sys.exit(rc if rc != 75 else 1)


if __name__ == "__main__":
    key, stage = sys.argv[1], sys.argv[2]
    {"gen": stage_gen, "tok": stage_tok, "trace": stage_trace}[stage](key)
