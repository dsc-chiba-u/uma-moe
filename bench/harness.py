"""Benchmark harness: start one inference server, run the prompt set against it, sample the machine, stop it.

Every run writes results/<name>/: run.json (summary), requests.jsonl (one line per request, with the full
generated text), samples.csv (CPU / memory / GPU every 0.25 s), server.log, and for Strata engine.log.

Safety on a 128 GB unified-memory machine: a run refuses to start when MemAvailable is below `need_gib`, and a
watchdog kills the server's whole process group if MemAvailable drops under WATCHDOG_GIB while it runs (the
known failure mode is the whole box freezing when GPU allocations exhaust memory).
"""
import json, os, re, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prompts as P
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import stats_pairing as SP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STRATA = Path.home() / "src/Strata"
STRATA_CFG = STRATA / "strata-q2_0.json"
LLAMA_MASTER = Path.home() / "src/llama.cpp-master/build/bin"
Q2_GGUF = Path.home() / "src/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
MTP_GGUF = Path.home() / "models/unsloth-Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
PORT = 8080
WATCHDOG_GIB = 6.0


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0]) / 1048576.0   # GiB
    return d


# ---------------------------------------------------------------- sampling
class Sampler:
    """CPU utilisation (all / big / little cores), MemAvailable, and nvidia-smi GPU metrics every 0.25 s."""
    BIG = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19}

    def __init__(self, path):
        self.path, self.stop_ev = path, threading.Event()
        self.gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm",
                                     "--format=csv,noheader,nounits", "-lms", "250"], stdout=subprocess.PIPE, text=True)
        self.gpu_lines = []
        threading.Thread(target=self._read_gpu, daemon=True).start()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _read_gpu(self):
        for line in self.gpu.stdout:
            self.gpu_lines.append((time.time(), line.strip()))

    @staticmethod
    def _nvme_sectors():
        return int(open("/sys/block/nvme0n1/stat").read().split()[2])

    @staticmethod
    def _cpu():
        out = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3] != " ":
                f = line.split(); c = int(f[0][3:]); v = list(map(int, f[1:8]))
                out[c] = (sum(v), v[3] + v[4])   # total, idle+iowait
        return out

    def _run(self):
        with open(self.path, "w") as f:
            f.write("t,cpu_all,cpu_big,cpu_little,mem_avail_gib,cached_gib,gpu_util,gpu_mem_util,gpu_power_w,sm_mhz,nvme_read_mbs\n")
            prev = self._cpu(); prev_rd = self._nvme_sectors(); prev_t = time.time()
            while not self.stop_ev.wait(0.25):
                cur = self._cpu(); m = meminfo()
                def util(cs):
                    tot = sum(cur[c][0] - prev[c][0] for c in cs); idle = sum(cur[c][1] - prev[c][1] for c in cs)
                    return 100.0 * (tot - idle) / tot if tot else 0.0
                allc = set(cur); big = self.BIG & allc; little = allc - big
                g = self.gpu_lines[-1][1].split(", ") if self.gpu_lines else ["", "", "", "", ""]
                rd, now = self._nvme_sectors(), time.time()
                mbs = (rd - prev_rd) * 512 / 1e6 / max(now - prev_t, 1e-3)
                f.write(f"{now:.3f},{util(allc):.1f},{util(big):.1f},{util(little):.1f},{m['MemAvailable']:.2f},"
                        f"{m['Cached']:.2f},{g[1]},{g[2]},{g[3]},{g[4]},{mbs:.1f}\n")
                prev, prev_rd, prev_t = cur, rd, now

    def stop(self):
        self.stop_ev.set(); self.th.join(); self.gpu.terminate()


# ---------------------------------------------------------------- servers
class Server:
    def __init__(self, name, cmd, cwd, log_path, ready, env=None):
        self.name, self.ready = name, ready
        self.log = open(log_path, "w")
        self.t0 = time.time()
        self.proc = subprocess.Popen(cmd, cwd=cwd, stdout=self.log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.killed_by_watchdog = False
        self.min_avail = meminfo()["MemAvailable"]
        self._wd_stop = threading.Event()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        while not self._wd_stop.wait(0.5):
            a = meminfo()["MemAvailable"]
            self.min_avail = min(self.min_avail, a)
            if a < WATCHDOG_GIB and self.proc.poll() is None:
                self.killed_by_watchdog = True
                print(f"WATCHDOG: MemAvailable {a:.1f} GiB < {WATCHDOG_GIB} GiB, killing {self.name}", flush=True)
                os.killpg(self.proc.pid, signal.SIGKILL)

    def wait_ready(self, timeout=900):
        while time.time() - self.t0 < timeout:
            if self.proc.poll() is not None:
                return False
            if self.ready():
                return True
            time.sleep(1)
        return False

    def stop(self):
        self._wd_stop.set()
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL); self.proc.wait()
        # the engine child can outlive server.py for a moment; wait until its memory is back
        for _ in range(60):
            try:
                os.killpg(self.proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
        self.log.close()


def health_ok():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _home(v):
    """Expand a leading ~ in a Strata config (the released configs write the home directory as ~)."""
    if isinstance(v, str):
        return os.path.expanduser(v) if v.startswith("~") else v
    if isinstance(v, list):
        return [_home(x) for x in v]
    if isinstance(v, dict):
        return {k: _home(x) for k, x in v.items()}
    return v


def start_strata(outdir, args, log_name="engine.log", env=None, exe=None, cfg_file=None, server_root=None):
    cfg = _home(json.load(open(cfg_file or STRATA_CFG)))
    cfg["args"] = args
    if exe:
        cfg["exe"] = str(exe)
    cfg["log"] = str(outdir / log_name)
    cfg_path = outdir / "strata-config.json"
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    root = Path(server_root) if server_root else STRATA
    cmd = [str(STRATA / ".venv/bin/python"), str(root / "serve/server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(PORT)]
    log_path = outdir / "server.log"
    full_env = dict(os.environ, **env) if env else None
    return Server("strata", cmd, root, log_path, ready=lambda: "ready: http" in open(log_path).read(), env=full_env)


def strata_args(expert_cache="auto", mtp=True, extra=(), spec=None, cfg_file=None):
    """Strata's own config args with the expert cache size, drafting and the MTP window (--spec T: the current
    token + T-1 drafts; the config's default is 4) switched."""
    base = _home(json.load(open(cfg_file or STRATA_CFG)))["args"]
    out, i = [], 0
    while i < len(base):
        if base[i] == "--expert-cache":
            out += ["--expert-cache", str(expert_cache)]; i += 2; continue
        if base[i] == "--spec" and spec is not None:
            out += ["--spec", str(spec)]; i += 2; continue
        out.append(base[i]); i += 1
    if not mtp:   # the native pack needs --spec and --mtp; a 1-token MTP window and no suffix drafts = no drafts
        out += ["--mtp-max-t", "1", "--suffix-draft", "0"]
    return out + list(extra)


def start_llama(outdir, args, binary="llama-server", env=None):
    cmd = [str(LLAMA_MASTER / binary), "--port", str(PORT), "--host", "127.0.0.1"] + list(args)
    full_env = dict(os.environ, **env) if env else None
    return Server("llama", cmd, None, outdir / "server.log", ready=health_ok, env=full_env)


# ---------------------------------------------------------------- client
def request(engine, msgs, max_tokens):
    body = {"model": "x", "messages": msgs, "max_tokens": max_tokens, "stream": True, "temperature": 0}
    if engine == "strata":
        body["reasoning_effort"] = "off"
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["stream_options"] = {"include_usage": True}
        body["cache_prompt"] = False
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                s = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if s:
                    if first is None:
                        first = time.time()
                    text += s; chunks += 1
    t1 = time.time()
    ct = (usage or {}).get("completion_tokens") or chunks
    return {"t_send": t0, "t_first": first, "t_end": t1, "ttft_s": (first - t0) if first else None,
            "completion_tokens": ct, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "decode_tok_s_client": ct / (t1 - first) if first and t1 > first else None, "text": text}


# ---------------------------------------------------------------- log parsing
STRATA_RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms "
                       r"\(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
STRATA_HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)%")
LLAMA_PROMPT = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
LLAMA_EVAL = re.compile(r"\s eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
LLAMA_DRAFT = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)(?:, mean len =\s*([\d.]+))?")


def parse_server_stats(engine, outdir):
    rows = []
    if engine == "strata":
        txt = open(outdir / "engine.log", errors="replace").read()
        hits = [float(h) for h in STRATA_HIT.findall(txt)]
        for i, m in enumerate(STRATA_RE.findall(txt)):
            pt, reused, read, pms, gen, gms, tps, acc, drafted = m
            rows.append({"prompt_tokens": int(pt), "prompt_read": int(read), "prefill_ms": int(pms), "gen_tokens": int(gen),
                         "gen_ms": int(gms), "decode_tok_s": float(tps), "drafts_accepted": int(acc), "drafts": int(drafted),
                         "hit_rate": hits[i] if i < len(hits) else None})
    else:
        txt = open(outdir / "server.log", errors="replace").read()
        pr = LLAMA_PROMPT.findall(txt); ev = LLAMA_EVAL.findall(txt); dr = LLAMA_DRAFT.findall(txt)
        for i in range(min(len(pr), len(ev))):
            # the server's own figure: llama.cpp's first output token comes out of the prompt pass, so its printed
            # tokens/s is (generated - 1) / generation time (tools/server/server-common.h, n_gen_tps); runs recorded
            # before 2026-11-22 stored generated / time here, which analysis/summarize_runs.decode_rate() corrects
            r = {"prompt_read": int(pr[i][1]), "prefill_ms": float(pr[i][0]), "gen_tokens": int(ev[i][1]), "gen_ms": float(ev[i][0]),
                 "decode_tok_s": (int(ev[i][1]) - 1) / float(ev[i][0]) * 1000}
            if i < len(dr):
                r.update({"draft_accept_rate": float(dr[i][0]), "drafts_accepted": int(dr[i][1]), "drafts": int(dr[i][2]),
                          "mean_len": float(dr[i][3]) if dr[i][3] else None})
            rows.append(r)
    return rows


# ---------------------------------------------------------------- one run
def run(name, engine, args, prompt_ids, repeats=3, need_gib=80, meta=None, max_tokens=None, env=None, exe=None,
        cfg_file=None, server_root=None):
    outdir = RESULTS / name
    outdir.mkdir(parents=True, exist_ok=True)
    m0 = meminfo()
    if m0["MemAvailable"] < need_gib:
        raise RuntimeError(f"{name}: MemAvailable {m0['MemAvailable']:.1f} GiB < {need_gib} GiB needed")
    t_start = time.time()
    srv = (start_strata(outdir, args, env=env, exe=exe, cfg_file=cfg_file, server_root=server_root) if engine == "strata"
           else start_llama(outdir, args, env=env))
    summary = {"name": name, "engine": engine, "exe": str(exe) if exe else None, "args": args, "env": env or {}, "meta": meta or {},
               "mem_avail_before_gib": m0["MemAvailable"],
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        if not srv.wait_ready():
            summary["error"] = "server did not become ready"
            return summary
        summary["load_s"] = time.time() - t_start
        time.sleep(3)
        summary["mem_avail_loaded_gib"] = meminfo()["MemAvailable"]
        sampler = Sampler(outdir / "samples.csv")
        reqs = []
        with open(outdir / "requests.jsonl", "w") as f:
            for rep in range(repeats):
                for pid in prompt_ids:
                    msgs, mt = P.messages(pid, nonce=f"{name}-{rep}")
                    if max_tokens and pid in P.SHORT:
                        mt = max_tokens
                    r = request(engine, msgs, mt)
                    r.update({"prompt_id": pid, "repeat": rep})
                    f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                    reqs.append(r)
                    if srv.killed_by_watchdog:
                        raise RuntimeError("watchdog")
        sampler.stop()
        summary["requests"] = len(reqs)
    except Exception as e:
        summary["error"] = repr(e)
    finally:
        summary["mem_avail_min_gib"] = srv.min_avail
        srv.stop()
        summary["watchdog"] = srv.killed_by_watchdog
        try:
            stats = parse_server_stats(engine, outdir)
            reqs_meta = [json.loads(l) for l in open(outdir / "requests.jsonl")] if (outdir / "requests.jsonl").exists() else []
            # a request that failed (no tokens, or broken off before the usage block) has no server line: the lines
            # belong to the requests that finished, in order (analysis/stats_pairing.py, shared with the analysis;
            # before 2026-11-22 a plain zip shifted every later request by one). When the pairing is undetermined the
            # rows are stored without labels and `server_stats_pairing` says so.
            summary["server_stats_pairing"] = SP.label_stats(engine, reqs_meta, stats)
            summary["server_stats"] = stats
        except Exception as e:
            summary["parse_error"] = repr(e)
        json.dump(summary, open(outdir / "run.json", "w"), indent=1, ensure_ascii=False)
    return summary
