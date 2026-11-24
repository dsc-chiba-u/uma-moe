"""Download one model GGUF into ~/models/locality/<key>/ (run under the shared GPU lock: large download).

    flock ~/work/uma-moe/.gpu.lock .venv/bin/python -I locality/download.py <key>
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODELS  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

key = sys.argv[1]
m = MODELS[key]
if "repo" not in m:
    sys.exit(f"{key}: local model, nothing to download")
dest = Path.home() / "models/locality" / key
dest.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for f in m["files"]:
    p = hf_hub_download(m["repo"], f, local_dir=dest)
    print(f"{key}: {f} -> {p} ({Path(p).stat().st_size / 1e9:.2f} GB)", flush=True)
print(f"{key}: done in {time.time() - t0:.0f} s", flush=True)
