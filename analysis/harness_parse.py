"""The llama-server log patterns, shared with bench/harness.py without importing the harness (which reads prompts)."""
import re
LLAMA_PROMPT = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
LLAMA_EVAL = re.compile(r"\s eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
LLAMA_DRAFT = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)(?:, mean len =\s*([\d.]+))?")
