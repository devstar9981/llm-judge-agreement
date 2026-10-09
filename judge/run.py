"""Grade the sample with a local judge model through llama.cpp's llama-server.

    python -m judge.run --judge qwen3.5-4b --condition full

Starts llama-server for the judge, sends each answer at temperature 0, and appends one JSON
line per answer to results/raw/<judge>__<condition>.jsonl. Answers already in that file are
skipped, so an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import requests

from judge import data, prompts

ROOT = data.ROOT
# llama.cpp's Windows CUDA 12.4 build (b11514); the Vulkan build is about half as fast on a GTX 1050 Ti.
SERVER = ROOT / "tools" / os.environ.get("LLAMA_DIR", "llama-cuda") / "llama-server.exe"
RAW = Path(os.environ.get("JUDGE_RAW_DIR", ROOT / "results" / "raw"))
PORT = 8091
# Talks only to the local server, so ignore HTTP(S)_PROXY from the environment.
http = requests.Session()
http.trust_env = False

JUDGES = {
    "qwen3.5-4b": ("Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5 4B"),
    "qwen3.5-2b": ("Qwen3.5-2B-Q4_K_M.gguf", "Qwen3.5 2B"),
    "gemma-4-e2b": ("gemma-4-E2B-it-Q4_K_M.gguf", "Gemma 4 E2B"),
    "ministral-3-3b": ("Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", "Ministral 3 3B"),
}

RESULT = re.compile(r"\[RESULT\]\s*\**\s*([1-5])")
FALLBACK = re.compile(r"(?:score|result)\s*[:=]?\s*\**\s*([1-5])\b", re.IGNORECASE)


def parse_score(text: str) -> int | None:
    """The score after the last `[RESULT]`, else the last 'Score: N'; None if neither is there."""
    for pattern in (RESULT, FALLBACK):
        found = pattern.findall(text)
        if found:
            return int(found[-1])
    return None


def start_server(judge: str, gpu_layers: int, ctx: int) -> subprocess.Popen:
    model = ROOT / "models" / JUDGES[judge][0]
    log = (ROOT / "results" / "logs" / f"server-{judge}.log").open("ab")
    proc = subprocess.Popen(
        [
            str(SERVER), "-m", str(model), "--port", str(PORT), "-c", str(ctx), "-ngl", str(gpu_layers),
            "-np", "1", "--jinja", "--no-webui", "--temp", "0", "--seed", "0",
            # Checkpoint the context every 256 tokens. Qwen3.5 (recurrent layers) and Gemma 4 (sliding
            # window) can only resume from a checkpoint, and the default spacing of 8,192 means the
            # 4 answers to a prompt would each reprocess the shared rubric and reference.
            "-ub", "256", "-cms", "0", "-ctxcp", "64",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )  # fmt: skip
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}; see {log.name}")
        try:
            if http.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError("llama-server did not become ready")


def grade(row, condition: str) -> dict:
    body = {
        "messages": prompts.messages(row, condition),
        "temperature": 0,
        "max_tokens": prompts.max_tokens(condition),
        "cache_prompt": True,
        # Qwen3.5 and Gemma 4 can think before answering; the judges here answer directly.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    r = http.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, timeout=900)
    r.raise_for_status()
    out = r.json()
    text = out["choices"][0]["message"]["content"] or ""
    timings = out.get("timings", {})
    return {
        "uuid": row["uuid"],
        "score": parse_score(text),
        "output": text,
        "seconds": round(time.perf_counter() - started, 2),
        "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
        "cached_tokens": timings.get("cache_n"),
        "completion_tokens": out.get("usage", {}).get("completion_tokens"),
        "prompt_ms": timings.get("prompt_ms"),
        "generation_ms": timings.get("predicted_ms"),
    }


def run(judge: str, condition: str, gpu_layers: int, ctx: int, limit: int | None) -> Path:
    sample = data.sample()
    if limit:
        sample = sample.head(limit)
    RAW.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    path = RAW / f"{judge}__{condition}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["uuid"] for line in path.read_text(encoding="utf8").splitlines() if line}
    todo = sample[~sample.uuid.isin(done)]
    print(f"{judge} / {condition}: {len(done)} done, {len(todo)} to go", flush=True)
    if todo.empty:
        return path

    proc = start_server(judge, gpu_layers, ctx)
    # On a shared 4 GB card, other apps can push part of the model out to system memory, and
    # everything runs ~5x slower until the server reloads. Track throughput against the first
    # answers of this server and restart it after 3 slow answers in a row.
    baseline: dict[str, list[float]] = {"prompt": [], "generation": []}
    slow = 0
    try:
        with path.open("a", encoding="utf8") as f:
            for i, (_, row) in enumerate(todo.iterrows(), 1):
                result = grade(row, condition)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  last {result['seconds']}s  score {result['score']}", flush=True)
                kind, speed = throughput(result)
                if kind is None:
                    continue
                if len(baseline[kind]) < 8:
                    baseline[kind].append(speed)
                    continue
                slow = slow + 1 if speed < 0.5 * sorted(baseline[kind])[len(baseline[kind]) // 2] else 0
                if slow >= 3:
                    print(f"  slowed down at {i}/{len(todo)}; restarting llama-server", flush=True)
                    proc.terminate()
                    proc.wait(timeout=30)
                    proc = start_server(judge, gpu_layers, ctx)
                    baseline, slow = {"prompt": [], "generation": []}, 0
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    return path


def throughput(result: dict) -> tuple[str | None, float]:
    """Generation tokens/s when the judge wrote enough, else prompt tokens/s when enough was new."""
    if (result["completion_tokens"] or 0) >= 20 and result["generation_ms"]:
        return "generation", result["completion_tokens"] / result["generation_ms"] * 1000
    fresh = (result["prompt_tokens"] or 0) - (result["cached_tokens"] or 0)
    if fresh >= 300 and result["prompt_ms"]:
        return "prompt", fresh / result["prompt_ms"] * 1000
    return None, 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=sorted(JUDGES), required=True)
    parser.add_argument("--condition", choices=prompts.CONDITIONS, default="full")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--limit", type=int, help="grade only the first N answers (for a quick check)")
    args = parser.parse_args()
    run(args.judge, args.condition, args.gpu_layers, args.ctx, args.limit)


if __name__ == "__main__":
    main()
