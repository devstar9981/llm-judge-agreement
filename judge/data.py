"""BiGGen-Bench human-eval data: download it and draw the balanced sample the judges grade.

The split has 695 prompts, each answered by 4 models (Llama-2-13B, Mistral-7B-Instruct,
Mixtral-8x7B-Instruct, GPT-3.5-Turbo), and every answer was scored 1-5 by a person against
a rubric written for that prompt. The same rows also carry the scores GPT-4, GPT-4 Turbo,
Claude 3 Opus and Prometheus 2 gave as judges, which we use as the large-model reference.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "human_eval.parquet"
SAMPLE = ROOT / "data" / "sample.parquet"
URL = (
    "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results/resolve/main/"
    "data/human_eval-00000-of-00001.parquet"
)

# Prompts drawn from each of the 8 capabilities; with 4 answers each that is 8 x 10 x 4 = 320 rows.
PER_CAPABILITY = 10
SEED = 7

# Scores the large judges gave, already in the dataset (names from the dataset card).
REFERENCE_JUDGES = {
    "gpt4_score": "GPT-4 (1106)",
    "gpt4_04_turbo_score": "GPT-4 Turbo (2024-04-09)",
    "claude_score": "Claude 3 Opus",
    "prometheus_score": "Prometheus 2 8x7B",
}


def download() -> Path:
    if RAW.exists():
        return RAW
    import requests

    RAW.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = RAW.with_suffix(".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(RAW)
    return RAW


def load_all() -> pd.DataFrame:
    df = pd.read_parquet(download())
    # Prometheus was sampled several times per answer; average its scores (a few rows have none).
    df["prometheus_score"] = df.prometheus_8x7b_score.map(
        lambda s: float(sum(s)) / len(s) if s is not None and len(s) else float("nan")
    )
    # Four rows have no human score (-1).
    return df[df.human_score > 0].reset_index(drop=True)


def sample() -> pd.DataFrame:
    """The same PER_CAPABILITY prompts per capability every time, keeping only prompts with all 4 answers scored."""
    if SAMPLE.exists():
        return pd.read_parquet(SAMPLE)
    df = load_all()
    complete = df.groupby("id").filter(lambda g: len(g) == 4)
    ids = (
        complete.drop_duplicates("id")
        .groupby("capability", group_keys=False)
        .apply(lambda g: g.sample(PER_CAPABILITY, random_state=SEED))
        .id
    )
    keep = [
        "id", "capability", "task", "system_prompt", "input", "reference_answer", "score_rubric",
        "response", "uuid", "model_name", "human_score", *REFERENCE_JUDGES,
    ]  # fmt: skip
    out = complete[complete.id.isin(set(ids))][keep].sort_values(["capability", "id", "model_name"])
    out = out.reset_index(drop=True)
    out.to_parquet(SAMPLE)
    return out


if __name__ == "__main__":
    s = sample()
    print(len(s), "rows,", s.id.nunique(), "prompts")
    print(s.capability.value_counts().to_string())
    print(s.human_score.value_counts().sort_index().to_string())
