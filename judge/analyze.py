"""How closely each judge's scores track the human scores, with 95% bootstrap intervals.

    python -m judge.analyze

Reads results/raw/*.jsonl plus the large-judge scores already in the dataset, and writes
results/summary.json and results/tables.md.

Metrics, per judge, on the answers it returned a parseable score for:
  pearson / spearman  correlation with the human score
  qwk                 quadratic weighted Cohen's kappa (scores rounded to 1-5)
  exact / within1     share of answers scored exactly the same as the human, or off by at most 1
  bias                mean (judge - human): above 0 means the judge is more lenient
  length_bias         Spearman correlation of answer length with (judge - human): above 0 means longer answers get extra credit
"Calibrated" rows map each score a judge gives to the average human score for that judge score,
learned with 5-fold cross-validation over prompts, so no prompt helps calibrate its own grades.
Intervals resample whole prompts (all 4 answers together), 2,000 times.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from judge import data, prompts
from judge.run import JUDGES, RAW

ROOT = data.ROOT
BOOT = 2000
rng = np.random.default_rng(0)


def qwk(a: np.ndarray, b: np.ndarray) -> float:
    a = np.clip(np.rint(a), 1, 5).astype(int) - 1
    b = np.clip(np.rint(b), 1, 5).astype(int) - 1
    observed = np.zeros((5, 5))
    np.add.at(observed, (a, b), 1)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    weights = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2) / 16
    return float(1 - (weights * observed).sum() / (weights * expected).sum())


def metrics(judge: np.ndarray, human: np.ndarray, length: np.ndarray) -> dict[str, float]:
    error = judge - human
    rounded = np.clip(np.rint(judge), 1, 5)
    return {
        "pearson": float(pearsonr(judge, human)[0]),
        "spearman": float(spearmanr(judge, human)[0]),
        "qwk": qwk(judge, human),
        "exact": float((rounded == human).mean()),
        "within1": float((np.abs(rounded - human) <= 1).mean()),
        "bias": float(error.mean()),
        "length_bias": float(spearmanr(length, error)[0]),
    }


def evaluate(frame: pd.DataFrame, column: str) -> dict:
    """Point estimates and prompt-level bootstrap intervals for one judge column."""
    rows = frame[frame[column].notna()]
    point = metrics(rows[column].to_numpy(float), rows.human_score.to_numpy(float), rows.length.to_numpy(float))
    groups = [g.index.to_numpy() for _, g in rows.groupby("id")]
    draws = {k: [] for k in point}
    for _ in range(BOOT):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sub = rows.loc[pick]
        for k, v in metrics(
            sub[column].to_numpy(float), sub.human_score.to_numpy(float), sub.length.to_numpy(float)
        ).items():
            draws[k].append(v)
    return {
        "n": len(rows),
        "missing": int(frame[column].isna().sum()),
        **{k: {"value": v, "ci": [float(np.nanpercentile(draws[k], 2.5)), float(np.nanpercentile(draws[k], 97.5))]}
           for k, v in point.items()},
    }  # fmt: skip


def paired_difference(frame: pd.DataFrame, a: str, b: str, metric: str = "pearson") -> dict:
    """Bootstrap interval for metric(a) - metric(b) on the answers both judges scored."""
    rows = frame[frame[a].notna() & frame[b].notna()]
    groups = [g.index.to_numpy() for _, g in rows.groupby("id")]

    def diff(sub: pd.DataFrame) -> float:
        h, n = sub.human_score.to_numpy(float), sub.length.to_numpy(float)
        return metrics(sub[a].to_numpy(float), h, n)[metric] - metrics(sub[b].to_numpy(float), h, n)[metric]

    draws = [diff(rows.loc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])]) for _ in range(BOOT)]
    return {"a": a, "b": b, "metric": metric, "value": diff(rows), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}  # fmt: skip


def load() -> tuple[pd.DataFrame, dict[str, str], dict[str, dict]]:
    """The sample with one score column per judge run, their display names, and per-run stats."""
    frame = data.sample().copy()
    frame["length"] = frame.response.str.split().str.len()
    names = dict(data.REFERENCE_JUDGES)
    runs = {}
    for path in sorted(RAW.glob("*.jsonl")):
        judge, condition = path.stem.split("__")
        lines = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
        if len(lines) < len(frame):
            print(f"skipping {path.name}: {len(lines)}/{len(frame)} answers graded")
            continue
        column = f"{judge}__{condition}"
        scores = {r["uuid"]: r["score"] for r in lines}
        frame[column] = frame.uuid.map(scores).astype(float)
        names[column] = f"{JUDGES[judge][1]} ({condition.replace('_', ' ')})"
        runs[column] = {
            "seconds_per_answer": float(np.mean([r["seconds"] for r in lines])),
            "parse_failures": int(sum(r["score"] is None for r in lines)),
        }
    return frame, names, runs


def calibrate(frame: pd.DataFrame, column: str, folds: int = 5) -> pd.Series:
    """Map each judge score to the mean human score for that judge score, learned on the other folds of prompts.

    Fixes a judge that is consistently harsh or lenient without ever seeing the prompt it is applied to.
    """
    ids = frame.id.drop_duplicates().sample(frac=1, random_state=0).to_numpy()
    fold_of = {pid: i % folds for i, pid in enumerate(ids)}
    fold = frame.id.map(fold_of)
    out = pd.Series(np.nan, index=frame.index)
    for k in range(folds):
        train = frame[(fold != k) & frame[column].notna()]
        level = np.clip(np.rint(train[column]), 1, 5)
        table = train.human_score.groupby(level).mean()
        test = (fold == k) & frame[column].notna()
        out[test] = np.clip(np.rint(frame.loc[test, column]), 1, 5).map(table).fillna(frame.loc[test, column])
    return out


def main() -> None:
    frame, names, runs = load()
    full = [c for c in names if c.endswith("__full")]
    # The small judges' average, on answers where all of them returned a score.
    if len(full) > 1:
        frame["ensemble__full"] = frame[full].mean(axis=1, skipna=False)
        names["ensemble__full"] = f"Average of {len(full)} small judges (full)"

    for c in [c for c in names if c.endswith("__full")]:
        frame[f"{c}__calibrated"] = calibrate(frame, c)
        names[f"{c}__calibrated"] = names[c].replace("(full)", "(full, calibrated)")

    results = {c: {"name": names[c], **evaluate(frame, c), **runs.get(c, {})} for c in names}
    by_capability = {
        c: {cap: float(pearsonr(g[c][g[c].notna()], g.human_score[g[c].notna()])[0]) for cap, g in frame.groupby("capability")}
        for c in names
    }  # fmt: skip

    comparisons = []
    best_small = max(full, key=lambda c: results[c]["pearson"]["value"]) if full else None
    if best_small:
        for ref in data.REFERENCE_JUDGES:
            comparisons.append(paired_difference(frame, best_small, ref))
        if "ensemble__full" in frame:
            comparisons.append(paired_difference(frame, "ensemble__full", "gpt4_04_turbo_score"))
        judge = best_small.split("__")[0]
        for condition in prompts.CONDITIONS[1:]:
            other = f"{judge}__{condition}"
            if other in frame:
                comparisons.append(paired_difference(frame, other, best_small))

    out = {
        "sample": {"answers": len(frame), "prompts": int(frame.id.nunique()), "human_mean": float(frame.human_score.mean())},
        "judges": results,
        "by_capability": by_capability,
        "comparisons": comparisons,
    }  # fmt: skip
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf8")

    def cell(m: dict, digits: int = 2) -> str:
        return f"{m['value']:.{digits}f} [{m['ci'][0]:.{digits}f}, {m['ci'][1]:.{digits}f}]"

    lines = [
        f"Sample: {len(frame)} answers to {frame.id.nunique()} prompts. 95% intervals from a prompt-level bootstrap.",
        "",
        "| Judge | Pearson | QWK | Exact | Within 1 | Bias | Length bias | Missing | s/answer |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c, r in sorted(results.items(), key=lambda kv: -kv[1]["pearson"]["value"]):
        speed = f"{r['seconds_per_answer']:.1f}" if "seconds_per_answer" in r else ""
        lines.append(
            f"| {r['name']} | {cell(r['pearson'])} | {cell(r['qwk'])} | {r['exact']['value']:.0%} | "
            f"{r['within1']['value']:.0%} | {r['bias']['value']:+.2f} | {r['length_bias']['value']:+.2f} | {r['missing']} | {speed} |"
        )
    lines += ["", "Paired differences in Pearson r (first minus second):", ""]
    for d in comparisons:
        lines.append(f"- {names[d['a']]} - {names[d['b']]}: {d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    lines += ["", "Pearson r by capability:", "", "| Judge | " + " | ".join(sorted(frame.capability.unique())) + " |"]
    lines.append("|---" * (frame.capability.nunique() + 1) + "|")
    for c, caps in by_capability.items():
        lines.append(f"| {names[c]} | " + " | ".join(f"{caps[k]:.2f}" for k in sorted(caps)) + " |")
    (ROOT / "results" / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
