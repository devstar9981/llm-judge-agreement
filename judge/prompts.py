"""Judge prompts for each experimental condition.

Everything that is the same for the 4 answers to a prompt (task, rubric, reference) comes
first and the answer comes last, so llama.cpp reuses the cached prefix for answers 2-4.

Conditions:
  full           rubric with all five level descriptions, plus the reference answer; short feedback, then the score
  no_reference   as full, without the reference answer
  criteria_only  only the rubric's one-line question, with a generic 1-5 scale; reference included
  score_only     as full, but the judge writes the score alone, with no feedback first
"""

from __future__ import annotations

CONDITIONS = ("full", "no_reference", "criteria_only", "score_only")

SYSTEM = (
    "You are a careful, strict grader. You grade one response against the rubric you are given. "
    "Judge only what the rubric asks for. Do not reward length or a confident tone."
)

GENERIC_SCALE = (
    "1: Fails the criterion.\n"
    "2: Mostly fails the criterion, with minor merit.\n"
    "3: Partly meets the criterion.\n"
    "4: Mostly meets the criterion, with minor gaps.\n"
    "5: Fully meets the criterion."
)


def _task(row) -> str:
    system = (row["system_prompt"] or "").strip()
    task = row["input"].strip()
    return f"System prompt given to the model:\n{system}\n\nUser request:\n{task}" if system else task


def _rubric(row, condition: str) -> str:
    rubric = row["score_rubric"]
    if condition == "criteria_only":
        return f"{rubric['criteria']}\n{GENERIC_SCALE}"
    levels = "\n".join(f"{n}: {rubric[f'score{n}_description']}" for n in range(1, 6))
    return f"{rubric['criteria']}\n{levels}"


def messages(row, condition: str) -> list[dict[str, str]]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    parts = [
        "### Task given to the model",
        _task(row),
        "### Rubric",
        _rubric(row, condition),
    ]
    if condition != "no_reference":
        parts += ["### Reference answer (scores a 5)", row["reference_answer"].strip()]
    parts += ["### Response to grade", row["response"].strip(), "### Your output"]
    if condition == "score_only":
        parts.append("Reply with exactly one line, `[RESULT] N`, where N is an integer from 1 to 5. Write nothing else.")
    else:
        parts.append(
            "First write feedback of at most 80 words that checks the response against the rubric. "
            "Then end with one line, `[RESULT] N`, where N is an integer from 1 to 5."
        )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def max_tokens(condition: str) -> int:
    return 12 if condition == "score_only" else 320
