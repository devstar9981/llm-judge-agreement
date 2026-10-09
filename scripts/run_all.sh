#!/usr/bin/env bash
# Every judge on the main condition first, then the three prompt variations. Resumable.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
JUDGES="qwen3.5-2b qwen3.5-4b ministral-3-3b gemma-4-e2b"
for condition in full score_only criteria_only no_reference; do
  for judge in $JUDGES; do
    $PY -m judge.run --judge "$judge" --condition "$condition"
  done
done
