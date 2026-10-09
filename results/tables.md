Sample: 320 answers to 80 prompts. 95% intervals from a prompt-level bootstrap.

| Judge | Pearson | QWK | Exact | Within 1 | Bias | Length bias | Missing | s/answer |
|---|---|---|---|---|---|---|---|---|
| GPT-4 Turbo (2024-04-09) | 0.61 [0.50, 0.69] | 0.59 [0.48, 0.67] | 39% | 83% | -0.05 | -0.11 | 0 |  |
| Qwen3.5 4B (full, calibrated) | 0.57 [0.46, 0.67] | 0.50 [0.39, 0.59] | 29% | 86% | +0.00 | -0.08 | 0 |  |
| Qwen3.5 4B (full) | 0.57 [0.47, 0.66] | 0.55 [0.44, 0.64] | 40% | 72% | -0.41 | +0.02 | 0 | 10.2 |
| GPT-4 (1106) | 0.56 [0.46, 0.66] | 0.55 [0.44, 0.65] | 38% | 79% | -0.12 | -0.14 | 0 |  |
| Average of 2 small judges (full, calibrated) | 0.56 [0.45, 0.66] | 0.49 [0.39, 0.59] | 30% | 81% | +0.00 | -0.09 | 3 |  |
| Prometheus 2 8x7B | 0.56 [0.47, 0.65] | 0.45 [0.35, 0.55] | 34% | 82% | +0.18 | -0.15 | 8 |  |
| Average of 2 small judges (full) | 0.56 [0.44, 0.66] | 0.52 [0.41, 0.62] | 37% | 71% | -0.57 | -0.00 | 3 |  |
| Claude 3 Opus | 0.55 [0.44, 0.65] | 0.52 [0.41, 0.62] | 34% | 81% | -0.02 | -0.07 | 0 |  |
| Qwen3.5 2B (full) | 0.43 [0.30, 0.55] | 0.38 [0.26, 0.50] | 38% | 62% | -0.73 | -0.02 | 3 | 4.1 |
| Qwen3.5 2B (full, calibrated) | 0.40 [0.27, 0.51] | 0.24 [0.16, 0.31] | 18% | 75% | -0.00 | -0.07 | 3 |  |

Paired differences in Pearson r (first minus second):

- Qwen3.5 4B (full) - GPT-4 (1106): +0.008 [-0.065, +0.087]
- Qwen3.5 4B (full) - GPT-4 Turbo (2024-04-09): -0.034 [-0.102, +0.033]
- Qwen3.5 4B (full) - Claude 3 Opus: +0.022 [-0.071, +0.122]
- Qwen3.5 4B (full) - Prometheus 2 8x7B: +0.016 [-0.078, +0.107]
- Average of 2 small judges (full) - GPT-4 Turbo (2024-04-09): -0.052 [-0.129, +0.023]

Pearson r by capability:

| Judge | grounding | instruction_following | planning | reasoning | refinement | safety | theory_of_mind | tool_usage |
|---|---|---|---|---|---|---|---|---|
| GPT-4 (1106) | 0.45 | 0.62 | 0.73 | 0.63 | 0.42 | 0.72 | 0.18 | 0.57 |
| GPT-4 Turbo (2024-04-09) | 0.47 | 0.58 | 0.69 | 0.72 | 0.53 | 0.71 | 0.34 | 0.60 |
| Claude 3 Opus | 0.60 | 0.46 | 0.74 | 0.59 | 0.08 | 0.66 | 0.43 | 0.67 |
| Prometheus 2 8x7B | 0.63 | 0.44 | 0.73 | 0.52 | 0.32 | 0.65 | 0.45 | 0.70 |
| Qwen3.5 2B (full) | 0.61 | 0.19 | 0.31 | 0.50 | 0.37 | 0.55 | 0.31 | 0.47 |
| Qwen3.5 4B (full) | 0.61 | 0.34 | 0.53 | 0.55 | 0.50 | 0.71 | 0.15 | 0.64 |
| Average of 2 small judges (full) | 0.68 | 0.28 | 0.45 | 0.59 | 0.51 | 0.68 | 0.26 | 0.66 |
| Qwen3.5 2B (full, calibrated) | 0.58 | 0.16 | 0.28 | 0.44 | 0.39 | 0.51 | 0.30 | 0.40 |
| Qwen3.5 4B (full, calibrated) | 0.57 | 0.28 | 0.55 | 0.53 | 0.50 | 0.70 | 0.13 | 0.66 |
| Average of 2 small judges (full, calibrated) | 0.68 | 0.21 | 0.48 | 0.56 | 0.51 | 0.67 | 0.22 | 0.68 |
