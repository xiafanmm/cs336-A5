# CS336 Assignment 5: Alignment Writeup

> Working draft for `writeup.pdf`. Problems are organized in handout order.

---

## Problem `prompting_baselines` (5 pts): Run OLMo-2-0425-1B on GSM8K

### Setup

Evaluate `allenai/OLMo-2-0425-1B` on GSM8K using the three provided prompt templates:

| Prompt | Template file | Reward function | Stop string |
|---|---|---|---|
| `question_only` | `cs336_alignment/prompts/question_only.prompt` | `question_only_reward_fn` | None |
| `r1_zero` | `cs336_alignment/prompts/r1_zero.prompt` | `r1_zero_reward_fn` | `</answer>` |
| `r1_zero_three_shot` | `cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt` | `r1_zero_reward_fn` | `</answer>` |

Generation hyperparameters:

| Parameter | Value |
|---|---|
| Temperature | `1.0` |
| Top-p | `1.0` |
| Max generation length | `512` |

GSM8K ground-truth answers should be extracted from the dataset answer field by splitting on `####` and stripping whitespace.

### (a) Evaluation Metrics and Parsing Audit

Deliverable: a few sentences of commentary, the evaluation metrics, and a few examples of prompts and responses.

For each prompt, count model generations in the following categories:

1. Format reward = `1` and correctness reward = `1`.
2. Format reward = `1` and correctness reward = `0`.
3. Format reward = `0` and correctness reward = `0`.

| Prompt | Total examples | Category 1: format 1, correct 1 | Category 2: format 1, correct 0 | Category 3: format 0, correct 0 | Accuracy / total reward |
|---|---:|---:|---:|---:|---:|
| `question_only` | TODO | TODO | TODO | TODO | TODO |
| `r1_zero` | TODO | TODO | TODO | TODO | TODO |
| `r1_zero_three_shot` | TODO | TODO | TODO | TODO | TODO |

Manual audit of at least ten Category 2 examples:

| Prompt | Examples inspected | Actually correct but not parsed | Notes |
|---|---:|---:|---|
| `question_only` | TODO | TODO | TODO |
| `r1_zero` | TODO | TODO | TODO |
| `r1_zero_three_shot` | TODO | TODO | TODO |

Manual audit of at least ten Category 3 examples:

| Prompt | Examples inspected | Actually correct but not parsed | Notes |
|---|---:|---:|---|
| `question_only` | TODO | TODO | TODO |
| `r1_zero` | TODO | TODO | TODO |
| `r1_zero_three_shot` | TODO | TODO | TODO |

Commentary:

TODO: Summarize the main quantitative findings. Mention which prompt produced the highest correctness, which prompt most reliably followed the expected answer format, and whether parser failures were a major source of undercounted correct answers.

#### Supporting Examples

Example 1:

```text
Prompt template: TODO
Question: TODO
Ground truth: TODO
Model response:
TODO
Format reward: TODO
Correctness reward: TODO
Manual judgment: TODO
```

Example 2:

```text
Prompt template: TODO
Question: TODO
Ground truth: TODO
Model response:
TODO
Format reward: TODO
Correctness reward: TODO
Manual judgment: TODO
```

Example 3:

```text
Prompt template: TODO
Question: TODO
Ground truth: TODO
Model response:
TODO
Format reward: TODO
Correctness reward: TODO
Manual judgment: TODO
```

### (b) Prompt-Conditioned Model Behavior

Deliverable: a few sentences of commentary with supporting examples.

Points to address:

- Whether the model answers the question reliably when given only `question_only`, or whether it shows other base-model behaviors such as continuing text, generating extra problems, omitting the boxed answer, or drifting from the requested output style.
- How the zero-shot `r1_zero` prompt changes behavior, especially whether it encourages chain-of-thought reasoning and use of `<think>` / `<answer>` tags.
- How the few-shot `r1_zero_three_shot` prompt changes behavior relative to zero-shot `r1_zero`, especially whether the examples improve task framing, output format, and final-answer extraction.

Commentary:

TODO: Characterize behavior for each prompt. Use concrete examples from the evaluation outputs, not just aggregate metrics.

Supporting example for `question_only`:

```text
Question: TODO
Model response:
TODO
Observation: TODO
```

Supporting example for `r1_zero`:

```text
Question: TODO
Model response:
TODO
Observation: TODO
```

Supporting example for `r1_zero_three_shot`:

```text
Question: TODO
Model response:
TODO
Observation: TODO
```

---

<!-- Continue with Problem `baseline_calcs` below after finishing `prompting_baselines`. -->
