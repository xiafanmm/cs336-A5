from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn


PROMPT_CONFIGS = {
    "question_only": {
        "template": "question_only.prompt",
        "reward_fn": question_only_reward_fn,
        "stop": None,
        "assistant_prefix": "",
    },
    "r1_zero": {
        "template": "r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": "</answer>",
        "assistant_prefix": "<think>",
    },
    "r1_zero_three_shot": {
        "template": "r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": "</answer>",
        "assistant_prefix": "<think>",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OLMo-2-0425-1B on GSM8K with the assignment prompt templates."
    )
    parser.add_argument(
        "--model-path",
        default="allenai/OLMo-2-0425-1B",
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data/gsm8k/test.jsonl",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=PROJECT_ROOT / "cs336_alignment/prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/prompting_baselines",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=sorted(PROMPT_CONFIGS),
        default=list(PROMPT_CONFIGS),
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Passed to from_pretrained. Use "auto" for GPU placement.',
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--slow-grade",
        action="store_true",
        help="Use slower math_verify fallback in the reward functions.",
    )
    parser.add_argument(
        "--audit-examples",
        type=int,
        default=20,
        help="Number of category 2/3 examples per prompt to write for manual audit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Format prompts and write a preview without loading the model.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def read_prompt_templates(prompt_dir: Path) -> dict[str, str]:
    templates = {}
    for prompt_name, config in PROMPT_CONFIGS.items():
        templates[prompt_name] = (prompt_dir / config["template"]).read_text()
    return templates


def gsm8k_ground_truth(answer: str) -> str:
    if "####" not in answer:
        raise ValueError(f"GSM8K answer does not contain ####: {answer!r}")
    return answer.split("####")[-1].strip()


def category_name(format_reward: float, answer_reward: float) -> str:
    if format_reward == 1.0 and answer_reward == 1.0:
        return "format1_correct1"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "format1_correct0"
    if format_reward == 0.0 and answer_reward == 0.0:
        return "format0_correct0"
    return f"format{format_reward:g}_correct{answer_reward:g}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dtype_from_arg(dtype: str):
    if dtype == "auto":
        return "auto"
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "device_map": args.device_map,
        "torch_dtype": dtype_from_arg(args.dtype),
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    except TypeError as exc:
        if "torch_dtype" not in str(exc):
            raise
        load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)

    model.eval()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(next(model.parameters()).device),
            }
        ),
        flush=True,
    )
    return model, tokenizer


def contains_subsequence(values: list[int], subsequence: list[int]) -> bool:
    if not subsequence or len(values) < len(subsequence):
        return False
    last_start = len(values) - len(subsequence)
    return any(values[start : start + len(subsequence)] == subsequence for start in range(last_start + 1))


def make_stop_criteria(tokenizer, stop: str | None, input_width: int):
    if stop is None:
        return None

    from transformers import StoppingCriteria, StoppingCriteriaList

    stop_ids = tokenizer(stop, add_special_tokens=False).input_ids
    if not stop_ids:
        return None

    class StopAfterAllRowsHaveStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            generated = input_ids[:, input_width:].tolist()
            return all(contains_subsequence(row, stop_ids) for row in generated)

    return StoppingCriteriaList([StopAfterAllRowsHaveStop()])


def trim_at_stop(text: str, stop: str | None) -> str:
    if stop is None:
        return text
    index = text.find(stop)
    if index == -1:
        return text
    return text[: index + len(stop)]


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    stop: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_width = encoded["input_ids"].shape[1]
    device = next(model.parameters()).device
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    stopping_criteria = make_stop_criteria(tokenizer, stop, input_width)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    if stopping_criteria is not None:
        generation_kwargs["stopping_criteria"] = stopping_criteria

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    completions = []
    for row in output_ids:
        generated_ids = row[input_width:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(trim_at_stop(text, stop))
    return completions


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def evaluate_prompt(
    prompt_name: str,
    examples: list[dict[str, Any]],
    templates: dict[str, str],
    args: argparse.Namespace,
    model=None,
    tokenizer=None,
) -> list[dict[str, Any]]:
    config = PROMPT_CONFIGS[prompt_name]
    template = templates[prompt_name]
    reward_fn = config["reward_fn"]
    stop = config["stop"]
    assistant_prefix = config["assistant_prefix"]

    rows = []
    prompts = [template.format(question=example["question"]) for example in examples]
    if args.dry_run:
        completions = ["DRY_RUN_COMPLETION" for _ in prompts]
    else:
        completions = []
        for start, prompt_batch in batched(prompts, args.batch_size):
            print(f"{prompt_name}: generating {start + len(prompt_batch)}/{len(prompts)}", flush=True)
            completions.extend(
                generate_batch(
                    model,
                    tokenizer,
                    prompt_batch,
                    stop=stop,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            )

    for index, (example, prompt, completion) in enumerate(zip(examples, prompts, completions)):
        ground_truth = gsm8k_ground_truth(example["answer"])
        assistant_response = f"{assistant_prefix}{completion}" if assistant_prefix else completion
        scores = reward_fn(assistant_response, ground_truth, fast=not args.slow_grade)
        category = category_name(scores["format_reward"], scores["answer_reward"])
        rows.append(
            {
                "index": index,
                "prompt_name": prompt_name,
                "question": example["question"],
                "ground_truth": ground_truth,
                "prompt": prompt,
                "completion": completion,
                "assistant_response": assistant_response,
                "format_reward": scores["format_reward"],
                "answer_reward": scores["answer_reward"],
                "reward": scores["reward"],
                "category": category,
            }
        )
    return rows


def summarize(rows_by_prompt: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {}
    for prompt_name, rows in rows_by_prompt.items():
        counts = Counter(row["category"] for row in rows)
        total = len(rows)
        correct = counts["format1_correct1"]
        formatted = counts["format1_correct1"] + counts["format1_correct0"]
        summary[prompt_name] = {
            "total": total,
            "category_counts": {
                "format1_correct1": counts["format1_correct1"],
                "format1_correct0": counts["format1_correct0"],
                "format0_correct0": counts["format0_correct0"],
            },
            "accuracy": correct / total if total else 0.0,
            "format_rate": formatted / total if total else 0.0,
        }
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_code_block(text: str) -> str:
    return "```text\n" + text.rstrip() + "\n```"


def write_audit_markdown(
    path: Path,
    rows_by_prompt: dict[str, list[dict[str, Any]]],
    audit_examples: int,
) -> None:
    lines = [
        "# Prompting Baselines Audit Examples",
        "",
        "Inspect at least ten examples for category 2 and category 3 where available.",
        "Record how many are actually correct but not parsed or graded correctly.",
        "",
    ]
    for prompt_name, rows in rows_by_prompt.items():
        lines.extend([f"## {prompt_name}", ""])
        for category in ["format1_correct0", "format0_correct0"]:
            selected = [row for row in rows if row["category"] == category][:audit_examples]
            lines.extend([f"### {category} ({len(selected)} shown)", ""])
            for row in selected:
                lines.extend(
                    [
                        f"Example index: {row['index']}",
                        f"Ground truth: {row['ground_truth']}",
                        "Question:",
                        markdown_code_block(row["question"]),
                        "Model response:",
                        markdown_code_block(row["assistant_response"]),
                        "Manual judgment: TODO",
                        "",
                    ]
                )
    path.write_text("\n".join(lines))


def write_report_markdown(path: Path, summary: dict[str, Any], rows_by_prompt: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "# Prompting Baselines Report Draft",
        "",
        "## Metrics",
        "",
        "| Prompt | Total | Format 1 Correct 1 | Format 1 Correct 0 | Format 0 Correct 0 | Accuracy | Format rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for prompt_name, stats in summary.items():
        counts = stats["category_counts"]
        lines.append(
            "| {prompt} | {total} | {c1} | {c2} | {c3} | {acc:.4f} | {fmt:.4f} |".format(
                prompt=prompt_name,
                total=stats["total"],
                c1=counts["format1_correct1"],
                c2=counts["format1_correct0"],
                c3=counts["format0_correct0"],
                acc=stats["accuracy"],
                fmt=stats["format_rate"],
            )
        )
    lines.extend(["", "## Supporting Examples", ""])
    for prompt_name, rows in rows_by_prompt.items():
        lines.extend([f"### {prompt_name}", ""])
        for row in rows[:3]:
            lines.extend(
                [
                    f"Ground truth: {row['ground_truth']}",
                    f"Format reward: {row['format_reward']}",
                    f"Correctness reward: {row['answer_reward']}",
                    "Question:",
                    markdown_code_block(row["question"]),
                    "Model response:",
                    markdown_code_block(row["assistant_response"]),
                    "",
                ]
            )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = read_jsonl(args.data_path)
    if args.limit is not None:
        examples = examples[: args.limit]
    templates = read_prompt_templates(args.prompt_dir)

    model = tokenizer = None
    if not args.dry_run:
        model, tokenizer = load_model_and_tokenizer(args)

    rows_by_prompt = {}
    for prompt_name in args.prompts:
        rows = evaluate_prompt(prompt_name, examples, templates, args, model, tokenizer)
        rows_by_prompt[prompt_name] = rows
        write_jsonl(args.output_dir / f"{prompt_name}.jsonl", rows)

    summary = summarize(rows_by_prompt)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_audit_markdown(args.output_dir / "audit_examples.md", rows_by_prompt, args.audit_examples)
    write_report_markdown(args.output_dir / "report_draft.md", summary, rows_by_prompt)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
