import argparse
import json
import os
import random
import re
from dataclasses import dataclass

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.grpo import grpo_train_step


DEFAULT_MODEL_NAME = "/mnt/workspace/models/allenai/OLMo-2-0425-1B"
DEFAULT_TRAIN_FILE = "data/gsm8k/train.jsonl"

ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def extract_gsm8k_answer(text: str) -> str | None:
    """
    GSM8K 的标准答案里通常有：
    #### 42
    """
    match = ANSWER_RE.search(text)
    if match is None:
        return None
    return match.group(1).replace(",", "").strip()


def extract_model_answer(text: str) -> str | None:
    """
    从模型输出里取最后一个数字，作为答案。
    这是最简单版本，够你先跑通实验。
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not numbers:
        return None
    return numbers[-1].strip()


def gsm8k_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    pred = extract_model_answer(response)
    gt = ground_truth.strip()

    answer_reward = 1.0 if pred == gt else 0.0

    # 简单 format reward：只要能抽出数字，就给格式分
    format_reward = 1.0 if pred is not None else 0.0

    return {
        "reward": answer_reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem. "
        "Give your reasoning, and end with the final numeric answer.\n\n"
        f"Problem: {question}\n\n"
        "Solution:"
    )


def load_gsm8k_examples(path: str) -> list[dict[str, str]]:
    train_examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            question = ex["question"]
            answer = extract_gsm8k_answer(ex["answer"])

            if answer is None:
                continue

            train_examples.append(
                {
                    "prompt": build_prompt(question),
                    "ground_truth": answer,
                }
            )

    return train_examples


@torch.no_grad()
def generate_rollouts_hf(
    model,
    tokenizer,
    prompts: list[str],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    """
    输入原始 prompts，每个 prompt 生成 group_size 个 responses。

    返回：
    repeated_prompts: 每个 prompt 重复 group_size 次
    rollout_responses: 对应生成的 response
    """
    model.eval()

    repeated_prompts: list[str] = []
    rollout_responses: list[str] = []

    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]

        for output_ids in outputs:
            response_ids = output_ids[prompt_len:]
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            )

            repeated_prompts.append(prompt)
            rollout_responses.append(response)

    model.train()
    return repeated_prompts, rollout_responses


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    output_dir: str = "outputs/grpo_olmo_gsm8k"
    train_file: str = DEFAULT_TRAIN_FILE
    local_files_only: bool = True

    train_batch_prompts: int = 4
    group_size: int = 4
    gradient_accumulation_steps: int = 2

    max_steps: int = 100
    lr: float = 1e-6
    max_grad_norm: float = 1.0

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0

    save_every: int = 50
    seed: int = 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="outputs/grpo_olmo_gsm8k")
    parser.add_argument("--train_file", type=str, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--train_batch_prompts", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main():
    cfg = parse_args()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        local_files_only=cfg.local_files_only,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=cfg.local_files_only,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
    )

    train_examples = load_gsm8k_examples(cfg.train_file)
    if not train_examples:
        raise ValueError(f"No usable GSM8K examples found in {cfg.train_file}")

    print(f"Loaded {len(train_examples)} GSM8K training examples from {cfg.train_file}.")

    pbar = tqdm(range(cfg.max_steps))

    for step in pbar:
        batch = random.sample(train_examples, cfg.train_batch_prompts)

        prompts = [ex["prompt"] for ex in batch]
        ground_truths = [ex["ground_truth"] for ex in batch]

        repeated_prompts, rollout_responses = generate_rollouts_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            group_size=cfg.group_size,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            device=device,
        )

        repeated_ground_truths = []
        for gt in ground_truths:
            repeated_ground_truths.extend([gt] * cfg.group_size)

        loss, metadata = grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_grad_norm=cfg.max_grad_norm,
            reward_fn=gsm8k_reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=cfg.group_size,

            # standard on-policy GRPO
            baseline="mean",
            advantage_eps=1e-6,
            advantage_normalizer="std",
            importance_reweighting_method="none",
            old_log_probs=None,
            cliprange=None,
            loss_normalization="sequence",
            normalization_constant=None,
        )

        reward_mean = metadata.get("reward_mean", 0.0)
        answer_reward_mean = metadata.get("answer_reward_mean", 0.0)
        grad_norm = metadata.get("grad_norm", 0.0)

        pbar.set_description(
            f"step={step} "
            f"loss={loss.item():.4f} "
            f"reward={reward_mean:.3f} "
            f"ans={answer_reward_mean:.3f} "
            f"grad={grad_norm:.3f}"
        )

        if (step + 1) % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"step_{step + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)

            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

            print(f"\nSaved checkpoint to {ckpt_dir}")

    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final model to {final_dir}")


if __name__ == "__main__":
    main()
