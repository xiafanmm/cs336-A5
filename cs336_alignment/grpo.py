import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch.nn.functional as F
import os
from typing import Any, Callable, Literal


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    tokenized = []
    prompt_lens = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        output_ids = tokenizer(output, add_special_tokens=False)['input_ids']
        prompt_lens.append(len(prompt_ids))
        tokenized.append(prompt_ids + output_ids)
    max_len = max(len(ids) for ids in tokenized)
    pad_id = tokenizer.pad_token_id

    padded = []
    response_mask = []

    for ids, prompt_len in zip(tokenized, prompt_lens):
        seq_len = len(ids)
        padded_ids = ids + [pad_id] * (max_len - seq_len)
        padded.append(padded_ids)
        mask = [0] * (prompt_len - 1) + [1] * (seq_len - prompt_len) + [0] * (max_len - seq_len)
        response_mask.append(mask)

    tokens = torch.tensor(padded, dtype = torch.long)

    return {
        'input_ids' : tokens[:, :-1],
        'labels' : tokens[:, 1:],
        'response_mask' : torch.tensor(response_mask, dtype=torch.bool)
    }



def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    outputs = model(input_ids)
    logits = outputs.logits

    log_distribution = F.log_softmax(logits, dim = -1)
    
    label_log_probs = torch.gather(
        log_distribution,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {
        "log_probs" : label_log_probs, 
    }

    if return_token_entropy:
        probabilities = F.softmax(logits, dim = -1)
        entropy = -torch.sum(probabilities * log_distribution, dim = -1)
        result["token_entropy"] = entropy
    return result

from typing import Callable
import torch

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    assert len(rollout_responses) == len(repeated_ground_truths)

    reward_dicts = [
        reward_fn(response, gt)
        for response, gt in zip(rollout_responses, repeated_ground_truths)
    ]

    raw_rewards = torch.tensor(
        [r["reward"] for r in reward_dicts],
        dtype=torch.float32,
    )

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "format_reward_mean": sum(r["format_reward"] for r in reward_dicts) / len(reward_dicts),
        "answer_reward_mean": sum(r["answer_reward"] for r in reward_dicts) / len(reward_dicts),
    }

    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    assert raw_rewards.numel() % group_size == 0

    grouped_rewards = raw_rewards.reshape(-1, group_size)
    metadata: dict[str, float] = {}

    # 1. baseline
    if baseline == "mean":
        group_baseline = grouped_rewards.mean(dim=-1, keepdim=True)
        advantages = grouped_rewards - group_baseline
    elif baseline == "none":
        advantages = grouped_rewards
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    # 2. normalizer
    if advantage_normalizer == "std":
        normalizer = advantages.std(dim=-1, keepdim=True)
        advantages = advantages / (normalizer + advantage_eps)

    elif advantage_normalizer == "mean":
        # 注意：如果 baseline == "mean"，这里不应该对 advantages 求 mean
        # 因为每组 advantages 的均值约等于 0
        normalizer = grouped_rewards.mean(dim=-1, keepdim=True).abs()
        advantages = advantages / (normalizer + advantage_eps)

    elif advantage_normalizer == "none":
        pass

    else:
        raise ValueError(f"Unknown advantage_normalizer: {advantage_normalizer}")

    # 3. metadata
    metadata["reward_mean"] = raw_rewards.mean().item()
    metadata["reward_std"] = raw_rewards.std().item()
    metadata["advantage_mean"] = advantages.mean().item()
    metadata["advantage_std"] = advantages.std().item()

    return advantages.flatten(), metadata


def compute_policy_gradient_loss_on_policy(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != 'none':
        raise ValueError("on_policy do not need reweighting method")
    
    per_token_loss = - raw_rewards_or_advantages.reshape(-1, 1) * policy_log_probs
    if response_mask:
        per_token_loss = per_token_loss * response_mask

    metadata : dict[str, torch.Tensor] = {}
    return per_token_loss, metadata