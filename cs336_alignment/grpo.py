import math
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(eps)


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)

    tokenized: list[list[int]] = []
    prompt_lens: list[int] = []
    output_lens: list[int] = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]

        prompt_lens.append(len(prompt_ids))
        output_lens.append(len(output_ids))
        tokenized.append(prompt_ids + output_ids)

    max_len = max(len(ids) for ids in tokenized)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id and tokenizer.eos_token_id are both None.")

    padded: list[list[int]] = []
    response_masks: list[list[bool]] = []

    for ids, prompt_len, output_len in zip(tokenized, prompt_lens, output_lens):
        seq_len = len(ids)

        padded_ids = ids + [pad_id] * (max_len - seq_len)
        padded.append(padded_ids)

        # response_mask 要和 labels = tokens[:, 1:] 对齐
        # labels 的第 prompt_len - 1 个位置开始，对应第一个 response token
        mask = (
            [False] * (prompt_len - 1)
            + [True] * output_len
            + [False] * (max_len - seq_len)
        )

        response_masks.append(mask)

    tokens = torch.tensor(padded, dtype=torch.long)
    response_mask = torch.tensor(response_masks, dtype=torch.bool)

    return {
        "input_ids": tokens[:, :-1],
        "labels": tokens[:, 1:],
        "response_mask": response_mask,
    }


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    log_distribution = F.log_softmax(logits, dim=-1)

    label_log_probs = torch.gather(
        log_distribution,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {
        "log_probs": label_log_probs,
    }

    if return_token_entropy:
        probs = log_distribution.exp()
        token_entropy = -(probs * log_distribution).sum(dim=-1)
        result["token_entropy"] = token_entropy

    return result


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

    metadata: dict[str, float] = {}

    # 自动统计 reward_fn 返回的所有 reward 项
    # 例如 reward / format_reward / answer_reward
    for key in reward_dicts[0].keys():
        values = torch.tensor(
            [r[key] for r in reward_dicts],
            dtype=torch.float32,
        )

        metadata[f"{key}_mean"] = values.mean().item()
        metadata[f"{key}_std"] = values.std(unbiased=False).item()
        metadata[f"{key}_min"] = values.min().item()
        metadata[f"{key}_max"] = values.max().item()

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

    if baseline == "mean":
        group_baseline = grouped_rewards.mean(dim=-1, keepdim=True)
        advantages = grouped_rewards - group_baseline
    elif baseline == "none":
        advantages = grouped_rewards
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    if advantage_normalizer == "std":
        # 作业里一般要求用 torch.std 默认行为
        normalizer = advantages.std(dim=-1, keepdim=True)
        advantages = advantages / (normalizer + advantage_eps)

    elif advantage_normalizer == "mean":
        # 不要对 advantages 求 mean，因为 baseline="mean" 后 advantage 均值接近 0
        normalizer = grouped_rewards.mean(dim=-1, keepdim=True).abs()
        advantages = advantages / (normalizer + advantage_eps)

    elif advantage_normalizer == "none":
        pass

    else:
        raise ValueError(f"Unknown advantage_normalizer: {advantage_normalizer}")

    flat_advantages = advantages.flatten()

    metadata = {
        "advantage_mean": flat_advantages.mean().item(),
        "advantage_std": flat_advantages.std(unbiased=False).item(),
        "advantage_min": flat_advantages.min().item(),
        "advantage_max": flat_advantages.max().item(),
        "group_reward_mean": grouped_rewards.mean().item(),
        "group_reward_std": grouped_rewards.std(unbiased=False).item(),
    }

    return flat_advantages, metadata


def compute_policy_gradient_loss_on_policy(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = raw_rewards_or_advantages.reshape(-1, 1)

    if response_mask is None:
        response_mask = torch.ones_like(policy_log_probs, dtype=torch.bool)

    mask = response_mask.to(dtype=policy_log_probs.dtype)

    metadata: dict[str, torch.Tensor] = {}

    if importance_reweighting_method == "none":
        # on-policy score-function estimator:
        # loss = - A * log pi_theta
        per_token_loss = -advantages * policy_log_probs

    else:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required when importance_reweighting_method != 'none'.")

        if old_log_probs.shape != policy_log_probs.shape:
            raise ValueError(
                f"old_log_probs shape {old_log_probs.shape} != policy_log_probs shape {policy_log_probs.shape}"
            )

        log_ratio = policy_log_probs - old_log_probs

        if importance_reweighting_method == "noclip":
            ratio = torch.exp(log_ratio)
            objective = ratio * advantages
            per_token_loss = -objective

        elif importance_reweighting_method == "grpo":
            if cliprange is None:
                raise ValueError("cliprange is required for GRPO clipping.")

            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

            unclipped_objective = ratio * advantages
            clipped_objective = clipped_ratio * advantages

            objective = torch.minimum(unclipped_objective, clipped_objective)
            per_token_loss = -objective

            clipped = (ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)
            metadata["clip_fraction"] = _masked_mean(clipped.float(), response_mask)

        elif importance_reweighting_method == "gspo":
            if cliprange is None:
                raise ValueError("cliprange is required for GSPO clipping.")

            # sequence-level ratio:
            # exp(mean_t(log pi_new - log pi_old))
            seq_log_ratio = (log_ratio * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
            seq_ratio = torch.exp(seq_log_ratio).reshape(-1, 1)

            clipped_seq_ratio = torch.clamp(
                seq_ratio,
                1.0 - cliprange,
                1.0 + cliprange,
            )

            unclipped_objective = seq_ratio * advantages
            clipped_objective = clipped_seq_ratio * advantages

            objective = torch.minimum(unclipped_objective, clipped_objective)
            per_token_loss = -objective.expand_as(policy_log_probs)

            clipped = (seq_ratio < 1.0 - cliprange) | (seq_ratio > 1.0 + cliprange)
            metadata["clip_fraction"] = clipped.float().mean()

            ratio = seq_ratio.expand_as(policy_log_probs)

        else:
            raise ValueError(f"Unknown importance_reweighting_method: {importance_reweighting_method}")

        metadata["ratio_mean"] = _masked_mean(ratio.detach(), response_mask)
        metadata["ratio_min"] = ratio.detach()[response_mask].min()
        metadata["ratio_max"] = ratio.detach()[response_mask].max()
        metadata["approx_kl"] = _masked_mean((old_log_probs - policy_log_probs).detach(), response_mask)

    per_token_loss = per_token_loss * mask

    metadata["policy_log_probs_mean"] = _masked_mean(policy_log_probs.detach(), response_mask)

    return per_token_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    mask = mask.to(dtype=per_token_policy_gradient_loss.dtype)

    per_token_loss = per_token_policy_gradient_loss * mask

    if loss_normalization == "sequence":
        denom = mask.sum(dim=-1).clamp_min(1.0)
        per_sequence_loss = per_token_loss.sum(dim=-1) / denom

    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant is required when loss_normalization='constant'.")

        per_sequence_loss = per_token_loss.sum(dim=-1) / normalization_constant

    else:
        raise ValueError(f"Unknown loss_normalization: {loss_normalization}")

    return per_sequence_loss.mean()


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    model.train()

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")

    d = tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer,
    )

    device = next(model.parameters()).device

    input_ids = d["input_ids"].to(device)
    labels = d["labels"].to(device)
    response_mask = d["response_mask"].to(device)

    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
    )

    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer,
    )

    advantages = advantages.to(device)

    batch_size = input_ids.shape[0]
    micro_batch_size = math.ceil(batch_size / gradient_accumulation_steps)


    total_loss = torch.tensor(0.0, device=device)
    pg_metadata_accum: dict[str, torch.Tensor] = {}

    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        weight = (end - start) / batch_size

        input_microbatch = input_ids[start:end]
        labels_microbatch = labels[start:end]
        mask_microbatch = response_mask[start:end]
        advantages_microbatch = advantages[start:end]

        if old_log_probs is None:
            old_log_probs_microbatch = None
        else:
            old_log_probs_microbatch = old_log_probs[start:end]

        result = get_response_log_probs(
            model=model,
            input_ids=input_microbatch,
            labels=labels_microbatch,
        )

        log_probs = result["log_probs"]

        per_token_loss, pg_metadata = compute_policy_gradient_loss_on_policy(
            raw_rewards_or_advantages=advantages_microbatch,
            policy_log_probs=log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs_microbatch,
            cliprange=cliprange,
            response_mask=mask_microbatch,
        )

        batch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=mask_microbatch,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        # 加权后再 backward，等价于 full batch mean loss
        weighted_batch_loss = batch_loss * weight

        weighted_batch_loss.backward()
        total_loss = total_loss + weighted_batch_loss.detach()

        for key, value in pg_metadata.items():
            value = value.detach()
            if key not in pg_metadata_accum:
                pg_metadata_accum[key] = value * weight
            else:
                pg_metadata_accum[key] = pg_metadata_accum[key] + value * weight

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
    else:
        grad_norm = None

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    response_lengths = response_mask.sum(dim=-1).float()

    metadata: dict[str, torch.Tensor | float] = {}

    metadata.update(reward_metadata)
    metadata.update(advantage_metadata)

    metadata["loss"] = total_loss.item()
    metadata["response_len_mean"] = response_lengths.mean().item()
    metadata["response_len_min"] = response_lengths.min().item()
    metadata["response_len_max"] = response_lengths.max().item()

    for key, value in pg_metadata_accum.items():
        metadata[key] = value.item()

    if grad_norm is not None:
        metadata["grad_norm"] = grad_norm.item()

    return total_loss, metadata