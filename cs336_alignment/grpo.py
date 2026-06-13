import torch
from transformers import PreTrainedTokenizerBase

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