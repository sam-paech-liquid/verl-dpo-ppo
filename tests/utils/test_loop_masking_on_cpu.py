from __future__ import annotations

import torch

from verl.trainer.ppo.loop_masking import apply_loop_tail_mask, find_inner_repetition_tokens


DEFAULT_CFG = {
    "base_min_repeats": 3,
    "short_period_threshold": 4,
    "short_period_extra_repeats": 3,
    "medium_period_threshold": 8,
    "medium_period_extra_repeats": 1,
    "min_total_repeated_tokens": 12,
    "sample_len": 2,
    "sample_interval": 1,
    "min_period": 1,
    "max_period": 64,
    "keep_repeats": 3,
    "short_period_keep_extra": 1,
}


class _FakeBatch:
    def __init__(self, responses: torch.Tensor, response_mask: torch.Tensor):
        self.batch = {
            "responses": responses,
            "response_mask": response_mask,
        }


def test_find_inner_repetition_tokens_detects_long_period():
    tokens = [1, 2, 3] + [11, 12, 13, 14] * 6 + [99, 100]
    hit = find_inner_repetition_tokens(tokens, DEFAULT_CFG)
    assert hit is not None
    assert hit.period == 4
    assert hit.repeats >= 6
    assert hit.start == 3


def test_find_inner_repetition_tokens_requires_more_repeats_for_short_period():
    tokens = [7] * 11 + [1, 2, 3]
    hit = find_inner_repetition_tokens(tokens, DEFAULT_CFG)
    assert hit is None

    tokens = [7] * 12 + [1, 2, 3]
    hit = find_inner_repetition_tokens(tokens, DEFAULT_CFG)
    assert hit is not None
    assert hit.period == 1


def test_apply_loop_tail_mask_masks_only_late_repeat_suffix():
    responses = torch.tensor([[10, 20, 30, 40, 30, 40, 30, 40, 30, 40, 30, 40, 30, 40, 99, 100]], dtype=torch.long)
    response_mask = torch.ones_like(responses)
    batch = _FakeBatch(responses, response_mask)

    metrics = apply_loop_tail_mask(batch, DEFAULT_CFG)

    masked_positions = torch.nonzero(batch.batch["response_mask"][0] == 0, as_tuple=False).flatten().tolist()
    assert masked_positions == [10, 11, 12, 13]
    assert batch.batch["response_mask"][0, 14].item() == 1
    assert metrics["loop_masking/samples_ratio"] == 1.0
    assert metrics["loop_masking/tokens_masked"] == 4.0


def test_apply_loop_tail_mask_preserves_recovery_after_repetition():
    responses = torch.tensor([[1, 2, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 70, 71, 72]], dtype=torch.long)
    response_mask = torch.ones_like(responses)
    batch = _FakeBatch(responses, response_mask)

    apply_loop_tail_mask(batch, DEFAULT_CFG)

    assert batch.batch["response_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]


def test_apply_loop_tail_mask_leaves_clean_sample_unchanged():
    responses = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    response_mask = torch.ones_like(responses)
    batch = _FakeBatch(responses, response_mask)

    metrics = apply_loop_tail_mask(batch, DEFAULT_CFG)

    assert torch.all(batch.batch["response_mask"] == 1)
    assert metrics["loop_masking/samples_ratio"] == 0.0
