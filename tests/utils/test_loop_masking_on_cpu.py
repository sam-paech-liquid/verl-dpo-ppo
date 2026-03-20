from __future__ import annotations

import numpy as np
import torch

from verl.trainer.ppo.loop_masking import apply_loop_tail_mask, find_inner_repetition_tokens


DEFAULT_CFG = {
    "min_repeats": 4,
    "max_period": 64,
    "min_period": 1,
    "min_total_repeated_tokens": 12,
    "sample_len": 2,
    "sample_interval": 1,
    "keep_segments": 3,
    "keep_tokens": 30,
}


class _FakeBatch:
    def __init__(self, responses: torch.Tensor, response_mask: torch.Tensor, doomloop: list[float]):
        self.batch = {
            "responses": responses,
            "response_mask": response_mask,
        }
        self.non_tensor_batch = {
            "doomloop": np.asarray(doomloop, dtype=np.float32),
        }


def test_find_inner_repetition_tokens_detects_long_period():
    tokens = [1, 2, 3] + [11, 12, 13, 14] * 6 + [99, 100]
    hit = find_inner_repetition_tokens(tokens, DEFAULT_CFG)
    assert hit is not None
    assert hit.period == 4
    assert hit.repeats >= 6
    assert hit.start == 3


def test_find_inner_repetition_tokens_requires_min_repeats():
    tokens = [7] * 11 + [1, 2, 3]
    hit = find_inner_repetition_tokens(tokens, DEFAULT_CFG)
    assert hit is None

    tokens = [7] * 12 + [1, 2, 3]
    hit = find_inner_repetition_tokens(tokens, DEFAULT_CFG)
    assert hit is not None
    assert hit.period == 1


def test_apply_loop_tail_mask_only_runs_for_doomloop_samples():
    responses = torch.tensor([[10, 20] + [30, 40] * 6 + [99, 100]], dtype=torch.long)
    response_mask = torch.ones_like(responses)
    batch = _FakeBatch(responses, response_mask, doomloop=[0.0])

    metrics = apply_loop_tail_mask(batch, DEFAULT_CFG)

    assert batch.batch["response_mask"][0].tolist() == [1] * responses.shape[1]
    assert metrics["loop_masking/eligible_ratio"] == 0.0
    assert metrics["loop_masking/samples_ratio"] == 0.0


def test_apply_loop_tail_mask_keeps_first_30_loop_tokens_then_masks_rest():
    responses = torch.tensor([[1, 2] + [8, 9] * 20 + [70, 71, 72]], dtype=torch.long)
    response_mask = torch.ones_like(responses)
    batch = _FakeBatch(responses, response_mask, doomloop=[1.0])

    metrics = apply_loop_tail_mask(batch, DEFAULT_CFG)

    expected = [1] * 32 + [0] * 13
    assert batch.batch["response_mask"][0].tolist() == expected
    assert metrics["loop_masking/eligible_ratio"] == 1.0
    assert metrics["loop_masking/samples_ratio"] == 1.0
    assert metrics["loop_masking/tokens_masked"] == 13.0


def test_apply_loop_tail_mask_masks_to_end_after_short_loop_budget():
    cfg = dict(DEFAULT_CFG)
    cfg["keep_tokens"] = 3
    responses = torch.tensor([[5, 6] + [21, 22, 23, 24] * 8 + [90, 91]], dtype=torch.long)
    response_mask = torch.ones_like(responses)
    batch = _FakeBatch(responses, response_mask, doomloop=[1.0])

    apply_loop_tail_mask(batch, cfg)

    expected_prefix = [1] * 14
    expected_suffix = [0] * (responses.shape[1] - len(expected_prefix))
    assert batch.batch["response_mask"][0].tolist() == expected_prefix + expected_suffix


def test_apply_loop_tail_mask_leaves_clean_sample_unchanged():
    responses = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    response_mask = torch.ones_like(responses)
    batch = _FakeBatch(responses, response_mask, doomloop=[1.0])

    metrics = apply_loop_tail_mask(batch, DEFAULT_CFG)

    assert torch.all(batch.batch["response_mask"] == 1)
    assert metrics["loop_masking/eligible_ratio"] == 1.0
    assert metrics["loop_masking/samples_ratio"] == 0.0
