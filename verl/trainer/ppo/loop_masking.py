from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TokenRepeatHit:
    start: int
    end: int
    period: int
    repeats: int


def _find_subsequence(tokens: list[int], needle: tuple[int, ...], start: int) -> int:
    max_start = len(tokens) - len(needle)
    for pos in range(max(start, 0), max_start + 1):
        if tuple(tokens[pos : pos + len(needle)]) == needle:
            return pos
    return -1


def _rfind_subsequence(tokens: list[int], needle: tuple[int, ...], end: int) -> int:
    max_start = min(end - len(needle), len(tokens) - len(needle))
    for pos in range(max_start, -1, -1):
        if tuple(tokens[pos : pos + len(needle)]) == needle:
            return pos
    return -1


def _verify_repetition_at(
    tokens: list[int],
    start_pos: int,
    period: int,
    min_repeats: int,
    min_total_repeated_tokens: int,
) -> TokenRepeatHit | None:
    if period < 1 or start_pos < 0 or start_pos + period > len(tokens):
        return None

    pattern = tokens[start_pos : start_pos + period]

    repeats = 0
    pos = start_pos
    while pos + period <= len(tokens) and tokens[pos : pos + period] == pattern:
        repeats += 1
        pos += period
    end_pos = pos

    pos = start_pos - period
    loop_start = start_pos
    while pos >= 0 and tokens[pos : pos + period] == pattern:
        repeats += 1
        loop_start = pos
        pos -= period

    total_repeated = repeats * period
    if repeats < min_repeats or total_repeated < min_total_repeated_tokens:
        return None

    return TokenRepeatHit(start=loop_start, end=end_pos, period=period, repeats=repeats)


def find_inner_repetition_tokens(tokens: list[int], config: dict[str, Any]) -> TokenRepeatHit | None:
    min_repeats = int(config.get("min_repeats", 4))
    max_period = int(config.get("max_period", 1024))
    min_period = int(config.get("min_period", 1))
    min_total_repeated_tokens = int(config.get("min_total_repeated_tokens", 60))
    sample_len = max(1, int(config.get("sample_len", 16)))
    sample_interval = max(1, int(config.get("sample_interval", 128)))

    if not tokens or len(tokens) < min_total_repeated_tokens or len(tokens) < sample_len:
        return None

    for sample_pos in range(0, len(tokens) - sample_len + 1, sample_interval):
        fingerprint = tuple(tokens[sample_pos : sample_pos + sample_len])

        other_pos = _find_subsequence(tokens, fingerprint, sample_pos + sample_len)
        if other_pos != -1:
            candidate_period = other_pos - sample_pos
            if min_period <= candidate_period <= max_period:
                hit = _verify_repetition_at(
                    tokens,
                    sample_pos,
                    candidate_period,
                    min_repeats=min_repeats,
                    min_total_repeated_tokens=min_total_repeated_tokens,
                )
                if hit is not None:
                    return hit

        other_pos = _rfind_subsequence(tokens, fingerprint, sample_pos)
        if other_pos != -1:
            candidate_period = sample_pos - other_pos
            if min_period <= candidate_period <= max_period:
                hit = _verify_repetition_at(
                    tokens,
                    other_pos,
                    candidate_period,
                    min_repeats=min_repeats,
                    min_total_repeated_tokens=min_total_repeated_tokens,
                )
                if hit is not None:
                    return hit

    return None


def _get_doomloop_flags(batch: Any) -> np.ndarray | None:
    doomloop = batch.non_tensor_batch.get("doomloop")
    if doomloop is None:
        return None
    doomloop = np.asarray(doomloop, dtype=np.float32).reshape(-1)
    return doomloop > 0.5


def apply_loop_tail_mask(batch: Any, loop_mask_cfg: dict[str, Any]) -> dict[str, float]:
    if "responses" not in batch.batch or "response_mask" not in batch.batch:
        return {}

    response_mask = batch.batch["response_mask"]
    if response_mask.numel() == 0:
        return {}

    doomloop_flags = _get_doomloop_flags(batch)
    if doomloop_flags is None:
        return {
            "loop_masking/eligible_ratio": 0.0,
            "loop_masking/samples_ratio": 0.0,
            "loop_masking/tokens_masked": 0.0,
            "loop_masking/token_ratio": 0.0,
        }

    keep_segments = int(loop_mask_cfg.get("keep_segments", 3))
    keep_tokens = int(loop_mask_cfg.get("keep_tokens", 30))

    eligible_sample_count = 0
    masked_sample_count = 0
    masked_token_count = 0
    first_mask_start_sum = 0.0
    repeat_period_sum = 0.0
    repeat_count_sum = 0.0
    valid_token_count = int(response_mask.sum().item())

    for row_idx in range(response_mask.shape[0]):
        if row_idx >= len(doomloop_flags) or not doomloop_flags[row_idx]:
            continue
        eligible_sample_count += 1

        valid_positions = torch.nonzero(response_mask[row_idx] > 0, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue

        token_sequence = batch.batch["responses"][row_idx, valid_positions].tolist()
        hit = find_inner_repetition_tokens(token_sequence, loop_mask_cfg)
        if hit is None:
            continue

        keep_loop_tokens = max(keep_segments * hit.period, keep_tokens)
        mask_start_in_valid = min(hit.start + keep_loop_tokens, len(token_sequence))
        if mask_start_in_valid >= len(token_sequence):
            continue

        mask_positions = valid_positions[mask_start_in_valid:]
        if mask_positions.numel() == 0:
            continue

        response_mask[row_idx, mask_positions] = 0
        masked_sample_count += 1
        masked_token_count += int(mask_positions.numel())
        first_mask_start_sum += float(mask_start_in_valid)
        repeat_period_sum += float(hit.period)
        repeat_count_sum += float(hit.repeats)

    base_metrics = {
        "loop_masking/eligible_ratio": eligible_sample_count / max(response_mask.shape[0], 1),
        "loop_masking/samples_ratio": masked_sample_count / max(response_mask.shape[0], 1),
        "loop_masking/tokens_masked": float(masked_token_count),
        "loop_masking/token_ratio": masked_token_count / max(valid_token_count, 1),
    }
    if masked_sample_count == 0:
        return base_metrics

    base_metrics.update(
        {
            "loop_masking/masked_tokens_per_hit": masked_token_count / masked_sample_count,
            "loop_masking/first_mask_start": first_mask_start_sum / masked_sample_count,
            "loop_masking/repeat_period": repeat_period_sum / masked_sample_count,
            "loop_masking/repeats": repeat_count_sum / masked_sample_count,
        }
    )
    return base_metrics
