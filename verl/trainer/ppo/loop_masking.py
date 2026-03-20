from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TokenRepeatHit:
    start: int
    end: int
    period: int
    repeats: int


def _required_repeats(period: int, config: dict[str, Any]) -> int:
    required = int(config.get("base_min_repeats", 3))

    short_threshold = int(config.get("short_period_threshold", 4))
    if period <= short_threshold:
        required += int(config.get("short_period_extra_repeats", 3))
        return required

    medium_threshold = int(config.get("medium_period_threshold", 8))
    if period <= medium_threshold:
        required += int(config.get("medium_period_extra_repeats", 1))

    return required


def _verify_repetition_at(
    tokens: list[int],
    start_pos: int,
    period: int,
    config: dict[str, Any],
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

    if repeats < _required_repeats(period, config):
        return None

    total_repeated = repeats * period
    if total_repeated < int(config.get("min_total_repeated_tokens", 24)):
        return None

    return TokenRepeatHit(start=loop_start, end=end_pos, period=period, repeats=repeats)


def find_inner_repetition_tokens(tokens: list[int], config: dict[str, Any]) -> TokenRepeatHit | None:
    sample_len = max(1, int(config.get("sample_len", 8)))
    sample_interval = max(1, int(config.get("sample_interval", 32)))
    min_period = int(config.get("min_period", 1))
    max_period = int(config.get("max_period", 256))
    min_total_repeated_tokens = int(config.get("min_total_repeated_tokens", 24))

    if len(tokens) < max(sample_len, min_total_repeated_tokens):
        return None

    positions_by_fingerprint: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for pos in range(0, len(tokens) - sample_len + 1):
        fingerprint = tuple(tokens[pos : pos + sample_len])
        positions_by_fingerprint[fingerprint].append(pos)

    sampled_positions = range(0, max(1, len(tokens) - sample_len + 1), sample_interval)
    for sample_pos in sampled_positions:
        fingerprint = tuple(tokens[sample_pos : sample_pos + sample_len])
        fingerprint_positions = positions_by_fingerprint.get(fingerprint, [])
        if len(fingerprint_positions) <= 1:
            continue

        sample_index = bisect_left(fingerprint_positions, sample_pos)

        if sample_index + 1 < len(fingerprint_positions):
            other_pos = fingerprint_positions[sample_index + 1]
            candidate_period = other_pos - sample_pos
            if not (min_period <= candidate_period <= max_period):
                other_pos = None
            if other_pos is not None:
                hit = _verify_repetition_at(tokens, sample_pos, candidate_period, config)
                if hit is not None:
                    return hit

        if sample_index > 0:
            other_pos = fingerprint_positions[sample_index - 1]
            candidate_period = sample_pos - other_pos
            if not (min_period <= candidate_period <= max_period):
                other_pos = None
            if other_pos is not None:
                hit = _verify_repetition_at(tokens, other_pos, candidate_period, config)
                if hit is not None:
                    return hit

    return None


def apply_loop_tail_mask(batch: Any, loop_mask_cfg: dict[str, Any]) -> dict[str, float]:
    if "responses" not in batch.batch or "response_mask" not in batch.batch:
        return {}

    response_mask = batch.batch["response_mask"]
    if response_mask.numel() == 0:
        return {}

    masked_sample_count = 0
    masked_token_count = 0
    first_mask_start_sum = 0.0
    repeat_period_sum = 0.0
    repeat_count_sum = 0.0
    valid_token_count = int(response_mask.sum().item())

    keep_repeats_base = int(loop_mask_cfg.get("keep_repeats", 3))
    short_threshold = int(loop_mask_cfg.get("short_period_threshold", 4))
    short_keep_extra = int(loop_mask_cfg.get("short_period_keep_extra", 1))

    for row_idx in range(response_mask.shape[0]):
        valid_positions = torch.nonzero(response_mask[row_idx] > 0, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue

        token_sequence = batch.batch["responses"][row_idx, valid_positions].tolist()
        hit = find_inner_repetition_tokens(token_sequence, loop_mask_cfg)
        if hit is None:
            continue

        keep_repeats = keep_repeats_base
        if hit.period <= short_threshold:
            keep_repeats += short_keep_extra

        if hit.repeats <= keep_repeats:
            continue

        mask_start_in_valid = hit.start + hit.period * keep_repeats
        mask_end_in_valid = hit.end
        if mask_start_in_valid >= mask_end_in_valid:
            continue

        mask_positions = valid_positions[mask_start_in_valid:mask_end_in_valid]
        if mask_positions.numel() == 0:
            continue

        response_mask[row_idx, mask_positions] = 0
        masked_sample_count += 1
        masked_token_count += int(mask_positions.numel())
        first_mask_start_sum += float(mask_start_in_valid)
        repeat_period_sum += float(hit.period)
        repeat_count_sum += float(hit.repeats)

    if masked_sample_count == 0:
        return {
            "loop_masking/samples_ratio": 0.0,
            "loop_masking/tokens_masked": 0.0,
            "loop_masking/token_ratio": 0.0,
        }

    return {
        "loop_masking/samples_ratio": masked_sample_count / response_mask.shape[0],
        "loop_masking/tokens_masked": float(masked_token_count),
        "loop_masking/token_ratio": masked_token_count / max(valid_token_count, 1),
        "loop_masking/masked_tokens_per_hit": masked_token_count / masked_sample_count,
        "loop_masking/first_mask_start": first_mask_start_sum / masked_sample_count,
        "loop_masking/repeat_period": repeat_period_sum / masked_sample_count,
        "loop_masking/repeats": repeat_count_sum / masked_sample_count,
    }
