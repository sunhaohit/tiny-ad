from __future__ import annotations

import math

import torch


def _top_values(values: torch.Tensor, count: int) -> torch.Tensor:
    count = max(1, min(int(count), values.shape[1]))
    return torch.topk(values, count, dim=1).values


def _mean_std(values: torch.Tensor, count: int, weight: float) -> torch.Tensor:
    top = _top_values(values, count)
    return top.mean(dim=1) + weight * top.std(dim=1, unbiased=False)


def _soft_top(values: torch.Tensor, count: int, temperature: float) -> torch.Tensor:
    top = _top_values(values, count)
    centered = top - top.mean(dim=1, keepdim=True)
    scale = top.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
    weights = torch.softmax(centered / (scale * temperature), dim=1)
    return (weights * top).sum(dim=1)


def _background_adjusted(
    values: torch.Tensor,
    count: int,
    weight: float,
    quantile: float,
) -> torch.Tensor:
    top_mean = _top_values(values, count).mean(dim=1)
    background = torch.quantile(values, quantile, dim=1)
    return top_mean - weight * background


def _logsumexp_mix(
    values: torch.Tensor,
    count: int,
    weight: float,
    temperature: float,
) -> torch.Tensor:
    top = _top_values(values, count)
    top_mean = top.mean(dim=1)
    scale = top.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
    smooth = torch.logsumexp(top / (scale * temperature), dim=1)
    smooth = smooth * scale.squeeze(1) * temperature
    smooth = smooth - math.log(top.shape[1]) * scale.squeeze(1) * temperature
    return (1.0 - weight) * top_mean + weight * smooth


def _multi_top(
    values: torch.Tensor,
    counts: tuple[int, ...],
    weights: tuple[float, ...],
) -> torch.Tensor:
    total = sum(weights)
    output = values.new_zeros(values.shape[0])
    for count, weight in zip(counts, weights):
        output = output + (weight / total) * _top_values(values, count).mean(dim=1)
    return output


def _contrast(values: torch.Tensor, count: int, weight: float) -> torch.Tensor:
    top_mean = _top_values(values, count).mean(dim=1)
    background = torch.quantile(values, 0.50, dim=1)
    return top_mean + weight * (top_mean - background)


def _tail_gap(values: torch.Tensor, count: int) -> torch.Tensor:
    top = _top_values(values, count)
    top_mean = top.mean(dim=1)
    split = max(1, top.shape[1] // 2)
    head = top[:, :split].mean(dim=1)
    tail = top[:, split:].mean(dim=1) if split < top.shape[1] else top_mean
    background = torch.quantile(values, 0.50, dim=1)
    return top_mean + 0.18 * (head - tail) + 0.18 * (top_mean - background)


def _soft_mix(values: torch.Tensor, count: int) -> torch.Tensor:
    top = _top_values(values, count)
    top_mean = top.mean(dim=1)
    centered = top - top_mean.unsqueeze(1)
    scale = top.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
    weights = torch.softmax(centered / (scale * 0.75), dim=1)
    soft_score = (weights * top).sum(dim=1)
    return 0.85 * top_mean + 0.15 * soft_score


def aggregate_image_score(
    patch_scores: torch.Tensor,
    category: str,
    top_count: int = 6,
) -> torch.Tensor:
    """Aggregate patch evidence using the checkpoint's fixed calibration."""
    name = category.casefold()
    if "macaroni2" in name:
        return _soft_top(patch_scores, top_count, 0.78)
    if "screw" in name:
        return _background_adjusted(patch_scores, top_count, 0.36, 0.45)
    if "2311694" in name:
        return _logsumexp_mix(patch_scores, top_count, 0.35, 0.70)
    if "macaroni1" in name:
        return _mean_std(patch_scores, top_count, 0.28)
    if "candle" in name or "pcb2" in name:
        return _multi_top(patch_scores, (5, 6, 8), (0.25, 0.55, 0.20))
    if "capsules" in name or "2306894" in name:
        return _mean_std(patch_scores, top_count, 0.25)
    if "chewinggum" in name:
        return _contrast(patch_scores, top_count, 0.35)
    if "grid" in name or "leather" in name:
        return _tail_gap(patch_scores, top_count)
    if "kolektor" in name:
        return _soft_mix(patch_scores, top_count)
    return _mean_std(patch_scores, top_count, 0.25)

