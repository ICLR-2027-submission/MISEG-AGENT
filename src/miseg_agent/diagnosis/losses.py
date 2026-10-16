"""Losses for the factorized ED V2 objective."""

from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    fp_weight: float = 0.7,
    fn_weight: float = 0.3,
    gamma: float = 0.75,
    smooth: float = 1.0,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=dims)
    false_positive = (probability * (1.0 - target)).sum(dim=dims)
    false_negative = ((1.0 - probability) * target).sum(dim=dims)
    score = (true_positive + smooth) / (
        true_positive + fp_weight * false_positive + fn_weight * false_negative + smooth
    )
    return torch.pow(1.0 - score, gamma).mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def balanced_masked_bce(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask > 0.5
    if not torch.any(valid):
        return logits.sum() * 0.0
    selected_logits = logits[valid]
    selected_target = target[valid]
    positives = selected_target.sum()
    negatives = selected_target.numel() - positives
    if positives > 0 and negatives > 0:
        positive_weight = (negatives / positives).clamp(0.25, 4.0)
        return F.binary_cross_entropy_with_logits(selected_logits, selected_target, pos_weight=positive_weight)
    return F.binary_cross_entropy_with_logits(selected_logits, selected_target)


def compute_losses(outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    error_target = targets["error"].float()
    polarity_target = targets["polarity"].float()
    gt_target = targets["gt_foreground"].float()
    error_bce = F.binary_cross_entropy_with_logits(outputs["error_logit"], error_target)
    error_tversky = focal_tversky_loss(outputs["error_logit"], error_target, fp_weight=0.7, fn_weight=0.3)
    error_loss = 0.5 * error_bce + 0.5 * error_tversky
    polarity_loss = balanced_masked_bce(outputs["polarity_logit"], polarity_target, error_target)
    gt_loss = F.binary_cross_entropy_with_logits(outputs["gt_foreground_logit"], gt_target) + soft_dice_loss(
        outputs["gt_foreground_logit"], gt_target
    )
    total = error_loss + 0.5 * polarity_loss + 0.25 * gt_loss
    return {
        "loss": total,
        "error_loss": error_loss,
        "polarity_loss": polarity_loss,
        "gt_loss": gt_loss,
        "error_bce": error_bce,
        "error_focal_tversky": error_tversky,
    }
