"""Pure padded action regression and distillation losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionLoss:
    total: torch.Tensor
    ground_truth: torch.Tensor
    teacher: torch.Tensor
    smoothness: torch.Tensor


def action_distillation_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor | None,
    padding_mask: torch.Tensor,
    *,
    teacher: torch.Tensor | None = None,
    teacher_variance: torch.Tensor | None = None,
    teacher_valid: torch.Tensor | None = None,
    ground_truth_weight: float = 1.0,
    teacher_weight: float = 0.0,
    smoothness_weight: float = 0.0,
    variance_floor: float = 1e-4,
    confidence_max: float = 100.0,
) -> ActionLoss:
    """Compute a strictly validated loss; ``padding_mask=True`` means padding."""
    if prediction.ndim != 3 or padding_mask.shape != prediction.shape[:2]:
        raise ValueError("prediction and padding_mask must have [B,T,D] and [B,T] shapes")
    if padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be boolean")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction must be finite")
    weights = (ground_truth_weight, teacher_weight, smoothness_weight)
    if any(not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value)) or value < 0 for value in weights):
        raise ValueError("loss weights must be finite and nonnegative")
    if ground_truth_weight <= 0:
        raise ValueError("ground_truth_weight must be positive; teacher supervision is additive only")
    if variance_floor <= 0 or confidence_max <= 0:
        raise ValueError("variance_floor and confidence_max must be positive")
    if ground_truth_weight > 0 and ground_truth is None:
        raise ValueError("ground_truth is required when ground_truth_weight > 0")
    if teacher_weight > 0 and teacher is None:
        raise ValueError("teacher is required when teacher_weight > 0")
    valid_steps = ~padding_mask
    if not bool(valid_steps.any()):
        raise ValueError("all-padded batches have no supervision")
    if teacher_valid is not None:
        if teacher_valid.dtype != torch.bool or teacher_valid.shape != (prediction.shape[0],):
            raise ValueError("teacher_valid must be boolean with shape [B]")
    elif teacher_weight > 0:
        teacher_valid = torch.ones(prediction.shape[0], dtype=torch.bool, device=prediction.device)

    zero = prediction.sum() * 0.0

    def masked_mse(target: torch.Tensor | None, mask: torch.Tensor, confidence=None) -> torch.Tensor:
        if target is None:
            return zero
        if target.shape != prediction.shape or not torch.isfinite(target).all():
            raise ValueError("action targets must be finite and match prediction shape")
        element_weight = mask.to(prediction.dtype).unsqueeze(-1).expand_as(prediction)
        if confidence is not None:
            element_weight = element_weight * confidence.expand_as(prediction)
        denominator = element_weight.sum()
        if not bool(denominator > 0):
            raise ValueError("requested target has no valid supervision")
        return ((prediction - target).square() * element_weight).sum() / denominator

    gt_loss = masked_mse(ground_truth, valid_steps) if ground_truth_weight > 0 else zero
    teacher_mask = valid_steps
    if teacher_valid is not None:
        teacher_mask = teacher_mask & teacher_valid.to(prediction.device).unsqueeze(1)
    confidence = None
    if teacher_variance is not None:
        if teacher is None or teacher_variance.shape not in (prediction.shape, prediction.shape[:2] + (1,)):
            raise ValueError("teacher_variance must broadcast over a supplied teacher target")
        if not torch.isfinite(teacher_variance).all() or bool((teacher_variance < 0).any()):
            raise ValueError("teacher_variance must be finite and nonnegative")
        confidence = teacher_variance.clamp_min(variance_floor).reciprocal().clamp_max(confidence_max)
        valid_confidence = confidence.expand_as(prediction)[teacher_mask.unsqueeze(-1).expand_as(prediction)]
        if valid_confidence.numel() == 0 and teacher_weight > 0:
            raise ValueError("teacher target has no valid supervision")
        if valid_confidence.numel() > 0:
            confidence = confidence / valid_confidence.mean().clamp_min(torch.finfo(prediction.dtype).eps)
    teacher_loss = (
        masked_mse(teacher, teacher_mask, confidence) if teacher_weight > 0 else zero
    )

    pair_valid = valid_steps[:, 1:] & valid_steps[:, :-1]
    delta = prediction[:, 1:] - prediction[:, :-1]
    if bool(pair_valid.any()):
        pair_weight = pair_valid.to(prediction.dtype).unsqueeze(-1)
        smoothness = (delta.square() * pair_weight).sum() / (pair_weight.sum() * prediction.shape[-1])
    else:
        smoothness = zero
    total = ground_truth_weight * gt_loss + teacher_weight * teacher_loss + smoothness_weight * smoothness
    if not torch.isfinite(total):
        raise ValueError("loss is non-finite")
    return ActionLoss(total, gt_loss, teacher_loss, smoothness)
