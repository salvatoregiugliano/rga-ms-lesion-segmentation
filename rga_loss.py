# Copyright (c) 2026 Salvatore Giugliano
# SPDX-License-Identifier: MIT

"""Source implementation of the Regularization via Gradient Attribution loss.

This module implements the model-independent components of the RGA framework:
saliency normalization, attribution-weighted FN/FP regularization, dynamic
lambda scheduling, and a trainer-facing integration wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

# Short aliases keep type annotations readable in the loss and wrapper below.
Tensor = torch.Tensor
SaliencyFn = Callable[[torch.nn.Module, Tensor, Tensor], Tensor]


def normalize_saliency(saliency: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize a dense attribution map to [0, 1] independently per sample."""
    if saliency.ndim < 3:
        raise ValueError("saliency must have shape (B, 1, ...)")

    # RGA uses saliency as a non-negative spatial weight.
    saliency = saliency.float().clamp_min(0.0)

    # Per-sample scaling prevents one case's attribution range from affecting
    # the loss scale of another case in the same batch.
    flat = saliency.reshape(saliency.shape[0], -1)
    scale = flat.amax(dim=1).clamp_min(eps)
    view_shape = (saliency.shape[0],) + (1,) * (saliency.ndim - 1)
    saliency = saliency / scale.view(view_shape)
    return saliency.clamp(0.0, 1.0).nan_to_num(0.0)


def rga_loss(
    logits: Tensor,
    target: Tensor,
    saliency: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute the attribution-normalized RGA loss.

    Args:
        logits: Raw segmentation logits with shape (B, C, D, H, W).
        target: Binary lesion mask with shape (B, D, H, W) or (B, 1, D, H, W).
        saliency: Dense attribution map with shape (B, 1, d, h, w), expected
            to be normalized to [0, 1] by the attribution helper. It is
            detached and clamped before being used as a spatial weight.

    Returns:
        Tuple `(loss, parts)` where `loss = L_FN + L_FP` and `parts` contains
        detached diagnostic tensors for the two components.
    """
    # Differentiable foreground probability for the lesion class.
    # RGA is optimized through this probability map, not through hard labels.
    pred_prob = torch.softmax(logits, dim=1)[:, 1:2]

    # Accept both common target layouts: (B, D, H, W) and (B, 1, D, H, W).
    if target.ndim == pred_prob.ndim - 1:
        target = target.unsqueeze(1)
    target_fg = (target == 1).float()

    # Attribution backends may operate on a lower-resolution layer; RGA is
    # evaluated on the same grid as the segmentation logits.
    if saliency.shape[2:] != pred_prob.shape[2:]:
        saliency = F.interpolate(
            saliency,
            size=pred_prob.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    # Detach prevents second-order gradients through the attribution method;
    # gradients still flow through pred_prob below. In the experimental
    # trainers, saliency is normalized by the attribution backend before the
    # RGA loss is evaluated, so the loss only enforces numeric bounds here.
    saliency = saliency.detach().float().clamp(0.0, 1.0).nan_to_num(0.0)

    # Differentiable soft error maps.
    soft_fn = target_fg * (1.0 - pred_prob)
    soft_fp = (1.0 - target_fg) * pred_prob

    # The normalization terms keep the penalty comparable across patches with
    # different lesion volumes or predicted foreground masses.
    spatial_dims = tuple(range(1, pred_prob.ndim))
    lesion_mass = target_fg.sum(dim=spatial_dims).clamp_min(1.0)
    pred_mass = pred_prob.sum(dim=spatial_dims).clamp_min(1.0)

    # L_FN: missed lesion probability weighted by low saliency.
    loss_fn_sample = ((1.0 - saliency) * soft_fn).sum(dim=spatial_dims)
    loss_fn_sample = loss_fn_sample / lesion_mass

    # L_FP: spurious lesion probability weighted by high saliency.
    loss_fp_sample = (saliency * soft_fp).sum(dim=spatial_dims)
    loss_fp_sample = loss_fp_sample / pred_mass

    # No FN term is defined for patches without lesion voxels.
    valid_fn = (target_fg.sum(dim=spatial_dims) > 0.5).float()
    loss_fn = (loss_fn_sample * valid_fn).sum()
    loss_fn = loss_fn / valid_fn.sum().clamp_min(1.0)
    loss_fp = loss_fp_sample.mean()

    parts = {"L_FN": loss_fn.detach(), "L_FP": loss_fp.detach()}
    return loss_fn + loss_fp, parts


def lambda_schedule(
    epoch: int,
    lambda_min: float,
    lambda_max: float,
    ramp_start: int,
    ramp_end: int,
) -> float:
    """Three-phase RGA schedule: warmup, square-root ramp, plateau."""
    # Warmup: keep RGA weak while the segmentation model is still unstable.
    if epoch < ramp_start:
        return float(lambda_min)

    # Plateau: use the final weight after the scheduled ramp.
    if epoch >= ramp_end:
        return float(lambda_max)

    # Square-root ramp increases early but avoids an abrupt jump at ramp_start.
    progress = (epoch - ramp_start) / max(1, ramp_end - ramp_start)
    progress = progress ** 0.5
    return float(lambda_min + progress * (lambda_max - lambda_min))


@dataclass
class RGARegularizer:
    """Wrapper for adding RGA to a segmentation loss every N batches."""

    lambda_min: float = 0.01
    lambda_max: float = 0.3
    ramp_start: int = 10
    ramp_end: int = 199
    # Attribution can be expensive; applying RGA periodically reproduces the
    # intended training use without computing saliency for every batch.
    every_n_batches: int = 8

    def weight(self, epoch: int) -> float:
        """Return the RGA weight for the current epoch."""
        return lambda_schedule(
            epoch,
            self.lambda_min,
            self.lambda_max,
            self.ramp_start,
            self.ramp_end,
        )

    def should_apply(self, batch_idx: int) -> bool:
        """Return True when this batch should include the attribution penalty."""
        return batch_idx % self.every_n_batches == 0

    def step_loss(
        self,
        base_loss: Tensor,
        model: torch.nn.Module,
        image: Tensor,
        target: Tensor,
        logits: Tensor,
        epoch: int,
        batch_idx: int,
        make_saliency: SaliencyFn,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return `base_loss + lambda * RGA` when the update frequency matches."""
        if not self.should_apply(batch_idx):
            # Unscheduled batches are ordinary segmentation updates.
            return base_loss, {}

        # The trainer supplies the attribution backend, so this wrapper stays
        # independent of model architecture and XAI method.
        saliency = make_saliency(model, image, logits)
        loss_rga, parts = rga_loss(logits, target, saliency)
        lambda_xai = self.weight(epoch)

        # RGA is an additive regularizer; it does not replace DiceCE or any
        # other segmentation loss used by the trainer.
        total_loss = base_loss + lambda_xai * loss_rga

        # Detached diagnostics are useful for logging without extending the
        # computation graph.
        parts = dict(parts)
        parts["L_RGA"] = loss_rga.detach()
        parts["lambda_xai"] = torch.as_tensor(
            lambda_xai,
            dtype=base_loss.dtype,
            device=base_loss.device,
        )
        return total_loss, parts
