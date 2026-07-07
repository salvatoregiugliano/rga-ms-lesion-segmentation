# Copyright (c) 2026 Salvatore Giugliano
# SPDX-License-Identifier: MIT

"""Attribution helpers for Regularization via Gradient Attribution.

The functions in this module implement the XAI maps used by the RGA framework:
LayerCAM and Integrated Gradients (IG). The IG helper uses Captum on a selected
target layer, matching the RGA (IG) experimental setting. Both helpers operate
on binary segmentation models with foreground class index 1.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from .rga_loss import normalize_saliency
except ImportError:  # Allows running this file from the repository root.
    from rga_loss import normalize_saliency

Tensor = torch.Tensor


def _integrated_gradients_cls():
    """Import Captum only when Integrated Gradients is requested."""
    # Captum is optional because LayerCAM does not need it.
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError(
            "captum is required for integrated_gradients_saliency. "
            "Install it with `pip install captum`."
        ) from exc
    return LayerIntegratedGradients


def _foreground_class_scores(
    model: torch.nn.Module,
    image: Tensor,
) -> Tensor:
    """Return per-sample foreground scores for Captum target=1."""
    output = model(image)
    # Some segmentation trainers return auxiliary outputs; RGA uses the main
    # logits tensor.
    logits = output[0] if isinstance(output, (list, tuple)) else output

    # Captum expects one score per class and per sample. Summing over spatial
    # locations gives a foreground lesion score for dense segmentation.
    return logits.float().flatten(2).sum(dim=2)


def layercam_saliency(
    foreground_logits: Tensor,
    activation: Tensor,
    output_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """Compute LayerCAM from foreground logits and decoder activations.

    Args:
        foreground_logits: Foreground logits with shape (B, D, H, W).
        activation: Target-layer activation with shape (B, C, d, h, w).
        output_shape: Optional spatial shape for trilinear upsampling.
    """
    if not activation.requires_grad:
        # Hooks in external trainers may provide activations without gradient
        # tracking enabled.
        activation.requires_grad_(True)

    # LayerCAM weights target-layer activations by positive gradients of the
    # foreground score.
    score = foreground_logits.float().sum()
    grad = torch.autograd.grad(
        score,
        activation,
        retain_graph=True,
        create_graph=False,
    )[0]

    saliency = (F.relu(grad) * activation).sum(dim=1, keepdim=True)
    # RGA consumes non-negative saliency weights in [0, 1].
    saliency = normalize_saliency(saliency.detach())

    if output_shape is not None and saliency.shape[2:] != output_shape:
        # Match the segmentation output grid before computing the RGA loss.
        saliency = F.interpolate(
            saliency,
            size=output_shape,
            mode="trilinear",
            align_corners=False,
        )
    return saliency


def integrated_gradients_saliency(
    model: torch.nn.Module,
    image: Tensor,
    target_layer: torch.nn.Module,
    steps: int = 5,
    output_shape: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """Compute Integrated Gradients saliency with Captum.

    A zero image is used as baseline. Captum receives per-class foreground
    scores, and `target=1` selects the lesion class.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    integrated_gradients_cls = _integrated_gradients_cls()

    # We need gradients with respect to the selected layer, not weight updates
    # through Captum's attribution graph.
    params_with_grad = [p for p in model.parameters() if p.requires_grad]
    for param in params_with_grad:
        param.requires_grad_(False)

    # nnU-Net style models may return deep-supervision outputs during training;
    # attribution uses only the main prediction.
    prev_deep_supervision = None
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "deep_supervision"):
        prev_deep_supervision = decoder.deep_supervision
        decoder.deep_supervision = False

    try:
        # Zero baseline matches the RGA (IG) experimental setting.
        lig = integrated_gradients_cls(
            lambda x: _foreground_class_scores(model, x),
            target_layer,
        )
        attr = lig.attribute(
            inputs=image,
            baselines=torch.zeros_like(image),
            target=1,
            n_steps=steps,
            internal_batch_size=1,
            attribute_to_layer_input=False,
        )

        # Collapse layer channels into one foreground saliency volume.
        saliency = F.relu(attr.float().sum(dim=1, keepdim=True))

        # Per-sample percentile normalization is robust to isolated attribution
        # outliers and keeps the RGA loss scale stable.
        batch = saliency.shape[0]
        flat = saliency.reshape(batch, -1)
        q99 = torch.quantile(flat, 0.99, dim=1).clamp_min(1e-8)
        view_shape = (batch,) + (1,) * (saliency.ndim - 1)
        saliency = (saliency / q99.view(view_shape)).clamp(0.0, 1.0)
        saliency = saliency.nan_to_num(0.0)

        if output_shape is not None and saliency.shape[2:] != output_shape:
            # Match the segmentation output grid before computing the RGA loss.
            saliency = F.interpolate(
                saliency,
                size=output_shape,
                mode="trilinear",
                align_corners=False,
            )
        return saliency.detach()
    finally:
        # Restore trainer state even if Captum raises.
        for param in params_with_grad:
            param.requires_grad_(True)
        if prev_deep_supervision is not None:
            decoder.deep_supervision = prev_deep_supervision
