# Copyright (c) 2026 Salvatore Giugliano
# SPDX-License-Identifier: MIT

"""Public API for the Regularization via Gradient Attribution framework."""

# Attribution backends used to build dense saliency maps for RGA.
from .attribution import (
    integrated_gradients_saliency,
    layercam_saliency,
)

# Core RGA components: saliency scaling, FN/FP regularization, scheduling, and
# the trainer-facing integration wrapper.
from .rga_loss import RGARegularizer, lambda_schedule, normalize_saliency, rga_loss

# Names exported when users import from the package or call `from rga_module import *`.
__all__ = [
    "RGARegularizer",
    "integrated_gradients_saliency",
    "lambda_schedule",
    "layercam_saliency",
    "normalize_saliency",
    "rga_loss",
]
