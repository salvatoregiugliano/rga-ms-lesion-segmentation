# Copyright (c) 2026 Salvatore Giugliano
# SPDX-License-Identifier: MIT

"""Source implementation of the Regularization via Gradient Attribution framework.

This compatibility module re-exports the public RGA API from `rga_loss.py` and
`attribution.py`.
"""

try:
    # Package-style imports, e.g. `from RELEASE_GITHUB import RGARegularizer`.
    from .attribution import (
        integrated_gradients_saliency,
        layercam_saliency,
    )
    from .rga_loss import RGARegularizer, lambda_schedule, normalize_saliency, rga_loss
except ImportError:  # Allows `python rga_module.py` style local imports.
    # Local-script imports, e.g. running `example_usage.py` from this directory.
    from attribution import (
        integrated_gradients_saliency,
        layercam_saliency,
    )
    from rga_loss import RGARegularizer, lambda_schedule, normalize_saliency, rga_loss

# Keep the compatibility module aligned with the package-level public API.
__all__ = [
    "RGARegularizer",
    "integrated_gradients_saliency",
    "lambda_schedule",
    "layercam_saliency",
    "normalize_saliency",
    "rga_loss",
]
