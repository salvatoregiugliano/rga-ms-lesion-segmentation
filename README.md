# Regularization via Gradient Attribution (RGA)

Source implementation of the Regularization via Gradient Attribution (RGA)
framework for:

**Regularization via Gradient Attribution for Multiple Sclerosis Lesion Segmentation**

**Authors:** Salvatore Giugliano and Giovanna Sannino  
**Code author:** Salvatore Giugliano  
**Affiliation:** Institute for High-Performance Computing and Networking
(ICAR), National Research Council of Italy (CNR), Naples, Italy  
**Journal:** Frontiers in Medicine  
**Research Topic:** Advancements and Applications of the Internet of Things
(IoT) in Medicine

**Paper status:** currently under review.

RGA is a model-independent regularization module for binary lesion segmentation.
It adds an attribution-weighted penalty to an existing segmentation objective,
such as Dice, cross-entropy, DiceCE, or another differentiable loss. The module
uses dense saliency maps to penalize soft false-negative and false-positive
regions during training.

The code is PyTorch-based and trainer-independent. It can be integrated into
nnU-Net, UNETR, TransBTS, or other segmentation pipelines that expose:

- segmentation logits;
- binary lesion masks;
- a differentiable segmentation loss;
- a saliency map from LayerCAM or Integrated Gradients.

## Repository Contents

- `rga_loss.py`
  - saliency normalization to `[0, 1]`;
  - attribution-normalized false-negative and false-positive RGA loss;
  - square-root lambda scheduling;
  - `RGARegularizer`, a small integration wrapper for training loops.
- `attribution.py`
  - LayerCAM saliency helper;
  - Integrated Gradients saliency helper implemented with Captum;
  - foreground-logit scoring for dense binary segmentation.
- `rga_module.py`
  - compatibility re-export of the public RGA API.
- `__init__.py`
  - package-level public API.
- `example_usage.py`
  - synthetic multi-seed train/test example comparing baseline DiceCE,
    DiceCE + RGA (LayerCAM), and DiceCE + RGA (IG).

## What Is Not Included

This release focuses on the RGA framework implementation. It does not include
datasets, trained weights, full nnU-Net/UNETR/TransBTS trainer classes,
preprocessing scripts, dataset-conversion utilities, experiment orchestration,
paper-revision utilities, or qualitative-figure generation code.

## Installation

The core RGA loss and LayerCAM helper require PyTorch:

```bash
pip install torch
```

The RGA (IG) configuration uses Captum:

```bash
pip install captum
```

No package installation step is required for this source release. The files can
be copied into a project or imported directly from the repository directory.

## Quick Start

Run the self-contained synthetic example:

```bash
python example_usage.py
```

The example trains a small 3D segmentation model over 10 deterministic synthetic
splits and reports mean +/- standard deviation of foreground DSC:

```text
Baseline: DiceCE only
  test DSC  = ...

Model + RGA (LayerCAM): DiceCE + lambda * RGA
  test DSC  = ...

Model + RGA (IG): DiceCE + lambda * RGA
  test DSC  = ...
```

This script demonstrates API usage and fair baseline/RGA integration. It is not
a benchmark and does not reproduce the paper experiments.

## Core Training Pattern

Baseline training is unchanged:

```python
logits = model(image)
base_loss = segmentation_loss(logits, target)
loss = base_loss
loss.backward()
optimizer.step()
```

RGA training keeps the same segmentation loss and adds the attribution penalty
on scheduled batches:

```python
from rga_module import RGARegularizer

rga = RGARegularizer(
    lambda_min=0.01,
    lambda_max=0.3,
    ramp_start=10,
    ramp_end=199,
    every_n_batches=8,
)

logits = model(image)
base_loss = segmentation_loss(logits, target)

loss, parts = rga.step_loss(
    base_loss=base_loss,
    model=model,
    image=image,
    target=target,
    logits=logits,
    epoch=epoch,
    batch_idx=batch_idx,
    make_saliency=make_saliency,
)

loss.backward()
optimizer.step()
```

`parts` contains detached diagnostic values such as `L_FN`, `L_FP`, `L_RGA`,
and `lambda_xai` when RGA is applied. On non-scheduled batches, `step_loss`
returns the original `base_loss` and an empty dictionary.

## Choosing RGA Parameters

The example script uses intentionally small values because the toy task is short
and runs for only a few epochs:

```python
RGARegularizer(
    lambda_min=0.01,
    lambda_max=0.03,
    ramp_start=2,
    ramp_end=EPOCHS - 1,
    every_n_batches=2,
)
```

For full training runs, the values used in the paper code are a more realistic
starting point:

```python
RGARegularizer(
    lambda_min=0.01,
    lambda_max=0.3,
    ramp_start=10,
    ramp_end=199,
    every_n_batches=8,
)
```

These parameters should be treated as hyperparameters, not constants. In
practice, vary them on a validation set while keeping the baseline training
setup unchanged. A small sweep is usually sufficient; a large grid search is not
required to use the module.

Useful ranges to try:

- `lambda_max`: controls the strength of RGA. Start around `0.1-0.3`; reduce it
  if RGA hurts precision or destabilizes training, and increase it only if the
  attribution penalty is too weak.
- `lambda_min`: usually small, e.g. `0.0-0.01`, to avoid over-regularizing early
  epochs.
- `ramp_start` and `ramp_end`: delay and smooth the RGA contribution. Start RGA
  after the segmentation loss has begun to decrease.
- `every_n_batches`: controls attribution cost. Smaller values apply RGA more
  often but are slower, especially for IG; larger values reduce overhead.

## LayerCAM Saliency

LayerCAM uses foreground logits and the activation tensor from a selected layer.
In a full trainer, this activation is usually captured with a forward hook.

```python
from rga_module import layercam_saliency

def make_saliency(model, image, logits):
    del image
    return layercam_saliency(
        foreground_logits=logits[:, 1],
        activation=selected_layer_activation,
        output_shape=logits.shape[2:],
    )
```

## Integrated Gradients Saliency

The IG configuration uses Captum on a selected target layer, matching the RGA
(IG) experimental setting. A zero image is used as the baseline.

```python
from rga_module import integrated_gradients_saliency

def make_saliency(model, image, logits):
    del logits
    return integrated_gradients_saliency(
        model=model,
        image=image,
        target_layer=model.feature_block,
        steps=5,
        output_shape=image.shape[2:],
    )
```

## Public API

```python
from rga_module import (
    RGARegularizer,
    integrated_gradients_saliency,
    lambda_schedule,
    layercam_saliency,
    normalize_saliency,
    rga_loss,
)
```

## Notes For Integration

- RGA is additive: it regularizes an existing segmentation objective and does
  not replace it.
- `target` can have shape `(B, D, H, W)` or `(B, 1, D, H, W)`.
- `logits` are expected to have shape `(B, C, D, H, W)` with foreground class
  index `1`.
- Saliency maps are detached inside `rga_loss`, so gradients flow through the
  segmentation probabilities but not through the attribution computation.
- Attribution maps are resized to the logit grid when needed.
- `every_n_batches` can reduce attribution overhead for expensive backends such
  as Integrated Gradients.

## Backbone Compatibility

The released code is model-independent, but the trainer must provide the target
layer used for attribution. The experimental integrations used the following
layers:

| Backbone | Target layer for attribution |
|---|---|
| nnU-Net v2 | `network.decoder.stages[-1]` |
| UNETR | `model.decoder2` |
| TransBTS | `model.DeBlock2` |

For LayerCAM, capture the activation of the selected layer during the same
forward pass and pass it to `layercam_saliency`. For IG, pass the selected layer
as `target_layer` to `integrated_gradients_saliency`.

The helper functions support standard 3D binary segmentation outputs with shape
`(B, C, D, H, W)` or an equivalent 5D spatial ordering, foreground class index
`1`, and optional auxiliary outputs returned as `list` or `tuple`. nnU-Net-style
`decoder.deep_supervision` is temporarily disabled during IG attribution when
that attribute is present. Other 3D binary segmentation models can be used by
providing an appropriate target layer and activation hook.

## License

This source code is released under the MIT License. See `LICENSE` for details.
