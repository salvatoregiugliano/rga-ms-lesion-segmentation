# Copyright (c) 2026 Salvatore Giugliano
# SPDX-License-Identifier: MIT

"""Small train/test example showing how to add RGA to DiceCE training.

The script trains the same toy 3D segmentation model on deterministic synthetic
volumes: once with Dice + cross-entropy only, and once with Dice + cross-entropy
plus RGA. The comparison is repeated over multiple seeds and reported as
mean +/- standard deviation. This is an API example, not a benchmark or a
reproduction of the paper results.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Captum may import Matplotlib internally; set a writable cache path so the
# example output stays clean on systems where the home config directory is read-only.
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib-rga-example"),
)

from rga_module import (
    RGARegularizer,
    integrated_gradients_saliency,
    layercam_saliency,
)

VOLUME_SIZE = 20
LESION_SIZE = 3
TRAIN_CASES = 20
TEST_CASES = 10
EPOCHS = 10
NUM_RUNS = 10


class ToySegmentationModel(torch.nn.Module):
    """Small binary 3D segmentation model used only for this example."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_block = torch.nn.Sequential(
            torch.nn.Conv3d(1, 6, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False),
        )
        self.classifier = torch.nn.Conv3d(6, 2, kernel_size=1)

        # LayerCAM uses this activation tensor as the target layer output.
        self.last_activation: torch.Tensor | None = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.feature_block(image)

        # A real trainer would usually capture this through a forward hook.
        self.last_activation = features
        return self.classifier(features)


@dataclass
class Metrics:
    dsc: float


def make_toy_dataset(
    num_cases: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic 3D cases with one small lesion and distractors."""
    # Fixed seeds make each run reproducible while still allowing multiple
    # independent train/test splits below.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    grid = torch.linspace(-1.0, 1.0, VOLUME_SIZE, device=device)
    zz, yy, xx = torch.meshgrid(grid, grid, grid, indexing="ij")

    # Smooth low-amplitude background prevents the task from being pure noise.
    background = 0.04 * (xx + yy + zz)

    images = []
    targets = []
    low = 2
    high = VOLUME_SIZE - LESION_SIZE - 2

    for _ in range(num_cases):
        image = background.clone().unsqueeze(0)
        target = torch.zeros(
            VOLUME_SIZE,
            VOLUME_SIZE,
            VOLUME_SIZE,
            dtype=torch.long,
            device=device,
        )

        lesion_origin = [
            int(torch.randint(low, high + 1, (1,), generator=generator).item())
            for _ in range(3)
        ]
        lesion = tuple(slice(start, start + LESION_SIZE) for start in lesion_origin)
        target[lesion] = 1

        # Lesion contrast varies across cases, closer to a real segmentation
        # setting than a single fixed intensity threshold.
        lesion_intensity = 0.85 + 0.15 * torch.rand(
            (1,),
            generator=generator,
            device=device,
        ).item()
        image[(slice(None),) + lesion] = lesion_intensity

        # Bright but unlabelled distractors make the toy task less
        # intensity-threshold-like while keeping the example self-contained.
        for _ in range(2):
            for _ in range(20):
                distractor_origin = [
                    int(torch.randint(low, high + 1, (1,), generator=generator).item())
                    for _ in range(3)
                ]
                distance = sum(
                    abs(a - b) for a, b in zip(lesion_origin, distractor_origin)
                )
                if distance > 2 * LESION_SIZE:
                    break
            distractor = tuple(
                slice(start, start + LESION_SIZE) for start in distractor_origin
            )
            distractor_intensity = 0.55 + 0.30 * torch.rand(
                (1,),
                generator=generator,
                device=device,
            ).item()
            image[(slice(None),) + distractor] = distractor_intensity

        # Add mild noise after lesions and distractors are inserted.
        image += 0.025 * torch.randn(
            image.shape,
            generator=generator,
            device=device,
        )
        images.append(image)
        targets.append(target)

    return torch.stack(images), torch.stack(targets)


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Foreground soft Dice loss for binary 3D segmentation."""
    # The example uses the same differentiable foreground probability that RGA
    # regularizes in rga_loss.py.
    foreground_prob = torch.softmax(logits, dim=1)[:, 1]
    foreground_target = (target == 1).float()
    intersection = (foreground_prob * foreground_target).flatten(1).sum(dim=1)
    denominator = foreground_prob.flatten(1).sum(dim=1)
    denominator = denominator + foreground_target.flatten(1).sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard Dice + cross-entropy segmentation objective."""
    # Baseline training uses only this objective; RGA training adds a scheduled
    # attribution penalty on top of it.
    return soft_dice_loss(logits, target) + F.cross_entropy(logits, target)


def evaluate(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
) -> Metrics:
    """Compute foreground DSC on a split."""
    with torch.no_grad():
        # Hard argmax segmentation is used only for reporting DSC.
        prediction = model(images).argmax(dim=1)
        tp = ((prediction == 1) & (targets == 1)).sum().item()
        fp = ((prediction == 1) & (targets == 0)).sum().item()
        fn = ((prediction == 0) & (targets == 1)).sum().item()

    dsc = 2 * tp / max(1, 2 * tp + fp + fn)
    return Metrics(dsc=dsc)


def train_model(
    rga_method: str | None,
    train_data: tuple[torch.Tensor, torch.Tensor],
    test_data: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    model_seed: int,
    shuffle_seed: int,
) -> tuple[Metrics, Metrics, dict[str, float]]:
    """Train a baseline model or an RGA-regularized model."""
    if rga_method not in {None, "layercam", "ig"}:
        raise ValueError("rga_method must be None, 'layercam', or 'ig'")

    # The same model_seed is passed to baseline, LayerCAM-RGA, and IG-RGA for a
    # fair within-run comparison.
    torch.manual_seed(model_seed)
    model = ToySegmentationModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    rga = RGARegularizer(
        lambda_min=0.01,
        lambda_max=0.03,
        ramp_start=2,
        ramp_end=EPOCHS - 1,
        every_n_batches=2,
    )

    train_images, train_targets = train_data
    last_parts: dict[str, float] = {}

    for epoch in range(EPOCHS):
        # The shuffle seed is also shared across methods within the same run.
        permutation = torch.randperm(
            train_images.shape[0],
            generator=torch.Generator().manual_seed(shuffle_seed + epoch),
        )
        for batch_idx, case_idx in enumerate(permutation):
            image = train_images[case_idx : case_idx + 1]
            target = train_targets[case_idx : case_idx + 1]

            optimizer.zero_grad(set_to_none=True)
            logits = model(image)

            # Baseline-only training uses this objective directly.
            base_loss = dice_ce_loss(logits, target)

            if rga_method is not None:

                def make_saliency(
                    model: torch.nn.Module,
                    image: torch.Tensor,
                    logits: torch.Tensor,
                ) -> torch.Tensor:
                    if rga_method == "layercam":
                        del image
                        if model.last_activation is None:
                            raise RuntimeError("missing target-layer activation")
                        # LayerCAM consumes the foreground logits and the saved
                        # target-layer activation from the current forward pass.
                        return layercam_saliency(
                            foreground_logits=logits[:, 1],
                            activation=model.last_activation,
                            output_shape=logits.shape[2:],
                        )

                    # RGA (IG) uses Captum Integrated Gradients on the same
                    # target layer. This is slower than LayerCAM but keeps the
                    # example close to the paper's IG configuration.
                    return integrated_gradients_saliency(
                        model=model,
                        image=image,
                        target_layer=model.feature_block,
                        steps=5,
                        output_shape=logits.shape[2:],
                    )

                # RGA adds lambda * (L_FN + L_FP) to the same DiceCE loss on
                # scheduled batches.
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
                if parts:
                    last_parts = {
                        key: float(value)
                        for key, value in parts.items()
                    }
            else:
                loss = base_loss

            loss.backward()
            optimizer.step()

    return evaluate(model, *train_data), evaluate(model, *test_data), last_parts


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return sample mean and standard deviation for printed summaries."""
    tensor = torch.as_tensor(values, dtype=torch.float64)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=True).item()) if tensor.numel() > 1 else 0.0
    return mean, std


def format_mean_std(values: list[float]) -> str:
    """Format a list of metric values as `mean +/- std`."""
    mean, std = mean_std(values)
    return f"{mean:.4f} +/- {std:.4f}"


def print_summary(label: str, train_runs: list[Metrics], test_runs: list[Metrics]) -> None:
    print(label)
    print(f"  train DSC = {format_mean_std([m.dsc for m in train_runs])}")
    print(f"  test DSC  = {format_mean_std([m.dsc for m in test_runs])}")


def main() -> None:
    # CPU keeps the example deterministic and quick on ordinary machines.
    device = torch.device("cpu")
    torch.set_num_threads(1)

    print("Toy DiceCE vs DiceCE + RGA training example")
    print(f"  train cases = {TRAIN_CASES}, test cases = {TEST_CASES}")
    print(f"  volume size = {VOLUME_SIZE}^3, lesion size = {LESION_SIZE}^3")
    print(f"  epochs = {EPOCHS}, batch size = 1")
    print(f"  repeated runs = {NUM_RUNS}")
    print("  RGA attribution = LayerCAM and Integrated Gradients (IG)")
    print("  RGA lambda = 0.01 -> 0.03, every 2 batches")
    print("  baseline and RGA use the same initialization within each run")
    print()

    baseline_train_runs: list[Metrics] = []
    baseline_test_runs: list[Metrics] = []
    layercam_train_runs: list[Metrics] = []
    layercam_test_runs: list[Metrics] = []
    ig_train_runs: list[Metrics] = []
    ig_test_runs: list[Metrics] = []

    print("Per-run test DSC")
    print("  run  baseline  RGA-LayerCAM  RGA-IG")
    for run_idx in range(NUM_RUNS):
        # Each run gets a different synthetic split, but the three methods
        # inside that run see exactly the same split and initialization.
        train_data = make_toy_dataset(TRAIN_CASES, seed=10 + 17 * run_idx, device=device)
        test_data = make_toy_dataset(TEST_CASES, seed=100 + 19 * run_idx, device=device)
        model_seed = 7 + 101 * run_idx
        shuffle_seed = 1000 + 53 * run_idx

        baseline_train, baseline_test, _ = train_model(
            rga_method=None,
            train_data=train_data,
            test_data=test_data,
            device=device,
            model_seed=model_seed,
            shuffle_seed=shuffle_seed,
        )
        layercam_train, layercam_test, _ = train_model(
            rga_method="layercam",
            train_data=train_data,
            test_data=test_data,
            device=device,
            model_seed=model_seed,
            shuffle_seed=shuffle_seed,
        )
        ig_train, ig_test, _ = train_model(
            rga_method="ig",
            train_data=train_data,
            test_data=test_data,
            device=device,
            model_seed=model_seed,
            shuffle_seed=shuffle_seed,
        )

        baseline_train_runs.append(baseline_train)
        baseline_test_runs.append(baseline_test)
        layercam_train_runs.append(layercam_train)
        layercam_test_runs.append(layercam_test)
        ig_train_runs.append(ig_train)
        ig_test_runs.append(ig_test)

        print(
            f"  {run_idx + 1:02d}   {baseline_test.dsc:.4f}    "
            f"{layercam_test.dsc:.4f}        {ig_test.dsc:.4f}"
        )

    print()
    print_summary("Baseline: DiceCE only", baseline_train_runs, baseline_test_runs)
    print()
    print_summary(
        "Model + RGA (LayerCAM): DiceCE + lambda * RGA",
        layercam_train_runs,
        layercam_test_runs,
    )
    print()
    print_summary(
        "Model + RGA (IG): DiceCE + lambda * RGA",
        ig_train_runs,
        ig_test_runs,
    )
    print()
    print("Note: this synthetic example demonstrates API usage, not benchmark performance.")


if __name__ == "__main__":
    main()
