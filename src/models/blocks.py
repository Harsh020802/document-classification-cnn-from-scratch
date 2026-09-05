"""Reusable building blocks: the conv block, and the four ablation heads.

The heads live here rather than inside DocCNN so the ablation is one config value
(`model.head.type`) selecting one factory function -- nothing else in the model can
drift between arms.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """[Conv3x3 -> BN -> ReLU] x2 -> MaxPool(2).

    Why two 3x3 convs rather than one 5x5: identical receptive field (5x5), fewer
    parameters (2*9=18 vs 25 per channel pair), and an extra non-linearity between
    them. This is the core VGG insight.

    Why BatchNorm: normalises each layer's inputs to roughly zero-mean/unit-variance
    across the batch, which permits higher learning rates without divergence and acts
    as a mild regulariser -- each sample's normalisation depends on its batch-mates, a
    small random perturbation. On 2,438 training images that regularisation is free
    and welcome. (This is the same Conv+BN+activation unit YOLOv8 uses everywhere.)

    Why MaxPool rather than a strided conv: parameter-free, and "is there a strong
    edge/text-block in this 2x2 window" is the right question for documents.

    Why bias=False on the convs: BatchNorm immediately subtracts the batch mean, which
    cancels any constant the conv bias adds. The bias term would be dead parameters.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Pool3x3MPS(nn.Module):
    """3x3 average pooling that runs on Apple's MPS backend.

    nn.AdaptiveAvgPool2d(3) is the operation we WANT, but it raises on MPS when the
    input is not divisible by the output size -- and our block-4 map is 14x14, with
    14 % 3 != 0. PYTORCH_ENABLE_MPS_FALLBACK=1 does NOT help: the op is implemented on
    MPS and explicitly rejects the input, so the fallback dispatcher never engages
    (measured, D-026).

    Fix: replicate-pad 14 -> 15, then a uniform non-overlapping AvgPool2d(5, 5).
      - full coverage (15/15), no dropped rows or columns
      - no double-counting
      - 0.9993 correlation with the true op on real activations
      - costs +0.011 s/epoch

    An earlier workaround, AvgPool2d(5, 4), was rejected: it dropped row and column 13,
    i.e. the bottom and right edges of the page -- where signature blocks and footers
    sit. That is precisely the spatial signal arm C exists to test, and dropping it
    would have put a confound inside the pre-registered primary comparison (D-025/D-026).

    Replicate rather than zero padding: zeros would inject artificial black pixels into
    a white-page region.
    """

    def __init__(self, out_size: int = 3) -> None:
        super().__init__()
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % self.out_size == 0 and w % self.out_size == 0:
            # Divisible -- use the exact op (this path is taken on CUDA/CPU too).
            return F.adaptive_avg_pool2d(x, self.out_size)
        # Pad up to the next multiple, then pool uniformly.
        th = ((h + self.out_size - 1) // self.out_size) * self.out_size
        tw = ((w + self.out_size - 1) // self.out_size) * self.out_size
        x = F.pad(x, (0, tw - w, 0, th - h), mode="replicate")
        return F.avg_pool2d(x, kernel_size=(th // self.out_size, tw // self.out_size))


def build_head(head_type: str, channels: int, num_classes: int,
               dropout: float, hidden_dim: int = 86) -> nn.Module:
    """The single switch that defines an ablation arm.

    Arms A and B carry NO spatial information (both pool to 1x1). Arms C and D keep a
    3x3 / 7x7 grid. B exists to match C's parameter count so the comparison isolates
    spatial information rather than head capacity:

        A gap1              1x1, no spatial       2,570 params
        B gap1_hidden(86)   1x1, no spatial      22,972 params  <-- matched to C
        C gap3              3x3 grid             23,050 params  <-- primary vs B
        D gap7              7x7 grid            125,450 params

    h=86 solves the capacity match exactly (86.29); 88 would be 2.0% off.

    Caveat carried from D-014: arm B has an extra ReLU that C lacks -- matched on
    parameters, not depth. Unavoidable (capacity cannot be added to a 256-vector
    without a layer), and it cuts in a direction that does not undermine either
    conclusion: a B~C tie strengthens the capacity explanation, a C win holds despite
    B's slight edge.
    """
    if head_type == "gap1":
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(channels, num_classes),
        )
    if head_type == "gap1_hidden":
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )
    if head_type == "gap3":
        return nn.Sequential(
            Pool3x3MPS(3), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(channels * 9, num_classes),
        )
    if head_type == "gap7":
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(7), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(channels * 49, num_classes),
        )
    raise ValueError(f"unknown head type {head_type!r}; "
                     "expected one of gap1, gap1_hidden, gap3, gap7")
