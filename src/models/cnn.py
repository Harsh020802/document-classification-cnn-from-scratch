"""DocCNN -- the from-scratch document classifier.

Backbone and head are deliberately separate attributes. Week 2's detection head plugs
into the same backbone without touching it, which is why `forward_features` exists as
a public method.

No pretrained weights, no torchvision.models. Every layer is defined here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import ConvBlock, build_head


class DocCNN(nn.Module):
    """VGG-style CNN for document image classification.

    Governing constraint: ~2,438 training images and no pretrained weights, so the
    dominant risk is overfitting, not lack of capacity. Hence a small model
    (~1.2M params), aggressive dropout, and a head that does not blow up the
    parameter count. For contrast, VGG11's classifier head alone is ~119.6M params.

    Args:
        in_channels: 1 (grayscale -- scans carry no colour signal, and the source
            images are already PIL mode L).
        block_channels: one entry per block; depth is a config value so Week 2 can
            try 5 blocks without a code change.
        num_classes: 10.
        dropout: 0.5.
        head_type: gap1 | gap1_hidden | gap3 | gap7 -- the ablation variable.
        hidden_dim: only used by gap1_hidden.
    """

    def __init__(self, in_channels: int = 1,
                 block_channels: tuple[int, ...] = (32, 64, 128, 256),
                 num_classes: int = 10, dropout: float = 0.5,
                 head_type: str = "gap3", hidden_dim: int = 86) -> None:
        super().__init__()
        self.block_channels = tuple(block_channels)
        self.head_type = head_type

        blocks, cin = [], in_channels
        for cout in block_channels:
            blocks.append(ConvBlock(cin, cout))
            cin = cout
        self.backbone = nn.Sequential(*blocks)
        self.head = build_head(head_type, cin, num_classes, dropout, hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal init for conv layers.

        Why not PyTorch's default: the default assumes leaky_relu with a=sqrt(5),
        which under-scales the variance for plain ReLU. Kaiming with
        nonlinearity='relu' keeps activation variance roughly constant through depth,
        so the signal neither vanishes nor explodes across 8 conv layers. This matters
        more here than in a pretrained model, where initialisation is overwritten.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone only. Public so Week 2's detection head can reuse it."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns RAW LOGITS -- no softmax.

        CrossEntropyLoss expects logits and applies log_softmax internally. Putting a
        softmax here would be a silent bug: it trains, just badly. It also keeps
        class weighting a one-argument change.
        """
        return self.head(self.backbone(x))

    def count_parameters(self) -> dict[str, int]:
        """Trainable parameters, split backbone vs head."""
        bb = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        hd = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {"backbone": bb, "head": hd, "total": bb + hd}


def build_model(cfg: dict) -> DocCNN:
    m = cfg["model"]
    return DocCNN(
        in_channels=m["in_channels"],
        block_channels=tuple(m["block_channels"]),
        num_classes=cfg["data"]["num_classes"],
        dropout=m["dropout"],
        head_type=m["head"]["type"],
        hidden_dim=m["head"]["hidden_dim"],
    )
