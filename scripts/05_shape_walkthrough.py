"""Phase 1.3 -- print REAL tensor shapes from an actual forward pass.

Not comments. Comments drift; a shape you watched is worth more than one you were
told. Given that the 14x14 divisibility problem produced D-024 through D-026, the
block-4 output in particular is confirmed here rather than assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F
import yaml

from src.models.blocks import Pool3x3MPS
from src.models.cnn import DocCNN, build_model
from src.utils.device import device_report, get_device
from src.utils.seed import set_seed

BS = 8


def rule(t: str = "") -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}" if t else "-" * 78)


def walk_backbone(model: DocCNN, x: torch.Tensor) -> torch.Tensor:
    """Forward hooks on each block, so every shape printed is one that really occurred."""
    rule("1. INPUT -> BLOCK 1 -> 2 -> 3 -> 4   (real shapes, captured by forward hooks)")
    seen: list[tuple[str, tuple[int, ...]]] = []
    handles = [
        blk.register_forward_hook(
            lambda m, i, o, n=f"block{k+1}": seen.append((n, tuple(o.shape)))
        )
        for k, blk in enumerate(model.backbone)
    ]
    with torch.no_grad():
        out = model.forward_features(x)
    for h in handles:
        h.remove()

    print(f"  {'stage':<10}{'shape (N,C,H,W)':<26}{'spatial':<12}{'channels':>9}{'elems/img':>12}")
    rule()
    print(f"  {'input':<10}{str(tuple(x.shape)):<26}{f'{x.shape[2]}x{x.shape[3]}':<12}"
          f"{x.shape[1]:>9}{x[0].numel():>12,}")
    for name, s in seen:
        print(f"  {name:<10}{str(s):<26}{f'{s[2]}x{s[3]}':<12}{s[1]:>9}"
              f"{s[1]*s[2]*s[3]:>12,}")
    return out


def walk_pool(feat: torch.Tensor) -> None:
    rule("2. BLOCK-4 OUTPUT -> PAD -> POOL   (the D-024/D-026 path, step by step)")
    h, w = feat.shape[-2:]
    print(f"  block4 output          : {tuple(feat.shape)}   spatial {h}x{w}")
    print(f"  is {h} divisible by 3?   : {h % 3 == 0}   -> AdaptiveAvgPool2d(3) "
          f"{'works' if h % 3 == 0 else 'RAISES on MPS'}")

    # Probe on EVERY available device: the restriction is MPS-specific, and running
    # this on a CPU tensor would silently succeed and hide the whole reason D-026 exists.
    for dname in ("cpu", "mps", "cuda"):
        if dname == "mps" and not torch.backends.mps.is_available():
            continue
        if dname == "cuda" and not torch.cuda.is_available():
            continue
        probe = feat[:2].to(dname)
        try:
            F.adaptive_avg_pool2d(probe, 3)
            print(f"  AdaptiveAvgPool2d(3) on {dname:<5}: OK")
        except RuntimeError as e:
            print(f"  AdaptiveAvgPool2d(3) on {dname:<5}: RAISES -- "
                  f"{str(e).split(chr(10))[0][:52]}...")

    th = ((h + 2) // 3) * 3
    padded = F.pad(feat, (0, th - w, 0, th - h), mode="replicate")
    print(f"  replicate-pad {h}->{th}    : {tuple(padded.shape)}")
    pooled = F.avg_pool2d(padded, kernel_size=th // 3)
    print(f"  AvgPool2d(k={th//3}, s={th//3})    : {tuple(pooled.shape)}   <- the 3x3 grid")
    print(f"  flatten                : {tuple(pooled.flatten(1).shape)}   "
          f"= {pooled.shape[1]} channels x 9 cells")

    same = Pool3x3MPS(3)(feat)
    print(f"\n  Pool3x3MPS module      : {tuple(same.shape)}  "
          f"max|diff| vs manual = {(same - pooled).abs().max():.2e}")

    rule()
    print("  RECEPTIVE FIELD -- what does each of the 9 cells actually see?\n")
    # r_out = r_in + (k-1)*jump_in ; jump_out = jump_in * stride
    r, j = 1, 1
    for _ in range(4):
        r += 2 * j              # conv 3x3 : (3-1)*jump
        r += 2 * j              # conv 3x3 : (3-1)*jump
        r += 1 * j              # maxpool 2: (2-1)*jump   <- must not be skipped
        j *= 2                  # maxpool stride 2
    print(f"    A single block-4 neuron has a theoretical receptive field of {r}x{r} px")
    print(f"    on the {224}px input -- {r/224:.1%} of the page. The EFFECTIVE receptive")
    print(f"    field is smaller still (~{int(r*0.4)}-{int(r*0.6)} px, Luo et al. 2016: the")
    print( "    gradient falls off roughly Gaussian from the centre).\n")
    print( "    So a block-4 neuron sees a PATCH, not a page. It can recognise 'dense")
    print( "    text block' or 'horizontal rule', but it cannot know that patch is at")
    print( "    the top of the page.\n")
    cell = 224 / 3
    print(f"    Each of the 9 cells averages a {th//3}x{th//3} region of the 15x15 padded map,")
    print(f"    which corresponds to roughly {cell:.0f}x{cell:.0f} px of the input image.")
    print( "    THAT is where global layout enters the model -- from the pooling grid,")
    print( "    not from the receptive field. Which is exactly why gap1, which averages")
    print( "    all 9 cells into 1, makes 'letterhead at top' unrepresentable (D-006).")


def head_table(cfg: dict, x: torch.Tensor) -> dict[str, dict]:
    rule("3-4. ALL FOUR HEADS -- output shapes and parameter counts, same backbone")
    arms = [("A", "gap1", "none"), ("B", "gap1_hidden", "none"),
            ("C", "gap3", "3x3"), ("D", "gap7", "7x7")]
    print(f"  {'arm':<5}{'head':<14}{'spatial':<9}{'head out':<14}"
          f"{'backbone':>11}{'head':>10}{'TOTAL':>12}")
    rule()
    res = {}
    for arm, ht, sp in arms:
        set_seed(cfg["seed"])
        m = DocCNN(head_type=ht, hidden_dim=cfg["model"]["head"]["hidden_dim"]).eval()
        with torch.no_grad():
            y = m(x)
        p = m.count_parameters()
        res[arm] = {"head": ht, "logits": tuple(y.shape),
                    "head_p": p["head"], "backbone_p": p["backbone"], "total_p": p["total"]}
        print(f"  {arm:<5}{ht:<14}{sp:<9}{str(tuple(y.shape)):<14}"
              f"{p['backbone']:>11,}{p['head']:>10,}{p['total']:>12,}")
    rule()
    b, c = res["B"]["head_p"], res["C"]["head_p"]
    print(f"  PRIMARY COMPARISON -- capacity match, MEASURED not asserted:")
    print(f"    B  gap1_hidden(86)  head = {b:>7,} params   spatial: NONE")
    print(f"    C  gap3             head = {c:>7,} params   spatial: 3x3 grid")
    print(f"    difference = {c - b:+,} params = {abs(c - b) / c:.2%} of C")
    print(f"    -> the two arms differ by <1% in capacity, so a difference in")
    print(f"       accuracy is attributable to spatial information, not head size.")
    print(f"\n    A  gap1  head = {res['A']['head_p']:>7,}  (capacity floor)")
    print(f"    D  gap7  head = {res['D']['head_p']:>7,}  ({res['D']['head_p']/c:.1f}x C)")
    return res


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])
    dev = get_device()
    print(f"device: {device_report(dev)}")

    size = cfg["data"]["input_size"]
    x = torch.randn(BS, cfg["model"]["in_channels"], size, size)

    model = build_model(cfg).eval()
    feat = walk_backbone(model, x)
    walk_pool(feat)
    head_table(cfg, x)
    print()


if __name__ == "__main__":
    main()
