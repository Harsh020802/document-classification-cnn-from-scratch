"""Phase 1.5 -- a full training run.

Flat and readable (D-003): the epoch loop is right here, not inherited from a base
class. Logs to CSV (not TensorBoard) so runs are diffable across ablation arms.

Usage:
  PYTHONPATH=. python scripts/09_train.py
  PYTHONPATH=. python scripts/09_train.py --head gap1 --epochs 5 --seed 1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets, class_weights
from src.engine.checkpoint import save_checkpoint
from src.engine.metrics import confusion, per_class_report
from src.engine.train import epoch_time, evaluate, train_one_epoch
from src.models.cnn import build_model
from src.utils.device import device_report, get_device
from src.utils.seed import set_seed

CLASSES = ["ADVE", "Email", "Form", "Letter", "Memo",
           "News", "Note", "Report", "Resume", "Scientific"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="configs/default.yaml")
    p.add_argument("--head", default=None,
                   choices=["gap1", "gap1_hidden", "gap3", "gap7"],
                   help="override model.head.type -- the ablation variable")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--tag", default="", help="suffix for the run directory name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load((REPO / args.config).read_text())

    # CLI overrides the config file; the config file is the source of truth otherwise.
    if args.head is not None:
        cfg["model"]["head"]["type"] = args.head
    if args.epochs is not None:
        cfg["trainer"]["epochs"] = args.epochs
        cfg["lr_scheduler"]["t_max"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch_size is not None:
        cfg["data"]["loader"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["loader"]["num_workers"] = args.num_workers

    set_seed(cfg["seed"], cfg["deterministic"])
    device = get_device()

    run_id = args.run_id or datetime.now().strftime("%m%d_%H%M%S")
    name = f"{cfg['model']['head']['type']}_seed{cfg['seed']}{('_' + args.tag) if args.tag else ''}"
    run_dir = REPO / cfg["trainer"]["save_dir"] / "runs" / f"{run_id}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"run     : {run_dir.relative_to(REPO)}")
    print(f"device  : {device_report(device)}")
    print(f"head    : {cfg['model']['head']['type']}   seed: {cfg['seed']}")

    train_ds, val_ds, test_ds = build_datasets(cfg, REPO)
    ld = cfg["data"]["loader"]
    common = dict(batch_size=ld["batch_size"], num_workers=ld["num_workers"],
                  pin_memory=ld["pin_memory"])
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = build_model(cfg).to(device)
    p = model.count_parameters()
    print(f"params  : backbone {p['backbone']:,}  head {p['head']:,}  "
          f"total {p['total']:,}")
    print(f"data    : train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

    w = class_weights(train_ds.class_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"],
                                  weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["lr_scheduler"]["t_max"])

    mode, metric = cfg["trainer"]["monitor"].split()
    best = -float("inf") if mode == "max" else float("inf")
    patience = cfg["trainer"]["early_stop"]
    stale = 0
    arch = {"type": "DocCNN", **cfg["model"]}

    csv_path = run_dir / "metrics.csv"
    fields = ["epoch", "lr", "train_loss", "train_acc", "train_macro_f1",
              "val_loss", "val_acc", "val_macro_f1", "epoch_sec"]
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()

    print(f"\n{'ep':>3} {'lr':>9} | {'tr loss':>8} {'tr acc':>7} {'tr F1':>7} | "
          f"{'va loss':>8} {'va acc':>7} {'va F1':>7} | {'time':>8}")
    print("-" * 82)

    t_start = time.perf_counter()
    per_epoch: list[float] = []
    for epoch in range(1, cfg["trainer"]["epochs"] + 1):
        t0 = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.perf_counter() - t0
        per_epoch.append(dt)

        print(f"{epoch:>3} {lr_now:>9.2e} | {tr['loss']:>8.4f} {tr['acc']:>7.4f} "
              f"{tr['macro_f1']:>7.4f} | {va['loss']:>8.4f} {va['acc']:>7.4f} "
              f"{va['macro_f1']:>7.4f} | {epoch_time(0, dt):>8}", end="")

        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fields).writerow({
                "epoch": epoch, "lr": lr_now,
                "train_loss": tr["loss"], "train_acc": tr["acc"],
                "train_macro_f1": tr["macro_f1"],
                "val_loss": va["loss"], "val_acc": va["acc"],
                "val_macro_f1": va["macro_f1"], "epoch_sec": dt,
            })

        score = va[metric.replace("val_", "")]
        improved = score > best if mode == "max" else score < best
        if improved:
            best, stale = score, 0
            save_checkpoint(run_dir / "best.pt", model=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch, monitor_best=best,
                            config=cfg, arch=arch)
            print("  * best")
        else:
            stale += 1
            print()
            if stale >= patience:
                print(f"\nearly stop: no improvement in {patience} epochs")
                break

    save_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=epoch, monitor_best=best,
                    config=cfg, arch=arch)

    total = time.perf_counter() - t_start
    print(f"\ntrained {epoch} epochs in {epoch_time(0, total)}  "
          f"(median {np.median(per_epoch):.1f}s/epoch)")
    print(f"best val {metric} = {best:.4f}")

    # Final test evaluation, from the BEST checkpoint -- not the last.
    from src.engine.checkpoint import load_checkpoint
    load_checkpoint(run_dir / "best.pt", model=model, map_location=device)
    te, y_true, y_pred, probs = evaluate(model, test_loader, criterion, device,
                                         return_predictions=True)
    print(f"\nTEST  loss {te['loss']:.4f}  acc {te['acc']:.4f}  "
          f"macro-F1 {te['macro_f1']:.4f}")

    rep = per_class_report(y_true, y_pred, CLASSES)
    print(f"\n  {'class':<12}{'n':>5}{'prec':>8}{'recall':>8}{'F1':>8}")
    print("  " + "-" * 41)
    for c in CLASSES:
        r = rep[c]
        print(f"  {c:<12}{r['support']:>5}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}")

    np.save(run_dir / "test_predictions.npy",
            {"y_true": y_true, "y_pred": y_pred, "probs": probs}, allow_pickle=True)
    (run_dir / "results.json").write_text(json.dumps({
        "head": cfg["model"]["head"]["type"], "seed": cfg["seed"],
        "epochs_run": epoch, f"best_val_{metric}": best,
        "test_loss": te["loss"], "test_acc": te["acc"],
        "test_macro_f1": te["macro_f1"],
        "per_class": rep,
        "confusion": confusion(y_true, y_pred).tolist(),
        "median_epoch_sec": float(np.median(per_epoch)),
        "total_sec": total, "params": p,
    }, indent=2))
    print(f"\nsaved -> {run_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
