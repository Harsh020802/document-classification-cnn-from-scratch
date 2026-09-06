"""Live view of every training run. Read-only, stdlib only, never imports torch.

    python scripts/watch.py            # refresh every 4 s
    python scripts/watch.py --once     # print one snapshot and exit
    python scripts/watch.py --every 10

Reads only outputs/runs/*/status.json and progress.jsonl. It cannot interfere with
training: no imports from the project, no writes, and every read is wrapped so a
half-written file during a flush degrades to a blank cell rather than a crash.

Ctrl-C exits cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs" / "runs"

BOLD, DIM, RED, GRN, YEL, OFF = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


def hms(sec: float) -> str:
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def read_json(p: Path):
    """Tolerate a torn read mid-flush."""
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def last_progress(run: Path) -> dict | None:
    p = run / "progress.jsonl"
    if not p.exists():
        return None
    try:
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception:
        return None


def collect() -> list[dict]:
    rows = []
    if not RUNS.exists():
        return rows
    for d in sorted(RUNS.iterdir()):
        st = read_json(d / "status.json")
        if st is None:
            # A run from before the logging existed, or mid-write. Show what we can.
            if (d / "metrics.csv").exists():
                rows.append({"arm": d.name, "seed": "", "epoch": "", "total_epochs": "",
                             "state": "legacy", "best_val_macro_f1": None,
                             "best_epoch": "", "start_time": None, "run_dir": d.name})
            continue
        st["_last"] = last_progress(d)
        rows.append(st)
    return rows


def render(rows: list[dict]) -> str:
    now = time.time()
    out = [f"{BOLD}ablation progress{OFF}   {time.strftime('%H:%M:%S')}   "
           f"{len(rows)} run(s)", ""]
    hdr = (f"  {'arm':<13}{'seed':>5}{'epoch':>10}{'val F1':>9}"
           f"{'best':>9}{'@ep':>5}{'s/ep':>7}{'elapsed':>9}{'eta':>9}  state")
    out += [hdr, "  " + "-" * (len(hdr) - 2)]

    for r in rows:
        state = r.get("state", "?")
        lp = r.get("_last") or {}
        ep, tot = r.get("epoch", ""), r.get("total_epochs", "")
        cur = lp.get("val_macro_f1")
        best = r.get("best_val_macro_f1")
        sec = lp.get("epoch_sec")
        elapsed = now - r["start_time"] if r.get("start_time") else None

        eta = ""
        if state == "running" and sec and isinstance(ep, int) and isinstance(tot, int) and ep:
            eta = hms(sec * (tot - ep))

        colour = {"running": YEL, "done": GRN, "failed": RED}.get(state, DIM)
        star = "*" if lp.get("new_best") else " "
        out.append(
            f"  {str(r.get('arm','')):<13}{str(r.get('seed','')):>5}"
            f"{f'{ep}/{tot}':>10}"
            f"{(f'{cur:.4f}' if cur is not None else '-'):>9}{star}"
            f"{(f'{best:.4f}' if best is not None else '-'):>8}"
            f"{str(r.get('best_epoch','')):>5}"
            f"{(f'{sec:.0f}' if sec else '-'):>7}"
            f"{(hms(elapsed) if elapsed else '-'):>9}"
            f"{eta:>9}  {colour}{state}{OFF}")

    done = sum(1 for r in rows if r.get("state") == "done")
    fail = sum(1 for r in rows if r.get("state") == "failed")
    run_ = sum(1 for r in rows if r.get("state") == "running")
    out += ["", f"  {done} done   {run_} running   {fail} failed"]
    if fail:
        out.append(f"  {RED}check the failed run's stdout{OFF}")
    out.append(f"\n  {DIM}* = new best this epoch. Ctrl-C to exit.{OFF}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=4.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    signal.signal(signal.SIGINT, lambda *_: (print("\n"), sys.exit(0)))
    while True:
        rows = collect()
        if a.once:
            print(render(rows))
            return
        os.system("clear")
        print(render(rows))
        time.sleep(a.every)


if __name__ == "__main__":
    main()
