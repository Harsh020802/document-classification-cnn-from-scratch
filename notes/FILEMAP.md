# Where everything lives

Absolute paths, measured sizes, and whether a fresh `git clone` would give it to you.
Project root: `/Users/harshagarwal/Desktop/ML_Model`

**NOTE: this is not a git repository yet** (`git init` has not been run). The
tracked/ignored column below describes what *would* happen given the existing
`.gitignore`.

---

## Raw dataset

| path | size | tracked? |
|---|---|---|
| `/Users/harshagarwal/Desktop/ML_Model/data/raw/` | **1.7 GB** | ignored |
| `~/.cache/huggingface/` | **1.8 GB** | outside the repo entirely |

The HuggingFace fetch lands in **two** places. `data/raw/` holds the Arrow-converted copy
(three shards: 970 MB + 666 MB + 100 MB) under
`data/raw/maveriq___tobacco3482/default/0.0.0/<hash>/`. The download cache in
`~/.cache/huggingface/` is a *separate* 1.8 GB that is no longer needed — deleting it
recovers that space, and `scripts/01_inspect_data.py` would re-fetch if you ever needed it.

## Cache and derived arrays

| path | size | what it is | tracked? |
|---|---|---|---|
| `data/cache/images.npy` | **217.6 MB** | 3,482 × 256 × 256 uint8, resized + white-padded | ignored |
| `data/cache/labels.npy` | 27.3 KB | int64 class index per image | ignored |
| `data/cache/boxes.npy` | 27.3 KB | (left, top, w, h) int16 — where real content sits inside the padded square | ignored |
| `outputs/norm_stats.json` | 284 B | train-split mean/std, content pixels only **and** the padded pair for comparison | ignored |

`boxes.npy` replaced a 217.6 MB per-pixel mask (D-032) — four numbers per image instead of
65,536.

## Splits

| path | size | what it is | tracked? |
|---|---|---|---|
| `data/splits/splits.json` | 33.8 KB | train/val/test index lists (2,438 / 523 / 521) | ignored |
| `data/splits/phash_groups.npy` | 27.3 KB | near-duplicate cluster id per image; the split grouping key | ignored |

## Checkpoints

Both live in the run directory `outputs/runs/0906_002054_gap3_seed0_first/`:

| path | size | which is which | tracked? |
|---|---|---|---|
| `best.pt` | **14.4 MB** | **epoch 16** — highest val macro-F1 (0.7857). This is the one `predict.py` loads and the one the test score came from | ignored (`outputs/`, and `*.pt`) |
| `last.pt` | 14.4 MB | epoch 26, the final state when early stopping fired. For resuming, not for inference | ignored |

14.4 MB not 4.8 MB because each holds weights **plus** AdamW's `exp_avg` and `exp_avg_sq` —
two full-size buffers per parameter — plus the arch config, scheduler state and
`monitor_best` (D-003).

## Run artifacts

All under `outputs/runs/0906_002054_gap3_seed0_first/`:

| file | size | contents |
|---|---|---|
| `metrics.csv` | 4.2 KB | per-epoch loss/acc/macro-F1 for train and val, plus epoch seconds |
| `results.json` | 2.9 KB | final test metrics, per-class P/R/F1, confusion matrix, timing, param counts |
| `test_predictions.npy` | 29.7 KB | `y_true`, `y_pred`, and the full 521 × 10 softmax matrix |
| `error_analysis.json` | 2.3 KB | the 12 most-confident errors, class-size/F1 correlation |
| `error_triage.json` | 9.8 KB | the 30-error hand triage with per-item verdicts and reasoning |
| `config.yaml` | 1.1 KB | the exact config this run used, snapshotted |

## Figures — `outputs/figures/`

| file | size | what it shows |
|---|---|---|
| `01_squash_vs_pad.png` | 569 KB | why pad-to-square beats a direct resize (D-007) |
| `02_augmentation_fill.png` | 503 KB | `fill=0` black corner wedges vs `fill=255` (D-008) |
| `03_cross_label_cluster.png` | 109 KB | the 5 "IMAGE NOT AVAILABLE" placeholder pages (D-033) |
| `04_confusion_matrix.png` | 128 KB | counts and row-normalised (recall per class) |
| `05_confident_errors.png` | 520 KB | the 12 most-confident errors with predicted/true probabilities |
| `06_training_curves.png` | 144 KB | loss, accuracy, macro-F1, and the train−val overfitting gap |

## Baselines and analysis

| path | size | contents |
|---|---|---|
| `outputs/baselines.json` | 3.7 KB | all three baselines: accuracy, macro-F1, selected C, per-class breakdown |
| `outputs/timing.json` | ~200 B | the four cache × num_workers timings |

There is one baseline **file**, not three — it holds all three models. Majority class 17.85%,
logreg 8×8 58.73%, logreg 32×32 61.61%.

---

## What survives a fresh clone

**Tracked** (everything needed to regenerate the rest): `src/`, `scripts/`, `tests/`,
`configs/`, `notes/`, `README.md`, `requirements.txt`, `watch_run.sh`, `.gitignore`, and the
`.gitkeep` files that preserve the empty directory layout.

**Ignored** — you would have to regenerate: `data/` (1.9 GB), `outputs/` (~30 MB),
`reference/` (the two reference repos, kept locally through Week 2).

**Regeneration cost from a clean clone:** ~4 minutes to download the dataset, ~1 minute for
dedup and split, 37 seconds for the cache, 2 seconds for baselines, 16 minutes to train.
Roughly **25 minutes end to end**, and every step is deterministic given the seed except for
MPS float non-determinism (D-021).

**The one thing that would NOT come back identically:** `best.pt`. MPS makes no
bit-reproducibility guarantee, so a re-run produces a very similar but not identical model.
That is exactly why the seed-variance floor is measured rather than assumed.
