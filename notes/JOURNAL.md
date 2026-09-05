# Project Journal

Append-only lab notebook. Newest at the bottom. Rough notes are fine. Failed runs,
tracebacks and dead ends belong here — they are the part that cannot be reconstructed later.

---

## 2026-09-05 — Phase 0: R&D

**Environment detection.** MacBook Air M2 (Mac14,2), 8 GB unified memory, 8 CPU cores
(4P+4E), 8 GPU cores, macOS 26.2. No NVIDIA. Disk 90% full — 21.5 GB free at first check,
22.7 GB after the user cleared some. Swap already at 5.1/6 GB before any work started, which
is the real constraint on this machine, more than disk.

Found three Python installs (python.org 3.13, miniconda base 3.13, conda `genai`) plus a
stray venv. PyTorch installed in none of them. Decided on a fresh conda env at Python 3.11 —
3.13 is new enough that ML wheels still lag, and debugging a wheel build is not how to spend
a Saturday.

**Verdict:** the M2 is sufficient. No cloud GPU needed. ~3.5k images against a 1.2M-param
model is small enough that a rented H100 would sit idle waiting on the dataloader.

**Dataset research.** Compared Tobacco-3482, RVL-CDIP, DocLayNet, PubLayNet, FUNSD, SROIE,
CORD. RVL-CDIP is 38.8 GB against 21 GB free — out on disk before time even enters.
Recommended Tobacco-3482 via HuggingFace (`maveriq/tobacco3482`) over Kaggle to skip the API
token setup.

Found the WACV 2025 label-error paper: 11.7% of Tobacco-3482 improperly annotated, 16.7% with
multiple valid labels. Sets a ~88% ceiling. Better to know this now than to spend Sunday
wondering why the model plateaus at 75%.

**Phase 0.4 review — the long one.** The plan went through five rounds of user review. Most
of the value in this project so far came out of that argument rather than out of the plan.
Substantive things that changed, all logged in DECISIONS.md:

- The GAP(1) head contradicted its own justification (D-006)
- Aspect ratio was unaddressed (D-007)
- Accuracy target was benchmarked against an incomparable protocol (D-012)
- No trivial baselines existed (D-013)
- **The ablation was confounded and could not answer its own question** (D-014)
- Rotation `fill=0` would have silently poisoned 4 of 9 spatial cells (D-008)
- Caching at 224 would double-resample (D-009)
- Normalization × padding interaction (D-010)
- The whole power analysis — underpowered by ~2.5×, McNemar, k consistency, variance
  decomposition, multiple comparisons, and the fitted-models-vs-architectures scope
  limit (D-016 → D-022)

Errors on my side worth remembering: designed a non-experiment and presented it as an
experiment; used k=1.96 and k=2.80 inconsistently in a way that flattered my own proposal by
~40%; claimed seed averaging shrinks test-set sampling variance (it cannot); wrote
"conservative" when I meant anti-conservative, in a protocol specifically about not fooling
ourselves. Errors on the user's side: h=88 vs 86; "may clear 80%"; the val+test proposal,
withdrawn after objection.

**Process note.** Two user messages were truncated in transit and one appears never to have
arrived. Switched to file-based replies under `notes/`. New rule: if a message reads as
incomplete, say so *before* acting rather than proceeding on a partial instruction. I
flagged both truncations but responded to the partial content anyway, which was the wrong
call.

---

## 2026-09-05 — Phase 1.1: environment and skeleton

**Done:**
- `conda create -n docclf python=3.11` → Python 3.11.16
- `requirements.txt` pinned with `~=` (compatible-release) — resolves cleanly today, stays
  reproducible. Install succeeded, exit 0.
  Key versions: torch 2.8.0, torchvision 0.23.0, numpy 2.4.6, sklearn 1.9.0, imagehash 4.3.2,
  datasets 3.6.0, pandas 2.3.3, matplotlib 3.11.1.
- Repo skeleton per the Phase 0.4 layout; `.gitkeep` in the ignored data/output dirs so a
  fresh clone still shows the structure.
- `.gitignore`: `data/`, `outputs/`, `reference/`, `__pycache__/`, `*.pt`, `.DS_Store`.
  `notes/` deliberately **tracked**.
- `src/utils/device.py` — mps → cuda → cpu, plus a one-line device report at run start
  (a silent CPU fallback is the most common cause of "why is this epoch so slow").
- `src/utils/seed.py` — seeds `random`, `numpy`, `torch`, `PYTHONHASHSEED`. Module docstring
  records that `cudnn.deterministic` is a **no-op on MPS**, so the standard seeding block
  copied from both reference repos gives false assurance.
- `configs/default.yaml` — every Phase 0 decision pinned, with the reasoning in comments.
- `configs/ablation.yaml` — **pre-registered before any run**, per D-020/D-022. Git timestamp
  makes the hypothesis auditable as predating the data.

**BLOCKER FOUND — `AdaptiveAvgPool2d(3)` fails on MPS.**

Environment smoke test:

```
RuntimeError: Adaptive pool MPS: input sizes must be divisible by output sizes.
Non-divisible input sizes are not implemented on MPS device yet.
```

Block-4 emits 14×14. 14 divides by 1, 2, 7, 14 — not 3. Measured directly:

| pool | 14 divisible? | MPS |
|---|---|---|
| `AdaptiveAvgPool2d(1)` | yes | OK |
| `AdaptiveAvgPool2d(3)` | **no** | **FAILS** |
| `AdaptiveAvgPool2d(7)` | yes | OK |

So arms A, B, D run and **arm C does not** — the arm in the pre-registered primary comparison.

Verified `AvgPool2d(kernel_size=5, stride=4)` and `AvgPool2d(kernel_size=4, stride=5)` both
produce 3×3 from 14×14 and both work on MPS, so a fix exists. But it changes what "3×3 spatial
pooling" means for the pre-registered arm, so per the drift rule: flagged, logged as **D-024
(open)**, not implemented pending a decision.

Everything else in the smoke test passed — conv, BN, ReLU, backward on MPS all fine.
This was exactly the MPS-immaturity risk noted in Phase 0.1; it just landed on the least
convenient op.

**Next:** resolve D-024, then Phase 1.2 (dataset download, inspection, dedup, cache).

---

## 2026-09-05 — D-024 resolved, Phase 1.1 closed

User chose `AvgPool2d(kernel_size=5, stride=4)` for arm C. Measured before committing to it
rather than assuming:

| option | cols used | coverage | corr. with `AdaptiveAvgPool2d(3)` |
|---|---|---|---|
| `AvgPool2d(k=5,s=4)` | 0–12 | **92.9%** | **0.816** |
| `AvgPool2d(k=4,s=5)` | 0–3, 5–8, 10–13 | 85.7% | 0.755 |

Cells are `[0-4] [4-8] [8-12]` — 1-column overlap, column 13 unused. Forward and backward
both confirmed on MPS. Output 3×3 → 2,304 features → 23,050 params, unchanged, so the
B-vs-C capacity match still holds at 0.3%.

**Correction:** I had described this option as using "all 14 rows and columns." It uses 13
of 14. Measured, not assumed. Logged in D-025.

Config comments updated in `default.yaml` and `ablation.yaml` so the deviation from textbook
adaptive pooling is visible at the point of use, not just in the decision record.

Phase 1.1 complete. Walked the user through `device.py`, `seed.py`, and `default.yaml`.

**Next:** Phase 1.2 — dataset download, inspection, phash dedup, squashed-vs-padded visual,
augmented-sample visual, cache build, and the timing experiment (cache × num_workers).

---

## 2026-09-05 — D-025 reversed, D-026 adopted

User caught that D-025 only affected arm C. Arms A, B, D divide 14 evenly (by 1, 1, 7) and
keep exact `AdaptiveAvgPool2d`; only C needed a substitute — and C is half the pre-registered
primary. So the workaround put an implementation asymmetry inside the one comparison fixed in
git before running. Same class of error as the original confounded 3-arm design, one layer down.

Tested both proposed fixes:

**(a) `PYTORCH_ENABLE_MPS_FALLBACK=1` — does not work.** Identical `RuntimeError` with and
without the flag. The flag routes *unimplemented* ops to CPU; `adaptive_avg_pool2d` is
implemented on MPS and explicitly rejects non-divisible inputs, so the fallback never engages.
Manual CPU round-trip works at 1.206 ms/batch vs 0.264 native — 4.6× — already rejected.

**(b) replicate-pad 14→15 + `AvgPool2d(5,5)` — works. Adopted.**

| | D-025 (k=5,s=4) | D-026 (pad→15, k=5,s=5) |
|---|---|---|
| coverage | 13/14, col 13 dropped | 15/15 full |
| double-counted | cols 4, 8 | none |
| corr (real activations) | 0.9993 | 0.9993 |
| cost | 0.264 ms/batch | 0.410 ms/batch |

+0.011 s/epoch. Eleven milliseconds to remove a confound from the primary.

**The correlation number was wrong and the user was right to question it.** D-025's 0.816 was
measured on `torch.randn`. Real block-4 activations have adjacent-column smoothness 0.991, so
windowing differences barely register — real correlation is 0.9993, mean |diff| 0.0001. I
benchmarked a pooling operator on the one input distribution it will never see. The corrected
figure doesn't change the decision; it clarifies that the objection was about coverage, not
numerical fidelity.

Also logged **D-027**: Phase 1.2 step order had the cache build (and normalization stats) at
step 4 with the split at step 6, which would have computed `compute_from: train` stats over all
3,482 images — val/test leaking into training normalization. Split moves to step 3, visuals
ahead of the cache build.

`SUMMARY.md` corrected now rather than deferred — a stale "unresolved" entry is exactly what
survives to the version read in four months.

**Next:** Phase 1.2 steps 1–6 (download, dedup, split, visuals, cache+stats). Step 7 (timing)
tomorrow.

---

## 2026-09-05 — Phase 1.2 steps 1-3

**Step 1, dataset verified.** 3,482 images, 10 classes, imbalance 5.17x (predicted ~5.16x).
Download took ~4 min; HF converts to Arrow, which is bigger than the JPEGs — `data/raw` is
1.7 GB and `~/.cache/huggingface` another 1.8 GB. Free space 22.7 -> 16 GB. My ~700 MB
estimate was for the raw JPEGs and was wrong for the on-disk format.

Two findings logged: **D-028** (measured class counts differ from the published table — Form
+59, ADVE +68; totals agree, so the published table is off, and class weights must use
measured values) and **D-029** (source images are median 2386x2292, far larger than the
assumed ~750x1000; aspect h/w spans **0.68 to 1.64**, so some documents are genuinely
landscape — this makes pad-to-square more justified than when it was argued, not less. All
sampled images are already mode `L`, confirming D-005).

**Steps 2-3, dedup then split — and the dedup was wrong at first.**

At threshold 5 the report said 39 clusters, 256 images (7.4%) near-duplicate. But the largest
cluster was **121 images**, labelled Email 119 / Letter 1 / News 1. That looked implausible,
so I rendered 24 of them and actually looked.

They are **different emails**. Different senders, subjects, body text. What they share is a
layout: small header block top-left, ~1.7% ink, bottom two-thirds blank.

Diagnosed it properly rather than guessing:
- median pairwise distance *within* the cluster = **14**, ~3x the threshold. Most pairs are
  not within threshold of each other -> the cluster formed by **transitive chaining**.
- ink density **0.018** in-cluster vs **0.124** elsewhere — 7x less ink. phash thresholds DCT
  coefficients against their median; on a near-blank page the coefficients cluster near that
  median and bits flip on noise. It was detecting emptiness, not duplication.
- threshold sweep showed the cliff: largest cluster 25 (thr 3) -> **121** (thr 4) -> 278
  (thr 6) -> 415 (thr 8). Real duplicate counts don't grow like that.

About **half** the reported 7.4% was this one artifact (121 of 256 images).

User chose threshold 3. Logged as **D-030**, including the correction that when I wrote
`dedup.py` I noted union-find "may over-merge, which is the safe direction for leakage" and
treated that as costless. It isn't — it locked 20% of the Email class into one split.

**Split at threshold 5 (before the fix)** was 2438/523/521 with every class within 0.2% of
target and the leakage assertion passing. Re-running at threshold 3.

**Lesson worth keeping:** the cluster count alone looked fine. Only looking at the images
revealed the problem. Same category as the user's standing instruction about the augmentation
visual — the number can be reassuring and wrong.

---

## 2026-09-05 — Phase 1.2 steps 4-6 complete

**Dedup re-run at threshold 3:** 31 clusters, 116 images (3.3%), largest 25, cross-label
clusters 5 -> 1. The survivor is `['ADVE','ADVE','ADVE','ADVE','News']`, a plausible genuine
label error. Split unchanged: 2438/523/521, every class within 0.2%, leakage assertion passes.
Test n=521 so the power analysis (computed for 522) holds.

**Visual checks — opened both, per the standing instruction.**

`02_augmentation_fill.png`: the fix works and is visible. At larger rotation angles the
`fill=0` row shows large black triangular wedges across the corners; the `fill=255` row is
clean white at the identical rotation. The near-zero-rotation column matches across both rows,
which is the correct control. D-008 confirmed empirically rather than argued.

`01_squash_vs_pad.png`: squashed row distorts each document by a different amount, padded row
preserves proportions. Caveat noted -- my sampling picked aspects 1.04-1.33 and missed the 0.68
landscape extreme where the effect is most dramatic.

**I made a wrong inference and then caught it.** The augmentation panel's document renders
sideways. I stated this was an EXIF orientation bug (PIL ignores the tag without
`ImageOps.exif_transpose`). Then measured: **0 of 498 sampled images carry a non-trivial EXIF
orientation tag.** No metadata is being ignored because there is none -- it is a genuinely
sideways-scanned advertisement. Landscape images are 2.6% of ADVE, 0% elsewhere. Jumped from
observation to mechanism without checking the mechanism existed. Same shape as the D-025
random-noise correlation. Logged as D-031.

**Cache built: 37 s, 217.6 MB, exactly as predicted.** First version also wrote a 218 MB
per-pixel `masks.npy`, which is 218 MB to encode 4 numbers per image -- replaced with
`boxes.npy` at 27 KB (D-032). Stats identical after the refactor.

**Normalization measured, D-010 confirmed:** padding is 23.4% of cached pixels; including it
shifts mean +0.0152 and **deflates std by 11%** (0.1726 -> 0.1535). The std effect is the
damaging one.

**Dataset smoke test passes:** augmentation random (max|d| 5.57 on two reads of one index),
eval deterministic (0.0000), class weights normalise to mean 1.000 with Resume 2.197 / Memo
0.425.

Disk: 18 GB free.

**Next:** step 7 (timing: cache x num_workers) tomorrow, then the model class.

---

## 2026-09-06 — Phase 1.3: the model class

**Cross-label question answered first.** At threshold 3 there is exactly **1** cross-label
cluster, and rendering it changed what it is: 5 images reading "IMAGE NOT AVAILABLE ONLINE"
-- archive placeholder pages where the scan was never digitised. Not mislabelled documents;
images with no class-relevant content at all, carrying labels for documents that are absent.
Swept the full dataset: exactly 5, all at phash distance 0, all landed in train (val 0,
test 0). Keeping them. D-033.

**Model written:** `src/models/blocks.py` (ConvBlock, Pool3x3MPS, build_head) and
`src/models/cnn.py` (DocCNN). Backbone and head are separate attributes with a public
`forward_features`, so Week 2's detection head plugs in without touching the backbone.

**Printed real shapes rather than annotating them, per instruction -- and it caught three
things a comment would have hidden:**

1. **Backbone is 1,172,640 params, not the 1,175,232 carried since Phase 0.4.** The estimate
   assumed conv bias; `bias=False` because BatchNorm subtracts the batch mean and cancels it.
   Head counts are exactly as pre-registered (22,972 / 23,050), so the ablation is unaffected.
   D-034.
2. **The MPS probe was running on CPU** and printed "succeeded on this device" one line after
   correctly saying it raises on MPS. Self-contradictory output. Now probes every device.
3. **Receptive field computed as 61 px instead of 76 px** -- the loop omitted the maxpool's own
   (k-1)*jump contribution. Pooling widens the RF, not just subsamples. A 20% understatement in
   the one explanation where the number is load-bearing. D-035.

Both bugs were visible only because the output was real. A comment saying "RF = 76" would have
stayed correct while the code was wrong.

**`tests/test_shapes.py` added and passing** -- block shapes, the 14%3 constraint, Pool3x3MPS
output and its 0.9964 correlation with the true op, all four arms' parameter counts, and the
B-vs-C capacity match (0.3384%, fails above 1%).

**Next:** the training loop.

---

## 2026-09-06 — Phase 1.4: the training loop

**Correlation reference pinned first.** User asked which distribution the 0.9964 in
`test_shapes.py` measured versus the 0.9993 reported earlier. They measure different things:

| input | corr |
|---|---|
| randn straight into the pool | 0.8323 |
| randn INPUT through a random-init backbone (what the test did) | 0.9962 |
| **real cached images** (what it asserts now) | **0.9860** |
| synthetic doc-like proxy | 0.9925 |

**And the 0.9993 does not reproduce.** That came from a hand-rolled backbone in a scratch
script, not `DocCNN`; the same synthetic input through the real model gives 0.9925. Third time
I have reported a number measured on a proxy rather than the real object (after the random-noise
correlation and the invented EXIF mechanism). Test now asserts on real images with the
reference distribution written into the docstring and a 0.98 floor (measured 0.9851-0.9873
across 5 seeds). D-036.

**Training loop written** -- `src/engine/metrics.py` and `src/engine/train.py`. Flat functions,
five steps visible in sequence, module docstring explaining what each does to the weights.

**Three demos, all real output:**

1. Gradient lifecycle. `.grad` None -> populated -> weights move by exactly 3.000e-04 = lr.
   The line worth keeping: after `backward()` the weights are byte-identical, max|diff|
   0.000e+00. Backward computes derivatives, it does not touch weights.
2. `zero_grad()` removed. Gradient magnitude RISES without it (1.69 -> 1.94 -> 2.23e-01,
   accumulating the sum of all prior steps) and FALLS with it (1.69 -> 1.33 -> 0.80e-01).
   Opposite directions by step 3.
3. Single-batch overfit: PASS, 2.324 -> 0.00012, acc 1.00.

**Caught myself on demo 3.** First version used `train_ds[0..7]`, and because split indices are
sorted that is **all class 0**. `y = [0,0,0,0,0,0,0,0]`. Loss hit 0.0002 in 5 steps and the demo
printed PASS -- but "memorise 8 images" had collapsed to "always predict class 0", a constant
function that cannot detect misaligned labels, which is one of the main bugs the check exists
for. A green result from a test that cannot fail. Fixed to span 8 distinct classes; the curve is
now a real one. D-037.

Nice incidental check: starting loss 2.324 ~= ln(10) = 2.303, which is what an untrained
10-class model must produce.

**Next:** validation loop is written; still needed are the checkpoint/config plumbing and the
full training script, then the timing experiment.

---

## 2026-09-06 (later) — probe, eval demos, checkpoints, baselines

**Per-cell correlation probe (D-038).** User's prediction that the disagreement concentrates at
the borders: CONFIRMED. Per-cell correlation on real images degrades monotonically toward the
bottom-right (top-left 1.0000, centre 0.9955, bottom-right 0.9684) — exactly where
replicate-padding acts.

Their proposed mechanism — real features carry MORE energy at the frame edges — is refuted, and
backwards. Real features carry **half** the edge energy (frame 0.0494 vs interior 0.1019);
column 13 specifically is at 36% of an average column, because the right edge of a padded page
is white margin. The real mechanism is the **spatial gradient**: replicate-padding duplicates
column 13, so the rightmost cell double-weights an atypically empty column. On a flat map
(noise) that changes nothing; on a steep gradient it shifts the value. Plus correlation is
scale-relative — real data has 2.8x the absolute error but only 1.4x the spread.

Both of us had the reasoning wrong. User was closer.

**The proxy pattern promoted to a standing rule (D-039).** Three occurrences: random-noise
correlation, invented EXIF mechanism, a figure from a hand-rolled backbone that did not
reproduce. Rule now written down: measure on the real object; label proxies at the point of
reporting; verify a mechanism exists before naming it as a cause.

**Checkpoint plumbing (D-040).** Round-trip passed first time — 13 checks including the arch
guard (loading a gap3 checkpoint into a gap1_hidden model raises rather than silently mixing
arms).

Resume-continuity test FAILED, and the test was wrong, not the code. It compared an interrupted
run against an uninterrupted one — but dropout draws a fresh mask every train() forward, so the
two consume different RNG streams and diverge from that alone (measured: 1.87 between two
forwards on identical input). **The giveaway was that "full resume" and "weights only" printed
byte-identical curves**, which is impossible if optimizer state were the cause.

Mirror image of D-037: there, a test that could not fail; here, a test that failed for the
wrong reason. Both give confident signals about something they are not measuring.

Fixed by reseeding at the boundary in every arm. Corrected result validates D-003:
**full resume 0.000e+00 deviation (bit-exact), weights-only 2.079e-02.**

**Eval demos (D-041).** train() varies by 1.5492 between two calls on identical input; eval()
is 0.0000. Forgetting eval() at validation: correct 12.50%, wrong-mode 0.00%/0.00%/mean 7.50%,
nothing errors. no_grad on a 32-image batch: 1569.2 MB vs 0.0 MB allocated (driver-reserved
2098.7 vs 1074.7 MB confirms it independently). The 0.0 is correct, not a measurement failure —
intermediates are freed as consumed. Block outputs alone are 91.9 MB, so the ~17x multiplier is
the other saved intermediates.

**BASELINES (D-042) — the important result of the session.**

| model | features | C | test acc | macro-F1 |
|---|---|---|---|---|
| majority | - | - | 17.85% | 0.0303 |
| logreg 8x8 | 64 | 0.1 | 58.73% | 0.5491 |
| logreg 32x32 | 1024 | 0.1 | **61.61%** | 0.5599 |

**My Phase 0.4 estimate of 30-45% was wrong by ~20 points.** The linear floor is 61.61%, which
puts it 8 points below the bottom of the 70-80% CNN target. A CNN at 65% would now be a failure
beaten by five lines of sklearn, not a marginal pass.

16x more features buys +2.88%, so both linear models are reading coarse ink distribution by
region. The CNN's margin over 61.61% is the number that shows it learned structure.

**First full training run launched** — gap3, seed 0, 40 epochs.

**Correction on the training run.** I briefly reported the process had died after epoch 1 and
started diagnosing an OOM kill, citing 12% free memory and 4.6 GB swap. Wrong on both counts:
`pgrep -fc '09_train'` returned 0 because of pattern matching, not because the process was gone
-- `ps aux` showed it alive and consuming CPU. There was no OOM kill and no entry in the system
log. The apparent silence was buffered stdout; only `metrics.csv` was updating.

Fourth instance of the same failure mode as D-039, and the most careless: I inferred a
mechanism (OOM) from an ambiguous signal (no output + high memory) and began reporting it
before checking whether the process was actually dead. One `ps` would have settled it. Noting
it here rather than as a new decision entry, since D-039 already states the rule -- what this
adds is that the rule is easiest to break under time pressure at the end of a session.

**Measured epoch 1: 38.7 s/epoch**, which is at the top of the original 20-45 s/epoch estimate
and confirms the cache is doing its job (source images are median 2386x2292, ~7x the pixels the
estimate assumed).

---

## 2026-09-06 — FIRST FULL RUN COMPLETE

gap3, seed 0. Early-stopped at epoch 26 of 40. 16m 12s, median 36.8 s/epoch.

**Test: 85.22% accuracy, 0.8286 macro-F1.** Against baselines of 17.85 / 58.73 / 61.61 that is
+23.6 points over the best linear model. Comfortably above the 70-80% target.

**And it crossed the >85% leakage tripwire, which then found a real bug (D-043).**

Checks 1-3 clean: no index overlap, no shared phash groups, no byte-identical cached images.
Check 4 was not clean: re-hashing the CACHED 256x256 images found 13 test images (2.5%) within
distance 3 of a training image, 4 at distance 0.

Cause: `02_dedup_and_split.py` hashes the ORIGINAL scans (median 2386x2292); the model trains on
the 256x256 cache. Downscaling ~9x destroys the detail that separated those pairs -- the four
distance-0 pairs are 4-12 bits apart as originals. The dedup answered "same document?" when the
question that governs leakage is "identical to the model?"

Impact measured: all 13 predicted correctly, 12 of 13 share a label. Excluding them,
**84.84% / 0.8282** (from 85.22 / 0.8286). Small but real. Reporting the clean figure.

Not re-splitting now -- that would break comparability with the baselines computed on this
split. The corrected dedup goes into next week's ablation, with baselines recomputed.

The tripwire was the user's requirement in D-012 and it earned its place on the first run that
crossed it.

**Error analysis found the more interesting result (D-044).** Scientific has F1 0.576, recall
0.487, and 6 of the 12 most-confident errors. Looking at the images: the class contains a
research letter, a data table, a contract cover sheet, a smoking poster, handwritten lab notes,
a research proposal. They share SUBJECT MATTER, not layout. Every other class here is
layout-defined.

A CNN reading page geometry cannot see "this is about science." That is a modality mismatch, not
a model defect, and it is the concrete argument for the Week 3-4 OCR + text branch -- the only
thing that can lift this class.

Also at least 3 of 12 confident errors are LABEL errors, not model errors: a page headed
"Inter-office Memorandum" labelled Report, one headed "INTEROFFICE MEMORANDUM" with
To/From/Date/Subject labelled Letter, a no-smoking poster labelled Scientific. Consistent with
the published 11.7% rate.

macro-F1 (0.8286) is only 2.4 points below accuracy (0.8522), better than the 5-10 predicted --
the inverse-frequency weighting worked. Visible cost: Resume recall 0.944 / precision 0.548, so
it over-predicts Resume. That is the intended trade.

---

## 2026-09-06 — WEEKEND CLOSE-OUT

**What got built.** A complete document-image classification pipeline, from scratch in PyTorch
on an 8 GB M2 MacBook Air, in two days. Dataset acquisition and verification; perceptual-hash
near-duplicate detection with union-find clustering; a grouped stratified 70/15/15 split that
keeps duplicate clusters inside one split; a 256x256 uint8 tensor cache with content-only
normalization statistics; a hand-written `Dataset` with the deterministic work cached and the
random augmentation applied per-epoch; a 1.2M-parameter VGG-style CNN with four interchangeable
pooling heads behind a single config key; a flat, visible training loop; validation with
macro-F1 as the selection metric; victoresque-format checkpoints carrying optimizer state and
the full arch config; CSV logging and matplotlib curves; three trivial baselines; confusion
matrix and confident-error analysis; two test suites (shapes and checkpoints); five teaching
demos that print real numbers from real forward passes; and a four-arm ablation harness with a
pre-registered primary hypothesis. Plus roughly 45 decision entries recording why each choice
was made, including the wrong ones.

**What holds.** The headline is **84.84% test accuracy / 0.8282 macro-F1** against a 61.61%
logistic-regression baseline on the same split — a +23.2 point margin that cannot be attributed
to ink density, since 16x more pixel features bought the linear model only +2.88 points. That is
the number the baselines were built to make interpretable, and they did their job. Training took
16 minutes. The checkpoint format is validated by a resume test showing bit-exact continuity
(0.000e+00 deviation) where a weights-only checkpoint diverges by 2.079e-02. Every shape and
parameter count in the writeup is asserted by a test that fails if it drifts. The >85% leakage
tripwire fired on the first run that crossed it and found a genuine defect — the dedup hashed
source images while the model consumes 256x256 tensors — which cost 0.38 accuracy points and is
reported rather than buried. A hand triage of the 30 most-confident errors found that 63% are
not the model's fault: 6 label errors, 13 genuinely ambiguous, 11 real model errors.

**What is open.** The four-arm ablation is designed, pre-registered and instrumented but not
run; it needs roughly four hours of compute and is next week's work. Its power analysis says an
inconclusive outcome is a live possibility, and that is written into the protocol as a valid
result. The Scientific-class hypothesis — that its 0.576 F1 is a modality ceiling rather than a
capacity limit — is an argument, not a measurement, and the ablation's per-class output tests it
for free. The dedup fix (hash the cached tensors, not the originals) is deferred deliberately so
this run stays comparable with its baselines; it lands with the ablation, along with recomputed
baselines. And the recurring failure mode of this project is now named and written down:
measuring something adjacent to the question and reporting it as the answer, four times over.
The rule that would have caught the fourth instance was written before it happened, which is
the best evidence available that the rule is worth keeping.

---

## 2026-09-06 — inference, timing, and the file map

**`scripts/predict.py` works.** Real output on test index 0: predicted **ADVE at 0.9999**,
true ADVE, CORRECT. Second run on test index 500 (a Scientific document): predicted Form at
**0.3391**, with Memo 0.3201 and Email 0.1824 — near-uniform and wrong, which is the right
behaviour for the class the model cannot resolve from layout.

Three things the script does deliberately:
- **`model.eval()`** — without it, dropout stays active and BatchNorm uses the batch's own
  statistics rather than its running estimates. On a batch of one that is catastrophic: BN
  would normalise the single image against itself, so every prediction would come from a
  degenerate distribution. Measured earlier (D-041): train() mode varies by 1.5492 between two
  calls on identical input, eval() by 0.0000.
- **`torch.no_grad()`** — the forward pass otherwise saves every intermediate activation for a
  backward pass that never comes. 1569 MB at batch 32 (D-041); trivial at batch 1, but it also
  removes any chance of an accidental gradient step.
- **Architecture rebuilt from the checkpoint's stored arch args**, not hardcoded. The moment
  the ablation produces four different heads, a hardcoded `DocCNN()` would silently load gap3
  weights into a gap1 model or fail on a shape mismatch.

Preprocessing asserts `exclude_padding is True` when reading norm_stats.json, so the
padding-inflated pair (0.9502 / 0.1535) cannot be picked up by accident.

**Checked the 0.9999 rather than assuming (D-047).** It is justified: 25.5% of test predictions
exceed 0.99, and in that bin the model is right 99.25% of the time. Expected calibration error
0.0343, and the model is slightly UNDER-confident in the 0.70–0.90 range. Suspicious-looking,
legitimate on inspection.

**Timing finally measured, and it refuted half my own reasoning (D-046).**

| | workers=0 | workers=2 |
|---|---|---|
| cached | **33.7 s** | 89.9 s |
| no cache | 34.9 s | 88.3 s |

The cache gives **~1.0x**, not the large speedup D-009 claimed. The GPU is not the bottleneck —
at 438 ms/batch the compute dominates so completely that JPEG decode hides inside it. The cache
is still worth keeping (determinism, and the baselines read it) but the stated performance
justification was wrong. `num_workers=2` being **2.7x slower** was predicted correctly, and by a
wider margin than expected.

**`notes/FILEMAP.md` written** — every artifact with its absolute path, measured size, and
tracked/ignored status. Total on disk: data 1.9 GB, outputs 30 MB, reference 131 MB.
Regeneration from a clean clone is ~25 minutes.

**Flagged: this is still not a git repository.** `git init` has not been run, so nothing is
actually tracked and the .gitignore is currently hypothetical. Worth doing before any further
work, since the pre-registration in configs/ablation.yaml depends on a git timestamp to be
auditable as predating the data.
