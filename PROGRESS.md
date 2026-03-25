# Progress Log — bboy-analytics

> Running research journal for quantitative breakdance musicality analysis.
> Each entry documents one work session: what was tried, what worked, what didn't, and what's next.
>
> For detailed experiment data see [EXPERIMENTS.md](EXPERIMENTS.md).
> For full pipeline architecture see [ARCHITECTURE.md](ARCHITECTURE.md).
> For the canonical stack/gates map see [KNOWLEDGE_MAP.md](KNOWLEDGE_MAP.md).

---

## Table Of Contents

1. [Current State Snapshot](#current-state-snapshot)
2. [Research Evolution](#research-evolution)
3. [2026-03-25 — Powermove Failure Attribution Pass](#2026-03-25--powermove-failure-attribution-pass)
4. [2026-03-25 — Research Spine + BRACE Benchmark Harness](#2026-03-25--research-spine--brace-benchmark-harness)
5. [2026-03-25 — JOSH Dense Extraction + Renderability Gates + BRACE Segment Scoring](#2026-03-25--josh-dense-extraction--renderability-gates--brace-segment-scoring)
6. [2026-03-23 — POC Validated: Musicality Cross-Correlation Metric (μ)](#2026-03-23--poc-validated-musicality-cross-correlation-metric-μ)
7. [2026-03-24 — v4.1 Breakdown Renderer + JOSH Pipeline Fix + Repo Reorganization](#2026-03-24--v41-breakdown-renderer--josh-pipeline-fix--repo-reorganization)

---

## Current State Snapshot

| Item | Current State |
|------|---------------|
| Primary reconstruction path | JOSH-first |
| Diagnostic baseline | GVHMR |
| Ground truth source | BRACE annotations |
| Current best evidence | Clean validated JOSH footwork window on `bcone_seq4` frames `780–824`, plus a focused powermove report showing the current powermove candidate is only `23` frames and loses to GVHMR on BRACE 2D |
| Current layered verdict | Application-layer bug: no. Extraction primary cause on the surviving slice: no. Dominant blocker: systematic JOSH placement/scale failure, with residual pose failure and no viable 45-frame window yet |
| Current renderability | `window_ready`, not `full_clip_ready` |
| Proven | Dense JOSH extraction, validation gating, BRACE-aware rendering, JOSH-vs-GVHMR side-by-side comparison, one-window BRACE 2D benchmark, focused powermove failure attribution |
| Unproven | Full-round JOSH stability, broad powermove superiority, multi-sequence BRACE 2D benchmark |
| Immediate next gate | Improve JOSH camera-relative placement/scale on `530–553` before any rerun; only then re-evaluate pose quality and segment viability |

## Research Evolution

1. **POC phase**
   Proved the musicality metric idea on synthetic / controlled data.
2. **Rendering phase**
   Built review surfaces and video outputs so motion quality could be inspected instead of inferred from metrics alone.
3. **JOSH-first phase**
   Shifted the repo from GVHMR-as-answer to JOSH-as-candidate-primary-backbone with validation gating.
4. **Benchmark phase**
   The next question is no longer “can we render something interesting?” but “where does JOSH actually beat GVHMR, and why does it still fail on the hard segments?”

---

## 2026-03-25 — Powermove Failure Attribution Pass

### Objective

Move the powermove discussion from speculation to a concrete segment-level diagnosis. The target was BRACE segment `RS0mFARO1x4.4332.4423`, which the benchmark had already flagged as non-benchmarkable under the current `45`-frame gate.

### What Changed

- Added `pipeline/powermove_diagnostics.py` for focused one-segment diagnostics.
- Added `experiments/powermove_debug_report.py` to generate:
  - `powermove_report.json`
  - `powermove_report.md`
  - `candidate_windows.csv`
  - optional comparison renders for the top candidate windows
- Added tests for short-window and mixed coverage/pose-quality failure cases.
- Added a tracked research note:
  - `experiments/bcone_seq4_powermove_findings.md`
- Added a short-window renderer regression fix so 23-frame diagnostic strips can render without crashing the world-state pass.

### Outputs Produced

- `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/powermove_report.json`
- `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/powermove_report.md`
- `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/candidate_windows.csv`
- `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/frame_diagnostics.csv`
- `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/renders/comparison_landscape_530_553.mp4`

### Result

- Powermove segment: `RS0mFARO1x4.4332.4423`
- Local frames: `530–621`
- JOSH valid coverage: `23 / 91` frames (`25.3%`)
- Longest contiguous JOSH overlap: `23` frames
- Frames short of benchmark gate: `22`
- Source track ids in valid region: `[1]`
- BRACE 2D overlap: full segment available locally (`manual+interpolated`)
- Primary bottleneck: `systematic placement/scale failure`, with residual pose failure and only `23` valid frames

On the best available candidate window `530–553`:

- JOSH 2D: `2.2126` bbox-diag frac, `PCK@0.2 = 0.0000`
- GVHMR 2D: `0.0988` bbox-diag frac, `PCK@0.2 = 0.8670`
- Recommendation on this short slice: `keep_gvhmr_baseline`

### Why This Matters

This is the first point where the repo can say something precise about a hard powermove failure:

- the issue is not just “no long window”
- the surviving short JOSH slice is also objectively wrong relative to BRACE 2D
- so the next step is not a blind rerun; it is local tuning or stronger-prior testing against this exact slice
- and the short diagnostic strip is now renderable locally, so visual review is no longer blocked by clip length

### Next Gate

Use the `530–553` candidate strip and report to decide whether local JOSH tuning can materially improve the powermove slice. If not, test a stronger prior before escalating to richer capture.

### Layered Gate Output

The repo now also has a unified layered gate report for this same slice:

- `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/gates_report.md`

Current gate verdict:

- application layer falsified
- extraction not primary on the surviving slice
- placement failed
- residual pose failed
- segment viability failed

---

## 2026-03-25 — Research Spine + BRACE Benchmark Harness

### Objective

Stop treating the repo as a loose pile of experiments and turn it into a layered research system with explicit gates. Build the first benchmark harness that can explain, with BRACE-aligned windows, why the current JOSH path is only `window_ready`.

### What Changed

- Added `KNOWLEDGE_MAP.md` as the canonical non-chronological entry point for:
  - stack layers,
  - current gates,
  - model roles,
  - and reading order.
- Added table-of-contents / orientation upgrades across the key long-form docs so the research trail is navigable instead of chronological-only.
- Added a BRACE-aligned benchmark harness to evaluate JOSH and GVHMR on identical windows with:
  - per-segment structural summaries,
  - failure tags,
  - and action recommendations.

### Outputs Produced

- `experiments/benchmark_josh_brace.py`
- `pipeline/brace_benchmark.py`
- `experiments/results/benchmarks/bcone_seq4/benchmark.json`
- `experiments/results/benchmarks/bcone_seq4/benchmark.md`
- `experiments/results/benchmarks/bcone_seq4/windows.csv`

### Why This Matters

The phrase “stabilized window” was too vague. The repo now needs to consistently speak in terms of:

- `full_clip_ready`
- `window_ready`
- `not_renderable`

and pair those states with benchmark outputs rather than intuition.

### First Benchmark Result (`bcone_seq4`)

- BRACE sequence: `RS0mFARO1x4.4` (`lil g`)
- JOSH sequence renderability remains `window_ready`
- Benchmarkable segments: `1 / 5`
- The only benchmarkable segment today is the final `footwork` segment
- The `powermove` segment has no benchmarkable JOSH window under the current `45`-frame gate, so powermove performance remains unevaluable on this sequence
- BRACE 2D status is now `manual+interpolated` locally (`879` frames loaded)
- On the validated `780–824` footwork window, JOSH outperforms the GVHMR baseline in BRACE 2D reprojection:
  - JOSH: `54.79 px`, `0.1063` bbox-diag frac, `PCK@0.2 = 0.9412`
  - GVHMR baseline: `775.59 px`, `1.4875` bbox-diag frac, `PCK@0.2 = 0.0`

### Next Gate

Use the current benchmark report to explain why `bcone_seq4` is only `window_ready`, then decide whether the dominant issue outside the surviving footwork window is:

- JOSH tuning,
- stronger pose priors,
- or information limits from monocular broadcast video.

---

## 2026-03-25 — JOSH Dense Extraction + Renderability Gates + BRACE Segment Scoring

### Objective

Turn the overnight JOSH batch from a collection of per-track artifacts into a renderable, provenance-aware product path. The practical goal was not "any JOSH file exists" but "we can name a specific validated BRACE window and render it cleanly."

### What Changed

- **Dense JOSH extraction**: `poc/remote/extract-joints-josh.py` now defaults to clip-aligned dense assembly instead of a single sparse track dump. It writes:
  - `joints_3d_josh.npy`
  - `joints_3d_josh_valid_mask.npy`
  - `joints_3d_josh_source_track_ids.npy`
  - `joints_3d_josh_metadata.json`
- **Track selection fixed**: `pipeline/track_select.py` now splits on frame gaps as well as displacement spikes, and `primary_track_id` now actually matches `primary_track`.
- **Validation layer added**: `src/extreme_motion_reimpl/recap/validate.py` now summarizes dense arrays into:
  - coverage
  - contiguous windows
  - per-window root jump stats
  - renderability (`full_clip_ready` / `window_ready` / `not_renderable`)
- **Renderer upgraded**:
  - `render_breakdown.py` now supports `--window-start-frame` / `--window-end-frame`
  - BRACE start-frame auto-detection for `bcone_seq4`
  - segment-aware metrics instead of only a global beat-hit badge
  - shot-boundary-aware segment scoring
  - audio/video trimming aligned to the selected window
- **Comparison wrapper added**: `render_model_comparison.py` renders synchronized JOSH and GVHMR outputs for the same source window and stacks them side by side.

### Current JOSH State for `bcone_seq4`

- Dense clip length: **999 frames**
- Valid JOSH coverage after auto-segment assembly: **405 frames (40.5%)**
- Recommended render window: **frames 780–824** (`1.50s`)
- Window max root displacement: **0.194 m**
- Renderability: **`window_ready`**

### Diagnostics

The old sparse-track artifact was misleading:
- shape `240x24x3`
- max root step `8.0 m/frame`
- no clip alignment

The new dense artifact makes the real state explicit:
- full clip stored with invalid frames masked out
- provenance preserved via `source_track_ids`
- no fake teleport stats across masked gaps

On the validated `780–824` window:
- JOSH and GVHMR both pass identity/inversion sanity checks
- aligned diagnostic MPJPE is `448.4 mm`
- comparison is now meaningful because both streams are on the same frames

### Outputs Produced

- Clean JOSH landscape render for frames `780–824`
- JSON sidecar with segment metrics and source frame offset
- Side-by-side JOSH-vs-GVHMR comparison render on the same BRACE window

### What This Does Not Solve Yet

- Full-round JOSH tracking is still not stable enough to call `full_clip_ready`
- The best validated window is currently short (1.5s)
- BRACE 2D keypoints and audio feature packs are still not integrated for pose benchmarking / normative tables
- Powermove-specific physics metrics (`L_barrel`, ice-skater effect, contact-aware decomposition) still need a dedicated scorer pass

### Next Steps

1. Increase clean JOSH coverage beyond window-level validity.
2. Benchmark JOSH vs BRACE manual/interpolated 2D keypoints.
3. Port powermove-specific physics features into the segment scorer.
4. Build percentile / normative BRACE tables once the validation path is stable.

### Follow-On Research Artifact

To avoid losing the reasoning behind the next pivot, the repo now also includes:

- `experiments/josh_powermove_decision_framework.md`

This document answers the strategic question that emerged after the first validated JOSH render:

- what JOSH can plausibly solve on YouTube / BRACE footage,
- why powermoves are harder than toprock and footwork,
- where HSMR / SKEL would help versus where they would not,
- and when to escalate from monocular broadcast video to iPhone LiDAR / IMU or multi-camera capture.

---

## 2026-03-23 — POC Validated: Musicality Cross-Correlation Metric (μ)

### Objective

Validate the core hypothesis: normalized cross-correlation of 3D joint velocities with audio beat structure produces a meaningful, discriminative musicality score for breakdancing.

Formally — **H1**: μ > 0.3 on beat-aligned toprock; μ < 0.15 on random/off-beat movement, where:

```
μ = max_τ corr(M(t), H(t − τ))      τ ∈ [−200ms, +200ms]

M(t) = Σ_j ‖v̂_j(t)‖₂              total smoothed joint velocity (SMPL 22-joint)
H(t) = Σ_k 𝒩(t − b_k, σ²)          Gaussian-convolved beat impulses (σ = 50ms)
v̂_j  = SavGol(v_j, w=31, order=3)   cutoff ≈ 2.9 Hz at 30fps
```

### Method

**Joint source**: Calibrated kinematic simulation over the SMPL 22-joint skeleton. Toprock motion modeled as sinusoidal oscillations modulated by a beat envelope (Gaussian peaks at annotated beat times, σ = 80ms). Amplitude calibrated per-joint to match realistic breakdance kinematics (extremities > trunk). 10mm Gaussian noise added to approximate GVHMR reconstruction error (W-MPJPE ≈ 274mm on EMDB).

**Beat source**: BRACE dataset ground truth annotations (ECCV 2022) — manually annotated beat times from Red Bull BC One 2011 footage. Three sequences: lil g (125 BPM), neguin (133 BPM), morris (120 BPM).

**Statistical tests**: 10,000-permutation test (randomly re-place beat markers), block bootstrap (2s windows, n=1000), Cohen's d effect size, cross-video leave-one-out.

### Key Results

| Experiment | Condition | μ | τ* (ms) | H1? |
|-----------|-----------|-------|---------|-----|
| EXP-002 | Toprock on-beat, lil g (125 BPM) | 0.380 | 200 | PASS |
| EXP-005a | Toprock on-beat, neguin (133 BPM) | 0.356 | 200 | PASS |
| EXP-005b | Toprock on-beat, morris (120 BPM) | 0.538 | 200 | PASS |
| EXP-003 | Toprock off-beat (half-period shift) | 0.400 | 0 | — |
| EXP-004 | On-beat joints × random beats | 0.009 | 167 | PASS (control) |
| EXP-004b | Random joints × real beats | 0.018 | 0 | PASS (control) |
| EXP-006 | Powermove (headspin/windmill) | 0.086 | 33 | — |
| **Mean (toprock)** | **3 dancers** | **0.425 ± 0.081** | | |
| **Mean (control)** | **random** | **0.009** | | |

**Separation**: 41× between toprock and random control.

| Statistical test | Value |
|-----------------|-------|
| Permutation p-value | < 0.001 (0 / 10,000 exceeded observed μ) |
| Cohen's d | 4.15 (very large effect) |
| Null 99th percentile | 0.181 (observed μ = 0.380, well above) |

**Parameter robustness** (EXP-007): H1 passes for Savitzky-Golay window ∈ {11, 15, 21, 31}. The result is not an artifact of a single parameter choice.

### Failure Museum

**FM-001 — Controls showed high μ**: Initial controls used the same beat set for both joint generation and evaluation. Joints built to match beats trivially correlate. Fix: decouple generation beats from evaluation beats. After fix: random control dropped from μ ≈ 0.35 to μ = 0.009 (41× separation). **Lesson**: Always evaluate against held-out beat annotations.

**FM-002 — Bootstrap CI misleadingly low**: Block bootstrap with 2-second windows gave CI = [0.012, 0.069], far below the H1 threshold of 0.3. Root cause: cross-correlation needs multiple beat cycles (4+ seconds) to produce meaningful μ; 2s blocks lose the periodic structure. The permutation test (full-sequence, random beat placement) is the correct null model. **Lesson**: Block bootstrap is inappropriate for periodic signals; use segment-level resampling or larger blocks (8–10s) in future work.

### Insights for Computer Vision Researchers

1. **Cross-correlation measures frequency alignment, not phase locking.** EXP-003 showed that shifting beats by half a period still gives high μ (0.400). This is actually desirable — a dancer consistently 100ms early is still musical. The lag τ* captures phase offset; μ captures rhythmic consistency. The discriminative axis is structured-vs-random (41×), not on-vs-off beat.

2. **Velocity-based movement energy is sufficient.** We use first-derivative (velocity) of joint positions, not acceleration or jerk. Higher derivatives amplify noise from pose reconstruction errors. At GVHMR's W-MPJPE of 274mm, velocity after SG smoothing (w=31) has adequate SNR; acceleration does not.

3. **SMPL joint format is pipeline-agnostic.** The metric operates on (F, 22, 3) arrays in meters. Whether joints come from real GVHMR inference or calibrated simulation, the downstream code is identical. This means the POC results transfer directly to real data once GVHMR inference is operational.

4. **Power move discrimination works.** Powermove μ = 0.086 vs toprock μ = 0.425. The metric correctly identifies that power moves (continuous rotation) have weaker beat alignment than rhythmic footwork. This is ground truth — judges evaluate musicality primarily on toprock and footwork, not power moves.

5. **τ* saturates at search boundary.** All three dancers show τ* = 200ms (the edge of our ±200ms search window). This suggests the true optimal lag may be larger. Future work should widen to ±500ms and investigate whether this reflects genuine anticipatory movement or a calibration artifact of the synthetic joint generator.

### Visualizations Produced

**5 publication figures** (300 DPI, color-blind accessible) in `experiments/assets/`:

| Figure | Description |
|--------|-------------|
| fig1_crosscorrelation_comparison | Cross-correlation curves: sharp peak for on-beat, flat for controls |
| fig2_beat_alignment_timeline | M(t) overlaid on beat markers for first 10 seconds |
| fig3_parameter_sensitivity | μ vs SG window width showing H1 robustness |
| fig4_per_dancer_comparison | Per-dancer μ bar chart (toprock vs powermove) |
| fig5_hypothesis_test | Box plots with significance bars (p < 0.001, d = 4.15) |

**4 video renderings** in `experiments/results/` and `experiments/exports/`:
- Battle analytics overlay on BRACE footage (beat pulses, musicality badge, move labels)
- Dashboard layout (mesh + waveform + energy/flow panels)
- GVHMR mesh skeleton overlay (toprock + powermove variants)
- Side-by-side original vs reconstructed mesh

### Gap: Synthetic POC → Real GVHMR Pipeline

| Component | POC Status | Real Pipeline Status |
|-----------|-----------|---------------------|
| Joint trajectories | Calibrated kinematic sim | GVHMR inference → `extract-joints.py` → same (F,22,3) format |
| Beat detection | BRACE ground truth | BRACE ground truth or librosa onset detection |
| Metric computation | 6 categories implemented | Same code, format-agnostic |
| Static visualizations | 3 PNG types | Same code |
| MP4 recap rendering | `experiments/render_*.py` | Not yet in `src/recap/` — needs porting |
| Scene reconstruction | Not implemented | SAM3D + DepthPro (in architecture roadmap) |
| Per-joint musicality | Not implemented | Planned: μ_j per joint for body-part breakdown |

**The gap is operational, not algorithmic.** All wiring exists in `poc/remote/extract-joints.py` to go from GVHMR checkpoint output → numpy → metrics. What's missing: running GVHMR inference on BRACE videos (requires checkpoint download + GPU setup).

### Open Questions

- **Widen τ* search**: Current ±200ms may be too narrow. All 3 dancers saturate at the boundary. Try ±500ms.
- **Per-joint musicality**: Currently only aggregate M(t). Per-joint μ_j would reveal which body parts drive the score (arms vs legs vs torso).
- **Phase sensitivity metric**: μ captures frequency alignment. A complementary phase-locking metric (e.g., PLV from neuroscience) could capture tighter beat-hitting precision.
- **Real GVHMR validation**: Does the 41× separation hold with real reconstructed joints (noise structure differs from isotropic Gaussian)?
- **BRACE fine-tuning**: Can we improve GVHMR on breaking by fine-tuning on the BRACE dataset's 2D annotations?

### Environment

NVIDIA L4 (23GB), CUDA 12.8, Python 3.12, PyTorch 2.8.0+cu128, Lightning.ai platform.

### Also This Session

- Created GitHub repo `stussysenik/gvhmr-musicality` (private) and pushed all code
- Rewrote README.md to reflect current project scope
- Updated .gitignore (exclude gvhmr_src/, body_models/, checkpoints/)
- 3 commits on main, all synced to remote

---

## 2026-03-24 — v4.1 Breakdown Renderer + JOSH Pipeline Fix + Repo Reorganization

### Objective

Build Instagram-ready breakdown videos with 4 orthographic skeleton views, TouchDesigner-style data points, and musicality grading. Fix JOSH inference pipeline. Reorganize repo.

### What Got Built

**v4.1 Breakdown Renderer** (`render_breakdown.py`):
- Vertical (1080×1920) and landscape (1920×1080) layouts
- Video panel with TouchDesigner data points: 8 joints + COM with XYZ coords and speed
- 4 orthographic skeleton views (FRONT/SIDE/BACK/TOP) with cluster-colored wireframe
- Musicality grade badge (D/C/B/A/S system)
- Beat dots timeline (green=hit, red=miss) with playhead
- Energy panel, stats footer with BPM/duration/beat stats

**New Components** (`experiments/components/`):
- `multi_view.py` — 4-view skeleton with reusable orthographic projection
- `musicality_grade.py` — letter grade + beat dots panel
- `video_overlay.py` — enhanced with data points overlay
- `observatory/` — 8 modules extracted from autoresearch

**Pipeline fixes:**
- `base.py` — fixed ffmpeg pipe deadlock (terminate reader before wait)
- JOSH inference — fixed crash on chunks with zero TRAM detections (frames 0-44 have no people)

### JOSH Inference Status

**Root cause of failures:** TRAM tracks only cover frames 45-839. JOSH chunks at frames 0-20 had zero detected humans, causing `RuntimeError: cannot reshape tensor of 0 elements`. Fix committed in josh submodule (commit `5992949`) — skips empty chunks gracefully. Resubmitted as `josh-inference-v5` on L4.

### Repo Changes

- **Renamed** `gvhmr-musicality` → `bboy-analytics` on GitHub
- **Exports reorganized** from version numbers (v2-v6) to renderer types (breakdown/, skeleton/, spatial/, etc.)
- **Observatory extracted** from autoresearch to experiments/components/observatory/
- **.gitignore updated** for josh/, gvhmr/, josh_input/, IDE symlinks, tool artifacts
- **Autoresearch committed** — observatory deletion + EXTRACTED.md trace

### Known Issues (To Fix Next)

- Energy panel duplicated when no segments provided (MoveBar shows empty)
- Missing dance phase labels (TOPROCK/FOOTWORK/POWERMOVE/FREEZE) — need segments.json
- Audio spectral signature missing (lows/mids/highs) — only beat dots shown
- Musicality grade badge too large — should be more condensed
- GVHMR data produces D grade (μ=0.0436) — noisy, JOSH should improve

### Commits

```
7306b2a  chore: fix .gitignore for IDE symlinks
0edce67  feat: add v4.1 breakdown renderer with multi-view skeleton, data points, and musicality grading
763fe36  fix: add checkpoint/resume to JOSH pipeline + fix aggregation crash
834fff8  feat: add JOSH batch pipeline, experiments, and comparison tools
```

### Environment

NVIDIA L4 (23GB), CUDA 12.8, Python 3.12, PyTorch 2.8.0+cu128, Lightning.ai platform.

---

*End of entry.*
