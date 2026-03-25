# MANIFEST: experiments/results/

**Generated**: 2026-03-23
**Purpose**: Derived outputs for both the older musicality experiments and the current JOSH-first validation path: 3D joints, audio features, benchmark reports, metrics, and render-ready artifacts.
**Total size**: ~199 MB

**Current operational state**: `bcone_seq4` is `window_ready`, not `full_clip_ready`. The newest benchmark artifacts live under `benchmarks/bcone_seq4/`, and the newest focused powermove diagnostics live under `powermove_debug/bcone_seq4/RS0mFARO1x4_seq4/RS0mFARO1x4.4332.4423/`. Together they are the canonical local evidence for the current JOSH-vs-GVHMR comparison pass.

---

## 3D Joint Data

| File | Size | Shape | Dtype | Description |
|------|------|-------|-------|-------------|
| `joints_3d_REAL_seq4.npy` | 287K | (999, 24, 3) | float32 | GVHMR output for BC One 2011 seq4 (frames 3802-4801 of full video). Full 24-joint SMPL skeleton in meters, gravity-view coords (Y=up). 999 frames at 30fps. Re-extracted 2026-03-24 with SMPL body model (was 22 joints, now 24). |
| `joints_3d_REAL.npy` | 2.4M | (8286, 24, 3) | float32 | Full GVHMR run for RS0mFARO1x4 (all sequences). 8286 frames = ~276s at 30fps. Re-extracted 2026-03-24 with 24 SMPL joints. |
| `joints_3d_CLEAN.npy` | 258K | (1000, 22, 3) | float32 | Cleaned/smoothed version of seq4 joints used in early musicality experiments. **STALE**: still uses old 22-joint format from SMPLX body model. |

---

## Audio Analysis

All audio features extracted from the BC One 2011 video soundtrack using librosa.

| File | Size | Shape | Dtype | Description |
|------|------|-------|-------|-------------|
| `audio_full.npy` | 8.0K | (1000,) | float64 | Combined audio energy envelope, 1 value per frame |
| `audio_harmonic.npy` | 8.0K | (1000,) | float64 | Harmonic component energy (HPSS), per frame |
| `audio_percussive.npy` | 8.0K | (1000,) | float64 | Percussive component energy (HPSS), per frame |
| `audio_onset.npy` | 8.0K | (1000,) | float64 | Onset strength envelope, per frame |
| `audio_waveform.npy` | 79K | (10000,) | float64 | Raw audio waveform samples (10x frame resolution) |
| `wave_harmonic.npy` | 40K | (5000,) | float64 | Harmonic waveform (5x frame resolution) |
| `wave_percussive.npy` | 40K | (5000,) | float64 | Percussive waveform (5x frame resolution) |
| `librosa_beats.npy` | 664B | (67,) | float64 | Librosa-detected beat times in seconds. 67 beats over ~33s = ~121 BPM. |

---

## Camera & Pose (seq4)

| File | Size | Shape | Dtype | Description |
|------|------|-------|-------|-------------|
| `camera_K_seq4.npy` | 36K | (999, 3, 3) | float32 | Camera intrinsic matrices, one per frame |
| `vitpose_2d_seq4.npy` | 200K | (999, 17, 3) | float32 | ViTPose 2D keypoints (17 COCO joints, x/y/confidence) |
| `bbx_seq4.npy` | 12K | (999, 3) | float32 | Bounding box data per frame (center_x, center_y, scale) |
| `joint_speed_seq4.npy` | 86K | (999, 22) | float32 | Per-joint speed magnitude in m/s. **STALE**: computed from old 22-joint data, not yet regenerated for 24 joints. |

---

## Metrics (JSON)

| File | Size | Date | Description |
|------|------|------|-------------|
| `CLEAN_metrics.json` | 130K | 2026-03-23 19:51 | Full metric report for cleaned joint data (CLEAN pipeline) |
| `REAL_metrics.json` | 132K | 2026-03-23 18:51 | Full metric report for raw GVHMR output (REAL pipeline) |
| `experiment_summary.json` | 3.0K | 2026-03-23 17:06 | Summary of all 8 experiments: mu, tau_star_ms, beat_alignment_pct, flow_score, stage_coverage, freeze/inversion counts |
| `statistical_validation.json` | 1.6K | 2026-03-23 17:11 | Permutation test (p=0.0, n=10000), bootstrap CI, Cohen's d=4.15 (large effect), cross-video consistency (3 videos, all pass H1) |

---

## Benchmark Reports

| Path | Date | Description |
|------|------|-------------|
| `benchmarks/bcone_seq4/benchmark.json` | 2026-03-25 13:23 | Machine-readable BRACE-aligned benchmark for `RS0mFARO1x4` seq4. JOSH is `window_ready`; only 1 of 5 segments is benchmarkable today, and that footwork window now has BRACE 2D metrics. |
| `benchmarks/bcone_seq4/benchmark.md` | 2026-03-25 13:23 | Human-readable benchmark summary. The only benchmarkable segment is the final footwork window, and BRACE 2D currently favors JOSH on that slice. |
| `benchmarks/bcone_seq4/windows.csv` | 2026-03-25 13:23 | One row per evaluated window with recommendation and failure tags. |

## Powermove Diagnostics

| Path | Date | Description |
|------|------|-------------|
| `powermove_debug/bcone_seq4/RS0mFARO1x4_seq4/RS0mFARO1x4.4332.4423/powermove_report.json` | 2026-03-25 13:47 | Machine-readable diagnostics for the failing powermove segment. Result: `coverage_and_pose_quality`, one 23-frame JOSH candidate, recommendation `keep_gvhmr_baseline` on that short slice. |
| `powermove_debug/bcone_seq4/RS0mFARO1x4_seq4/RS0mFARO1x4.4332.4423/powermove_report.md` | 2026-03-25 13:47 | Human-readable summary of the same result, including the window ladder and local artifact paths. |
| `powermove_debug/bcone_seq4/RS0mFARO1x4_seq4/RS0mFARO1x4.4332.4423/candidate_windows.csv` | 2026-03-25 13:47 | One row per candidate JOSH window inside the powermove segment. |
| `powermove_debug/bcone_seq4/RS0mFARO1x4_seq4/RS0mFARO1x4.4332.4423/frame_diagnostics.csv` | 2026-03-25 13:47 | Per-frame availability table for the powermove segment: JOSH validity, BRACE GT overlap, and shot-boundary flags. |
| `powermove_debug/bcone_seq4/RS0mFARO1x4_seq4/RS0mFARO1x4.4332.4423/renders/comparison_landscape_530_553.mp4` | 2026-03-25 13:47 | Comparison strip for the best surviving short powermove candidate window. |

---

## Videos

| File | Size | Resolution | FPS | Duration | Description |
|------|------|------------|-----|----------|-------------|
| `combined_analytics.mp4` | 38M | 1920x1080 | 29.97 | 33.4s | Multi-panel analytics overlay: video + skeleton + waveforms + metrics |
| `overlay_RS0mFARO1x4_seq4.mp4` | 50M | 1920x1080 | 29.97 | 33.4s | Original video with GVHMR mesh overlay |
| `gvhmr_mesh_seq4.mp4` | 42M | 1920x1080 | 30 | 33.4s | GVHMR mesh rendering (raw, uncleaned) |
| `gvhmr_mesh_clean_seq4.mp4` | 35M | 1920x1080 | 30 | 33.3s | GVHMR mesh rendering (cleaned/smoothed joints) |
| `gvhmr_global_clean_seq4.mp4` | 256K | -- | -- | -- | Global trajectory view (cleaned). Note: file may be truncated (256K, ffprobe returns empty streams). |
| `sidebyside_original_mesh.mp4` | 32M | 1920x600 | 25 | 33.4s | Side-by-side: original video (left) + mesh (right), half-height |

---

## Experiment Folders

Each experiment folder contains: `joints_3d.npy`, `meta.json`, `metrics.json`, `com_trajectory.png`, `energy_flow.png`, `spatial_heatmap.png`.

| Folder | Experiment | Joint Shape | Source | BPM | Key Result |
|--------|-----------|-------------|--------|-----|------------|
| `EXP-001_synthetic_baseline/` | Synthetic baseline (calibration) | (900, 22, 3) | Synthetic 120 BPM sine | -- | mu=0.42, strong (confirms pipeline works) |
| `EXP-002_toprock_on-beat_(lil_g)/` | Toprock on-beat (Lil G) [original] | (1057, 22, 3) | RS0mFARO1x4 seq4 | 125.3 | mu=0.38, moderate |
| `EXP-002_toprock_on-beat_lil_g/` | Toprock on-beat (Lil G) [rename] | (1057, 22, 3) | RS0mFARO1x4 seq4 | 125.3 | mu=0.38, moderate (duplicate of above, filesystem-safe name) |
| `EXP-003_toprock_off-beat_control/` | Off-beat control (beats shifted +240ms) | (1057, 22, 3) | RS0mFARO1x4 seq4 | 125.3 | mu=0.40, but tau_star=0 (no lag = no real sync) |
| `EXP-004_random_phase_control/` | Random phase control (shuffled beat times) | (1057, 22, 3) | RS0mFARO1x4 seq4 | 125.3 | mu=0.009, weak (confirms null) |
| `EXP-004b_random_motion_control/` | Random motion control (Gaussian noise joints) | (1057, 22, 3) | RS0mFARO1x4 seq4 | 125.3 | mu=0.018, weak (confirms null) |
| `EXP-005a_cross-video_neguin/` | Cross-video: Neguin | (663, 22, 3) | HQbI8aWRU7o seq3 | 133.2 | mu=0.36, moderate |
| `EXP-005b_cross-video_morris/` | Cross-video: Morris | (676, 22, 3) | k1RTNQxNt6Q seq1 | 120.3 | mu=0.54, strong |
| `EXP-006_powermove_stress_test/` | Powermove stress test | (325, 22, 3) | RS0mFARO1x4 seq6 | 96.8 | mu=0.086, weak (expected -- powermoves are acyclic) |

### Savitzky-Golay Sweep (EXP-007)

`EXP-007_sg_sweep/sweep_results.json` contains aggregated results. Individual window-size runs:

| Folder | Window | mu | tau_star_ms | beat_alignment_pct |
|--------|--------|----|-------------|---------------------|
| `EXP-007-w11_sg_sweep_w=11/` | 11 | 0.649 | 0 | 91.3% |
| `EXP-007-w15_sg_sweep_w=15/` | 15 | 0.644 | 0 | 100.0% |
| `EXP-007-w21_sg_sweep_w=21/` | 21 | 0.440 | 0 | 91.3% |
| `EXP-007-w31_sg_sweep_w=31/` | 31 | 0.380 | 200 | 23.2% |
| `EXP-007-w41_sg_sweep_w=41/` | 41 | 0.123 | 133 | 53.6% |
| `EXP-007-w61_sg_sweep_w=61/` | 61 | 0.254 | 200 | 23.2% |

Each subfolder has the standard 6-file set (joints_3d.npy, meta.json, metrics.json, 3x PNG).

### Librosa Baseline (EXP-008)

| Folder | Description |
|--------|-------------|
| `EXP-008_librosa_vs_ground_truth/` | Compares librosa beat detection vs BRACE ground-truth annotations. joints_3d shape (1057, 22, 3). |

---

## Locked Segments (`locked/`)

Pre-split segments from `joints_3d_REAL.npy` (full 8286-frame sequence). Each segment represents a contiguous detection window separated by detection gaps. Metadata in `segments.json`.

| File | Size | Shape | Dtype | Frame Range | Duration |
|------|------|-------|-------|-------------|----------|
| `seg_00_f0-76.npy` | 20K | (77, 22, 3) | float32 | 0-76 | 2.6s |
| `seg_01_f102-263.npy` | 42K | (162, 22, 3) | float32 | 102-263 | 5.4s |
| `seg_02_f264-296.npy` | 8.7K | (33, 22, 3) | float32 | 264-296 | 1.1s |
| `seg_03_f297-368.npy` | 19K | (72, 22, 3) | float32 | 297-368 | 2.4s |
| `seg_04_f377-2052.npy` | 433K | (1676, 22, 3) | float32 | 377-2052 | 55.9s |
| `seg_05_f2053-2655.npy` | 156K | (603, 22, 3) | float32 | 2053-2655 | 20.1s |
| `seg_06_f2658-3675.npy` | 263K | (1018, 22, 3) | float32 | 2658-3675 | 33.9s |
| `seg_07_f3676-5248.npy` | 406K | (1573, 22, 3) | float32 | 3676-5248 | 52.4s |
| `seg_08_f5249-5429.npy` | 47K | (181, 22, 3) | float32 | 5249-5429 | 6.0s |
| `seg_09_f5430-5599.npy` | 44K | (170, 22, 3) | float32 | 5430-5599 | 5.7s |
| `seg_10_f5600-7707.npy` | 544K | (2108, 22, 3) | float32 | 5600-7707 | 70.3s |
| `seg_11_f7708-8285.npy` | 150K | (578, 22, 3) | float32 | 7708-8285 | 19.3s |
| `segments.json` | 3.9K | -- | JSON | Metadata for all 12 segments (frame ranges, height stats, smoothness) |

---

## Directory Structure

```
results/
  joints_3d_REAL.npy          2.1M   Full GVHMR 3D joints (8286 frames)
  joints_3d_REAL_seq4.npy     258K   Seq4 GVHMR 3D joints (999 frames)
  joints_3d_CLEAN.npy         258K   Cleaned seq4 joints (1000 frames)
  audio_full.npy              8.0K   Combined audio energy
  audio_harmonic.npy          8.0K   Harmonic audio energy
  audio_percussive.npy        8.0K   Percussive audio energy
  audio_onset.npy             8.0K   Onset strength
  audio_waveform.npy          79K    Raw waveform
  wave_harmonic.npy           40K    Harmonic waveform
  wave_percussive.npy         40K    Percussive waveform
  librosa_beats.npy           664B   Beat timestamps (67 beats)
  camera_K_seq4.npy           36K    Camera intrinsics
  vitpose_2d_seq4.npy         200K   2D pose keypoints
  bbx_seq4.npy                12K    Bounding boxes
  joint_speed_seq4.npy        86K    Joint speeds
  CLEAN_metrics.json          130K   Cleaned pipeline metrics
  REAL_metrics.json           132K   Raw pipeline metrics
  experiment_summary.json     3.0K   Experiment summary
  statistical_validation.json 1.6K   Statistical tests
  benchmarks/                 --     BRACE-aligned benchmark reports
  powermove_debug/            --     Focused failing-segment diagnostics and short review renders
  combined_analytics.mp4      38M    Multi-panel analytics video
  overlay_RS0mFARO1x4_seq4.mp4  50M  Video + mesh overlay
  gvhmr_mesh_seq4.mp4         42M    Mesh render (raw)
  gvhmr_mesh_clean_seq4.mp4   35M    Mesh render (cleaned)
  gvhmr_global_clean_seq4.mp4 256K   Global trajectory view
  sidebyside_original_mesh.mp4 32M   Side-by-side comparison
  EXP-001_synthetic_baseline/       Synthetic calibration
  EXP-002_toprock_on-beat_(lil_g)/  On-beat toprock (original name)
  EXP-002_toprock_on-beat_lil_g/    On-beat toprock (safe name)
  EXP-003_toprock_off-beat_control/ Off-beat control
  EXP-004_random_phase_control/     Random beat phase control
  EXP-004b_random_motion_control/   Random motion control
  EXP-005a_cross-video_neguin/      Neguin cross-video
  EXP-005b_cross-video_morris/      Morris cross-video
  EXP-006_powermove_stress_test/    Powermove acyclicity test
  EXP-007_sg_sweep/                 Sweep summary
  EXP-007-w{11,15,21,31,41,61}_*/  Individual sweep runs
  EXP-008_librosa_vs_ground_truth/  Librosa vs BRACE beats
  locked/                           Pre-split REAL segments (12 npy)
```
