# MANIFEST: experiments/

**Generated**: 2026-03-23
**Purpose**: Bboy musicality analysis -- from raw GVHMR 3D joints to validated musicality metrics and Instagram-shareable video exports. Contains the experiment harness, rendering pipeline, component library, and all outputs.

---

## Sub-directories with Detailed Manifests

| Directory | Description | Manifest |
|-----------|-------------|----------|
| `results/` | All experiment outputs: 3D joints, audio features, metrics, videos, 8 validated experiments, locked segments | [results/MANIFEST.md](results/MANIFEST.md) |
| `exports/` | Polished video exports (v3 skeleton/spatial/timelines, v5 pitch/worldstate) | [exports/MANIFEST.md](exports/MANIFEST.md) |

### Related Manifests (External)

| Path | Description |
|------|-------------|
| `autoresearch/experiments/bboy-battle-analysis/OVERNIGHT_RESULTS.md` | TRIVIUM scoring engine: overnight Karpathy loop results. `analyze_motion.py` (859 LOC), `match_beats.py` (889 LOC), `analyze_track.py` (548 LOC). Score: 100/100, 9/9 variants. |

---

## Scripts

| File | Size | Description |
|------|------|-------------|
| `harness.py` | 15K | Experiment harness: runs EXP-001 through EXP-008, computes musicality metrics (mu, tau_star, beat alignment), writes per-experiment results |
| `synthetic_joints.py` | 12K | Generates synthetic SMPL joint sequences for calibration experiments (sine-wave motion, random noise, etc.) |
| `statistics.py` | 11K | Statistical validation: permutation tests (n=10000), bootstrap confidence intervals, Cohen's d effect size, cross-video consistency |
| `publication_figs.py` | 15K | Generates publication-quality figures (fig1-fig5 in assets/) |
| `render_video.py` | 14K | Core video renderer: GVHMR mesh overlay on original footage |
| `render_combined.py` | 18K | Multi-panel analytics video (combines skeleton, audio waveforms, metrics into single frame) |
| `render_skeleton.py` | 11K | 3D skeleton joint overlay renderer (v3 export) |
| `render_spatial.py` | 12K | Spatial coverage / heatmap overlay renderer (v3 export) |
| `render_timelines.py` | 14K | Multi-track timeline renderer: beats, energy, joint speed, musicality ribbon (v3 export) |
| `render_pitch.py` | 13K | Elevated-camera pitch view renderer (v5 export) |
| `render_worldstate.py` | 7.8K | Top-down world-state renderer (v5 export) |
| `world_state.py` | 17K | World-state data model: floor plane, dancer position/orientation, movement trail computation |
| `person_lock.py` | 4.1K | Person locking: segments `joints_3d_REAL.npy` into contiguous detection windows (produces `locked/` segments) |
| `extract_2d.py` | 4.0K | Extracts 2D pose data (vitpose, bounding boxes, camera intrinsics) from GVHMR results into standalone numpy arrays |

---

## Components Library (`components/`)

Reusable visualization components for building analytics video panels.

| File | Description |
|------|-------------|
| `__init__.py` | Package init |
| `base.py` | Base component class / interface |
| `panel.py` | Panel layout manager (arranges components in grid) |
| `video_overlay.py` | Original video frame with overlay support |
| `skeleton_overlay.py` | 3D skeleton bone drawing on video |
| `com_tracker.py` | Center-of-mass trajectory visualization |
| `contact_light.py` | Ground contact indicator light |
| `data_points.py` | Data point scatter/line plots |
| `energy_flow.py` | Energy flow waveform strip |
| `move_bar.py` | Move-type progress bar |
| `musicality_ribbon.py` | Musicality score ribbon (beat-aligned color strip) |
| `pattern_detect.py` | Movement pattern detection display |
| `scalar_strip.py` | Generic scalar value strip chart |

---

## Assets (`assets/`)

Static images: publication figures, video preview thumbnails, and visualization screenshots.

### Publication Figures (from `publication_figs.py`)

| File | Size | Description |
|------|------|-------------|
| `fig1_crosscorrelation_comparison.png` | 235K | Cross-correlation comparison: on-beat vs controls |
| `fig1_crosscorrelation_comparison.pdf` | 23K | Same, vector PDF |
| `fig2_beat_alignment_timeline.png` | 268K | Beat alignment timeline visualization |
| `fig2_beat_alignment_timeline.pdf` | 20K | Same, vector PDF |
| `fig3_parameter_sensitivity.png` | 176K | Savitzky-Golay window sensitivity sweep |
| `fig3_parameter_sensitivity.pdf` | 19K | Same, vector PDF |
| `fig4_per_dancer_comparison.png` | 141K | Per-dancer musicality comparison (Lil G, Neguin, Morris) |
| `fig4_per_dancer_comparison.pdf` | 22K | Same, vector PDF |
| `fig5_hypothesis_test.png` | 121K | Hypothesis test visualization (observed vs null distribution) |
| `fig5_hypothesis_test.pdf` | 21K | Same, vector PDF |

### GVHMR Preview Thumbnails

| File | Size | Description |
|------|------|-------------|
| `gvhmr_mesh_toprock.jpg` | 201K | GVHMR mesh render during toprock segment |
| `gvhmr_mesh_powermove.jpg` | 202K | GVHMR mesh render during powermove segment |
| `clean_mesh_5s.jpg` | 206K | Cleaned mesh at t=5s |
| `overlay_preview_5s.jpg` | 231K | Video overlay preview at t=5s |
| `overlay_preview_18s.jpg` | 217K | Video overlay preview at t=18s |
| `sidebyside_preview.jpg` | 188K | Side-by-side (original vs mesh) preview |
| `sidebyside_powermove.jpg` | 175K | Side-by-side during powermove |

### Visualization Screenshots

| File | Size | Description |
|------|------|-------------|
| `combined_5s.jpg` | 258K | Combined analytics panel at t=5s |
| `combined_18s.jpg` | 243K | Combined analytics panel at t=18s |
| `timelines_preview.jpg` | 263K | Timeline view preview |
| `timelines_powermove.jpg` | 291K | Timeline view during powermove |
| `v2_skeleton_5s.jpg` | 236K | Skeleton overlay at t=5s |
| `v2_skeleton_18s.jpg` | 225K | Skeleton overlay at t=18s |
| `v2_spatial_5s.jpg` | 213K | Spatial heatmap at t=5s |
| `v2_spatial_25s.jpg` | 193K | Spatial heatmap at t=25s |

### Metrics

| File | Size | Description |
|------|------|-------------|
| `metrics_summary.json` | 2.0K | Compact summary of hypothesis test, on-beat vs control metrics, and per-video results |

---

## Pipeline Flow

```
GVHMR (gvhmr_src/outputs/)
  |
  v
extract_2d.py          -- Extract 2D pose, bboxes, camera K
person_lock.py         -- Segment into locked detection windows
  |
  v
synthetic_joints.py    -- Generate calibration data
harness.py             -- Run all experiments (EXP-001..008)
statistics.py          -- Statistical validation
publication_figs.py    -- Generate figures
  |
  v
render_*.py            -- Render video exports
  |
  v
results/               -- All data outputs
exports/               -- Polished video exports
assets/                -- Static images
```
