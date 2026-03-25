# bboy-analytics

Quantitative breakdance battle analytics with a JOSH-first motion pipeline, BRACE ground-truth segmentation, and renderable review outputs for difficult movement classes.

## Table Of Contents

1. [Start Here](#start-here)
2. [Current Direction](#current-direction)
3. [What Works Now](#what-works-now)
4. [Scoring Model](#scoring-model)
5. [Data Sources](#data-sources)
6. [Pipeline](#pipeline)
7. [Current Gates](#current-gates)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [Research Log](#research-log)
11. [Environment](#environment)
12. [Related](#related)

## Start Here

Use the docs in this order:

1. [KNOWLEDGE_MAP.md](KNOWLEDGE_MAP.md) — canonical map of the stack, gates, and model roles
2. [README.md](README.md) — operational summary and commands
3. [ARCHITECTURE.md](ARCHITECTURE.md) — broader research architecture and layer survey
4. [experiments/josh_research_report.md](experiments/josh_research_report.md) — detailed JOSH batch audit
5. [experiments/josh_powermove_decision_framework.md](experiments/josh_powermove_decision_framework.md) — powermove strategy and pivot criteria
6. [experiments/bcone_seq4_powermove_findings.md](experiments/bcone_seq4_powermove_findings.md) — tracked summary of the current powermove failure-attribution result

## Current Direction

The repo is no longer treating GVHMR as the main answer. The current engineering objective is:

1. Run JOSH on BRACE clips.
2. Extract a dense clip-aligned joint artifact with explicit validity/provenance.
3. Gate rendering on physical sanity checks.
4. Score by segment type instead of one global beat-hit number.
5. Produce clean review renders and JOSH-vs-GVHMR comparisons on the same validated window.

## What Works Now

- JOSH extraction now writes a dense `joints_3d_josh.npy` plus:
  - `joints_3d_josh_valid_mask.npy`
  - `joints_3d_josh_source_track_ids.npy`
  - `joints_3d_josh_metadata.json`
- The metadata includes renderability, contiguous recommended windows, and track provenance.
- `render_breakdown.py` supports clip windows, BRACE segment labels, and segment-aware grades.
- `render_model_comparison.py` renders synchronized JOSH-vs-GVHMR side-by-side videos for the same window.
- `benchmark_josh_brace.py` now produces BRACE-aligned benchmark artifacts under `experiments/results/benchmarks/`, including BRACE 2D scoring when keypoints are local.
- `fetch_brace_assets.py` downloads BRACE manual/interpolated keypoints for local benchmarking.
- `export_josh_2d.py` projects dense JOSH joints into full-frame COCO-17 coordinates for BRACE 2D evaluation.
- `powermove_debug_report.py` produces a focused BRACE-segment report with candidate-window tables, BRACE 2D comparisons, and optional review renders for failing powermoves.

## Scoring Model

The repo still keeps the original global musicality experiments, but the active render path is now segment-aware:

- `toprock`: soft beat alignment, groove consistency, control
- `footwork`: speed, contact, syncopation, COM stability
- `powermove`: cyclic consistency, energy, duration
- `freeze`: heuristic-only stability/duration for now

This is driven by BRACE dance-type annotations and should be treated as the operative product path, not the old single-number `μ` prototype alone.

## Data Sources

| Source | Status | Handles Inversions? |
|--------|--------|-------------------|
| **JOSH** | Primary | Yes — scene/human optimization; requires validation and windowing because tracks are sparse |
| **GVHMR** | Diagnostic baseline | Useful for comparison, not the final target on hard powermoves |
| **BRACE** | Ground truth labels | Segments, beats, dancer names, shot boundaries; 2D keypoints still available for future benchmarking |

## Pipeline

```
Video → Track/Scene (TRAM + DECO + JOSH)
  → Dense clip-aligned joints + validity mask
  → Validation / recommended render window
  → BRACE segments + segment-aware metrics
  → Breakdown render / comparison render
```

## Current Gates

| Gate | Question | Current State |
|------|----------|---------------|
| Extraction | Do we have a dense clip-aligned JOSH artifact with provenance? | Yes |
| Validation | Is the sequence safe to render end-to-end? | `window_ready`, not `full_clip_ready` |
| Segment scoring | Are BRACE semantics wired into the render path? | Yes |
| Objective benchmark | Can we quantify JOSH vs GVHMR per segment/window? | Yes for `bcone_seq4`; manual+interpolated BRACE 2D is now local and the validated footwork window favors JOSH |
| Powermove attribution | Do we know why the current powermove segment fails? | Yes for `bcone_seq4`: the surviving `530–553` JOSH slice is only 23 frames and still loses to GVHMR on BRACE 2D |

## Project Structure

```
.
├── experiments/                   # Visualization + analysis
│   ├── components/                #   Modular render panels (26 components)
│   │   ├── base.py                #     RendererBase: ffmpeg pipe compositor
│   │   ├── panel.py               #     Panel ABC, fonts, colors, bones
│   │   ├── multi_view.py          #     4 orthographic skeleton views
│   │   ├── video_overlay.py       #     TouchDesigner-style data points
│   │   ├── musicality_grade.py    #     D/C/B/A/S grade + beat dots
│   │   ├── energy_flow.py         #     Kinetic energy visualization
│   │   ├── move_bar.py            #     Dance phase bar
│   │   └── observatory/           #     Real-time dashboard (8 modules)
│   ├── render_breakdown.py        #   v4.1 composite (Instagram + landscape)
│   ├── benchmark_josh_brace.py    #   BRACE-aligned JOSH vs GVHMR benchmark CLI
│   ├── powermove_debug_report.py  #   Focused powermove failure-attribution CLI
│   ├── render_skeleton.py         #   Skeleton overlay on video
│   ├── render_spatial.py          #   Spatial coverage heatmap
│   ├── render_timelines.py        #   Beat/energy/musicality timeline
│   ├── render_trails.py           #   Ghost trail multi-view
│   ├── render_pitch.py            #   Elevated camera angle
│   ├── render_worldstate.py       #   Top-down floor plan
│   ├── world_state.py             #   Deterministic per-frame state
│   ├── exports/                   #   Rendered outputs (see MANIFEST.md)
│       ├── breakdown/             #     Instagram composites
│       ├── skeleton/              #     Skeleton overlay
│       ├── spatial/               #     Spatial heatmap
│       ├── timelines/             #     Beat/energy timelines
│       ├── trails/                #     Ghost trails
│       ├── pitch/                 #     Elevated camera
│       ├── worldstate/            #     Top-down view
│   └── results/                   #   Derived data artifacts
│       ├── benchmarks/            #     Segment-class benchmark reports
│       └── powermove_diagnostics/ #     Focused failing-segment diagnostics
├── pipeline/                      # Batch data processing
│   ├── extract.py                 #   GVHMR/JOSH → joints .npy
│   ├── compare.py                 #   GVHMR vs JOSH comparison
│   ├── track_select.py            #   Multi-person track selection
│   └── config.py                  #   Pipeline configuration
├── poc/                           # Proof-of-concept scripts
│   ├── compare_josh_gvhmr.py      #   JOSH vs GVHMR comparison
│   └── remote/                    #   GPU batch job scripts
├── src/extreme_motion_reimpl/     # Core production code
│   ├── recap/                     #   Battle recap CLI (bboy-recap)
│   ├── audio_motion.py            #   Cross-modal metrics
│   └── scoring.py                 #   Utility/parity/economy scoring
├── ARCHITECTURE.md                # Full pipeline architecture
├── EXPERIMENTS.md                 # 9 experiments + 6 sensitivity sweeps
├── PROGRESS.md                    # Research journal
└── pyproject.toml                 # Python config + CLI entry points
```

## Quick Start

```bash
# Rebuild dense clip-aligned JOSH joints from aggregated output
python poc/remote/extract-joints-josh.py \
  --josh-dir josh_input/bcone_seq4 \
  --body-model-path gvhmr_src/inputs/checkpoints/body_models \
  --fps 29.97

# Render a validated JOSH window from the original footage
python experiments/render_breakdown.py \
  --joints josh_input/bcone_seq4/joints_3d_josh.npy \
  --video josh_input/bcone_seq4/video.mp4 \
  --beats experiments/results/beats.npy \
  --audio josh_input/bcone_seq4/audio.wav \
  --layout landscape \
  --window-start-frame 780 \
  --window-end-frame 825

# Render JOSH vs GVHMR side by side on the same window
python experiments/render_model_comparison.py \
  --josh-joints josh_input/bcone_seq4/joints_3d_josh.npy \
  --josh-meta josh_input/bcone_seq4/joints_3d_josh_metadata.json \
  --gvhmr-joints experiments/results/joints_3d_REAL_seq4.npy \
  --video josh_input/bcone_seq4/video.mp4 \
  --beats experiments/results/beats.npy \
  --audio josh_input/bcone_seq4/audio.wav \
  --layout landscape \
  --brace-video-id RS0mFARO1x4 \
  --brace-start-frame 3802

# Generate the BRACE-aligned benchmark report for the current sequence
python experiments/fetch_brace_assets.py \
  --artifacts manual_keypoints interpolated_keypoints \
  --year 2011 \
  --video-id RS0mFARO1x4

python experiments/export_josh_2d.py \
  --joints josh_input/bcone_seq4/joints_3d_josh.npy \
  --video josh_input/bcone_seq4/video.mp4

python experiments/benchmark_josh_brace.py \
  --josh-joints josh_input/bcone_seq4/joints_3d_josh.npy \
  --josh-meta josh_input/bcone_seq4/joints_3d_josh_metadata.json \
  --gvhmr-joints experiments/results/joints_3d_REAL_seq4.npy \
  --gvhmr-2d experiments/results/vitpose_2d_seq4.npy \
  --video-id RS0mFARO1x4 \
  --seq-idx 4 \
  --sequence-name bcone_seq4

# Diagnose the failing powermove segment and render the top short candidate
python experiments/powermove_debug_report.py \
  --brace-dir data/brace \
  --video-id RS0mFARO1x4 \
  --seq-idx 4 \
  --segment-uid RS0mFARO1x4.4332.4423 \
  --josh-joints josh_input/bcone_seq4/joints_3d_josh.npy \
  --josh-meta josh_input/bcone_seq4/joints_3d_josh_metadata.json \
  --gvhmr-joints experiments/results/joints_3d_REAL_seq4.npy \
  --gvhmr-2d experiments/results/vitpose_2d_seq4.npy \
  --video josh_input/bcone_seq4/video.mp4 \
  --beats experiments/results/beats.npy \
  --audio josh_input/bcone_seq4/audio.wav \
  --render-top-k 1

# Individual renderers
python experiments/render_skeleton.py --joints ... --mesh-video ...
python experiments/render_spatial.py --joints ... --metrics ... --mesh-video ...
python experiments/render_timelines.py --joints ... --metrics ... --mesh-video ... --audio ...

# Battle recap CLI
uv run bboy-recap --help
```

## Research Log

See [PROGRESS.md](PROGRESS.md) for the running research journal.

Key active research artifacts:

- [KNOWLEDGE_MAP.md](KNOWLEDGE_MAP.md) — canonical stack map, gates, model roles, and reading order
- [experiments/josh_research_report.md](experiments/josh_research_report.md) — JOSH batch audit, tuning findings, and pipeline failure analysis
- [experiments/josh_powermove_decision_framework.md](experiments/josh_powermove_decision_framework.md) — next-stage experiment logic: JOSH stabilization, BRACE benchmarking, HSMR/SKEL role, and when to escalate to sensor-rich capture
- [experiments/results/benchmarks/bcone_seq4/benchmark.md](experiments/results/benchmarks/bcone_seq4/benchmark.md) — current BRACE 2D-backed benchmark report for `bcone_seq4`
- [experiments/bcone_seq4_powermove_findings.md](experiments/bcone_seq4_powermove_findings.md) — tracked summary of the current powermove failure-attribution pass and the local artifact paths

## Environment

Developed on Lightning.ai with NVIDIA L4 (23GB VRAM), CUDA 12.8, Python 3.12, PyTorch 2.8.0.

## Related

- [GVHMR](https://github.com/zju3dv/GVHMR) — World-grounded human motion recovery (SIGGRAPH Asia 2024)
- [JOSH](https://github.com/jfzhang95/JOSH) — Joint optimization of scene and humans (ICLR 2026)
- [BRACE](https://github.com/dmoltisanti/brace) — Red Bull BC One breakdancing dataset (ECCV 2022)
