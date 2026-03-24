# bboy-analytics

Quantitative breakdance battle analytics — cross-correlating 3D joint velocities with audio beat structure to produce musicality scores, spatial analysis, and Instagram-shareable breakdown videos.

## The Core Metric

```
μ = max_τ corr(M(t), H(t−τ))
```

**M(t)** = total movement energy from 3D joint velocities (SMPL 22-joint skeleton)
**H(t)** = audio hotness signal (beat strength, bass energy, spectral flux, rhythm complexity)
**μ** = peak cross-correlation — how tightly a dancer's movement locks to the music

## Musicality Grades

| Grade | μ Range | Label |
|-------|---------|-------|
| D | < 0.10 | Off-Beat |
| C | 0.10–0.25 | Catching It |
| B | 0.25–0.40 | Grooving |
| A | 0.40–0.60 | Locked In |
| S | > 0.60 | Surgical |

## Validation Status

**Hypothesis H1: SUPPORTED** (p < 0.001, Cohen's d = 4.15)

| Metric | Value |
|--------|-------|
| Mean μ (3 dancers) | 0.425 ± 0.081 |
| Random control μ | 0.009 |
| Separation | 41× |

## Data Sources

| Source | Status | Handles Inversions? |
|--------|--------|-------------------|
| **GVHMR** | Available | No — single-pass regression fails on headspins/windmills |
| **JOSH** | Batch job running on L4 | Yes — per-track scene optimization with contact constraints |

## Pipeline

```
Video → Segment (SAM3) → Track (TRAM) → Pose (GVHMR or JOSH)
  → WorldState (joints, energy, COM, cyclic) → Renderers → Exports
```

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
│   ├── render_skeleton.py         #   Skeleton overlay on video
│   ├── render_spatial.py          #   Spatial coverage heatmap
│   ├── render_timelines.py        #   Beat/energy/musicality timeline
│   ├── render_trails.py           #   Ghost trail multi-view
│   ├── render_pitch.py            #   Elevated camera angle
│   ├── render_worldstate.py       #   Top-down floor plan
│   ├── world_state.py             #   Deterministic per-frame state
│   └── exports/                   #   Rendered outputs (see MANIFEST.md)
│       ├── breakdown/             #     Instagram composites
│       ├── skeleton/              #     Skeleton overlay
│       ├── spatial/               #     Spatial heatmap
│       ├── timelines/             #     Beat/energy timelines
│       ├── trails/                #     Ghost trails
│       ├── pitch/                 #     Elevated camera
│       └── worldstate/            #     Top-down view
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
# Render the v4.1 breakdown (Instagram vertical + landscape)
python experiments/render_breakdown.py \
  --joints experiments/results/joints_3d_REAL_seq4.npy \
  --video experiments/results/gvhmr_mesh_clean_seq4.mp4 \
  --beats experiments/results/beats.npy \
  --layout both

# Individual renderers
python experiments/render_skeleton.py --joints ... --mesh-video ...
python experiments/render_spatial.py --joints ... --metrics ... --mesh-video ...
python experiments/render_timelines.py --joints ... --metrics ... --mesh-video ... --audio ...

# Battle recap CLI
uv run bboy-recap --help
```

## Research Log

See [PROGRESS.md](PROGRESS.md) for the running research journal.

## Environment

Developed on Lightning.ai with NVIDIA L4 (23GB VRAM), CUDA 12.8, Python 3.12, PyTorch 2.8.0.

## Related

- [GVHMR](https://github.com/zju3dv/GVHMR) — World-grounded human motion recovery (SIGGRAPH Asia 2024)
- [JOSH](https://github.com/jfzhang95/JOSH) — Joint optimization of scene and humans (ICLR 2026)
- [BRACE](https://github.com/dmoltisanti/brace) — Red Bull BC One breakdancing dataset (ECCV 2022)
