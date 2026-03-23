# gvhmr-musicality

Quantitative breakdance battle analytics — cross-correlating 3D joint velocities with audio beat structure to produce the first musicality score for breaking.

## The Core Metric

```
μ = max_τ corr(M(t), H(t−τ))
```

**M(t)** = total movement energy from 3D joint velocities (SMPL 22-joint skeleton)
**H(t)** = audio hotness signal (beat strength, bass energy, spectral flux, rhythm complexity)
**μ** = peak cross-correlation — how tightly a dancer's movement locks to the music

## Validation Status

**Hypothesis H1: SUPPORTED** (p < 0.001, Cohen's d = 4.15)

| Metric | Value |
|--------|-------|
| Mean μ (3 dancers) | 0.425 ± 0.081 |
| Random control μ | 0.009 |
| Separation | 41× |
| Experiments completed | 9 + 6 sensitivity sweeps |

Cross-video consistency on Red Bull BC One footage (BRACE dataset):

| Dancer | BPM | μ |
|--------|-----|-------|
| lil g | 125 | 0.380 |
| neguin | 133 | 0.356 |
| morris | 120 | 0.538 |

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experiment journal.

## Pipeline Overview

```
① Capture (iPhone/GH5 @ 120fps)
② Segment (SAM 3 → dancer mask)
③ Track (CoTracker3 → dense trajectories)
④ 2D Pose (Sapiens 1B / RTMPose)
⑤ 3D World (GVHMR → SMPL mesh + global trajectory)
⑥ Scene (SAM3D + DepthPro → floor plane)
⑦ Audio (BeatNet+ → H(t) hotness signal)
⑧ Core (μ = cross-correlation of movement × audio)
⑨ Output (scoring, 3D playback, heatmaps)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for model selection rationale, deployment tiers, and the power move problem.

## Project Structure

```
.
├── experiments/               # Musicality analysis scripts & results
│   ├── harness.py             #   Experiment runner
│   ├── statistics.py          #   Statistical validation
│   ├── synthetic_joints.py    #   Calibrated kinematic simulation
│   ├── publication_figs.py    #   Figure generation
│   ├── render_video.py        #   Video rendering
│   └── render_combined.py     #   Combined overlay rendering
├── src/extreme_motion_reimpl/
│   ├── audio_motion.py        # Movement spectrogram + cross-modal metrics
│   ├── cli.py                 # CLI entrypoint (extreme-motion-reimpl)
│   ├── recap/                 # Battle recap CLI (bboy-recap)
│   │   ├── cli.py
│   │   ├── audio.py
│   │   ├── metrics.py
│   │   ├── render.py
│   │   └── validate.py
│   ├── scoring.py             # Utility/parity/economy scoring
│   └── ...                    # Paper ladder runners & oracles
├── gvhmr/inputs/              # GVHMR input data
├── ARCHITECTURE.md            # Full pipeline architecture document
├── EXPERIMENTS.md             # Experiment journal (9 experiments)
├── POC.md                     # Original proof-of-concept spec
└── pyproject.toml             # Python project config
```

## Quick Start

```bash
# Install with experiment/recap dependencies
uv sync --extra recap --extra dev

# Run the paper reimplementation ladder (dry run with fixtures)
uv run extreme-motion-reimpl run --continue-on-fail

# Run battle recap CLI
uv run bboy-recap --help

# Run tests
make test
```

## Current Status

- **POC validated** — musicality metric works on calibrated kinematic simulation with BRACE ground-truth beats
- **Next phase** — run GVHMR on real video footage, build Instagram-shareable MP4 recap overlays
- **Known limitation** — no 3D model handles inverted poses (headspins, windmills); upright moves (toprock, footwork, freezes) work well

## Environment

Developed on Lightning.ai with NVIDIA L4 (23GB VRAM), CUDA 12.8, Python 3.12, PyTorch 2.8.0.

## Related

- [GVHMR](https://github.com/zju3dv/GVHMR) — World-grounded human motion recovery (SIGGRAPH Asia 2024)
- [BRACE](https://github.com/dmoltisanti/brace) — Red Bull BC One breakdancing dataset (ECCV 2022)
