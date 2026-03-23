# Extreme Motion Reimplementation Ladder

Paper-faithful, minimal-code overnight experiment for:

- `CoTracker3`
- `MotionBERT`
- `SAM3D`
- `SAM4D`

The primary objective is not generic leaderboard chasing. The ladder optimizes for breakdance-adjacent utility: inverse poses, extreme articulation, occlusion-heavy motion, segmentation quality, and audio-motion signature stability.

## What This Experiment Does

1. Loads a paper ladder from `papers.yaml`
2. Loads a curated applied scenario bank from `scenarios.json`
3. Runs an oracle command and a reimplementation command per paper
4. Scores each paper on:
   - applied utility
   - canonical parity
   - code economy
5. Generates:
   - per-paper result JSON
   - `ANALYSIS.md`
   - one author packet per paper

The command contract is intentionally small: each runner prints one JSON object to stdout with canonical metrics, applied metrics, runtime cost, code stats, artifacts, and open questions.

## Quick Start

```bash
cd experiments/extreme-motion-reimpl
uv sync --extra dev
uv run extreme-motion-reimpl run --continue-on-fail
```

That command uses local fixture adapters by default, so it works as a dry run before wiring real CUDA jobs.

## Layout

```text
.
├── fixtures/                     # Deterministic sample metric payloads
├── papers.yaml                   # Paper ladder, gates, sources, commands
├── scenarios.json                # Applied scenario bank
├── src/extreme_motion_reimpl/
│   ├── audio_motion.py           # Movement spectrogram + cross-modal metrics
│   ├── cli.py                    # CLI entrypoint
│   ├── fixture_runner.py         # Fixture adapter used for dry runs
│   ├── manifest.py               # Manifest loading
│   ├── models.py                 # Dataclasses and run schema
│   ├── reporting.py              # ANALYSIS.md + author packets
│   ├── runner.py                 # Command execution + ladder orchestration
│   └── scoring.py                # Utility/parity/economy scoring
├── tests/
└── Makefile
```

## Real Compute Wiring

The default manifests use local fixture runners. To move onto a rented CUDA box, replace the `oracle_cmd` and `reimpl_cmd` entries in `papers.yaml` with real shell commands.

Supported command executors:

- `local`: execute in a local working directory
- `ssh`: execute remotely through `ssh target "..."` and parse JSON from stdout

Example shape:

```yaml
reimpl_cmd:
  executor: ssh
  target: ${REMOTE_CUDA_HOST}
  workdir: ${REMOTE_WORKDIR}/cotracker3-min
  cmd: python eval_reimpl.py --scenarios breakdance-clips-v1
```

## Scoring

The surrogate objective is fixed to:

- `0.50` applied utility
- `0.30` canonical parity
- `0.20` code economy

Promotion requires both gate sets to pass:

- canonical gate
- applied gate

The default ladder stops on first failure. Use `--continue-on-fail` to collect all paper packets in one pass.

## Audio-Motion Evaluation

`audio_motion.py` provides reusable utilities for:

- smoothing pose sequences
- deriving velocity and acceleration traces
- building a lightweight movement spectrogram
- measuring audio-motion alignment lag and stability
- estimating derivative cleanliness for noisy pose lifts

That logic is independent of any specific paper and is intended for the downstream breakdance analysis stack.

## Commands

```bash
make dry-run
make test
make score-synthetic
```

## Outputs

Generated files land in `runs/<timestamp>/`:

- `summary.json`
- `ANALYSIS.md`
- `paper_results/*.json`
- `author_packets/*.md`

`runs/latest.txt` stores the most recent run directory.
