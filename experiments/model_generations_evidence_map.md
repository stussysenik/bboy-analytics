# Model Generations And Evidence Map

**Status:** Canonical trust map for model-history claims  
**Date:** 2026-03-25  
**Purpose:** Distinguish what this studio actually computed from what it only proposed, remembered, or imported from papers.

## Table Of Contents

1. [Why This Exists](#why-this-exists)
2. [How To Read The Evidence](#how-to-read-the-evidence)
3. [Generations At A Glance](#generations-at-a-glance)
4. [Generation-by-Generation Assessment](#generation-by-generation-assessment)
5. [Nearby Workspace Findings](#nearby-workspace-findings)
6. [What The Problem Actually Is Right Now](#what-the-problem-actually-is-right-now)
7. [What To Do Next](#what-to-do-next)
8. [Bottom Line](#bottom-line)

## Why This Exists

The repo has several waves of work:

- synthetic POC and musicality experiments,
- remembered PromptHMR/WHAC failures,
- GVHMR-first proof-of-concept work,
- JOSH-first reconstruction work,
- autoresearch architecture/physics documents,
- and nearby studio workspaces with segmentation or older WHAC artifacts.

Those are not all the same kind of evidence.

This file is the canonical answer to:

1. What generation are we talking about?
2. What was actually computed locally?
3. What conclusion are we allowed to draw?

## How To Read The Evidence

Use this trust scale:

- `A — benchmarked locally`
  The repo has concrete artifacts, metrics, or reports on local data.
- `B — computed locally, but not benchmark-complete`
  There are outputs or videos, but not enough preserved evaluation to call it a rigorous benchmark.
- `C — historical negative signal`
  The repo preserves the conclusion, but not the original artifact bundle well enough to re-audit it.
- `D — paper-derived / research-only`
  Useful for planning, not evidence of local success or failure.

Do not promote a `C` or `D` result into an `A` claim.

## Generations At A Glance

| Generation | Core idea | What exists locally | Trust |
|------------|-----------|---------------------|-------|
| `G0` Synthetic POC | Prove the scoring/renderer path on controlled joints | POC docs, synthetic runs, metric outputs | `A` for scoring concept, `D` for real reconstruction |
| `G1` PromptHMR + WHAC era | Early attempt at breakdance 3D | Historical references to bad results, weak surviving artifacts | `C` |
| `G2` GVHMR-first | First real monocular 3D backbone in this repo | Real seq4 outputs, render artifacts, metadata | `B` |
| `G3` JOSH-first | Scene-coupled reconstruction with dense extraction | Real JOSH runs, dense artifacts, audits, validation gates | `A` |
| `G4` BRACE benchmark | Objective JOSH-vs-GVHMR comparison | BRACE 2D-backed benchmark outputs | `A` |
| `G5` Powermove gates | Numeric root-cause analysis on hard segment | Diagnostics, root-cause report, gate verdict | `A` |
| `R1` Autoresearch model survey | Search broader model space | Strong research notes, limited local execution | `D` |
| `R2` Nearby studio helpers | Segmentation / older WHAC / scene code | Real code and some run caches in sibling workspaces | mixed |

## Generation-by-Generation Assessment

### G0. Synthetic POC

What it proved:

- The downstream analysis stack is computationally coherent once joints exist.
- The renderer/scoring path can be exercised end to end.

What it did not prove:

- That monocular reconstruction is accurate enough on battle footage.
- That powermoves can be reconstructed.

Primary artifacts:

- [POC.md](/teamspace/studios/this_studio/POC.md)
- [PROGRESS.md](/teamspace/studios/this_studio/PROGRESS.md)
- [EXP-006_powermove_stress_test/meta.json](/teamspace/studios/this_studio/experiments/results/EXP-006_powermove_stress_test/meta.json)
- [EXP-006_powermove_stress_test/metrics.json](/teamspace/studios/this_studio/experiments/results/EXP-006_powermove_stress_test/metrics.json)

Assessment:

> Strong evidence that the scoring idea is worth building. Not evidence that the vision stack solves powermoves.

### G1. PromptHMR + WHAC Era

What is preserved:

- The repo repeatedly records this as a negative result on breaking footage.
- The central failure mode was already identified as inversion / out-of-distribution motion.

What is missing:

- A benchmark-complete local report.
- A canonical artifact bundle that can be re-run or re-scored today.

Primary references:

- [ARCHITECTURE.md](/teamspace/studios/this_studio/ARCHITECTURE.md)
- [powermove_research_consolidation.md](/teamspace/studios/this_studio/experiments/powermove_research_consolidation.md)

Assessment:

> Treat this as a real warning, not as a reproducible benchmark. It tells us “upright-trained SMPL regressors already hurt us here,” but not exactly how much by current standards.

### G2. GVHMR-First

What it proved:

- The product direction is renderable.
- World-grounded monocular motion is useful enough for review and downstream experimentation.
- A diagnostic baseline exists.

What it did not prove:

- Reliable powermove reconstruction.
- Correct contact/support interpretation.
- Full-round robustness.

Primary artifacts:

- [joints_3d_REAL_seq4_metadata.json](/teamspace/studios/this_studio/experiments/results/joints_3d_REAL_seq4_metadata.json)
- [REAL_metrics.json](/teamspace/studios/this_studio/experiments/results/REAL_metrics.json)
- [gvhmr_mesh_seq4.mp4](/teamspace/studios/this_studio/experiments/results/gvhmr_mesh_seq4.mp4)

Assessment:

> GVHMR is the first real local backbone with preserved outputs, but it remains a proof-of-concept baseline rather than the answer for hard powermoves.

### G3. JOSH-First

What it proved:

- Dense clip-aligned extraction works.
- Validation can separate `full_clip_ready`, `window_ready`, and `not_renderable`.
- One short footwork window is objectively good.

What it did not prove:

- Full-round stability.
- Broad superiority on powermoves.

Primary artifacts:

- [josh_research_report.md](/teamspace/studios/this_studio/experiments/josh_research_report.md)
- [joints_3d_josh.npy](/teamspace/studios/this_studio/josh_input/bcone_seq4/joints_3d_josh.npy)
- [joints_3d_josh_metadata.json](/teamspace/studios/this_studio/josh_input/bcone_seq4/joints_3d_josh_metadata.json)
- [breakdown_landscape_780_825_20260325_123023.mp4](/teamspace/studios/this_studio/experiments/exports/breakdown/josh_validated/breakdown_landscape_780_825_20260325_123023.mp4)

Assessment:

> This is the first generation with enough preserved local machinery to make hard yes/no statements.

### G4. BRACE Benchmark

What it proved:

- The repo can compare JOSH and GVHMR on the same BRACE-aligned window.
- The validated footwork window favors JOSH.
- The current sequence remains only `window_ready`.

Primary artifacts:

- [benchmark.md](/teamspace/studios/this_studio/experiments/results/benchmarks/bcone_seq4/benchmark.md)
- [benchmark.json](/teamspace/studios/this_studio/experiments/results/benchmarks/bcone_seq4/benchmark.json)

Assessment:

> This is where the repo stopped relying on pretty renders as proof.

### G5. Powermove Failure Attribution

What it proved:

- The hard powermove failure can be decomposed numerically.
- On the surviving slice, the application layer is not the primary problem.
- Extraction is not the primary problem on that slice.
- The dominant current failure is camera-relative placement/scale, followed by residual pose error, then viability.

Primary artifacts:

- [powermove_report.md](/teamspace/studios/this_studio/experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/powermove_report.md)
- [root_cause_report.md](/teamspace/studios/this_studio/experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/root_cause_report.md)
- [gates_report.md](/teamspace/studios/this_studio/experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/gates_report.md)
- [comparison_landscape_530_553.mp4](/teamspace/studios/this_studio/experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/renders/comparison_landscape_530_553.mp4)

Assessment:

> This is the current frontier. The open problem is no longer vague. It is a specific layered failure on a specific segment.

### R1. Autoresearch Model Survey

What it contributes:

- good literature synthesis,
- useful architecture decomposition,
- honest warnings when a paper claim was overstated,
- candidate future directions.

What it does not contribute:

- local benchmark evidence for JOSH, HSMR, SKEL, SAM-Body4D, or GenHMR on BRACE.

Primary references:

- [TECH_STACK_REEVALUATION.md](/teamspace/studios/this_studio/autoresearch/experiments/bboy-battle-analysis/TECH_STACK_REEVALUATION.md)
- [PHYSICS_STACK_DEEP_RESEARCH.md](/teamspace/studios/this_studio/autoresearch/experiments/bboy-battle-analysis/PHYSICS_STACK_DEEP_RESEARCH.md)
- [SAM_BODY4D_VIABILITY_SPIKE.md](/teamspace/studios/this_studio/autoresearch/experiments/bboy-battle-analysis/SAM_BODY4D_VIABILITY_SPIKE.md)

Assessment:

> Use this layer for ideas and constraints, not for claims that a model already works in this repo.

## Nearby Workspace Findings

### `/teamspace/studios/whac`

What is useful:

- old run caches,
- tracking videos,
- scale outputs that can be inspected as failure cases,
- one concrete example of scale collapse in a prior world-grounded pipeline.

What is not preserved:

- a clean breakdance failure report with modern benchmarking.

Key local evidence:

- [OPTW_09_30_25_halo](/teamspace/studios/whac/WHAC/demo/OPTW_09_30_25_halo)
- [OPTW_09_30_25_part](/teamspace/studios/whac/WHAC/demo/OPTW_09_30_25_part)
- [OPTW-08-21-2025-cropped](/teamspace/studios/whac/WHAC/demo/OPTW-08-21-2025-cropped)

Assessment:

> Useful as a historical scale/placement cautionary tale, not as a current benchmark path.

### `/teamspace/studios/sam-3-playground`

What is useful:

- real runnable segmentation/tracking code for dancers and body parts,
- direct support for text prompts like `breakdancer` and `head`,
- a possible path for floor/head/body masks on the failing powermove slice.

Important caveat:

- the default API path uses `frame_skip=3`, which is likely too sparse for fast support/contact transitions.

Key local evidence:

- [inference.py](/teamspace/studios/sam-3-playground/breaking_cv/api/inference.py)
- [models.py](/teamspace/studios/sam-3-playground/breaking_cv/api/models.py)
- [app.py](/teamspace/studios/sam-3-playground/breaking_cv/app.py)

Assessment:

> This is potentially useful as a dense mask/intermediary layer, not as a full 3D solution.

## What The Problem Actually Is Right Now

The repo’s main open question is not:

> Which new model name should we try next?

It is:

> Can we maintain a coherent 3D body + placement + support explanation through fast inverted motion on monocular battle footage?

The current failure tree is:

1. information quality,
2. tracking continuity,
3. camera/scale/venue geometry,
4. contact/support interpretation,
5. pose prior under inversion,
6. cycle viability over enough frames.

Current local evidence says the active powermove failure sits mainly at:

- `camera/scale/venue geometry`
- then `pose prior`
- then `segment viability`

and only secondarily at extraction on the surviving slice.

## What To Do Next

Do not restart the model-shopping loop.

The next clean work items are:

1. Build one canonical physical intermediary layer for:
   - ground / floor plane
   - support contact
   - mass-weighted COM
   - powermove cycle state
2. Use that layer on the two fixed controls:
   - footwork success `780–824`
   - powermove failure `530–553`
3. Only after that decide:
   - local JOSH rescue,
   - stronger prior test,
   - or richer capture escalation.

## Bottom Line

The studio has tried a lot, but not all attempts carry the same evidentiary weight.

The only generations that currently justify strong technical claims are:

- the JOSH-first reconstruction path,
- the BRACE-backed benchmark path,
- and the powermove failure-attribution path.

Everything else should be used as:

- historical warning,
- design inspiration,
- or supporting context,

not as proof that the powermove problem is already solved or that the next model swap will solve it automatically.
