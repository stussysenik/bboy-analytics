# Knowledge Map — Bboy Computer Vision Stack

**Date:** 2026-03-25  
**Purpose:** Canonical orientation document for the repo. Read this before diving into implementation details or research notes.

---

## Table Of Contents

1. [What This Project Is](#what-this-project-is)
2. [Why This Problem Is Hard](#why-this-problem-is-hard)
3. [The Canonical Stack](#the-canonical-stack)
4. [What Is Interchangeable And What Is Not](#what-is-interchangeable-and-what-is-not)
5. [Current Operational Path](#current-operational-path)
6. [Current Gates](#current-gates)
7. [Current Model Roles](#current-model-roles)
8. [What The Repo Proves Today](#what-the-repo-proves-today)
9. [What Remains Unproven](#what-remains-unproven)
10. [Recommended Reading Order](#recommended-reading-order)
11. [Near-Term Research Tracks](#near-term-research-tracks)
12. [Bottom Line](#bottom-line)

---

## What This Project Is

This project is trying to turn raw breakdance footage into:

- a stable 3D motion representation,
- physically sane clip windows,
- segment-aware analysis grounded in BRACE labels,
- and review-quality renders that can be trusted enough to guide product and research decisions.

The operative output is **not** just “a 3D skeleton exists.”  
The operative output is:

> a validated motion window that survives physical checks, aligns to the source clip, and supports meaningful per-segment analysis.

---

## Why This Problem Is Hard

This is near the hard end of monocular human motion analysis.

The task combines:

- monocular 3D reconstruction,
- fast nonrigid human motion,
- repeated inversion,
- severe self-occlusion,
- contact changes across many body parts,
- broadcast-camera artifacts,
- and an out-of-distribution motion class relative to most training corpora.

Older computer vision systems achieved “crazy things” when the task geometry was more constrained:

- static cameras,
- upright humans,
- slower motion,
- cleaner backgrounds,
- fewer occlusions,
- or stronger priors about the scene.

Breakdance powermoves remove many of those conveniences at once.

So the honest answer is:

> yes, this is one of the harder practical computer vision settings for monocular human recovery.

That does **not** mean it is impossible. It means the project has to be organized as a stack with gates, not as a vague “try better models” loop.

---

## The Canonical Stack

The project should be understood as a layered system:

1. **Data / Capture**
   Video, audio, frame rate, clip boundaries, source provenance.

2. **Person Isolation / Tracking**
   Masks, person tracks, dense points, identity continuity.

3. **Human Motion Reconstruction**
   3D body, trajectory, scene coupling, pose prior.

4. **Dense Clip Assembly**
   Convert fragmented outputs into a clip-aligned artifact with provenance and validity masks.

5. **Validation / Renderability**
   Decide whether the result is physically sane enough to use.

6. **Ground Truth / Evaluation**
   BRACE labels, beats, shot boundaries, and eventually BRACE 2D keypoints.

7. **Scoring / Semantics**
   Segment-type-aware metrics for toprock, footwork, powermove, freeze.

8. **Rendering / Review**
   Clean human-review outputs and synchronized comparisons.

9. **Decision / Pivot**
   Decide whether to tune the pipeline, add a better prior, or escalate capture.

Each layer has a different job. Confusion usually happens when these layers get mixed together.

---

## What Is Interchangeable And What Is Not

Not everything in the repo is the same kind of component.

### Interchangeable within a layer

- `GVHMR` and `JOSH` are both **3D human reconstruction backbones**
- `TRAM` and future stronger trackers are both **tracking / identity inputs**
- different renderers are all **review surfaces**

### Not interchangeable because they live in different layers

- `BRACE` is a **dataset / ground truth source**, not a reconstruction model
- `HSMR / SKEL` is a **human prior / body model direction**, not a full scene-grounded replacement for JOSH
- `CoTracker3` is **dense point tracking**, not 3D body recovery
- `DECO` is a **contact estimator**, not a pose estimator
- `render_breakdown.py` is a **review tool**, not a validator

This distinction matters because “swap in HSMR” is not the same kind of move as “swap GVHMR for JOSH.”  
One is a prior/body-model change; the other is a backbone change.

---

## Current Operational Path

The current repo is operating on this path:

`BRACE clip -> TRAM/DECO/JOSH outputs -> dense clip-aligned joints -> validation gate -> BRACE-aware scoring -> review render`

The primary implementation for that path lives in:

- [pipeline/extract.py](/teamspace/studios/this_studio/pipeline/extract.py)
- [src/extreme_motion_reimpl/recap/validate.py](/teamspace/studios/this_studio/src/extreme_motion_reimpl/recap/validate.py)
- [experiments/world_state.py](/teamspace/studios/this_studio/experiments/world_state.py)
- [experiments/render_breakdown.py](/teamspace/studios/this_studio/experiments/render_breakdown.py)
- [experiments/render_model_comparison.py](/teamspace/studios/this_studio/experiments/render_model_comparison.py)

The current sequence anchor is `bcone_seq4`.

---

## Current Gates

These are the important gates right now.

### Gate 1: Extraction

Question:

> Did we build a dense clip-aligned JOSH artifact with provenance?

Current answer:

- **Yes**

### Gate 2: Validation

Question:

> Is the motion physically sane enough to trust for rendering?

Current answer:

- **Partially**
- `bcone_seq4` is `window_ready`, not `full_clip_ready`

### Gate 3: Segment-aware scoring

Question:

> Are we grading the right move type with the right semantics?

Current answer:

- **Yes, at first-pass heuristic level**
- BRACE segment labels are wired in

### Gate 4: Objective evaluation

Question:

> Can we quantify whether JOSH is better than GVHMR on hard segments?

Current answer:

- **Partially**
- BRACE 2D-backed benchmarking now exists for `bcone_seq4`
- Broader multi-sequence evaluation is still missing

### Gate 5: Powermove-specific understanding

Question:

> Do we understand why powermoves fail?

Current answer:

- **Partially**
- We have the first segment-class structural benchmark
- The current powermove evidence is still weak because the `bcone_seq4` powermove segment has no benchmarkable JOSH window yet

---

## Current Model Roles

| Component | Layer | Current Role | Status |
|-----------|-------|--------------|--------|
| **BRACE** | Ground truth / evaluation | Segment labels, beats, dancer IDs, shot boundaries | Active |
| **TRAM** | Tracking input | Supplies fragmented person tracks into JOSH | Active but weak |
| **DECO** | Contact input | Supplies contact cues into JOSH | Active but suspect on breaking |
| **JOSH** | 3D reconstruction backbone | Primary motion path | Active |
| **GVHMR** | 3D reconstruction backbone | Diagnostic baseline / comparison | Active |
| **HSMR / SKEL** | Human prior / body model | Candidate future refinement path | Not integrated |
| **Render breakdown** | Review surface | Human inspection and side-by-side comparison | Active |

---

## What The Repo Proves Today

The repo now proves:

- JOSH outputs can be assembled into a dense, provenance-aware artifact.
- Validation can distinguish bad full clips from usable windows.
- A clean validated JOSH render can be produced on a BRACE-aligned window.
- BRACE labels can drive segment-aware scoring and overlays.
- JOSH and GVHMR can be compared on the same source window.
- BRACE 2D can now score the validated `bcone_seq4` footwork window directly.

That is real progress.

---

## What Remains Unproven

The repo does **not** yet prove:

- that JOSH is stable across full battle rounds,
- that JOSH outperforms GVHMR on powermoves broadly,
- that monocular YouTube footage is sufficient for reliable full-cycle powermove reconstruction,
- that HSMR / SKEL would materially improve the right failure mode,
- or that the current JOSH advantage generalizes beyond the surviving `bcone_seq4` footwork window.

Those are the next empirical questions.

---

## Recommended Reading Order

If you want the cleanest mental model, read in this order:

1. [README.md](/teamspace/studios/this_studio/README.md)
   Fast operational summary and commands.

2. [KNOWLEDGE_MAP.md](/teamspace/studios/this_studio/KNOWLEDGE_MAP.md)
   This file. Canonical orientation, stack, gates, and doc roles.

3. [ARCHITECTURE.md](/teamspace/studios/this_studio/ARCHITECTURE.md)
   Broad research architecture and layer survey. More exploratory and historically broader than the current shipping path.

4. [experiments/josh_research_report.md](/teamspace/studios/this_studio/experiments/josh_research_report.md)
   Detailed audit of the overnight JOSH run and why it broke.

5. [experiments/josh_powermove_decision_framework.md](/teamspace/studios/this_studio/experiments/josh_powermove_decision_framework.md)
   How to decide whether to keep pushing JOSH, try HSMR / SKEL, or escalate capture.

6. [PROGRESS.md](/teamspace/studios/this_studio/PROGRESS.md)
   Chronological research log.

7. [POC.md](/teamspace/studios/this_studio/POC.md)
   Historical proof-of-concept framing.

---

## Near-Term Research Tracks

The next phase should run on three tracks:

1. **JOSH stabilization**
   Improve coverage, continuity, and failure attribution.

2. **BRACE benchmarking**
   Add 2D reprojection evaluation and segment-class reports.

3. **Powermove-specific diagnostics**
   Determine whether failures are mainly tracking, prior, contact, or information-limited.

Only after those tracks do we have enough evidence to make a serious pivot decision.

---

## Bottom Line

This repo is no longer “a pile of CV models.”

It is a layered system with:

- a current operational path,
- explicit gates,
- distinct component roles,
- and a small number of real unresolved questions.

The hardest unresolved question is still powermoves.

The clean framing is:

> JOSH-first for reconstruction, BRACE for ground truth, validation before rendering, segment-aware scoring after validation, and benchmark-driven pivots rather than model roulette.
