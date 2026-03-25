# Powermove Research Consolidation

**Status:** Active synthesis memo  
**Date:** 2026-03-25  
**Purpose:** Consolidate what this studio has actually tried, what is proven, what remains speculative, and what mathematical / computer-vision intermediaries are still missing for powermove understanding.

For the repo-wide generation trust map, read [model_generations_evidence_map.md](/teamspace/studios/this_studio/experiments/model_generations_evidence_map.md) alongside this memo.

## Table Of Contents

1. [Why This Exists](#why-this-exists)
2. [What The Studio Has Actually Tried](#what-the-studio-has-actually-tried)
3. [What Is Proven vs Speculative](#what-is-proven-vs-speculative)
4. [Current Powermove Diagnosis](#current-powermove-diagnosis)
5. [Failure Tree](#failure-tree)
6. [Missing Intermediary Layers](#missing-intermediary-layers)
7. [Recommended Next Scripts](#recommended-next-scripts)
8. [What Not To Do](#what-not-to-do)
9. [Bottom Line](#bottom-line)

## Why This Exists

The repo has accumulated multiple waves of work:

- synthetic POC work,
- GVHMR-first vertical-slice work,
- JOSH-first reconstruction work,
- autoresearch architecture and physics-stack exploration,
- and several model-spike ideas that were never all validated.

Without a consolidation pass, it is too easy to confuse:

- a paper result with a local result,
- a planned spike with a completed spike,
- a rendering improvement with a reconstruction improvement,
- or a model swap with a real root-cause fix.

This memo is the current source of truth for that distinction.

## What The Studio Has Actually Tried

### 1. Synthetic POC + musicality stack

This phase proved that the downstream scoring idea was computationally coherent on synthetic or calibrated kinematic data.

Evidence:

- [POC.md](/teamspace/studios/this_studio/POC.md)
- [PROGRESS.md](/teamspace/studios/this_studio/PROGRESS.md)

What it established:

- the spectrogram / cross-correlation idea can be computed,
- the renderer stack can be built,
- the metric code path is format-agnostic once joints exist.

What it did **not** establish:

- that real monocular reconstruction quality is good enough,
- that powermoves can be reconstructed,
- or that musicality is validated on real battle footage.

### 2. PromptHMR + WHAC era

This studio repeatedly cites a prior PromptHMR + WHAC attempt as a negative result on breaking footage.

Evidence:

- [ARCHITECTURE.md](/teamspace/studios/this_studio/ARCHITECTURE.md)

What is actually evidenced in the current repo:

- the attempt is referenced as having poor results on bboy footage,
- the inversion problem was already recognized as the central failure mode.

What is **not** cleanly preserved here:

- a first-class local evaluation report with quantitative BRACE benchmarking,
- a canonical artifact bundle showing those failures.

So the PromptHMR + WHAC attempt should be treated as a **historical negative signal**, not as a benchmark-complete experiment.

### 3. GVHMR-first phase

GVHMR was the first serious end-to-end monocular 3D backbone used here.

Evidence:

- [POC.md](/teamspace/studios/this_studio/POC.md)
- [ARCHITECTURE.md](/teamspace/studios/this_studio/ARCHITECTURE.md)
- [PROGRESS.md](/teamspace/studios/this_studio/PROGRESS.md)

What GVHMR proved:

- the product direction is renderable,
- world-grounded monocular reconstruction is useful enough for upright / simpler motion,
- a practical baseline exists for comparison.

What GVHMR did **not** prove:

- reliable powermove reconstruction,
- robust inversion handling,
- or correct full-round contact / support interpretation.

### 4. JOSH-first phase

This is the first phase with strong local evidence and explicit gates.

Evidence:

- [josh_research_report.md](/teamspace/studios/this_studio/experiments/josh_research_report.md)
- [benchmark.md](/teamspace/studios/this_studio/experiments/results/benchmarks/bcone_seq4/benchmark.md)
- [bcone_seq4_powermove_findings.md](/teamspace/studios/this_studio/experiments/bcone_seq4_powermove_findings.md)
- [gates_report.md](/teamspace/studios/this_studio/experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/gates_report.md)

What JOSH proved locally:

- dense clip-aligned extraction works,
- validation can distinguish `full_clip_ready` from `window_ready`,
- one short footwork window is genuinely good,
- the current powermove failure can be decomposed numerically.

What JOSH has **not** proved:

- full-round stability,
- broad powermove superiority,
- or that its current scene/contact stack is sufficient for breakdance contacts.

### 5. Autoresearch physics-stack exploration

This contains the best future-direction thinking, but most of it is not yet locally benchmarked on BRACE.

Evidence:

- [TECH_STACK_REEVALUATION.md](/teamspace/studios/this_studio/autoresearch/experiments/bboy-battle-analysis/TECH_STACK_REEVALUATION.md)
- [PHYSICS_STACK_DEEP_RESEARCH.md](/teamspace/studios/this_studio/autoresearch/experiments/bboy-battle-analysis/PHYSICS_STACK_DEEP_RESEARCH.md)
- [SAM_BODY4D_VIABILITY_SPIKE.md](/teamspace/studios/this_studio/autoresearch/experiments/bboy-battle-analysis/SAM_BODY4D_VIABILITY_SPIKE.md)
- [ARC_101_FEASIBILITY.md](/teamspace/studios/this_studio/autoresearch/experiments/bboy-battle-analysis/ARC_101_FEASIBILITY.md)

The strongest insight from that layer is not “swap models.” It is:

> powermoves need a stronger intermediate representation of contact, support, floor geometry, and biomechanical plausibility than the current pipeline exposes.

## What Is Proven vs Speculative

### Proven locally

- JOSH has one validated footwork success window on `bcone_seq4`.
- JOSH currently fails the focused powermove slice on both coverage and quality.
- The current powermove failure is **not mainly an evaluation bug**.
- The surviving powermove slice is **not mainly an extraction bug**.
- The dominant current failure is **camera-relative placement / scale**, with a residual pose problem after alignment.

### Strongly suggested, but not yet proven locally

- DECO / BSTRO-style contact assumptions are weak for breakdance-specific contacts.
- A better human prior could reduce impossible rotations on extreme poses.
- Better floor / support reasoning would improve powermove diagnosis and maybe reconstruction tuning.

### Still speculative

- SAM-Body4D solves inversions well enough for breaking.
- HSMR materially improves the exact failure mode we now see in JOSH.
- Monocular YouTube footage is sufficient for full-cycle, full-round powermove reconstruction.
- Fine-tuning on BRACE alone will solve the hard cases.

## Current Powermove Diagnosis

The active failure case is BRACE segment `RS0mFARO1x4.4332.4423` on `bcone_seq4`.

Current local evidence:

- only `23` valid JOSH frames survive in the segment,
- the best surviving run is local `530–553`,
- application layer: falsified,
- extraction on the surviving slice: not primary,
- placement / scale: failed,
- residual pose: failed,
- segment viability: failed.

Evidence:

- [powermove_report.md](/teamspace/studios/this_studio/experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/powermove_report.md)
- [root_cause_report.md](/teamspace/studios/this_studio/experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/root_cause_report.md)
- [gates_report.md](/teamspace/studios/this_studio/experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/gates_report.md)

This means the repo’s current best answer is:

> the dominant blocker is not simply “we lost a few frames.” The dominant blocker is that the current system fails to maintain a coherent camera-relative body placement / support story through the powermove.

## Failure Tree

The powermove problem should now be understood as a layered failure tree.

### Layer 0: Information

- motion blur,
- low frame rate relative to angular speed,
- self-occlusion,
- reflective floors / poor scene texture,
- camera movement and crowd clutter.

This is where monocular YouTube may simply be insufficient on some clips.

### Layer 1: Tracking / identity

- person fragmentation,
- missed detections,
- track handoffs,
- discontinuous windows.

This was a major earlier JOSH problem and still affects coverage.

### Layer 2: Camera / venue / scale geometry

- inconsistent focal estimation,
- poor scene depth,
- weak floor-plane understanding,
- bad world-to-image placement,
- scale drift.

This is the dominant current local powermove failure mode.

### Layer 3: Contact / support interpretation

- feet-only contact assumptions,
- no strong support-surface understanding for head, back, shoulder, elbow, knee,
- no explicit per-frame support label,
- no support-transition reasoning.

This is currently under-modeled in the stable pipeline.

### Layer 4: Pose / body prior

- upright-biased priors,
- implausible rotations under inversion,
- ambiguity in self-occluded configurations.

This is the remaining residual error after placement is removed.

### Layer 5: Phase / cycle understanding

- where the powermove starts,
- which cycle stage is active,
- which support mode is active,
- whether the system is maintaining a coherent move cycle.

The repo has segment labels, but not a mature cycle-state representation for powermoves.

### Layer 6: Evaluation / product layer

- renderability gates,
- BRACE-aligned benchmarking,
- visual overlays,
- interpretable diagnostics.

This layer is now much stronger than it was before.

## Missing Intermediary Layers

The biggest gap is not “we need another model.” The biggest gap is that we still skip over the mathematical middle.

### 1. Venue geometry layer

We do not yet have a first-class, trusted representation of:

- floor plane,
- floor normal,
- venue scale confidence,
- camera-to-floor relation,
- manual or semi-manual ground anchors.

Current contact logic in [world_state.py](/teamspace/studios/this_studio/experiments/world_state.py) estimates ground as the minimum foot height in the clip, which is too weak for powermoves.

### 2. Support-contact layer

We do not yet emit:

- which body region is supporting the dancer,
- when support transitions occur,
- confidence for head/back/shoulder/elbow/knee support,
- whether contact is plausible relative to the floor plane.

This is necessary for windmills, headspins, hand hops, flares, and freezes.

### 3. COM / dynamics layer

We need explicit intermediate quantities such as:

- COM height relative to the floor plane,
- COM velocity and acceleration,
- angular velocity around floor-normal and body axes,
- radius-vs-angular-speed coupling,
- cycle-consistency features for rotational motion.

These are the quantities that tell us whether the motion is really a powermove and whether the reconstruction remains physically coherent.

### 4. Phase / cycle-state layer

BRACE segment labels tell us `powermove`, but not:

- entry,
- support exchange,
- airborne transition,
- spin axis lock,
- exit.

Powermove understanding needs cycle-state, not just coarse segment type.

### 5. Decomposed evaluation layer

The benchmark is already good, but the next diagnostics should score:

- floor-plane consistency,
- support consistency,
- COM plausibility,
- angular-momentum proxy consistency,
- projection error conditioned on support state.

## Recommended Next Scripts

Before another large model pivot, the studio should add these scripts.

### 1. `pipeline/venue_geometry.py`

Goal:

- fit a floor plane,
- expose plane normal / offset / confidence,
- support manual anchor points when scene depth is unreliable.

Output:

- floor plane parameters,
- per-frame distance-to-floor diagnostics,
- venue geometry confidence score.

### 2. `pipeline/support_contacts.py`

Goal:

- infer support contact for feet, hands, knees, elbows, shoulders, back, and head.

Suggested signal:

- distance to fitted floor plane,
- local speed,
- local acceleration,
- pose visibility,
- learned or heuristic support priors by move phase.

### 3. `pipeline/powermove_kinematics.py`

Goal:

- compute powermove-specific intermediaries:
- COM-to-floor,
- angular velocity proxies,
- spin radius,
- cycle periodicity,
- support transition events.

This should produce the first mathematically meaningful powermove state vector.

### 4. `experiments/render_support_debug.py`

Goal:

- render the floor plane,
- active support joint/body region,
- COM track,
- support-confidence overlays,
- keyframe labels for the current phase / cycle.

### 5. `experiments/run_powermove_forensics.py`

Goal:

- run the entire diagnostic chain on one segment,
- generate one consolidated markdown + csv + short render bundle,
- compare JOSH and GVHMR under the same support / geometry decomposition.

This should become the standard no-rerun forensic entrypoint.

## What Not To Do

Do not:

- keep swapping models without isolating the layer that failed,
- assume a paper with extreme-pose gains solves breakdance inversions,
- treat render improvements as reconstruction improvements,
- or pay for broad JOSH reruns before the local `530–553` failure is better understood.

Do not assume “better pose prior” solves:

- motion blur,
- missing surface evidence,
- or absent floor / support geometry.

## Bottom Line

The studio’s accumulated work points to one clear conclusion:

> Powermoves are not failing because the team lacks imagination. They are failing because the current pipeline still lacks a strong geometric-support intermediate representation between raw reconstruction and final judgment.

The next serious step is therefore:

1. stop treating the problem as model roulette,
2. build floor / support / COM / cycle intermediaries,
3. then use those intermediaries to decide whether JOSH tuning, HSMR-style priors, or richer capture is the right next move.

That is the most evidence-based path currently available in this studio.
