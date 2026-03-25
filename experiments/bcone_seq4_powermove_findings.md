# bcone_seq4 Powermove Findings

**Date:** 2026-03-25  
**Status:** Active tracked research note  
**Scope:** Focused failure-attribution pass for BRACE segment `RS0mFARO1x4.4332.4423`

---

## Table Of Contents

1. [Summary](#summary)
2. [Current Evidence](#current-evidence)
3. [Local Artifacts](#local-artifacts)
4. [Interpretation](#interpretation)
5. [Next Actions](#next-actions)

---

## Summary

The current `bcone_seq4` powermove problem is not just “no long JOSH window.”

The focused diagnostics pass shows a mixed failure:

- JOSH only survives for **23 / 91** frames inside the powermove segment (`25.3%` coverage)
- all valid JOSH frames come from **one track** (`track 1`)
- the best raw JOSH overlap is **local frames `530–553`**
- that window is **22 frames short** of the current `45`-frame benchmark gate
- and on that same short window, JOSH is **objectively worse than GVHMR** on BRACE 2D

So the dominant bottleneck for this segment is:

`coverage_and_pose_quality`

Not just continuity. Not just information limits. The surviving short JOSH slice is also wrong.

## Current Evidence

On the best available candidate window `530–553`:

- JOSH 2D: `mean_error_bbox_diag_frac = 2.2126`, `PCK@0.2 = 0.0000`
- GVHMR 2D: `mean_error_bbox_diag_frac = 0.0988`, `PCK@0.2 = 0.8670`
- JOSH recommendation on this slice: `keep_gvhmr_baseline`

Additional structural notes:

- BRACE 2D overlap exists on the whole segment: `91` overlapping frames
- the failure is **not** an in-segment identity handoff
- the immediate JOSH issue is early termination plus poor pose quality on the surviving candidate

## Local Artifacts

These outputs are generated locally under the ignored `experiments/results/` tree:

- `experiments/results/powermove_diagnostics/bcone_seq4/RS0mFARO1x4.4332.4423/powermove_report.json`
- `experiments/results/powermove_diagnostics/bcone_seq4/RS0mFARO1x4.4332.4423/powermove_report.md`
- `experiments/results/powermove_diagnostics/bcone_seq4/RS0mFARO1x4.4332.4423/candidate_windows.csv`
- `experiments/results/powermove_diagnostics/bcone_seq4/RS0mFARO1x4.4332.4423/renders/comparison_landscape_530_553.mp4`

The CLI that generated them is:

- `experiments/powermove_debug_report.py`

## Interpretation

The useful update is conceptual:

- footwork is now objectively favorable to JOSH on at least one validated window
- this powermove segment is not
- the repo can now distinguish:
  - `coverage-only` failure
  - from `coverage + pose-quality` failure

That means the next powermove decisions should be evidence-based:

- first inspect/tune local JOSH assembly around the short candidate
- then decide whether the failure looks fixable by JOSH tuning
- only after that decide whether to try a stronger prior such as HSMR / SKEL

## Next Actions

1. Use the short comparison strip to inspect whether the bad JOSH 2D comes from wrong orientation, bad contact assumptions, or collapse in inversion handling.
2. Try local JOSH continuity/tuning changes on the `530–553` neighborhood before scheduling a more expensive full rerun.
3. If the short slice still loses badly after local cleanup, test a stronger prior before escalating to richer capture.
