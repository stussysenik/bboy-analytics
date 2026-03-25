# bcone_seq4 Powermove Findings

**Date:** 2026-03-25  
**Status:** Current local ground truth for the failing powermove segment

## Summary

The current powermove target is BRACE segment `RS0mFARO1x4.4332.4423` on `bcone_seq4`.

- Local frames: `530–621`
- JOSH valid coverage: `23 / 91` frames (`25.3%`)
- Longest contiguous JOSH run: `23` frames (`530–553`)
- Frames short of the `45`-frame benchmark gate: `22`
- BRACE 2D overlap: available locally across the segment (`manual+interpolated`)
- Current diagnosis: `coverage_and_pose_quality`

The key result is that the current blocker is no longer ambiguous:

- JOSH does not yet provide a benchmarkable powermove window on this segment
- the one surviving short JOSH slice is also objectively worse than the current baseline 2D path on BRACE 2D

## Best Candidate Window

Window: `530–553` local (`4332–4355` global, end exclusive)

- JOSH 2D: `2.2126` bbox-diag frac, `PCK@0.2 = 0.0000`
- Baseline 2D path: `0.0988` bbox-diag frac, `PCK@0.2 = 0.8670`
- Recommendation on this slice: `keep_gvhmr_baseline`

This means the immediate next task is not a blind rerun. The next task is to improve or replace JOSH on this exact short slice first.

## Root Cause Attribution

The numerical root-cause pass says this is **not** a generic application-side projection bug.

- Changing camera intrinsics does **not** change the powermove error:
  - JOSH default projection: `1199.64 px`
  - JOSH with `camera_K_seq4.npy`: `1199.64 px`
- The same projection/evaluation code works on the control footwork window:
  - Powermove JOSH raw error: `1199.64 px`
  - Footwork JOSH raw error: `54.79 px`
- Most of the powermove error collapses under 2D alignment:
  - raw: `1199.64 px`
  - translation-aligned: `438.64 px`
  - similarity-aligned: `77.92 px`

That means the dominant issue on this slice is **bad camera-relative placement / scale inside the JOSH reconstruction**, with a secondary pose problem still remaining after alignment.

The evidence for that is concrete:

- Mean 2D center offset vs BRACE GT: `1062.99 px`
- Mean scale ratio vs BRACE GT: `2.4142x`
- Fraction of projected JOSH joints outside the image: `0.8031`
- Mean projected JOSH bbox center: `(-49.01, 411.08)`
- Mean BRACE/baseline bbox center on the same frames: approximately `(995, 541)` / `(1018, 521)`

So the current powermove failure is best described as:

1. **Coverage failure**: only `23 / 91` valid JOSH frames in the segment
2. **Camera-relative placement / scale failure**: JOSH projects far off the subject
3. **Residual pose failure**: even after similarity alignment, JOSH is still worse than the control footwork case

This makes the next no-rerun experiment precise: inspect and improve the JOSH camera-relative placement around frames `530–553` before paying for broader reruns.

## Canonical Artifacts

- Report: `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/powermove_report.md`
- Machine-readable report: `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/powermove_report.json`
- Candidate table: `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/candidate_windows.csv`
- Frame diagnostics: `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/frame_diagnostics.csv`
- Root-cause analysis: `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/root_cause_report.md`
- Review render: `experiments/results/powermove_debug/bcone_seq4/RS0mFARO1x4.4332.4423/renders/comparison_landscape_530_553.mp4`

## Interpretation

The powermove path currently has two separate problems:

1. **Continuity / coverage**
   JOSH stops after 23 valid frames in the target segment.
2. **Camera-relative placement / scale**
   On the surviving 23-frame run, JOSH projects far off the subject and far too large.
3. **Residual pose quality**
   Even after 2D alignment, the surviving JOSH run is still materially worse than the footwork control.

That is why the repo now treats this as a mixed `coverage + placement/scale + pose quality` gate rather than a pure “missing frames” issue.
