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
- the one surviving short JOSH slice is also objectively worse than GVHMR on BRACE 2D

## Best Candidate Window

Window: `530–553` local (`4332–4355` global, end exclusive)

- JOSH 2D: `2.2126` bbox-diag frac, `PCK@0.2 = 0.0000`
- GVHMR 2D: `0.0988` bbox-diag frac, `PCK@0.2 = 0.8670`
- Recommendation on this slice: `keep_gvhmr_baseline`

This means the immediate next task is not a blind rerun. The next task is to improve or replace JOSH on this exact short slice first.

## Canonical Artifacts

- Report: `experiments/results/powermove_debug/bcone_seq4/powermove_report.md`
- Machine-readable report: `experiments/results/powermove_debug/bcone_seq4/powermove_report.json`
- Candidate table: `experiments/results/powermove_debug/bcone_seq4/candidate_windows.csv`
- Frame diagnostics: `experiments/results/powermove_debug/bcone_seq4/frame_diagnostics.csv`
- Review render: `experiments/results/powermove_debug/bcone_seq4/renders/comparison_landscape_530_553.mp4`

## Interpretation

The powermove path currently has two separate problems:

1. **Continuity / coverage**
   JOSH stops after 23 valid frames in the target segment.
2. **Pose quality**
   On the surviving 23-frame run, the BRACE 2D comparison still favors GVHMR strongly.

That is why the repo now treats this as a mixed `coverage + pose quality` gate rather than a pure “missing frames” issue.
