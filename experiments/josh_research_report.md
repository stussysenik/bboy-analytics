# JOSH Deep Research Report: Bboy Motion Capture Quality

**Date:** 2026-03-25
**Scope:** Full pipeline audit for bcone_seq4 (Red Bull BC One, 999 frames @ 30fps)

---

## Executive Summary

1. **JOSH is currently producing WORSE output than GVHMR for bboy.** Acceleration is 7x higher, jerk is 8.8x higher, and identity tracking FAILS with 8m teleportation spikes. The root cause is not the optimizer itself but the pipeline: fragmented TRAM tracks (only 24% frame coverage), a missing chunk (600-700 crashed on a DECO bug), and wildly inconsistent focal lengths across chunks (2.3x variation).

2. **The JOSH hyperparameters are tuned for slow walking demos and actively harm bboy capture.** `prior_loss_weight=100` anchors SMPL poses to TRAM's initial estimates (which are wrong for inversions). `smooth_loss_weight=0.1` is far too weak to prevent chunk-boundary discontinuities. `max_frames=21` with `opt_interval=5` creates 101-frame chunks with only 1-frame overlap -- zero stitching margin.

3. **Three high-impact fixes can transform quality:** (a) Fix the DECO crash bug to recover chunk 600-700, (b) lock focal length across all chunks since this is a fixed broadcast camera, (c) increase chunk overlap from 1 frame to 20+ frames with blended interpolation at boundaries.

---

## Q1: JOSH Hyperparameters for Bboy

**File:** `/teamspace/studios/this_studio/josh/josh/config.py`

### Current Defaults vs. Bboy Requirements

| Parameter | Default | Issue for Bboy | Recommendation |
|-----------|---------|---------------|----------------|
| `max_frames` | 21 | Only 21 images per chunk. With `opt_interval=5`, covers 101 frames. Adequate for slow motion, insufficient temporal context for fast transitions. | Keep 21 (GPU memory constrained), but increase overlap (see Q3). |
| `opt_interval` | 5 | Samples every 5th frame. At 30fps, that's 6fps effective input to MASt3R. A bboy doing windmills completes a full rotation in ~0.5s (15 frames). At 6fps, MASt3R sees only 3 images per rotation -- barely enough for scene reconstruction. | Reduce to 3 for power move segments (10fps effective). |
| `scene_graph` | `"window-10"` | Each image paired with its 10 nearest neighbors. For 21 images this means each frame connects to roughly half the chunk. Reasonable. | Keep as-is. |
| `smooth_loss_weight` | 0.1 | **Way too low.** The smooth loss (`loss_smooth` at `josh_utils.py:529-546`) penalizes acceleration (second-order finite differences of joint velocities). At 0.1, the optimizer barely enforces temporal coherence. GVHMR's built-in temporal model implicitly has much stronger smoothness. | Increase to 1.0-5.0 for bboy. |
| `prior_loss_weight` | 100 | **Disastrously high.** The prior loss (`josh_utils.py:517-527`) penalizes deviation from TRAM's initial SMPL estimates with equal weight on quaternion, translation, body pose, and shape. For bboy, TRAM's estimates are often wrong (inversions, unusual poses). A weight of 100 locks the optimizer to TRAM's mistakes. | Reduce to 10.0-20.0. Let the scene-human coupling do more work. |
| `static_loss_weight` | 0.1 | Penalizes contact vertex displacement between consecutive frames. Reasonable but too weak relative to prior. | Increase to 0.5-1.0. |
| `scale_loss_weight` | 1.0 | Penalizes depth mismatch between SMPL contact vertices and the scene point cloud. This is the core scene-human coupling. | Keep 1.0 but ensure it has data (see DECO analysis). |
| `optimize_depth` | **False** | Depth maps from MASt3R/Pi3X are frozen during refinement (line 707: `core_depth[i].requires_grad_(opt_depth)`). For a dynamic scene with crowd movement and stage lighting, the initial depth estimate may be wrong. | Test with True -- allows depth correction during 2D reprojection refinement. |
| `update_correspondences` | **False** | When True, enables dynamic re-projection of SMPL contact vertices at each optimization step (`loss_scale_opt_corres` at line 434). When False, uses pre-computed static correspondences. Dynamic is more accurate but slower. | Test with True for quality improvement. |
| `conf_thres` | 0.1 | Very low confidence threshold for MASt3R points. Allows noisy points into the scene reconstruction. | Increase to 0.3 for broadcast footage (plenty of signal). |
| `depth_filter_ratio` | 1.01 | Filters out depth correspondences where SMPL depth and scene depth differ by more than 1%. **Extremely tight** -- even slight depth errors cause rejection. | Increase to 1.1-1.2 for bboy (body parts at various depths). |

### Critical Observation: Optimization Iterations

In `inference.py:198-215`, the optimization runs:
- **Coarse (3D matching):** `lr1=0.07, niter1=500`
- **Fine (2D reprojection):** `lr2=0.014, niter2=200`

These are hardcoded in the `inference()` function, not in config. 500+200 = 700 total iterations. For fast-moving subjects, more iterations may be needed. However, the coarse phase only optimizes camera poses (core_depth frozen), and the fine phase has `opt_depth=False` by default, so depth is NEVER optimized.

---

## Q2: JOSH Optimization Core

**Files:**
- `/teamspace/studios/this_studio/josh/josh/joint_opt.py` (setup + result extraction)
- `/teamspace/studios/this_studio/josh/josh/utils/josh_utils.py:180-718` (optimizer)

### Architecture

JOSH jointly optimizes:
1. **Scene reconstruction** (MASt3R sparse alignment + Pi3X depth)
2. **Camera intrinsics/extrinsics** (focal, pose per sampled frame)
3. **SMPL human parameters** (quaternion, translation, body pose, shape)

### Loss Functions (6 total)

The optimizer in `sparse_scene_optimizer` runs two phases:

**Phase 1 (Coarse): 3D Point Matching** -- `loss_3d` (line 561-589)
- Matches 3D points between image pairs using MASt3R correspondences
- Weighted by correspondence confidence
- Camera poses (quats, trans, log_sizes) are trainable; depth is frozen

**Phase 2 (Fine): 2D Reprojection** -- `loss_2d` (line 591-605)
- Projects 3D points to 2D and compares with MASt3R pixel correspondences
- Focal length becomes trainable; depth remains frozen by default
- `loss_dust3r_w=0.0` (disabled) -- fallback DUSt3R regression for low-conf pairs

**Human-Scene Coupling Losses** (added in both phases):
1. **`loss_scale`** (line 494-505): Matches SMPL contact vertex depth to scene depthmap depth. This is the PRIMARY scene-human coupling. Uses DECO contacts projected to the closest valid scene point.
2. **`loss_static`** (line 507-515): Penalizes movement of contact vertices between consecutive frames in world space. Enforces "if your foot is on the ground, it stays on the ground."
3. **`loss_prior`** (line 517-527): L2 penalty between optimized SMPL params and TRAM initial estimates. Equal weight on quaternion, translation, pose, and shape. At weight=100, this dominates.
4. **`loss_smooth`** (line 529-546): Second-order smoothness on JOINT positions in world space. Penalizes acceleration: `(v_{t+1} - v_t) vs (v_t - v_{t-1})`, normalized by frame spacing.

### Key Insight: Contact-Scene Coupling is the Innovation

The `loss_scale` function (lines 494-505 for static mode) works by:
1. Finding SMPL vertices marked as "in contact" by DECO
2. Projecting those vertices to 2D
3. Finding the closest valid (non-human-masked) scene point
4. Comparing depth: scene depth vs SMPL vertex depth
5. Weighted by scene confidence

This is what makes JOSH different from TRAM alone -- it grounds the human in the 3D scene. But this only works if:
- DECO provides accurate contacts (questionable for bboy -- see Q5)
- Scene reconstruction is good (questionable -- see Q6)
- The depth_filter_ratio doesn't reject everything (at 1.01, it likely does)

---

## Q3: Chunk Processing & Aggregation

**Files:**
- `/teamspace/studios/this_studio/josh/josh/inference_long_video.py` (chunk dispatch)
- `/teamspace/studios/this_studio/josh/josh/aggregate_results.py` (stitching)

### Chunk Structure

Chunks step by 100 frames with `CHUNK_FRAMES=21` images sampled every 5 frames:
```
Chunk 0:   frames 0-100   (21 images: 0, 5, 10, ..., 100)
Chunk 100: frames 100-200 (21 images: 100, 105, ..., 200)
Chunk 200: frames 200-300 (21 images: 200, 205, ..., 300)
...
```

### Overlap: Only 1 Frame

Each chunk shares only frame 100 with the next chunk. The aggregation in `aggregate_results.py:50-52` chains camera poses:
```python
pred_cam = np.einsum("ij, bjk->bik", pred_cams[-1][-1], pred_cam)
pred_cams.append(pred_cam[1:])  # Skip the first (shared) frame
```

This means:
- Camera alignment depends entirely on that single shared frame
- No blending or averaging at boundaries
- Errors in any single chunk propagate to all subsequent chunks
- **For bboy: if the shared frame has poor scene reconstruction (e.g., motion blur during a power move), the entire downstream chain drifts**

### SMPL Aggregation (lines 93-121)

For each track, per-chunk JOSH results overwrite the TRAM originals:
```python
pred_rotmat[frame_mask] = josh_smpl_dict['pred_rotmat']
pred_shape[frame_mask] = josh_smpl_dict['pred_shape']
pred_trans[frame_mask] = josh_smpl_dict['pred_trans'].squeeze(1)
```

No blending at overlapping frames. Last-writer-wins.

### Missing Chunk 600-700

Chunk 600 crashed (see Q9), leaving a gap. The aggregation skips this chunk entirely. For TRAM tracks with frames in 600-700, JOSH reverts to unoptimized TRAM estimates, creating a quality discontinuity.

### Current Chunk Status

| Chunk | Frames | Status | Focal |
|-------|--------|--------|-------|
| josh_0-100 | 0-100 | OK | 1010.31 |
| josh_100-200 | 100-200 | OK | 773.57 |
| josh_200-300 | 200-300 | OK | 560.09 |
| josh_300-400 | 300-400 | OK | 586.29 |
| josh_400-500 | 400-500 | OK | 600.96 |
| josh_500-600 | 500-600 | OK | 552.97 |
| josh_600-700 | 600-700 | **CRASHED** | N/A |
| josh_700-800 | 700-800 | OK | 633.27 |
| josh_800-998 | 800-998 | OK (oversized) | N/A |
| josh_900-998 | 900-998 | OK | 1248.42 |

**Note:** josh_800-998 and josh_900-998 overlap. The 800-998 chunk processes 199 remaining frames (~40 images), which is 2x the normal chunk size.

---

## Q4: TRAM Track Selection

**Data:** `/teamspace/studios/this_studio/josh_input/bcone_seq4/tram/`

### 22 Tracks, Massive Fragmentation

| Track | Frames | Range | Likely Identity |
|-------|--------|-------|----------------|
| track_0 | 240 | 45-824 | Primary dancer (largest) |
| track_1 | 210 | 420-764 | Second dancer or re-ID of same |
| track_2 | 115 | 525-839 | Likely another re-ID |
| All others | 17-66 each | Various | Crowd, judges, cameramen |

### Critical Problems

1. **Track 0 only covers 240 of 999 frames (24%)** with 4 major gaps:
   - Frames 75-269 (195 frames, 6.5 seconds)
   - Frames 300-434 (135 frames, 4.5 seconds)
   - Frames 480-644 (165 frames, 5.5 seconds)
   - Frames 735-779 (45 frames, 1.5 seconds)

2. **468 of 999 frames have NO track at all** -- nearly half the video has zero human detection.

3. **Identity tracking = FAIL.** The metadata confirms max root displacement of 8.0m at frame 59. Track 0 likely conflates the active bboy with other people when the dancer exits/re-enters frame or gets occluded by camera cuts.

4. **No multi-track fusion.** The joint extraction script (`extract-joints-josh.py`) only extracts a single track (default track_id=0). It doesn't attempt to stitch fragmented tracks from the same person.

### Root Cause

TRAM's person tracker loses the bboy during:
- Power moves (body inverted, tracker can't match)
- Camera cuts/transitions (BC One broadcast footage)
- Crowd occlusions

### Recommendation

Need a track merging strategy:
1. Use bounding box IoU and temporal proximity to merge fragmented tracks
2. Or use the `--auto-select` mode in extract-joints-josh.py which calls `pipeline.track_select.select_best_segments`
3. Long-term: pre-process with a more robust tracker (e.g., CoTracker3 + re-ID)

---

## Q5: Contact Estimation (DECO)

**Data:** `/teamspace/studios/this_studio/josh_input/bcone_seq4/deco/`
**Code:** `/teamspace/studios/this_studio/josh/preprocess/run_deco.py`

### What DECO Provides

DECO estimates per-vertex contact probability on the 6890-vertex SMPL mesh. It produces a binary mask (threshold 0.5) per frame.

**Track 0 statistics:**
- Mean contact vertices per frame: 706.6 (10.3% of mesh)
- Range: 282 - 1769 vertices
- High contact frames (>1000 verts): 50/240 (21%)
- Low contact frames (<400 verts): 80/240 (33%)

### How DECO is Used in JOSH

In `joint_opt.py:362-434`, DECO contacts are used to build `smpl_scale_info`:
1. For each frame, find SMPL vertices where `contact[frame] == 1`
2. For each of 45 joints, find the closest contact vertex within 0.05m
3. Project these joint-proximate contact points to 2D
4. Find the closest valid (non-masked) scene point
5. Compute depth offset ratio for `loss_scale`

Additionally, `smpl_static_info` (line 405-406) tracks contact vertices that persist between consecutive frames for `loss_static`.

### Problems for Bboy

1. **DECO was trained on standing/walking poses.** It doesn't understand headspins (head-ground contact) or handstands (hand-ground contact in unusual orientations). The high contact count (700+ vertices) suggests it's over-predicting contact.

2. **The 0.05m proximity threshold is extremely tight.** SMPL joints are at bone centers, not surface contact points. For a bboy in a handstand, the wrist joint is ~0.05m from the palm surface -- barely within threshold. For a headspin, the head joint is ~0.10m from the scalp -- outside threshold.

3. **The crash at chunk 600** (`joint_opt.py:379`) happens because `points` is empty after filtering. DECO marks vertices as contacting, but none are within 0.05m of any joint. The `torch.min()` call on an empty tensor raises IndexError. This is a **code bug** -- needs a guard.

---

## Q6: Scene Reconstruction Quality

### Pi3X Depth Model

JOSH uses Pi3X (a video depth estimator) to initialize depth maps before MASt3R optimization. Each chunk independently runs Pi3X on its 21 images.

### Focal Length Instability

The optimized focal length varies wildly across chunks:
```
Chunk 0-100:   1010.31
Chunk 100-200:  773.57
Chunk 200-300:  560.09
Chunk 300-400:  586.29
Chunk 400-500:  600.96
Chunk 500-600:  552.97
Chunk 700-800:  633.27
Chunk 900-998: 1248.42
```

**Range: 553 to 1248 (2.3x variation)**

This is physically impossible for a fixed broadcast camera. The actual focal length is constant across the entire video. This variation means:
- Scene scale is inconsistent between chunks
- Camera pose chain accumulates scale drift
- Human translation in world space is wrong

**Root cause:** Each chunk independently estimates focal length using MASt3R confidence weighting, then optimizes it. Dynamic content (crowd, dancers) confuses the estimator differently in each chunk.

### Recommendation

**Lock focal length across all chunks.** Either:
1. Set `optimize_focal: False` with a known focal from camera metadata
2. Estimate focal once from the best chunk (e.g., a static establishing shot) and pass via `init_focal` to all chunks
3. Average the per-chunk estimates: ~700 pixels for 512-wide images (reasonable for broadcast)

---

## Q7: What We're NOT Using

### `optimize_depth = False` (IMPACTFUL)

In the fine phase (2D reprojection, `josh_utils.py:702-707`):
```python
core_depth[i].requires_grad_(opt_depth)  # False by default
```

Depth maps are frozen. The scene geometry from MASt3R/Pi3X is taken as ground truth. For a crowded BC One stage with dynamic lighting, reflective floor, and people moving in the background, the initial depth is likely wrong in places. Enabling depth optimization would allow the optimizer to correct these errors.

**Recommendation:** Test with `optimize_depth=True`. May improve scene-human coupling accuracy.

### `update_correspondences = False` (MODERATE IMPACT)

When False (default), SMPL-scene depth correspondences are computed once before optimization and stored as static tensors (`smpl_scale_info` entries at line 433). The projected 2D positions are fixed.

When True, correspondences are recomputed at each iteration using the current SMPL vertex positions (`loss_scale_opt_corres` at line 434-492). This is more accurate because:
- As the optimizer moves the SMPL body, the projected 2D contact points shift
- Static correspondences become stale as the body moves

**Recommendation:** Test with `update_correspondences=True`. More compute per iteration but better coupling.

### Post-Processing

The JOSH codebase has **no post-processing** after optimization:
- No temporal smoothing across chunks
- No physics-based refinement (ground plane constraint, penetration avoidance)
- No confidence-weighted blending at chunk boundaries

The aggregation is purely mechanical -- camera chain multiplication + SMPL parameter overwriting.

### `loss_dust3r_w = 0.0` (DISABLED)

The DUSt3R regression fallback for low-confidence pairs is disabled. For challenging BC One footage, some pairs may have low matching confidence, and the fallback could provide useful constraints.

---

## Q8: Existing Output Quality

### JOSH Output

- **Shape:** (240, 24, 3) -- 240 frames, 24 joints, XYZ
- **Coverage:** Frames 45-824 with 4 major gaps (only track 0)
- **Root speed:** mean 0.225 m/frame, max 8.00 m/frame (physically impossible)
- **Identity tracking:** FAIL (8m teleport at frame 59-60)
- **Inverted frames:** 10/240 (4.2%)

### GVHMR Output

- **Shape:** (999, 24, 3) -- full video coverage
- **Root speed:** mean 0.024 m/frame, max 0.118 m/frame (plausible)
- **Smoothness:** 7x less acceleration, 8.8x less jerk

### Quantitative Comparison

| Metric | JOSH | GVHMR | Winner |
|--------|------|-------|--------|
| Frame coverage | 240/999 (24%) | 999/999 (100%) | GVHMR |
| Root speed mean | 0.225 m/frame | 0.024 m/frame | GVHMR |
| Root speed max | 8.00 m/frame | 0.118 m/frame | GVHMR |
| Acceleration (mean) | 0.287 | 0.041 | GVHMR |
| Jerk (mean) | 0.556 | 0.063 | GVHMR |
| Z-range | 9.54m | 2.96m | GVHMR (camera-relative depth should be stable) |
| Identity tracking | FAIL | N/A (single-person) | GVHMR |

### Speed Spike Analysis

Major discontinuities in JOSH output:
- **Frame 59->60:** 8.0 m/frame (Z jumps from 9.25 to 1.89 -- 7.36m depth teleport)
- **Frame 25->26:** 5.24 m/frame (Z jumps from 4.84 to 10.08)
- **Frames 40-45:** Consistent 1.56 m/frame drift (linear interpolation artifact)

These spikes correspond to:
1. Track identity swaps (different person detected)
2. Chunk boundary camera pose discontinuities
3. Interpolation across frames where TRAM lost tracking

### Verdict

JOSH output is **currently unusable** for musicality analysis. The 8m teleportation spikes would dominate any velocity-based metric. GVHMR, despite lacking scene grounding, produces far smoother and more plausible trajectories.

---

## Q9: Pipeline Gaps

**Script:** `/teamspace/studios/this_studio/josh_batch_job.sh`

### Full Pipeline

1. **DECO** contact estimation (per-track, per-frame)
2. **JOSH inference** (per-chunk: Pi3X depth -> MASt3R scene -> joint optimization)
3. **Aggregation** (stitch chunks, merge camera chains)
4. **Joint extraction** (SMPL forward kinematics -> 3D joints)

### Issues Found

1. **Chunk 600-700 crashed** with `IndexError: min(): Expected reduction dim 0 to have non-zero size.` at `joint_opt.py:379`. Root cause: DECO contact vertices exist but none are within 0.05m of any SMPL joint. Code needs a guard for empty `points` tensor.

2. **SAM3 OOM in preprocessing** (josh_pipeline.log tail): SAM3 video segmentation ran out of memory at frame 7 of a 15-frame chunk. This means masks may be incomplete for some frames, degrading scene reconstruction.

3. **Missing SAM3 preprocessing checkpoint/resume**: The SAM3 crash was not gracefully handled. Masks should be verified before JOSH inference.

4. **Overlapping last chunks**: Both josh_800-998 and josh_900-998 exist. The aggregation processes josh_800-998 first, then josh_900-998 overwrites its frames 900-998. This is wasteful and the 800-998 oversized chunk (40 images) may have different quality characteristics than standard 21-image chunks.

5. **No focal consistency enforcement**: Each chunk independently estimates focal length. There's no mechanism to share or constrain focal across chunks.

---

## Prioritized Recommendations

### P0: Critical Fixes (Must Do)

| # | Change | Expected Impact | Effort |
|---|--------|----------------|--------|
| 1 | **Fix DECO crash bug** at `joint_opt.py:379` -- add `if len(points) == 0: continue` guard | Recovers chunk 600-700 | 5 min |
| 2 | **Lock focal length** -- add `init_focal=700.0` to config (or compute once and pass to all chunks) | Eliminates scale inconsistency across chunks, fixes depth teleportation | 15 min |
| 3 | **Re-run chunk 600-700** after bug fix | Fills the gap | 30 min (GPU) |

### P1: High Impact Tuning

| # | Change | Expected Impact | Effort |
|---|--------|----------------|--------|
| 4 | **Reduce `prior_loss_weight`** from 100.0 to 10.0-20.0 | Allows optimizer to correct TRAM errors for unusual poses | Config change |
| 5 | **Increase `smooth_loss_weight`** from 0.1 to 1.0-5.0 | Reduces inter-frame jitter and chunk boundary discontinuities | Config change |
| 6 | **Increase `depth_filter_ratio`** from 1.01 to 1.1-1.2 | Allows more SMPL-scene depth correspondences (currently rejects almost everything) | Config change |
| 7 | **Increase chunk overlap** from 1 frame to 20+ frames, add blended interpolation at boundaries | Eliminates chunk boundary teleportation | Moderate code change in aggregate_results.py |

### P2: Quality Improvements

| # | Change | Expected Impact | Effort |
|---|--------|----------------|--------|
| 8 | **Enable `optimize_depth=True`** | Corrects depth errors from Pi3X, better scene-human coupling | Config change |
| 9 | **Enable `update_correspondences=True`** | Dynamic SMPL-scene correspondences, better coupling | Config change |
| 10 | **Increase contact proximity threshold** from 0.05m to 0.10m in `joint_opt.py:380` | More contact constraints for unusual poses (headspins, handstands) | Code change |
| 11 | **Add post-processing smoothing** -- Savitzky-Golay or Butterworth filter on joint positions after aggregation | Removes residual jitter | New code |
| 12 | **Increase `conf_thres`** from 0.1 to 0.3 | Fewer noisy scene points | Config change |

### P3: Track Quality (Upstream)

| # | Change | Expected Impact | Effort |
|---|--------|----------------|--------|
| 13 | **Implement track merging** -- stitch fragmented TRAM tracks from the same person | Increases coverage from 24% to potentially 50%+ | Significant |
| 14 | **Use robust person tracker** (e.g., ByteTrack with re-ID) as TRAM input | Fixes identity tracking failure | Significant |
| 15 | **Reduce `opt_interval`** from 5 to 3 for power move segments | 10fps input to MASt3R instead of 6fps | Config + GPU cost |

---

## Specific Config Changes to Test

### Experiment A: Conservative Fix (P0 only)
```python
# Fix bug, lock focal, re-run 600-700
JOSHConfig(
    init_focal=700.0,         # Lock focal (was: None / auto-estimated)
    # Everything else unchanged
)
```

### Experiment B: Bboy-Tuned (P0 + P1)
```python
JOSHConfig(
    init_focal=700.0,
    prior_loss_weight=15.0,    # Was: 100.0
    smooth_loss_weight=2.0,    # Was: 0.1
    static_loss_weight=0.5,    # Was: 0.1
    depth_filter_ratio=1.15,   # Was: 1.01
    conf_thres=0.2,            # Was: 0.1
)
```

### Experiment C: Full Quality (P0 + P1 + P2)
```python
JOSHConfig(
    init_focal=700.0,
    prior_loss_weight=15.0,
    smooth_loss_weight=2.0,
    static_loss_weight=0.5,
    depth_filter_ratio=1.15,
    conf_thres=0.2,
    optimize_depth=True,        # Was: False
    update_correspondences=True, # Was: False
)
# Also: change contact threshold in joint_opt.py:380 from 0.05 to 0.10
```

### Code Changes Required

**1. Bug fix** (`joint_opt.py:375-380`):
```python
# Before line 379, add:
if len(points) == 0:
    continue
```

**2. Chunk overlap** (`inference_long_video.py:56`):
```python
# Change step from 100 to 80 (20-frame overlap)
for start_frame in range(0, num_frames, 80):
```

**3. Aggregation blending** (`aggregate_results.py:50-52`):
```python
# Instead of skipping first frame, blend overlapping 20 frames
# Weighted average: w = overlap_position / overlap_length
```

---

## Appendix: Key File Paths

| File | Purpose |
|------|---------|
| `/teamspace/studios/this_studio/josh/josh/config.py` | All hyperparameters |
| `/teamspace/studios/this_studio/josh/josh/joint_opt.py` | SMPL-scene coupling setup, bug at line 379 |
| `/teamspace/studios/this_studio/josh/josh/utils/josh_utils.py:180-718` | Core optimizer with all loss functions |
| `/teamspace/studios/this_studio/josh/josh/inference.py` | Per-chunk inference, hardcoded lr/niter |
| `/teamspace/studios/this_studio/josh/josh/inference_long_video.py` | Chunk dispatch, 100-frame stepping |
| `/teamspace/studios/this_studio/josh/josh/aggregate_results.py` | Chunk stitching, camera chain, SMPL merging |
| `/teamspace/studios/this_studio/josh/preprocess/run_deco.py` | DECO contact estimation |
| `/teamspace/studios/this_studio/poc/remote/extract-joints-josh.py` | Joint extraction from JOSH output |
| `/teamspace/studios/this_studio/josh_batch_job.sh` | Pipeline orchestration |
| `/teamspace/studios/this_studio/josh_input/bcone_seq4/logs/600.log` | Crash log for missing chunk |
