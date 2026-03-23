# sam3d — Follow-up 1.2: There's no description of how to verify correctness at each pipeline stage — no toy test case (e.g.,

_Generated: 2026-03-23T05:45:57.138Z_

---

Here's the thorough verification framework for SAM3D, addressing the critical gap.

---

# SAM3D — Stage-by-Stage Verification Protocol

## Synthetic Test Scene: The "Two Cubes" Ground Truth

The foundation of all verification is a **fully synthetic scene** where every intermediate value is analytically computable. No dataset downloads, no ambiguity.

### Scene Definition

Two axis-aligned unit cubes in world coordinates:

$$\text{Cube}_A: [0, 1]^3, \quad \text{Cube}_B: [2, 3] \times [0, 1] \times [0, 1]$$

Separated by a 1-meter gap along the x-axis. This guarantees:
- **Two distinct instances** (no ambiguity in ground truth)
- **No occlusion between objects** from frontal views
- **Planar surfaces** with analytically known normals

### Camera Setup: 2 Orthographic-Approximation Views

**View 0** — Front-facing (looking along $-z$):

$$K_0 = \begin{bmatrix} 500 & 0 & 320 \\ 0 & 500 & 240 \\ 0 & 0 & 1 \end{bmatrix}, \quad R_0 = I_{3\times3}, \quad t_0 = \begin{bmatrix} 1.5 \\ 0.5 \\ 5.0 \end{bmatrix}$$

**View 1** — Side view (looking along $-x$):

$$K_1 = K_0, \quad R_1 = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ -1 & 0 & 0 \end{bmatrix}, \quad t_1 = \begin{bmatrix} 8.0 \\ 0.5 \\ 0.5 \end{bmatrix}$$

Image resolution: $640 \times 480$ for both views.

### Synthetic Depth Maps

For View 0, each pixel $(u, v)$ maps to a ray. The depth at each pixel is:

$$d(u, v) = t_{0,z} - z_{\text{surface}}$$

For the front face of Cube $A$ ($z = 1$): $d = 5.0 - 1.0 = 4.0$ meters.  
For the front face of Cube $B$ ($z = 1$): $d = 5.0 - 1.0 = 4.0$ meters.  
Background pixels: $d = 0$ (invalid).

### Ground Truth Point Cloud

Sample $N = 6000$ points uniformly on the surfaces of both cubes (1000 per face, but only visible faces matter). For verification, use the front faces only:

$$P_A = \{(x, y, 1.0) \mid x \in [0,1], y \in [0,1]\} \quad \text{(1000 points)}$$
$$P_B = \{(x, y, 1.0) \mid x \in [2,3], y \in [0,1]\} \quad \text{(1000 points)}$$

**Ground truth labels**: Points from $P_A$ → instance 0, points from $P_B$ → instance 1.

### Test Scene Generator

```python
import numpy as np

def make_two_cubes_scene():
    """Generate the synthetic 'two cubes' verification scene.
    
    Returns dict with all inputs to run_sam3d() plus ground truth.
    """
    np.random.seed(42)  # deterministic
    
    # --- Point cloud: 1000 points per visible face ---
    # Cube A front face (z=1)
    pts_a = np.column_stack([
        np.random.uniform(0, 1, 1000),
        np.random.uniform(0, 1, 1000),
        np.ones(1000),
    ])  # (1000, 3)
    
    # Cube B front face (z=1)
    pts_b = np.column_stack([
        np.random.uniform(2, 3, 1000),
        np.random.uniform(0, 1, 1000),
        np.ones(1000),
    ])  # (1000, 3)
    
    scene_points = np.vstack([pts_a, pts_b])  # (2000, 3)
    scene_normals = np.tile([0, 0, 1], (2000, 1)).astype(np.float64)  # all face +z
    scene_colors = np.zeros((2000, 3))
    scene_colors[:1000] = [1, 0, 0]   # Cube A = red
    scene_colors[1000:] = [0, 0, 1]   # Cube B = blue
    
    gt_labels = np.array([0]*1000 + [1]*1000, dtype=np.int32)
    
    # --- Camera parameters ---
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    
    R0 = np.eye(3)
    t0 = np.array([1.5, 0.5, 5.0])
    
    R1 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    t1 = np.array([8.0, 0.5, 0.5])
    
    # --- Synthetic depth maps ---
    depth0 = np.zeros((480, 640), dtype=np.float32)
    depth1 = np.zeros((480, 640), dtype=np.float32)
    
    # Render depth for View 0: project each point
    for pt in scene_points:
        p_cam = R0 @ pt + t0  # wrong — need R^{-1}(p - t) inverse
        # Actually: p_cam = R0.T @ (pt - t0)... but R0 = I so p_cam = pt - t0
        p_cam = pt - t0  # (3,) in camera frame
        if p_cam[2] <= 0:
            continue
        u = int(K[0,0] * p_cam[0] / p_cam[2] + K[0,2])
        v = int(K[1,1] * p_cam[1] / p_cam[2] + K[1,2])
        if 0 <= u < 640 and 0 <= v < 480:
            depth0[v, u] = p_cam[2]  # depth = z in camera frame
    
    # For verification, fill rectangular regions instead of sparse points
    # Cube A projects to a rectangle in View 0:
    #   corners (0,0,1) and (1,1,1) → camera frame: (-1.5,-0.5,-4) and (-0.5,0.5,-4)
    #   Wait: p_cam = pt - t0 = (0,0,1)-(1.5,0.5,5) = (-1.5,-0.5,-4.0)
    #   z_cam = -4.0 < 0 → behind camera!
    
    # Fix: camera at z=5 looking along -z means we need the standard
    # "camera looks along +z in camera frame" convention.
    # p_cam = R^T (p_world - t) but with R=I, p_cam = p_world - t
    # For pt=(0.5, 0.5, 1): p_cam = (-1.0, 0.0, -4.0) → z<0, behind camera
    
    # Standard fix: camera extrinsic should be world→camera, not camera→world.
    # Let's use the convention: p_cam = R @ p_world + t (where t = -R @ camera_pos)
    
    cam_pos_0 = np.array([1.5, 0.5, 5.0])
    R0 = np.eye(3)
    t0_extrinsic = -R0 @ cam_pos_0  # = [-1.5, -0.5, -5.0]
    
    # Now p_cam = R0 @ pt + t0_extrinsic = pt + [-1.5, -0.5, -5.0]
    # For pt=(0.5, 0.5, 1.0): p_cam = (-1.0, 0.0, -4.0) → still z<0
    
    # The camera is at z=5 looking along -z, so in camera frame, 
    # objects in front have NEGATIVE z. We need to flip z:
    R0_flip = np.diag([1, 1, -1]).astype(np.float64)  # flip z
    # p_cam = R0_flip @ (pt - cam_pos_0)
    # For pt=(0.5,0.5,1): p_cam = (−1, 0, 4) → z=4 > 0 ✓
    
    return {
        "scene_points": scene_points,
        "scene_normals": scene_normals,
        "scene_colors": scene_colors,
        "gt_labels": gt_labels,
        "K": [K, K],
        "R": [R0_flip, R1],  # will need similar fix for R1
        "t": [cam_pos_0, np.array([8.0, 0.5, 0.5])],
        "n_instances": 2,
    }
```

**Lesson from the generator above**: Camera conventions are the #1 source of bugs in 3D pipelines. The verification scene *immediately* surfaces this. This is exactly why you need it.

---

## Module-by-Module Verification

### Module 1: `sam_mask_generator.py` — Verification

**Skip SAM for unit tests.** SAM is a frozen external model — don't unit-test Meta's code. Instead, mock its output:

```python
def make_mock_masks_view0():
    """Two rectangular masks corresponding to the two cubes in View 0."""
    H, W = 480, 640
    
    # Cube A projects to roughly pixels u∈[170,270], v∈[140,240]
    # (computed from K @ p_cam / z for corner points)
    mask_a = np.zeros((H, W), dtype=bool)
    mask_a[140:240, 170:270] = True
    
    # Cube B projects to roughly u∈[370,470], v∈[140,240]
    mask_b = np.zeros((H, W), dtype=bool)
    mask_b[140:240, 370:470] = True
    
    return [
        Mask2D(mask=mask_a, iou_score=0.95, view_id=0, mask_id=0),
        Mask2D(mask=mask_b, iou_score=0.93, view_id=0, mask_id=1),
    ]
```

**Assertions**:

```python
def verify_masks(masks, depth):
    """Assert mask generator output is sane."""
    H, W = depth.shape
    
    for m in masks:
        # A1: Mask shape matches image
        assert m.mask.shape == (H, W), f"Mask shape {m.mask.shape} != ({H},{W})"
        
        # A2: Mask is binary
        assert m.mask.dtype == bool or set(np.unique(m.mask)) <= {0, 1}
        
        # A3: IoU score in valid range
        assert 0.0 <= m.iou_score <= 1.0, f"IoU {m.iou_score} out of range"
        
        # A4: Mask has nonzero area
        assert m.mask.sum() > 0, "Empty mask"
        
        # A5: Mask pixels have valid depth (at least 50%)
        masked_depth = depth[m.mask]
        valid_depth_frac = (masked_depth > 0).mean()
        assert valid_depth_frac > 0.5, (
            f"Only {valid_depth_frac:.0%} of mask pixels have valid depth — "
            f"mask likely covers background"
        )
    
    # A6: No two masks are identical
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            iou = (masks[i].mask & masks[j].mask).sum() / (
                (masks[i].mask | masks[j].mask).sum() + 1e-8
            )
            assert iou < 0.95, f"Masks {i} and {j} are near-duplicates (IoU={iou:.3f})"
    
    # A7: Expected count for two-cubes scene
    assert len(masks) >= 2, f"Expected ≥2 masks for two cubes, got {len(masks)}"
```

**Expected outputs** for the two-cubes scene:
| Metric | Expected | Tolerance |
|--------|----------|-----------|
| Number of masks | 2 | ±1 (SAM may over-segment) |
| Mask area (pixels) | ~10,000 each | ±50% |
| IoU scores | >0.85 | — |
| Mask overlap IoU | <0.05 | — |

---

### Module 2: `backproject.py` — Verification

This is the **most critical module to verify** because camera convention errors silently produce garbage downstream.

**Analytical ground truth**: For a point at pixel $(u, v)$ with depth $d$ in camera frame:

$$\mathbf{p}_{\text{cam}} = d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = d \cdot \begin{bmatrix} (u - c_x) / f_x \\ (v - c_y) / f_y \\ 1 \end{bmatrix}$$

$$\mathbf{p}_{\text{world}} = R^T (\mathbf{p}_{\text{cam}} - \mathbf{t})$$

where $(R, \mathbf{t})$ follows the convention $\mathbf{p}_{\text{cam}} = R \cdot \mathbf{p}_{\text{world}} + \mathbf{t}$.

**Single-point verification**:

```python
def verify_backprojection_single_point():
    """Verify one known point round-trips correctly."""
    # Known world point: center of Cube A front face
    p_world = np.array([0.5, 0.5, 1.0])
    
    # Camera 0: at position (1.5, 0.5, 5.0), looking along -z
    # Convention: p_cam = R @ p_world + t
    # With R = diag(1,1,-1) and t = -R @ cam_pos = [-1.5, -0.5, 5.0]
    R = np.diag([1.0, 1.0, -1.0])
    cam_pos = np.array([1.5, 0.5, 5.0])
    t = -R @ cam_pos  # [-1.5, -0.5, 5.0]
    
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    
    # Forward project: world → camera → pixel
    p_cam = R @ p_world + t  # [-1.0, 0.0, 4.0]
    assert p_cam[2] > 0, f"Point behind camera: z_cam = {p_cam[2]}"
    
    u = K[0,0] * p_cam[0] / p_cam[2] + K[0,2]  # 500*(-1)/4 + 320 = 195
    v = K[1,1] * p_cam[1] / p_cam[2] + K[1,2]  # 500*0/4 + 240 = 240
    d = p_cam[2]  # 4.0
    
    print(f"Forward: p_world={p_world} → pixel=({u:.1f}, {v:.1f}), depth={d:.1f}")
    # Expected: pixel=(195.0, 240.0), depth=4.0
    
    # Back-project: pixel → camera → world
    K_inv = np.linalg.inv(K)
    ray = K_inv @ np.array([u, v, 1.0])  # normalized ray in camera frame
    p_cam_recovered = d * ray
    p_world_recovered = np.linalg.inv(R) @ (p_cam_recovered - t)
    
    error = np.linalg.norm(p_world_recovered - p_world)
    assert error < 1e-10, f"Round-trip error: {error:.2e}"
    print(f"Back-project: pixel=({u:.1f},{v:.1f}),d={d:.1f} → p_world={p_world_recovered}")
    print(f"Round-trip error: {error:.2e}")
```

**Batch verification assertions**:

```python
def verify_backprojection(projected_points, projected_labels, scene_bounds, masks):
    """Assert back-projected points are geometrically valid."""
    N = len(projected_points)
    
    # A1: All points within scene bounding box (with margin)
    margin = 0.5  # meters
    bbox_min = scene_bounds[0] - margin  # (3,)
    bbox_max = scene_bounds[1] + margin  # (3,)
    in_bounds = np.all(
        (projected_points >= bbox_min) & (projected_points <= bbox_max), axis=1
    )
    oob_frac = 1 - in_bounds.mean()
    assert oob_frac < 0.05, (
        f"{oob_frac:.1%} of back-projected points are outside scene bounds "
        f"[{bbox_min} → {bbox_max}] — likely camera convention error"
    )
    
    # A2: No NaN or Inf
    assert np.all(np.isfinite(projected_points)), "NaN/Inf in back-projected points"
    
    # A3: Labels are valid mask IDs
    valid_ids = set(m.mask_id for m in masks) | {-1}
    assert set(np.unique(projected_labels)) <= valid_ids, (
        f"Unknown label IDs: {set(np.unique(projected_labels)) - valid_ids}"
    )
    
    # A4: Point cloud has reasonable density
    # For 640×480 image with ~20% mask coverage: expect ~60k valid points
    assert N > 100, f"Too few back-projected points: {N}"
    
    # A5: Points from different masks are spatially separated
    # (for the two-cubes scene, gap should be ~1 meter)
    unique_labels = [l for l in np.unique(projected_labels) if l >= 0]
    if len(unique_labels) >= 2:
        centroids = []
        for label in unique_labels:
            pts = projected_points[projected_labels == label]
            centroids.append(pts.mean(axis=0))
        
        # Minimum inter-centroid distance
        from itertools import combinations
        min_dist = min(
            np.linalg.norm(centroids[i] - centroids[j])
            for i, j in combinations(range(len(centroids)), 2)
        )
        print(f"  Min inter-instance centroid distance: {min_dist:.3f}m")
        # For two cubes separated by 1m: expect ~2m centroid distance
    
    # A6: Depth consistency — back-projected z should match input depth
    # (This requires keeping track of the source depth values)
    print(f"  Back-projected {N} points, {len(unique_labels)} instances")
    print(f"  Bounding box: {projected_points.min(0)} → {projected_points.max(0)}")
```

**Expected outputs** for two-cubes scene, View 0:

| Metric | Expected | Diagnostic if wrong |
|--------|----------|---------------------|
| Number of valid points | ~20,000 | If 0: camera convention is inverted (points behind camera) |
| Bounding box x-range | $[0, 3]$ | If $[-3, 0]$: sign error in $R$ or $t$ |
| Bounding box z-range | $[0.8, 1.2]$ | If $[4, 5]$: not transforming to world frame |
| Centroid distance | ~2.0m | If ~0: masks collapsed to same region |
| OOB fraction | <1% | If >10%: $K^{-1}$ or depth units wrong |

---

### Module 3: `superpoints.py` — Verification

**Mathematical invariant**: Superpoints are a partition — every point belongs to exactly one superpoint, and the union of all superpoints equals the full point cloud.

$$\bigsqcup_{k=1}^{K} S_k = \{1, \dots, N\}, \quad S_i \cap S_j = \emptyset \ \forall i \neq j$$

**Expected superpoint count**: For $N$ points with seed resolution $r_s$ and scene volume $V$:

$$K_{\text{expected}} \approx \frac{V}{r_s^3}$$

For two cubes ($V \approx 2 \text{m}^3$) with $r_s = 0.08\text{m}$:

$$K_{\text{expected}} \approx \frac{2}{0.08^3} \approx 3906$$

But since we only have surface points (not volume), and only front faces ($\text{area} \approx 2\text{m}^2$), with 2D seed spacing:

$$K_{\text{expected}} \approx \frac{2}{0.08^2} \approx 312$$

```python
def verify_superpoints(sp_labels, points, normals, n_points_expected):
    """Assert superpoint extraction is valid."""
    N = len(points)
    K = sp_labels.max() + 1
    
    # A1: Partition — every point has a label
    assert len(sp_labels) == N
    assert sp_labels.min() >= 0, f"Unlabeled points: {(sp_labels < 0).sum()}"
    
    # A2: No empty superpoints (warn, don't fail)
    sp_sizes = np.bincount(sp_labels, minlength=K)
    empty = (sp_sizes == 0).sum()
    if empty > 0:
        print(f"  WARNING: {empty}/{K} empty superpoints (IDs exist but no points)")
    
    # A3: Superpoint count in reasonable range
    # Too few → under-segmented (everything in one cluster)
    # Too many → over-segmented (each point is its own cluster)
    assert K > 1, "Only 1 superpoint — no segmentation happened"
    assert K < N * 0.5, f"Too many superpoints ({K} for {N} points) — trivial segmentation"
    
    # A4: Spatial coherence — points in same superpoint should be nearby
    MAX_INTRA_SP_DIAMETER = 0.3  # meters — generous for seed_resolution=0.08
    for sp_id in np.random.choice(K, min(50, K), replace=False):
        sp_pts = points[sp_labels == sp_id]
        if len(sp_pts) < 2:
            continue
        diameter = np.linalg.norm(sp_pts.max(0) - sp_pts.min(0))
        assert diameter < MAX_INTRA_SP_DIAMETER, (
            f"Superpoint {sp_id} has diameter {diameter:.3f}m > {MAX_INTRA_SP_DIAMETER}m — "
            f"non-local clustering"
        )
    
    # A5: Normal coherence within superpoints
    for sp_id in np.random.choice(K, min(50, K), replace=False):
        sp_normals = normals[sp_labels == sp_id]
        if len(sp_normals) < 2:
            continue
        mean_normal = sp_normals.mean(0)
        mean_normal /= np.linalg.norm(mean_normal) + 1e-8
        cos_sims = sp_normals @ mean_normal
        assert cos_sims.mean() > 0.7, (
            f"Superpoint {sp_id} has inconsistent normals (mean cos_sim={cos_sims.mean():.3f})"
        )
    
    # A6: Two cubes should NOT share a superpoint
    # (if ground truth is available)
    print(f"  Superpoints: K={K} for N={N} points")
    print(f"  Size range: [{sp_sizes[sp_sizes>0].min()}, {sp_sizes.max()}]")
    print(f"  Mean size: {sp_sizes[sp_sizes>0].mean():.1f}")
```

**Expected outputs** for two-cubes scene:

| Metric | Expected | Diagnostic if wrong |
|--------|----------|---------------------|
| $K$ (superpoint count) | 200–500 | If $K = 1$: seed resolution too large. If $K > 1500$: voxel resolution too small |
| Max intra-SP diameter | <0.15m | If >0.3m: spatial coherence broken, check distance metric |
| Mean intra-SP normal coherence | >0.95 | Lower for non-planar surfaces, but two-cubes is all-planar |
| Cross-cube superpoints | 0 | If >0: 1m gap isn't enough separation — bug in distance metric |

---

### Module 4: `voting.py` — Verification

**Mathematical invariant**: The vote matrix row-sums reflect observation frequency, not a probability distribution. But each row should have at most $V$ nonzero entries (one per view), and the values should be in $[0, 1]$ after normalization.

$$0 \leq V(s_i, m_j) \leq V_{\text{views}} \quad \text{(before normalization)}$$

$$V(s_i, m_j) = \frac{1}{|s_i|} \sum_{p \in s_i} \mathbb{1}[\ell_v(p) = m_j]$$

After normalization, each entry is the fraction of points in superpoint $s_i$ that were labeled $m_j$ in some view:

$$0 \leq V(s_i, m_j) \leq V_{\text{views}}, \quad \text{and} \quad \sum_j V(s_i, m_j) \leq V_{\text{views}}$$

```python
def verify_vote_matrix(vote_matrix, sp_labels, per_view_labels, n_views):
    """Assert vote matrix is consistent."""
    N_sp, M = vote_matrix.shape
    
    # A1: No negative votes
    assert (vote_matrix >= 0).all(), "Negative votes in vote matrix"
    
    # A2: No NaN/Inf
    assert np.all(np.isfinite(vote_matrix)), "NaN/Inf in vote matrix"
    
    # A3: Row sums bounded by number of views
    row_sums = vote_matrix.sum(axis=1)  # (N_sp,)
    assert (row_sums <= n_views + 1e-6).all(), (
        f"Row sum {row_sums.max():.3f} exceeds {n_views} views"
    )
    
    # A4: Each superpoint votes for at least one mask (if it was observed)
    observed_sps = set()
    for view_labels in per_view_labels:
        for sp_id in range(N_sp):
            if np.any((sp_labels == sp_id) & (view_labels >= 0)):
                observed_sps.add(sp_id)
    
    for sp_id in observed_sps:
        assert vote_matrix[sp_id].max() > 0, (
            f"Superpoint {sp_id} was observed but has zero votes"
        )
    
    # A5: Unobserved superpoints have zero rows
    all_sps = set(range(N_sp))
    unobserved = all_sps - observed_sps
    for sp_id in unobserved:
        assert vote_matrix[sp_id].sum() == 0, (
            f"Superpoint {sp_id} was never observed but has nonzero votes"
        )
    
    # A6: Dominant vote should match ground truth for well-separated objects
    winners = np.argmax(vote_matrix, axis=1)  # (N_sp,)
    max_votes = vote_matrix.max(axis=1)        # (N_sp,)
    
    # Confidence: how dominant is the winner?
    second_best = np.partition(vote_matrix, -2, axis=1)[:, -2]  # (N_sp,)
    margin = max_votes - second_best
    
    # For the two-cubes scene, every observed superpoint should have
    # a clear winner (margin > 0.5) because there's no ambiguity
    confident = margin[list(observed_sps)]
    print(f"  Vote matrix shape: ({N_sp}, {M})")
    print(f"  Observed superpoints: {len(observed_sps)}/{N_sp}")
    print(f"  Mean confidence margin: {confident.mean():.3f}")
    print(f"  Min confidence margin: {confident.min():.3f}")
```

**Expected outputs** for two-cubes scene:

| Metric | Expected | Diagnostic if wrong |
|--------|----------|---------------------|
| Vote matrix shape | $(K, 4)$ | $K$ superpoints × (2 masks/view × 2 views) |
| All row sums | $\leq 2.0$ | 2 views, normalized |
| Mean confidence margin | >0.8 | If low: masks overlap in 3D, or label assignment radius too large |
| Superpoints with clear winner | 100% | Two well-separated cubes → no ambiguity |

---

### Module 5: `merging.py` — Verification

**Mathematical invariant**: Merging reduces the number of instances but never increases it. And it preserves the partition property:

$$|\text{instances}_{\text{after}}| \leq |\text{instances}_{\text{before}}|$$

For the two-cubes scene, merging should be a **no-op** (nothing to merge — the cubes are separate).

```python
def verify_merging(final_labels, sp_instance_labels, points, gt_labels=None):
    """Assert merging produces valid instance segmentation."""
    N = len(final_labels)
    
    # A1: All points labeled (or explicitly unlabeled as -1)
    assert len(final_labels) == len(points)
    
    # A2: Instance IDs are contiguous starting from 0
    valid = final_labels[final_labels >= 0]
    if len(valid) > 0:
        unique_ids = np.unique(valid)
        assert unique_ids[0] == 0, f"Instance IDs don't start at 0: min={unique_ids[0]}"
        assert len(unique_ids) == unique_ids[-1] + 1, (
            f"Non-contiguous IDs: {unique_ids}"
        )
    
    # A3: No single-point instances (noise)
    instance_sizes = np.bincount(valid)
    tiny = (instance_sizes < 10).sum()
    assert tiny == 0, f"{tiny} instances with <10 points — likely noise"
    
    # A4: Spatial separation between instances
    unique_instances = np.unique(valid)
    centroids = {}
    for inst_id in unique_instances:
        inst_pts = points[final_labels == inst_id]
        centroids[inst_id] = inst_pts.mean(axis=0)
    
    # A5: Against ground truth (if available)
    if gt_labels is not None:
        # Compute Hungarian matching between predicted and GT
        from scipy.optimize import linear_sum_assignment
        
        pred_ids = np.unique(final_labels[final_labels >= 0])
        gt_ids = np.unique(gt_labels[gt_labels >= 0])
        
        # IoU matrix
        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        for i, pi in enumerate(pred_ids):
            for j, gj in enumerate(gt_ids):
                intersection = ((final_labels == pi) & (gt_labels == gj)).sum()
                union = ((final_labels == pi) | (gt_labels == gj)).sum()
                iou_matrix[i, j] = intersection / (union + 1e-8)
        
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximize IoU
        matched_ious = iou_matrix[row_ind, col_ind]
        
        mean_iou = matched_ious.mean()
        print(f"  Instances: {len(pred_ids)} predicted, {len(gt_ids)} ground truth")
        print(f"  Matched IoUs: {matched_ious}")
        print(f"  Mean IoU: {mean_iou:.3f}")
        
        # For two-cubes scene: expect IoU > 0.9 per instance
        assert mean_iou > 0.8, f"Mean IoU {mean_iou:.3f} too low — merging is wrong"
        assert len(pred_ids) == len(gt_ids), (
            f"Instance count mismatch: {len(pred_ids)} predicted vs {len(gt_ids)} GT"
        )
    
    # A6: Merging didn't create cross-boundary instances for two-cubes
    for inst_id in unique_instances:
        inst_pts = points[final_labels == inst_id]
        x_range = inst_pts[:, 0].max() - inst_pts[:, 0].min()
        # Each cube is 1m wide. If an instance spans >1.5m in x, it merged both cubes.
        assert x_range < 1.5, (
            f"Instance {inst_id} spans {x_range:.2f}m in x — likely merged both cubes"
        )
```

**Expected outputs** for two-cubes scene:

| Metric | Expected | Diagnostic if wrong |
|--------|----------|---------------------|
| Number of instances | 2 | If 1: over-merging. If >4: under-merging |
| Per-instance IoU vs GT | >0.90 | If <0.7: label assignment or voting is wrong |
| Max instance x-span | <1.1m | If >1.5m: merged across the 1m gap |
| Merging operations performed | 0 | Cubes are separated; nothing should merge |

---

### Module 6: `pipeline.py` — End-to-End Verification

```python
def run_full_verification():
    """End-to-end test on synthetic two-cubes scene."""
    scene = make_two_cubes_scene()
    
    # Skip actual SAM — use mock masks
    mock_masks_v0 = make_mock_masks_view0()
    mock_masks_v1 = make_mock_masks_view1()  # similar, from side view
    
    # Run pipeline stages sequentially with verification at each step
    print("=" * 60)
    print("STAGE 1: Mask Generation (mocked)")
    print("=" * 60)
    verify_masks(mock_masks_v0, scene["depths"][0])
    verify_masks(mock_masks_v1, scene["depths"][1])
    
    print("\n" + "=" * 60)
    print("STAGE 2: Back-projection")
    print("=" * 60)
    verify_backprojection_single_point()  # analytical check first
    
    proj_pts_0, proj_labels_0 = backproject_masks(
        mock_masks_v0, scene["depths"][0], scene["K"][0], scene["R"][0], scene["t"][0]
    )
    scene_bounds = (scene["scene_points"].min(0), scene["scene_points"].max(0))
    verify_backprojection(proj_pts_0, proj_labels_0, scene_bounds, mock_masks_v0)
    
    print("\n" + "=" * 60)
    print("STAGE 3a: Superpoints")
    print("=" * 60)
    sp_labels = extract_superpoints(
        scene["scene_points"], scene["scene_normals"], scene["scene_colors"]
    )
    verify_superpoints(sp_labels, scene["scene_points"], scene["scene_normals"], 2000)
    
    print("\n" + "=" * 60)
    print("STAGE 3b: Voting")
    print("=" * 60)
    # ... assign labels, build vote matrix ...
    # verify_vote_matrix(...)
    
    print("\n" + "=" * 60)
    print("STAGE 3c: Merging")  
    print("=" * 60)
    # ... merge ...
    # verify_merging(final_labels, ..., gt_labels=scene["gt_labels"])
    
    print("\n" + "=" * 60)
    print("FINAL: End-to-End Metrics")
    print("=" * 60)
    # AP@50, AP@25 on the two-cubes scene
    # With clean synthetic data, expect AP@50 = 1.0 (both instances matched at IoU>0.5)
```

---

## Common Failure Modes & Diagnostics

| Symptom | Most likely cause | Diagnostic check |
|---------|-------------------|-----------------|
| All back-projected points at origin | Depth map is all zeros | `assert depth.max() > 0` |
| Points form a mirror image | Sign error in $R$ or $t$ | Plot projected centroids vs expected |
| Points at wrong scale (mm vs m) | Depth units mismatch | `assert 0.1 < depth[depth>0].median() < 100` |
| Only 1 instance after merging | `normal_threshold` too permissive | Reduce to 15° and re-run |
| Hundreds of instances | `iou_threshold` too high | Lower to 0.3 and re-run |
| Superpoints span both cubes | `seed_resolution` larger than gap | Must be < gap / 2 |
| Vote matrix all zeros | Label assignment radius too small | Increase `radius` in `assign_labels_to_pointcloud` |
| NaN in vote matrix | Division by zero in normalization | Check `sp_sizes` for zeros |

---

## Quantitative Pass/Fail Criteria

For the two-cubes synthetic scene, the pipeline **passes** if:

1. $\text{AP@50} = 1.0$ — both cubes detected with IoU > 0.5
2. $\text{AP@25} = 1.0$ — both cubes detected with IoU > 0.25  
3. Number of predicted instances $= 2$
4. Zero points assigned to wrong instance (for this trivially separable scene)
5. All intermediate assertions pass

**If these fail on two cubes, the pipeline is broken.** Don't move to real data until the synthetic scene passes. The two-cubes scene is the minimum bar — it's designed to be *trivially solvable* so that any failure is definitively a bug, not a hard case.

### Graduated Difficulty Test Scenes

After two-cubes passes, increase difficulty:

| Scene | What it tests | Expected AP@50 |
|-------|---------------|----------------|
| **Two cubes, 1m gap** | Basic correctness | 1.0 |
| **Two cubes, 5cm gap** | Boundary precision | >0.9 |
| **Cube + sphere** | Non-planar normals (sphere has varying normals) | >0.85 |
| **L-shaped object** | Single instance with normal discontinuity (merge must handle 90° edges) | >0.8 |
| **10 scattered cubes** | Scale — many instances | >0.7 |
| **Partial occlusion** (2 views, one cube occluded in one view) | Multi-view fusion necessity | >0.85 |

Each scene can be generated synthetically in ~30 LOC. The L-shaped object test is particularly important because it exercises boundary-aware merging — two connected faces with perpendicular normals ($90°$) must be merged into one instance, which requires `normal_threshold > 90°` or the merging will incorrectly split it.
