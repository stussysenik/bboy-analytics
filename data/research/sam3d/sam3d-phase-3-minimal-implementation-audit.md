# sam3d — Minimal Implementation Audit

_Generated: 2026-03-23T05:39:20.711Z_

---

I'll work directly from the detailed architecture and math context provided. Here's the minimal reimplementation plan for SAM3D.

---

# SAM3D — Minimal Reimplementation Plan

## Architecture Overview

SAM3D is **training-free**. Zero learnable parameters. It's a 3-stage pipeline:

```
Multi-view RGB + Depth + Camera Params
  → SAM 2D masks (frozen ViT-H)
  → Back-project to 3D point cloud
  → Superpoint voting + boundary-aware merging
  → 3D instance segmentation labels
```

**Total trainable parameters: 0.** This is an engineering pipeline, not a learning problem.

---

## ESSENTIAL — Critical Path

### Module 1: `sam_mask_generator.py` — 2D Mask Extraction

**What it does**: Runs frozen SAM ViT-H on each view with grid prompts, filters by IoU confidence, applies NMS.

**Key functions**:
```python
def generate_masks(image: np.ndarray, 
                   grid_size: int = 64,
                   iou_threshold: float = 0.88,
                   nms_threshold: float = 0.7) -> list[Mask2D]:
    """Run SAM automatic mask generator on a single view."""

def process_all_views(images: list[np.ndarray]) -> dict[int, list[Mask2D]]:
    """Generate masks for all views. Returns {view_id: [masks]}."""
```

**Data structures**:
```python
@dataclass
class Mask2D:
    mask: np.ndarray        # (H, W) binary mask
    iou_score: float        # predicted IoU quality
    view_id: int            # which view this came from
    mask_id: int            # unique ID within this view
```

**Pseudocode**:
```python
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def generate_masks(image, grid_size=64, iou_threshold=0.88, nms_threshold=0.7):
    # image: (H, W, 3) uint8 RGB
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to("cuda")
    
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=grid_size,       # 64×64 = 4096 grid prompts
        pred_iou_thresh=iou_threshold,   # filter low-quality masks
        box_nms_thresh=nms_threshold,     # suppress overlapping masks
        min_mask_region_area=100,         # ignore tiny fragments
    )
    
    # raw_masks: list of dicts with keys: 'segmentation', 'predicted_iou', ...
    raw_masks = generator.generate(image)
    
    return [
        Mask2D(
            mask=m["segmentation"],       # (H, W) bool
            iou_score=m["predicted_iou"], # float in [0, 1]
            view_id=-1,                   # set by caller
            mask_id=i,
        )
        for i, m in enumerate(raw_masks)
    ]
```

**Estimated LOC**: ~40 (mostly SAM API wrapping)

---

### Module 2: `backproject.py` — 2D→3D Projection

**What it does**: Unprojects 2D mask pixels to 3D points using depth maps and camera intrinsics/extrinsics. Assigns each 3D point its mask label.

**Key functions**:
```python
def backproject_masks(
    masks: list[Mask2D],
    depth: np.ndarray,          # (H, W) metric depth in meters
    K: np.ndarray,              # (3, 3) camera intrinsic matrix
    R: np.ndarray,              # (3, 3) rotation (world ← camera)
    t: np.ndarray,              # (3,) translation
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (points_3d: (N, 3), labels: (N,))"""

def assign_labels_to_pointcloud(
    scene_points: np.ndarray,   # (N, 3) existing point cloud
    projected_points: np.ndarray,  # (M, 3) back-projected points
    projected_labels: np.ndarray,  # (M,) mask IDs
    radius: float = 0.02,       # assignment radius in meters
) -> np.ndarray:
    """Assign mask labels to nearest scene points. Returns (N,) labels."""
```

**Pseudocode**:
```python
def backproject_masks(masks, depth, K, R, t):
    # depth: (H, W) float32, meters
    # K: (3, 3) intrinsic — [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # R: (3, 3) rotation world←camera
    # t: (3,) translation
    
    H, W = depth.shape
    K_inv = np.linalg.inv(K)  # (3, 3)
    
    # Build pixel grid
    # u_coords: (H, W), v_coords: (H, W)
    v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Homogeneous pixel coords: (3, H*W)
    ones = np.ones((H, W))
    pixels = np.stack([u_coords, v_coords, ones], axis=0).reshape(3, -1)
    
    # Unproject: p_cam = d * K^{-1} * [u, v, 1]^T
    # rays: (3, H*W) — normalized ray directions in camera frame
    rays = K_inv @ pixels                       # (3, H*W)
    d_flat = depth.reshape(1, -1)               # (1, H*W)
    points_cam = rays * d_flat                  # (3, H*W) — 3D in camera frame
    
    # Transform to world: p_world = R^{-1} * (p_cam - t)
    # NOTE: convention from paper: R_v^{-1}(d·K_v^{-1}[u,v,1]^T - t_v)
    points_world = R.T @ (points_cam - t.reshape(3, 1))  # (3, H*W)
    points_world = points_world.T               # (H*W, 3)
    
    # Assign labels from masks
    # labels: (H*W,) — -1 for unmasked, mask_id otherwise
    labels = np.full(H * W, -1, dtype=np.int32)
    for m in masks:
        mask_flat = m.mask.reshape(-1)          # (H*W,) bool
        labels[mask_flat] = m.mask_id
    
    # Filter out invalid depth and unmasked points
    valid = (d_flat.squeeze() > 0) & (labels >= 0)
    return points_world[valid], labels[valid]


def assign_labels_to_pointcloud(scene_points, projected_points, projected_labels, radius=0.02):
    # scene_points: (N, 3)
    # projected_points: (M, 3)
    # projected_labels: (M,)
    from scipy.spatial import cKDTree
    
    tree = cKDTree(projected_points)            # build KD-tree on projected points
    dists, idxs = tree.query(scene_points, k=1) # (N,), (N,) — nearest neighbor
    
    labels = np.full(len(scene_points), -1, dtype=np.int32)
    within_radius = dists < radius
    labels[within_radius] = projected_labels[idxs[within_radius]]
    return labels                               # (N,)
```

**Estimated LOC**: ~60

---

### Module 3: `superpoints.py` — VCCS Superpoint Extraction

**What it does**: Over-segments the 3D point cloud into spatially coherent superpoints using Voxel Cloud Connectivity Segmentation. Each superpoint is a cluster of nearby, similarly-oriented points.

**Key functions**:
```python
def extract_superpoints(
    points: np.ndarray,         # (N, 3) xyz
    normals: np.ndarray,        # (N, 3) surface normals
    colors: np.ndarray,         # (N, 3) RGB [0,1]
    voxel_resolution: float = 0.02,   # 2cm seed resolution
    seed_resolution: float = 0.08,    # 8cm seed spacing
    normal_weight: float = 1.0,
) -> np.ndarray:
    """Returns superpoint_labels: (N,) — cluster ID per point."""
```

**Pseudocode**:
```python
def extract_superpoints(points, normals, colors, 
                         voxel_resolution=0.02, seed_resolution=0.08,
                         normal_weight=1.0):
    # Use Open3D or PCL bindings for VCCS
    # If no VCCS available, fall back to simple voxel grid clustering
    
    import open3d as o3d
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)      # (N, 3)
    pcd.normals = o3d.utility.Vector3dVector(normals)     # (N, 3)
    pcd.colors = o3d.utility.Vector3dVector(colors)       # (N, 3)
    
    # Open3D doesn't have native VCCS, so we use voxel downsampling
    # + region growing as an approximation
    
    # SIMPLIFIED VCCS: voxelize → seed selection → region growing
    
    # Step 1: Voxelize
    voxel_grid = {}  # maps (ix, iy, iz) → list of point indices
    for i, p in enumerate(points):
        key = tuple((p / voxel_resolution).astype(int))
        voxel_grid.setdefault(key, []).append(i)
    
    # Step 2: Select seeds at seed_resolution spacing
    seeds = []
    occupied = set()
    seed_spacing = int(seed_resolution / voxel_resolution)
    for key in sorted(voxel_grid.keys()):
        coarse_key = tuple(k // seed_spacing for k in key)
        if coarse_key not in occupied:
            occupied.add(coarse_key)
            seeds.append(key)
    
    # Step 3: Assign each voxel to nearest seed (weighted by normal similarity)
    # sp_labels: (N,) — superpoint ID per point
    sp_labels = np.full(len(points), -1, dtype=np.int32)
    seed_centers = np.array([
        np.mean(points[voxel_grid[s]], axis=0) for s in seeds
    ])  # (S, 3)
    seed_normals = np.array([
        np.mean(normals[voxel_grid[s]], axis=0) for s in seeds
    ])  # (S, 3)
    
    from scipy.spatial import cKDTree
    tree = cKDTree(seed_centers)
    
    for key, indices in voxel_grid.items():
        center = np.mean(points[indices], axis=0)     # (3,)
        avg_normal = np.mean(normals[indices], axis=0) # (3,)
        
        _, candidates = tree.query(center, k=5)
        
        # Score = spatial_dist + normal_weight * (1 - cos_similarity)
        best_seed, best_score = -1, float('inf')
        for c in candidates:
            spatial = np.linalg.norm(center - seed_centers[c])
            normal_diff = normal_weight * (1 - np.dot(avg_normal, seed_normals[c]))
            score = spatial + normal_diff
            if score < best_score:
                best_score = score
                best_seed = c
        
        for idx in indices:
            sp_labels[idx] = best_seed
    
    return sp_labels  # (N,)
```

**Estimated LOC**: ~80

**Note**: A production implementation would use PCL's VCCS directly via `python-pcl` or `pclpy`. The above is a pure-Python approximation that captures the algorithm. For minimum keystrokes, using Open3D's voxel downsampling as proxy is even simpler (~15 LOC) but less accurate.

---

### Module 4: `voting.py` — Multi-View Vote Aggregation

**What it does**: Accumulates mask votes per superpoint across all views. Each superpoint votes for which 2D mask it most frequently belongs to.

**Key functions**:
```python
def build_vote_matrix(
    sp_labels: np.ndarray,      # (N,) superpoint IDs
    point_mask_labels: list[np.ndarray],  # per-view: (N,) mask IDs (-1 if unseen)
    n_superpoints: int,
    n_total_masks: int,
) -> np.ndarray:
    """Returns vote_matrix: (N_sp, M_total) — normalized vote counts."""

def resolve_votes(vote_matrix: np.ndarray) -> np.ndarray:
    """Returns sp_instance_labels: (N_sp,) — winning mask ID per superpoint."""
```

**Pseudocode**:
```python
def build_vote_matrix(sp_labels, point_mask_labels, n_superpoints, n_total_masks):
    # sp_labels: (N,) — superpoint ID per point
    # point_mask_labels: list of V arrays, each (N,) — mask label per point per view
    #                    -1 means point not visible in that view
    
    # vote_matrix: (N_sp, M_total) — how strongly each superpoint votes for each mask
    vote_matrix = np.zeros((n_superpoints, n_total_masks), dtype=np.float32)
    
    # Count points in each superpoint for normalization
    sp_sizes = np.bincount(sp_labels, minlength=n_superpoints)  # (N_sp,)
    sp_sizes = np.maximum(sp_sizes, 1)  # avoid div-by-zero
    
    for view_labels in point_mask_labels:
        # view_labels: (N,) — mask ID per point for this view
        valid = view_labels >= 0
        for sp_id in range(n_superpoints):
            in_sp = (sp_labels == sp_id) & valid   # (N,) bool
            if not np.any(in_sp):
                continue
            mask_ids = view_labels[in_sp]
            for mid in mask_ids:
                vote_matrix[sp_id, mid] += 1.0
    
    # Normalize: V(s_i, m_j) = (1/|s_i|) * count
    vote_matrix /= sp_sizes[:, None]               # (N_sp, M_total)
    
    return vote_matrix


def resolve_votes(vote_matrix):
    # vote_matrix: (N_sp, M_total)
    # For each superpoint, pick the mask with highest vote
    sp_instance_labels = np.argmax(vote_matrix, axis=1)  # (N_sp,)
    
    # Mark superpoints with no votes as unlabeled (-1)
    max_votes = np.max(vote_matrix, axis=1)               # (N_sp,)
    sp_instance_labels[max_votes == 0] = -1
    
    return sp_instance_labels  # (N_sp,)
```

**Estimated LOC**: ~50

**Vectorized version** (~25 LOC, production-quality):
```python
def build_vote_matrix_fast(sp_labels, point_mask_labels, n_sp, n_masks):
    # Vectorized with np.add.at
    V = np.zeros((n_sp, n_masks), dtype=np.float32)
    sp_sizes = np.bincount(sp_labels, minlength=n_sp).astype(np.float32)
    sp_sizes = np.maximum(sp_sizes, 1)
    
    for view_labels in point_mask_labels:
        valid = view_labels >= 0                     # (N,)
        np.add.at(V, (sp_labels[valid], view_labels[valid]), 1.0)
    
    V /= sp_sizes[:, None]
    return V
```

---

### Module 5: `merging.py` — Boundary-Aware Region Merging

**What it does**: Merges adjacent superpoint groups that likely belong to the same 3D instance, using IoU overlap and surface normal discontinuity as the merge criterion.

**Key functions**:
```python
def build_adjacency(
    points: np.ndarray,         # (N, 3)
    sp_labels: np.ndarray,      # (N,) superpoint IDs
    radius: float = 0.05,       # adjacency radius
) -> dict[int, set[int]]:
    """Returns adjacency graph: {sp_id: {neighbor_sp_ids}}."""

def merge_regions(
    sp_instance_labels: np.ndarray,  # (N_sp,) from voting
    adjacency: dict[int, set[int]],
    points: np.ndarray,              # (N, 3)
    normals: np.ndarray,             # (N, 3)
    sp_labels: np.ndarray,           # (N,) superpoint IDs
    iou_threshold: float = 0.5,      # τ_merge
    normal_threshold: float = 30.0,  # τ_boundary in degrees
) -> np.ndarray:
    """Returns final_labels: (N,) — instance ID per point."""
```

**Pseudocode**:
```python
def build_adjacency(points, sp_labels, radius=0.05):
    from scipy.spatial import cKDTree
    
    n_sp = sp_labels.max() + 1
    # Compute superpoint centroids
    centroids = np.zeros((n_sp, 3))               # (N_sp, 3)
    for s in range(n_sp):
        mask = sp_labels == s
        if np.any(mask):
            centroids[s] = points[mask].mean(axis=0)
    
    tree = cKDTree(centroids)
    adjacency = {}
    for s in range(n_sp):
        neighbors = tree.query_ball_point(centroids[s], r=radius * 5)
        adjacency[s] = set(neighbors) - {s}
    
    return adjacency  # {sp_id: {neighbor_ids}}


def compute_3d_iou(points, sp_labels, group_a_sps, group_b_sps):
    """Approximate 3D IoU via bounding box overlap."""
    pts_a = points[np.isin(sp_labels, list(group_a_sps))]
    pts_b = points[np.isin(sp_labels, list(group_b_sps))]
    
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0.0
    
    # Axis-aligned bounding box IoU
    min_a, max_a = pts_a.min(0), pts_a.max(0)    # (3,), (3,)
    min_b, max_b = pts_b.min(0), pts_b.max(0)
    
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_vol = np.prod(np.maximum(inter_max - inter_min, 0))
    
    vol_a = np.prod(max_a - min_a)
    vol_b = np.prod(max_b - min_b)
    
    return inter_vol / (vol_a + vol_b - inter_vol + 1e-8)


def merge_regions(sp_instance_labels, adjacency, points, normals, 
                  sp_labels, iou_threshold=0.5, normal_threshold=30.0):
    # Union-Find for merging
    parent = list(range(len(sp_instance_labels)))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    cos_threshold = np.cos(np.radians(normal_threshold))
    
    for sp_a, neighbors in adjacency.items():
        for sp_b in neighbors:
            if sp_instance_labels[sp_a] < 0 or sp_instance_labels[sp_b] < 0:
                continue
            if find(sp_a) == find(sp_b):
                continue
            
            # Check normal discontinuity
            normals_a = normals[sp_labels == sp_a].mean(0)   # (3,)
            normals_b = normals[sp_labels == sp_b].mean(0)   # (3,)
            cos_sim = np.dot(normals_a, normals_b) / (
                np.linalg.norm(normals_a) * np.linalg.norm(normals_b) + 1e-8
            )
            
            if cos_sim < cos_threshold:
                continue  # boundary — don't merge
            
            # Check 3D IoU between the groups
            group_a = {s for s in range(len(parent)) if find(s) == find(sp_a)}
            group_b = {s for s in range(len(parent)) if find(s) == find(sp_b)}
            
            iou = compute_3d_iou(points, sp_labels, group_a, group_b)
            if iou > iou_threshold:
                union(sp_a, sp_b)
    
    # Flatten to per-point labels
    # Map each superpoint to its merged group root
    sp_to_instance = np.array([find(s) for s in range(len(parent))])
    
    # Remap to contiguous instance IDs
    unique_roots = np.unique(sp_to_instance[sp_to_instance >= 0])
    root_to_id = {r: i for i, r in enumerate(unique_roots)}
    
    final_labels = np.full(len(points), -1, dtype=np.int32)
    for pt_idx in range(len(points)):
        sp = sp_labels[pt_idx]
        root = find(sp)
        if root in root_to_id:
            final_labels[pt_idx] = root_to_id[root]
    
    return final_labels  # (N,) — instance ID per point
```

**Estimated LOC**: ~90

---

### Module 6: `pipeline.py` — End-to-End Orchestrator

**What it does**: Connects all modules into the full SAM3D pipeline.

**Key functions**:
```python
def run_sam3d(
    images: list[np.ndarray],       # V × (H, W, 3) RGB
    depths: list[np.ndarray],       # V × (H, W) metric depth
    intrinsics: list[np.ndarray],   # V × (3, 3)
    extrinsics: list[tuple[np.ndarray, np.ndarray]],  # V × (R, t)
    scene_points: np.ndarray,       # (N, 3) point cloud
    scene_normals: np.ndarray,      # (N, 3)
    scene_colors: np.ndarray,       # (N, 3)
) -> np.ndarray:
    """Returns instance_labels: (N,) — instance ID per point."""
```

**Pseudocode**:
```python
def run_sam3d(images, depths, intrinsics, extrinsics, 
              scene_points, scene_normals, scene_colors):
    V = len(images)
    
    # Stage 1: Generate 2D masks for all views
    all_masks = []
    mask_offset = 0
    for v in range(V):
        masks = generate_masks(images[v])                # list[Mask2D]
        for m in masks:
            m.view_id = v
            m.mask_id += mask_offset                     # globally unique mask IDs
        mask_offset += len(masks)
        all_masks.append(masks)
    
    n_total_masks = mask_offset
    
    # Stage 2: Back-project and assign labels to scene points
    per_view_labels = []
    for v in range(V):
        R, t = extrinsics[v]
        proj_pts, proj_labels = backproject_masks(
            all_masks[v], depths[v], intrinsics[v], R, t
        )
        view_labels = assign_labels_to_pointcloud(
            scene_points, proj_pts, proj_labels, radius=0.02
        )                                                # (N,)
        per_view_labels.append(view_labels)
    
    # Stage 3a: Extract superpoints
    sp_labels = extract_superpoints(
        scene_points, scene_normals, scene_colors,
        voxel_resolution=0.02, seed_resolution=0.08
    )                                                    # (N,)
    n_sp = sp_labels.max() + 1
    
    # Stage 3b: Vote aggregation
    vote_matrix = build_vote_matrix(
        sp_labels, per_view_labels, n_sp, n_total_masks
    )                                                    # (N_sp, M_total)
    sp_instance_labels = resolve_votes(vote_matrix)      # (N_sp,)
    
    # Stage 3c: Boundary-aware merging
    adjacency = build_adjacency(scene_points, sp_labels)
    final_labels = merge_regions(
        sp_instance_labels, adjacency,
        scene_points, scene_normals, sp_labels,
        iou_threshold=0.5, normal_threshold=30.0
    )                                                    # (N,)
    
    return final_labels
```

**Estimated LOC**: ~40

---

## NICE-TO-HAVE

### 1. Hierarchical Mask Generation (multi-scale grid prompts)
- **What you lose**: Misses small objects; SAM's default single-scale grid may under-segment
- **Quality impact**: +3–5 mAP@50 on small objects
- **LOC**: ~30 (adjust `points_per_side` parameter, add multi-scale pass)

### 2. Weighted Vote Aggregation (IoU-score weighting)
- **What you lose**: Low-confidence masks count equally with high-confidence ones
- **Quality impact**: +1–2 mAP@50
- **LOC**: ~10 (multiply vote by `m.iou_score` in the vote accumulation loop)

### 3. Iterative Merging with Decreasing Thresholds
- **What you lose**: Single-pass merging may under-merge distant parts of the same object
- **Quality impact**: +2–3 mAP@50
- **LOC**: ~20 (wrap merge_regions in a loop with relaxing thresholds)

### 4. GPU-Accelerated KD-Tree / Ball Tree
- **What you lose**: 10–50× slower on large scenes (>1M points)
- **Quality impact**: None (same output, just speed)
- **LOC**: ~15 (swap scipy for cuml or faiss)

### 5. Normal Estimation from Depth Maps
- **What you lose**: Need normals as input; this derives them from depth
- **Quality impact**: Required if dataset doesn't provide normals
- **LOC**: ~25 (cross-product of depth gradients)

---

## SKIP

| Component | Why it's safe to skip |
|-----------|----------------------|
| **ScanNet/ScanNet200 benchmark harness** | Paper-specific evaluation. We only need the pipeline, not the benchmark infrastructure. |
| **Distributed/multi-GPU processing** | Engineering optimization. Single-GPU handles the pipeline fine for reasonable scene sizes. |
| **Visualization (point cloud rendering, mask overlays)** | Debug tool only. Open3D's built-in viewer covers this ad-hoc. |
| **Paper ablation code** (varying grid size, threshold sweeps) | Reproduces Table 2/3 from the paper. Not needed for the core capability. |
| **ScanNet data loader / preprocessing** | Dataset-specific I/O. Generic numpy input is sufficient. |
| **SAM model training/fine-tuning** | SAM3D uses frozen SAM. Zero training by design. |
| **Class-specific evaluation** (semantic segmentation mode) | SAM3D is class-agnostic instance segmentation. Class labels are a separate problem. |

---

## Dependency Audit

### PyTorch
- **Minimum**: PyTorch ≥ 1.12 (for SAM compatibility)
- **Recommended**: PyTorch 2.0+ (for `torch.compile` speedup on SAM encoder)

### Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `segment-anything` | 1.0 | SAM ViT-H model + automatic mask generator |
| `numpy` | ≥ 1.21 | Point cloud operations |
| `scipy` | ≥ 1.7 | `cKDTree` for nearest-neighbor queries |
| `open3d` | ≥ 0.17 | Point cloud I/O, optional VCCS, visualization |

**No einops, no timm, no exotic dependencies.** This is a geometry pipeline, not a learning pipeline.

### Pretrained Weights

| Checkpoint | URL | Size | Dataset | Notes |
|-----------|-----|------|---------|-------|
| SAM ViT-H | `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` | 2.4 GB | SA-1B (11M images, 1.1B masks) | **Frozen — never retrained.** This is the only checkpoint needed. |

**Does pre-training dataset matter?** No. SAM was trained on diverse images for universal segmentation. SAM3D uses it as-is with zero adaptation. The 3D assembly is purely geometric.

---

## Total Estimate

| Category | LOC |
|----------|-----|
| `sam_mask_generator.py` | ~40 |
| `backproject.py` | ~60 |
| `superpoints.py` | ~80 |
| `voting.py` | ~50 |
| `merging.py` | ~90 |
| `pipeline.py` | ~40 |
| **Essential total** | **~360 lines** |
| + Nice-to-have | ~460 lines |

### Implementation time
- Essential (360 LOC): One focused session for someone comfortable with 3D geometry and numpy. The hardest part is getting VCCS right — consider using PCL bindings instead of reimplementing.
- The pipeline is entirely **inference-time geometry** — no training loops, no loss functions, no optimizers. The only "model" is frozen SAM, consumed via its public API.

### Key Insight for Minimum Keystrokes

The absolute minimum viable SAM3D is **three operations**:

1. `SamAutomaticMaskGenerator.generate()` — already implemented by Meta
2. `depth * K_inv @ pixel_coords` — one matrix multiply for back-projection  
3. `np.argmax(vote_matrix, axis=1)` — majority vote per superpoint

Everything else (VCCS, boundary merging, NMS tuning) is refinement. A bare-bones prototype that skips VCCS (uses voxel grid instead) and skips boundary merging (raw voting only) would be **~150 LOC**.
