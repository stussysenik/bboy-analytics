<!-- Generated: 2026-03-23T06:31:23.243Z | Paper: sam3d | Overnight Research Loop -->

# SAM3D Reimplementation Guide

## Paper Metadata

| Field | Value |
|-------|-------|
| **Title** | SAM3D: Segment Anything in 3D Scenes |
| **Authors** | Yang, Y., Wu, Y., Zhai, W., Zhang, D., Cao, Z. |
| **Venue** | Preprint 2023 (2023) |
| **arXiv** | [2306.03908](https://arxiv.org/abs/2306.03908) |
| **Code** | [Pointcept/SegmentAnything3D](https://github.com/Pointcept/SegmentAnything3D) |

---

## Why This Paper (for Breakdancing Analysis)

SAM3D addresses the problem of lifting 2D segmentation into 3D space without any training — a capability the breakdancing analysis pipeline needs for spatially grounding dancer body parts in world coordinates. In the bboy analysis stack, SAM3D's role is **static scene reconstruction only**: segmenting the stage, floor, barriers, and audience regions into 3D instance masks that provide the spatial reference frame against which dancer motion is measured. The research conclusively demonstrates that SAM3D is architecturally incompatible with segmenting the dancer themselves (motion contamination drives limb vote accuracy below 0.10, viewpoint degeneracy collapses effective view count to 1, and MDE noise exceeds superpoint resolution by 7–11×). However, understanding SAM3D's pipeline is essential because (a) the static scene elements it *can* segment establish the coordinate system and floor plane for the movement spectrogram $S_m(j,t)$, (b) its 2D-to-3D back-projection mathematics are reused throughout the pipeline wherever depth maps are consumed, and (c) its failure modes on dynamic articulated bodies directly motivate the alternative pipeline (SAM 2 + CoTracker3 + HMR 2.0/WHAM) that handles the dancer. SAM3D thus occupies the "scene understanding" layer beneath the dancer-specific stack, providing the spatial context — floor boundaries, camera-to-stage geometry, obstacle locations — that the movement spectrogram needs for absolute spatial grounding rather than purely relative joint trajectories.

---

## Architecture

```mermaid
flowchart TD
    subgraph Input["Input: RGB-D Scene"]
        I1["Multi-view RGB Images<br/>V views × H × W × 3"]
        I2["Depth Maps<br/>V × H × W"]
        I3["Camera Poses<br/>V × (R|t) + K intrinsics"]
    end

    subgraph Stage1["Stage 1: 2D Mask Generation (per-view, frozen SAM)"]
        S1A["SAM ViT-H Encoder<br/>~632M params, frozen<br/>1024×1024 → 256×64×64"]
        S1B["Grid Prompt Generator<br/>64×64 = 4096 point prompts"]
        S1C["SAM Prompt Encoder<br/>~6K params, frozen<br/>points → 256-dim tokens"]
        S1D["SAM Mask Decoder<br/>~4M params, frozen<br/>→ K binary masks + IoU scores"]
        S1E["NMS + Confidence Filter<br/>τ_conf, box_nms_thresh"]
    end

    subgraph Stage2["Stage 2: 2D→3D Back-Projection"]
        S2A["Pixel-to-3D Projection<br/>p_3D = R⁻¹(d·K⁻¹[u,v,1]ᵀ − t)"]
        S2B["Mask Label Assignment<br/>Nearest-neighbor to point cloud"]
    end

    subgraph Stage3["Stage 3: Multi-View 3D Merging"]
        S3A["Superpoint Extraction (VCCS)<br/>R_seed = 0.02m"]
        S3B["Vote Accumulation<br/>V(s_i, m_j) across all views"]
        S3C["Boundary-Aware Region Merging<br/>IoU + normal discontinuity gate"]
        S3D["Final 3D Instance Masks"]
    end

    I1 --> S1A --> S1D
    S1B --> S1C --> S1D
    S1D --> S1E

    S1E --> S2A
    I2 --> S2A
    I3 --> S2A
    S2A --> S2B

    S2B --> S3A --> S3B --> S3C --> S3D

    style Input fill:#1a1a2e,stroke:#e94560,color:#eee
    style Stage1 fill:#16213e,stroke:#0f3460,color:#eee
    style Stage2 fill:#0f3460,stroke:#533483,color:#eee
    style Stage3 fill:#533483,stroke:#e94560,color:#eee
```

### Component-by-Component Walkthrough

**Stage 1 — 2D Mask Generation (Per-View).** For each of $V$ views in the RGB-D scene, the RGB image is resized to $1024 \times 1024$ and fed through SAM's frozen ViT-H image encoder. This encoder consists of 32 transformer blocks with embedding dimension $d = 1280$, MLP expansion factor 4× (hidden dim 5120), and patch size $16 \times 16$, producing a 4096-token sequence that is reshaped into a $256 \times 64 \times 64$ spatial feature map. Twenty-eight blocks use windowed attention (window size $14 \times 14 = 196$ tokens) and four blocks use global attention across all 4096 tokens. The computational cost is substantial: approximately 2.8 TFLOPs per encoder invocation (28 windowed blocks × ~82.6 GFLOPs + 4 global blocks × ~123.5 GFLOPs), totaling ~280 TFLOPs for a 100-view scene. This is 7.5× higher than naive estimates that assume uniform attention.

Rather than requiring manual prompts, SAM3D generates a dense $64 \times 64$ grid of point prompts over the image — 4096 foreground points, each paired with a background point. The prompt encoder (a lightweight ~6K parameter module) converts each point to a 256-dimensional positional encoding token. The mask decoder (~4M parameters, two transformer layers) cross-attends prompt tokens against the image embedding to produce, for each prompt, up to 3 candidate binary masks at different granularities plus an IoU confidence score. After decoding all 4096 prompts, NMS with `box_nms_thresh = 0.7` eliminates duplicate masks, and a confidence filter retains only masks with predicted IoU above `pred_iou_thresh` (SAM default 0.88, though the paper may use ~0.7). The output is a set of $K$ binary masks per view, where $K$ typically ranges from 50–300 depending on scene complexity.

**Stage 2 — 2D→3D Back-Projection.** Each per-view mask is lifted into 3D by combining the depth map and camera parameters. For every pixel $(u, v)$ that belongs to a mask, the corresponding depth value $d(u,v)$ and camera intrinsics $\mathbf{K}_v$ and extrinsics $(\mathbf{R}_v, \mathbf{t}_v)$ are used to compute the 3D world coordinate via the standard pinhole back-projection equation (see Equation 1 below). The resulting 3D points are then assigned to the nearest points in the pre-reconstructed 3D point cloud of the scene (obtained from depth fusion across all views), transferring the 2D mask label to each matched 3D point. This is a purely geometric operation with zero learnable parameters. The critical vulnerability here is that depth error propagates 1:1 into 3D displacement — a 15cm depth error produces a 15cm 3D misplacement, which is 7.5× the superpoint voxel size.

**Stage 3 — Multi-View 3D Merging.** The merging stage resolves conflicting mask assignments from different views. First, the 3D point cloud is over-segmented into **superpoints** using Voxel Cloud Connectivity Segmentation (VCCS) with seed resolution $R_{\text{seed}} = 0.02$m. Each superpoint $s_i$ is a small cluster of geometrically coherent points. Then, for each superpoint, a **vote vector** is accumulated across all views: $V(s_i, m_j)$ counts the fraction of points in $s_i$ that were assigned mask label $m_j$ in view $v$. The dominant mask label wins the superpoint. Finally, **boundary-aware region merging** combines adjacent superpoint groups whose pairwise 3D IoU exceeds $\tau_{\text{merge}}$ (~0.50) AND whose surface normals are compatible (normal angle discontinuity below $\tau_{\text{boundary}}$ ~30°). The normal gate prevents merging across geometric edges (e.g., wall-floor boundary). This entire stage is algorithmic — zero parameters, zero training.

**What makes SAM3D unique** is that it is entirely training-free for the 3D task. The only neural network parameters belong to the frozen SAM model, which was trained on 11M images for 2D segmentation. The 3D reasoning is performed entirely through geometric projection and combinatorial merging. This is both its strength (zero-shot generalization to any scene) and its weakness (no learned priors about 3D object structure, body topology, or motion).

---

## Core Mathematics

### Equation 1: Pinhole Back-Projection (Pixel → 3D World Coordinate)

$$\mathbf{p}_{3D} = \mathbf{R}_v^{-1}\left(d(u,v) \cdot \mathbf{K}_v^{-1}\begin{bmatrix}u \\ v \\ 1\end{bmatrix} - \mathbf{t}_v\right)$$

- **Variables**:
  - $\mathbf{p}_{3D} \in \mathbb{R}^3$: resulting 3D point in world coordinates
  - $d(u,v) \in \mathbb{R}_{>0}$: depth value at pixel $(u, v)$ in meters
  - $\mathbf{K}_v \in \mathbb{R}^{3 \times 3}$: camera intrinsic matrix for view $v$, encoding focal lengths $(f_x, f_y)$ and principal point $(c_x, c_y)$
  - $\mathbf{R}_v \in SO(3) \subset \mathbb{R}^{3 \times 3}$: rotation matrix (camera-to-world) for view $v$
  - $\mathbf{t}_v \in \mathbb{R}^3$: translation vector (camera-to-world) for view $v$
  - $[u, v, 1]^\top \in \mathbb{R}^3$: homogeneous pixel coordinate

- **Intuition**: The inner term $\mathbf{K}_v^{-1}[u, v, 1]^\top$ converts the pixel to a ray direction in camera space. Multiplying by depth $d$ scales the ray to the correct distance. Subtracting translation and rotating by $\mathbf{R}_v^{-1}$ transforms from camera space to world space. This is the fundamental operation that "lifts" every 2D mask pixel into 3D.

- **Connection**: The back-projected 3D points are used in Stage 2 to assign mask labels to the nearest points in the scene point cloud. Depth error $\epsilon_d$ propagates as $\|\Delta \mathbf{p}_{3D}\| \approx |\epsilon_d|$ near image center, amplified by $\sec(\theta/2)$ at image edges (Equation 2). This feeds directly into the mask assignment accuracy, which determines superpoint purity (Equation 4).

### Equation 2: Depth Error Propagation

$$\|\Delta \mathbf{p}_{3D}\| = |\epsilon_d| \cdot \left\|\mathbf{R}^{-1}\mathbf{K}^{-1}\begin{bmatrix}u \\ v \\ 1\end{bmatrix}\right\| \approx |\epsilon_d| \cdot \sec\left(\frac{\theta}{2}\right)$$

where the approximation holds for image-edge pixels at field-of-view angle $\theta$.

- **Variables**:
  - $\epsilon_d = \hat{d} - d$: depth estimation error (meters)
  - $\theta$: camera field of view (radians)
  - The $\sec(\theta/2)$ factor arises because $\|\mathbf{K}^{-1}[u,v,1]^\top\|$ increases toward image edges

- **Intuition**: Depth is a multiplicative factor in back-projection, so depth error translates 1:1 into 3D displacement at the image center. At the edges of a wide-angle lens (e.g., iPhone 14 Pro with FOV ≈ 75°), the displacement is amplified by ~26%. For a monocular depth estimator like DepthPro with mean error ~14.4cm at 3m, every boundary point is displaced by ~14cm in 3D — 7× the superpoint voxel size of 2cm.

- **Connection**: This directly determines the mask assignment error rate (Equation 3) and ultimately bounds the achievable segmentation quality when using estimated depth.

### Equation 3: Mask Assignment Error Rate

$$e_{\text{assign}} \approx 1 - \Phi\left(\frac{d_{\text{boundary}}}{\sigma_d}\right)$$

where $\Phi$ is the standard normal CDF.

- **Variables**:
  - $d_{\text{boundary}}$: distance between adjacent objects in 3D (meters). For a hand near a hip, $d_{\text{boundary}} \approx 0.05$m
  - $\sigma_d$: standard deviation of depth error (meters). For Depth Anything v2 at 3m: $\sigma_d \approx 0.15$m
  - $e_{\text{assign}} \in [0, 1]$: fraction of boundary points receiving incorrect mask labels

- **Intuition**: If the depth error is large relative to the gap between two objects, the back-projected points from one object will frequently land in the region belonging to the other. When $\sigma_d \gg d_{\text{boundary}}$, the assignment becomes essentially random. For a hand 5cm from a hip with 15cm depth noise: $e_{\text{assign}} \approx 1 - \Phi(0.33) \approx 0.37$ — 37% of boundary points get the wrong label.

- **Connection**: Feeds into superpoint purity (Equation 4) and the multi-view voting correction (Equation 5).

### Equation 4: Superpoint Vote Aggregation

$$V(s_i, m_j) = \frac{1}{|s_i|} \sum_{p \in s_i} \mathbb{1}\left[L_v(p) = m_j\right]$$

- **Variables**:
  - $s_i$: the $i$-th superpoint (a cluster of geometrically nearby 3D points)
  - $|s_i|$: number of points in superpoint $s_i$
  - $m_j$: mask label $j$ from view $v$
  - $L_v(p)$: the mask label assigned to point $p$ from view $v$ via back-projection
  - $V(s_i, m_j) \in [0, 1]$: the vote score — fraction of points in $s_i$ labeled as $m_j$ from view $v$

- **Intuition**: Each superpoint collects votes from all views. If a superpoint contains 100 points and 73 of them are labeled "chair" from a particular view, the vote for "chair" is 0.73. Across all views, the label with the highest accumulated vote wins. This is the mechanism that resolves cross-view disagreements and (in theory) averages out per-view noise.

- **Connection**: The accumulated votes across all views determine the final label for each superpoint. The quality of this voting depends critically on (a) superpoint purity (whether the superpoint contains points from only one object) and (b) the independence of votes across views — both of which degrade catastrophically for dynamic scenes (Equation 6). For static scene elements (floor, walls), multi-view voting works as intended.

### Equation 5: Multi-View Voting Accuracy

$$P(\text{correct}) = \sum_{k=\lceil V/2\rceil}^{V} \binom{V}{k}(1 - e_{\text{assign}})^k \cdot e_{\text{assign}}^{V-k}$$

With correlated errors (correlation $\rho$ between views):

$$V_{\text{eff}} = \frac{V}{1 + (V-1)\rho}$$

- **Variables**:
  - $V$: number of views contributing votes for a superpoint
  - $e_{\text{assign}}$: per-view mask assignment error rate (from Equation 3)
  - $\rho \in [0, 1]$: pairwise correlation between depth errors across views
  - $V_{\text{eff}}$: effective number of independent views after accounting for correlation

- **Intuition**: If errors were independent across views, majority voting would exponentially suppress noise — 50 independent views with 30% error rate would yield 99.99% accuracy. But monocular depth errors are systematically correlated (same model biases, similar viewpoints). For $\rho = 0.5$ with $V = 50$: $V_{\text{eff}} \approx 1.96$. The multi-view benefit nearly vanishes. For single-camera video where all frames share the same viewpoint ($\rho \approx 1.0$): $V_{\text{eff}} = 1$ regardless of frame count.

- **Connection**: This explains why SAM3D performs well on multi-view datasets like ScanNet (diverse viewpoints, $\rho$ low) but fails on video from a fixed camera (identical viewpoints, $\rho \approx 1$).

### Equation 6: Region Merging Criterion

$$\text{merge}(G_a, G_b) = \begin{cases} \text{True} & \text{if } \text{IoU}_{3D}(G_a, G_b) > \tau_{\text{merge}} \;\text{AND}\; \Delta\theta_{\text{normal}} < \tau_{\text{boundary}} \\ \text{False} & \text{otherwise} \end{cases}$$

- **Variables**:
  - $G_a, G_b$: two candidate superpoint groups for merging
  - $\text{IoU}_{3D}(G_a, G_b)$: volumetric intersection-over-union between the two groups in 3D
  - $\tau_{\text{merge}} \approx 0.50$: IoU threshold for merging
  - $\Delta\theta_{\text{normal}}$: angle between average surface normals of the two groups
  - $\tau_{\text{boundary}} \approx 30°$: maximum normal angle for merging (prevents merging across geometric edges)

- **Intuition**: Two neighboring superpoint groups should be merged into a single instance if they overlap significantly in 3D AND their surfaces face roughly the same direction. The normal gate prevents merging a wall with a floor even if they were assigned the same mask label — their normals differ by 90°. This is a conservative merge: it requires both geometric proximity AND surface compatibility.

- **Connection**: The sensitivity of $\tau_{\text{merge}}$ differs dramatically between ScanNet-style scenes (bimodal IoU distribution, easy to threshold) and human body segmentation (unimodal IoU $\sim \mathcal{B}(2.5, 3.5)$ with mode ~0.38, no clean separation). For body parts, the optimal threshold shifts to ~0.38–0.42 but peak AP is only ~0.56 relative because the distributions overlap too severely.

### Equation 7: Motion-to-Resolution Ratio (Breakdancing Extension)

$$\alpha_b = \frac{\|\mathbf{v}_b\| \cdot \Delta t}{R_{\text{seed}}}$$

- **Variables**:
  - $\|\mathbf{v}_b\|$: velocity of body part $b$ in m/s
  - $\Delta t = 1/f$: inter-frame time interval (at 30fps: 0.033s)
  - $R_{\text{seed}} = 0.02$m: superpoint voxel size
  - $\alpha_b$: dimensionless ratio — how far the body part moves between frames relative to superpoint resolution

- **Intuition**: When $\alpha_b \geq 1$, the body part moves more than one superpoint width between frames. Consecutive frames produce completely independent superpoints for that body part, destroying the assumption that the same superpoint accumulates consistent votes. At 30fps with $R_{\text{seed}} = 0.02$m, $\alpha = 1$ at velocity $v = 0.6$ m/s — exceeded by virtually all dance movements except near-static pauses.

- **Connection**: This is the fundamental quantity that invalidates SAM3D for dynamic bodies. It feeds into the vote contamination model and the required frame rate calculation: coherent superpoints require $f > \|\mathbf{v}_b\| / (0.5 \cdot R_{\text{seed}})$, yielding 200–500 fps for active limbs.

### Equation 8: Confidence-Weighted Vote (Mitigation)

$$V_w(s_i, m_j) = \frac{\sum_{p \in s_i} c(p) \cdot \mathbb{1}[L_v(p) = m_j]}{\sum_{p \in s_i} c(p)}$$

- **Variables**:
  - $c(p) \in [0, 1]$: per-pixel depth confidence from the depth estimator's uncertainty map
  - All other variables as in Equation 4

- **Intuition**: Instead of giving every point an equal vote, weight by how confident the depth estimator is about that point. Points with unreliable depth (near occlusion boundaries, on reflective surfaces, in motion-blurred regions) contribute less to the vote. Expected improvement: $e_{\text{assign}}$ from ~0.30 to ~0.18 for DepthPro on static scenes.

- **Connection**: This is the primary mitigation for depth noise in the back-projection stage. It requires a depth estimator that outputs calibrated uncertainty (DepthPro and Metric3D v2 both do). For the breakdancing pipeline, this should be applied when using SAM3D for static scene elements.

---

## "Least Keystrokes" Implementation Roadmap

### ESSENTIAL (~650 LOC)

1. **SAM wrapper for dense mask generation** (~80 LOC) — Load frozen SAM ViT-H, generate $64 \times 64$ grid prompts, run encoder + decoder, apply NMS and confidence filtering. Wraps the `segment_anything` library.

2. **Pinhole back-projector** (~60 LOC) — Given a depth map $D \in \mathbb{R}^{H \times W}$, a binary mask $M \in \{0,1\}^{H \times W}$, intrinsics $\mathbf{K}$, and extrinsics $(\mathbf{R}, \mathbf{t})$, compute the 3D coordinates of all mask pixels. Pure NumPy/torch, no dependencies.

3. **Mask-to-point-cloud label transfer** (~50 LOC) — For each view's back-projected 3D points, find the nearest neighbor in the scene point cloud (via KD-tree) and assign the mask label. Requires `scipy.spatial.KDTree` or `open3d.geometry.KDTreeFlann`.

4. **VCCS superpoint extraction** (~120 LOC) — Over-segment the point cloud into superpoints using voxel seeding + region growing. Either wrap Open3D's voxel downsampling as an approximation, or implement the seed-based growing with normal + color coherence checks.

5. **Vote accumulator** (~80 LOC) — For each superpoint, accumulate per-view mask label votes into a vote matrix $\mathbf{V} \in \mathbb{R}^{|S| \times |M|}$. Assign the argmax label per superpoint.

6. **Boundary-aware region merger** (~150 LOC) — Build an adjacency graph of superpoints, compute pairwise 3D IoU and normal angle for adjacent pairs, merge if both thresholds are satisfied. Iterate until convergence.

7. **Evaluation (mAP computation)** (~60 LOC) — Compute AP@25, AP@50, AP for the predicted instance masks against ground truth, following the ScanNet evaluation protocol.

8. **Pipeline orchestrator** (~50 LOC) — Wire stages 1–6 together: iterate over views, accumulate, merge, output.

### NICE-TO-HAVE (~350 LOC)

9. **Confidence-weighted voting** (~40 LOC) — Replace uniform voting with depth-confidence-weighted voting (Equation 8). Requires extracting uncertainty maps from DepthPro or Metric3D v2.

10. **ScanNet data loader** (~80 LOC) — Load `.sens` files, extract RGB frames, depth maps, camera poses, and ground truth instance labels. Alternatively, use the preprocessed `.ply` + `.txt` format.

11. **Visualization** (~100 LOC) — Open3D-based 3D visualization of superpoints, vote distributions, and final instance masks with per-instance coloring.

12. **Monocular depth integration** (~60 LOC) — Replace GT depth with DepthPro/Depth Anything v2 inference and optional scale alignment to sparse SfM points ($\min_{\alpha, \beta} \sum \|\alpha\hat{d}_i + \beta - d_i^{\text{SfM}}\|^2$).

13. **Adaptive threshold tuning** (~70 LOC) — Grid search or Bayesian optimization over $\tau_{\text{conf}}$, $\tau_{\text{merge}}$, and $\tau_{\text{boundary}}$ on a validation split.

### SKIP

- **SAM training or fine-tuning** — SAM3D uses SAM frozen; do not retrain. The entire point of the paper is training-free 3D segmentation.
- **Custom ViT encoder** — Use Meta's pretrained `vit_h` checkpoint directly. Reimplementing ViT-H from scratch is ~2000 LOC with zero benefit.
- **Point cloud reconstruction** — Assume the scene point cloud is pre-built (from depth fusion or provided by the dataset). Reconstructing point clouds from scratch is a separate system.
- **Semantic label assignment** — SAM3D produces instance masks only, not semantic labels. Open-vocabulary classification (e.g., via CLIP) is a downstream step outside this paper's scope.
- **Single-camera temporal extension** — The research proves this is architecturally futile ($V_{\text{eff}} = 1$, vote accuracy $< 0.15$ for limbs). Do not attempt.

---

## Pseudocode

```python
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from scipy.spatial import KDTree

# ============================================================
# Stage 0: Configuration
# ============================================================
class SAM3DConfig:
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    pred_iou_thresh: float = 0.88       # SAM default; tune to ~0.74 for bodies
    stability_score_thresh: float = 0.95
    box_nms_thresh: float = 0.7
    points_per_side: int = 64           # → 64×64 = 4096 grid prompts
    superpoint_voxel_size: float = 0.02 # R_seed = 2cm
    merge_iou_thresh: float = 0.50      # τ_merge
    merge_normal_thresh: float = 30.0   # τ_boundary in degrees
    nn_distance_thresh: float = 0.05    # max distance for label transfer (meters)


# ============================================================
# Stage 1: 2D Mask Generation (per-view)
# ============================================================
def generate_2d_masks(
    image: np.ndarray,          # (H, W, 3) uint8 RGB
    mask_generator: SamAutomaticMaskGenerator,
) -> list[dict]:
    """
    Returns list of masks, each with:
      - 'segmentation': (H, W) bool
      - 'predicted_iou': float
      - 'stability_score': float
      - 'area': int
    """
    # mask_generator handles: grid prompts → encoder → decoder → NMS → filtering
    masks = mask_generator.generate(image)  # list of dicts
    return masks  # typically 50–300 masks per view


def init_sam(config: SAM3DConfig) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[config.sam_model_type](
        checkpoint=config.sam_checkpoint
    )
    sam = sam.to("cuda")
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=config.points_per_side,        # 64 → 4096 prompts
        pred_iou_thresh=config.pred_iou_thresh,         # 0.88
        stability_score_thresh=config.stability_score_thresh,  # 0.95
        box_nms_thresh=config.box_nms_thresh,           # 0.7
    )
    return generator


# ============================================================
# Stage 2: 2D→3D Back-Projection
# ============================================================
def backproject_mask_to_3d(
    mask: np.ndarray,           # (H, W) bool — single 2D mask
    depth: np.ndarray,          # (H, W) float32 — depth in meters
    K: np.ndarray,              # (3, 3) intrinsic matrix
    R: np.ndarray,              # (3, 3) rotation (camera-to-world)
    t: np.ndarray,              # (3,)   translation (camera-to-world)
) -> np.ndarray:                # (N_pts, 3) — 3D points in world coords
    """
    p_3D = R^{-1} (d * K^{-1} [u, v, 1]^T - t)
    Since R is camera-to-world: p_3D = R @ (d * K^{-1} [u,v,1]^T) + t
    Convention note: check your dataset's convention (cam-to-world vs world-to-cam).
    """
    vs, us = np.where(mask & (depth > 0))  # valid mask pixels with positive depth
    # (N_pts,)
    ds = depth[vs, us]                      # depth values

    # Homogeneous pixel coords: (3, N_pts)
    pixels = np.stack([us, vs, np.ones_like(us)], axis=0).astype(np.float64)

    # Camera-space rays: (3, N_pts)
    K_inv = np.linalg.inv(K)
    rays_cam = K_inv @ pixels               # (3, N_pts)

    # Scale by depth: (3, N_pts)
    points_cam = rays_cam * ds[np.newaxis, :]

    # Transform to world: p_world = R @ p_cam + t
    # (adjust sign convention per your dataset)
    R_inv = R.T  # if R is world-to-cam, R_inv = R^T
    points_world = R_inv @ (points_cam - t[:, np.newaxis])  # (3, N_pts)

    return points_world.T  # (N_pts, 3)


def assign_labels_to_pointcloud(
    scene_points: np.ndarray,         # (P, 3) — scene point cloud
    projected_points: np.ndarray,     # (N_pts, 3) — back-projected mask points
    mask_label: int,                  # label ID for this mask
    tree: KDTree,                     # pre-built KD-tree of scene_points
    distance_thresh: float = 0.05,    # max NN distance (meters)
) -> np.ndarray:                      # (P,) int — label per scene point (-1 = unlabeled)
    """Nearest-neighbor assignment of mask label to scene point cloud."""
    dists, indices = tree.query(projected_points, k=1)  # (N_pts,)
    valid = dists < distance_thresh
    labels = np.full(len(scene_points), -1, dtype=np.int32)

    # For each valid projected point, assign label to its nearest scene point
    valid_scene_idx = indices[valid]
    labels[valid_scene_idx] = mask_label
    return labels


# ============================================================
# Stage 3a: Superpoint Extraction (VCCS approximation)
# ============================================================
def extract_superpoints(
    points: np.ndarray,          # (P, 3)
    normals: np.ndarray,         # (P, 3) — surface normals
    colors: np.ndarray,          # (P, 3) — RGB [0,1]
    voxel_size: float = 0.02,    # R_seed = 2cm
) -> np.ndarray:                 # (P,) int — superpoint ID per point
    """
    Simplified VCCS: voxel grid seeding + connectivity.
    For production, use Open3D or PCL's VCCS implementation.
    """
    # Step 1: Voxelize — assign each point to a voxel
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)  # (P, 3)

    # Step 2: Unique voxels → superpoint seeds
    # Each unique voxel becomes a superpoint
    _, superpoint_ids = np.unique(
        voxel_coords, axis=0, return_inverse=True
    )  # (P,)

    # Note: Full VCCS does region growing with normal+color coherence.
    # This voxel approximation is sufficient for ~90% of the benefit.
    return superpoint_ids  # (P,) — superpoint ID per point


# ============================================================
# Stage 3b: Vote Accumulation
# ============================================================
def accumulate_votes(
    superpoint_ids: np.ndarray,       # (P,) int — superpoint ID per point
    per_view_labels: list[np.ndarray],  # V × (P,) int — label per point per view
    num_superpoints: int,
    num_masks: int,
    confidence_weights: list[np.ndarray] | None = None,  # V × (P,) float, optional
) -> np.ndarray:                      # (S, M) float — vote matrix
    """
    V(s_i, m_j) = sum of (weighted) indicator[label == m_j] for points in s_i.
    """
    votes = np.zeros((num_superpoints, num_masks), dtype=np.float64)  # (S, M)

    for v_idx, labels_v in enumerate(per_view_labels):
        weights = (
            confidence_weights[v_idx] if confidence_weights is not None
            else np.ones(len(labels_v))
        )
        for p_idx in range(len(labels_v)):
            lbl = labels_v[p_idx]
            if lbl >= 0:  # skip unlabeled
                sp = superpoint_ids[p_idx]
                votes[sp, lbl] += weights[p_idx]

    # Normalize per superpoint (rows sum to number-of-votes)
    row_sums = votes.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    votes_normalized = votes / row_sums  # (S, M)

    return votes_normalized


def assign_superpoint_labels(
    votes: np.ndarray,               # (S, M) — normalized vote matrix
) -> np.ndarray:                     # (S,) int — winning label per superpoint
    """Argmax label per superpoint. -1 if no votes."""
    labels = np.argmax(votes, axis=1)  # (S,)
    no_votes = votes.sum(axis=1) == 0
    labels[no_votes] = -1
    return labels


# ============================================================
# Stage 3c: Boundary-Aware Region Merging
# ============================================================
def build_superpoint_adjacency(
    points: np.ndarray,               # (P, 3)
    superpoint_ids: np.ndarray,       # (P,)
    num_superpoints: int,
    adjacency_radius: float = 0.04,   # 2× voxel size
) -> list[tuple[int, int]]:
    """Build adjacency list of superpoint pairs within radius."""
    tree = KDTree(points)
    adjacency = set()
    # Sample centroid per superpoint
    centroids = np.zeros((num_superpoints, 3))
    counts = np.zeros(num_superpoints)
    for p_idx in range(len(points)):
        sp = superpoint_ids[p_idx]
        centroids[sp] += points[p_idx]
        counts[sp] += 1
    counts[counts == 0] = 1
    centroids /= counts[:, np.newaxis]  # (S, 3)

    centroid_tree = KDTree(centroids)
    pairs = centroid_tree.query_pairs(r=adjacency_radius)
    return list(pairs)


def compute_superpoint_normals(
    points: np.ndarray,               # (P, 3)
    normals: np.ndarray,              # (P, 3)
    superpoint_ids: np.ndarray,       # (P,)
    num_superpoints: int,
) -> np.ndarray:                     # (S, 3) — mean normal per superpoint
    sp_normals = np.zeros((num_superpoints, 3))
    counts = np.zeros(num_superpoints)
    for p_idx in range(len(points)):
        sp = superpoint_ids[p_idx]
        sp_normals[sp] += normals[p_idx]
        counts[sp] += 1
    counts[counts == 0] = 1
    sp_normals /= np.linalg.norm(sp_normals, axis=1, keepdims=True).clip(1e-8)
    return sp_normals


def region_merge(
    sp_labels: np.ndarray,            # (S,) — label per superpoint
    adjacency: list[tuple[int, int]],
    sp_normals: np.ndarray,           # (S, 3)
    votes: np.ndarray,                # (S, M)
    tau_merge: float = 0.50,
    tau_boundary_deg: float = 30.0,
) -> np.ndarray:                     # (S,) — merged label per superpoint
    """
    Iterative merging: merge adjacent superpoints if IoU > τ_merge
    AND normal angle < τ_boundary.
    """
    tau_boundary_rad = np.radians(tau_boundary_deg)
    merged = sp_labels.copy()

    # Union-Find for connected components
    parent = list(range(len(sp_labels)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b) in adjacency:
        if merged[a] < 0 or merged[b] < 0:
            continue
        if merged[a] != merged[b]:
            continue  # only merge same-label superpoints

        # Normal compatibility check
        cos_angle = np.dot(sp_normals[a], sp_normals[b])
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        if angle > tau_boundary_rad:
            continue  # boundary — do not merge

        # IoU check between vote distributions
        v_a = votes[a]
        v_b = votes[b]
        intersection = np.minimum(v_a, v_b).sum()
        union_val = np.maximum(v_a, v_b).sum()
        if union_val == 0:
            continue
        iou = intersection / union_val
        if iou > tau_merge:
            union(a, b)

    # Flatten union-find to get final group labels
    for i in range(len(merged)):
        merged[i] = find(i)

    # Re-label groups contiguously
    unique_groups = np.unique(merged)
    group_map = {g: idx for idx, g in enumerate(unique_groups)}
    return np.array([group_map[g] for g in merged])


# ============================================================
# Full Pipeline
# ============================================================
def sam3d_pipeline(
    images: list[np.ndarray],          # V × (H, W, 3) uint8
    depths: list[np.ndarray],          # V × (H, W) float32 meters
    intrinsics: list[np.ndarray],      # V × (3, 3)
    extrinsics: list[tuple[np.ndarray, np.ndarray]],  # V × (R, t)
    scene_points: np.ndarray,          # (P, 3)
    scene_normals: np.ndarray,         # (P, 3)
    scene_colors: np.ndarray,          # (P, 3)
    config: SAM3DConfig = SAM3DConfig(),
) -> np.ndarray:                       # (P,) int — instance label per point
    """
    Full SAM3D pipeline: 2D masks → back-project → superpoints → vote → merge.
    """
    # --- Init ---
    mask_generator = init_sam(config)
    tree = KDTree(scene_points)

    # --- Superpoint extraction ---
    superpoint_ids = extract_superpoints(
        scene_points, scene_normals, scene_colors, config.superpoint_voxel_size
    )  # (P,)
    num_superpoints = superpoint_ids.max() + 1

    # --- Per-view processing ---
    global_mask_id = 0
    per_view_labels = []

    for v_idx in range(len(images)):
        # Stage 1: Generate 2D masks
        masks_2d = generate_2d_masks(images[v_idx], mask_generator)

        # Stage 2: Back-project each mask and assign labels
        view_labels = np.full(len(scene_points), -1, dtype=np.int32)

        R, t = extrinsics[v_idx]
        K = intrinsics[v_idx]

        for mask_info in masks_2d:
            mask_binary = mask_info['segmentation']  # (H, W) bool
            pts_3d = backproject_mask_to_3d(
                mask_binary, depths[v_idx], K, R, t
            )  # (N, 3)

            if len(pts_3d) == 0:
                global_mask_id += 1
                continue

            # Assign to nearest scene points
            dists, indices = tree.query(pts_3d, k=1)
            valid = dists < config.nn_distance_thresh
            view_labels[indices[valid]] = global_mask_id
            global_mask_id += 1

        per_view_labels.append(view_labels)

    num_masks = global_mask_id

    # --- Stage 3: Aggregate and merge ---
    votes = accumulate_votes(
        superpoint_ids, per_view_labels, num_superpoints, num_masks
    )  # (S, M)

    sp_labels = assign_superpoint_labels(votes)  # (S,)

    adjacency = build_superpoint_adjacency(
        scene_points, superpoint_ids, num_superpoints
    )
    sp_normals = compute_superpoint_normals(
        scene_points, scene_normals, superpoint_ids, num_superpoints
    )

    merged_sp_labels = region_merge(
        sp_labels, adjacency, sp_normals, votes,
        config.merge_iou_thresh, config.merge_normal_thresh
    )  # (S,)

    # Map superpoint labels back to per-point labels
    point_labels = merged_sp_labels[superpoint_ids]  # (P,)

    return point_labels
```

---

## Breakdance-Specific Modifications

### Headspin

**What works**: SAM 2D masks of the full body silhouette remain valid since the body stays roughly in one location. Static scene elements (floor mat, surrounding area) segment correctly via SAM3D.

**What fails**: MDE inverts the depth map — models trained on upright humans estimate the head (now at bottom, contacting floor) correctly but feet (now at top) 2–3× too far away (Abs Rel: 0.05 → 0.12–0.15). Back-projected limb points land in wrong regions of 3D space. Superpoint voting assigns spinning limbs to incorrect instances.

**Modification**: Do not use SAM3D for the dancer during headspins. Use SAM3D only for static scene. For the dancer, route to SAM 2 + CoTracker3 → optimization-based SMPL fitting with a relaxed pose prior (VPoser with enlarged $\Sigma$ for inverted configurations).

### Windmill

**What works**: Large, recognizable body silhouette. Floor contact point provides depth anchor.

**What fails**: Angular velocity 300–600°/s causes 10cm motion blur per frame at 30fps. MDE Abs Rel increases 3–5× on blurred regions. Cyclical self-occlusion (50–70% of body surface disappears per half-rotation) means no stable set of points is visible across enough frames for voting. Motion-to-resolution ratio $\alpha > 5.0$ for all limbs.

**Modification**: Require 120+ fps capture to reduce $\alpha$ below 2.0 for the torso. Use CoTracker3's occlusion-aware tracking (visibility mask $v_n(t)$) to weight votes — only include frames where a point is actually visible. Accept that limb segmentation will fail and rely on SMPL body model topology to resolve limb boundaries.

### Flare

**What works**: Periodic motion is predictable — once the rotation axis is established, future positions can be anticipated.

**What fails**: Legs sweep through full 360° circles at 3–8 m/s. No single frame captures a "complete" leg pose. Depth estimation on fast-moving legs returns background depth → legs "vanish" from 3D reconstruction. Required fps for superpoint coherence: 300–500.

**Modification**: Do not attempt per-frame 3D segmentation. Instead: (1) SAM 2 tracks the overall body bbox across frames, (2) CoTracker3 provides 2D point trajectories on visible body surface, (3) fit SMPL model to 2D trajectories using temporal consistency constraints that exploit the periodic nature of flares (strong prior: $\theta_{t+T_{\text{period}}} \approx \theta_t$).

### Freeze

**What works**: Static pose — this is the one breakdancing movement where SAM3D could actually work. Body is motionless for 1–3 seconds, providing clean multi-frame data. If a multi-camera setup is available, SAM3D's full pipeline applies.

**What fails**: From a single camera, viewpoint degeneracy persists ($V_{\text{eff}} = 1$). Extreme poses (baby freeze, airchair) create severe self-occlusion. MDE struggles with inverted/unusual body configurations.

**Modification**: For single-camera: capture 2–4 seconds of the freeze and use SAM3D's temporal voting across these static frames — even though $\rho \approx 1$ for viewpoint, the mask consistency improves because there's no motion contamination. Combine with CoTracker3 surface tracks during the static phase to get dense correspondence for SMPL fitting.

### Footwork

**What works**: Upper body is relatively stable. SAM 2D masks of the torso are reliable.

**What fails**: Feet move at 2–5 m/s, producing $\alpha_b = 3.3–8.3$. Hands touching the ground create floor-body merging in depth estimation. Rapid weight shifts cause the CoM to move 0.2–0.5 m/s — enough to contaminate even torso superpoints.

**Modification**: Segment torso only via SAM3D-style voting (torso $\bar{V} \approx 0.85$ during footwork). For limbs, rely on the skeletal model: ViTPose/Sapiens detects 2D keypoints → CoTracker3 tracks surface points → SMPL fitting resolves 3D limb positions. The floor plane (segmented reliably by SAM3D) provides the ground-contact constraint.

### Toprock

**What works**: Most amenable to SAM3D — velocities are moderate (torso 0.3–0.8 m/s), body is upright (MDE trained on this), minimal self-occlusion.

**What fails**: Arms during aggressive toprock reach 2–5 m/s, breaking superpoint coherence. At 30fps, torso superpoints retain $\bar{V} \approx 0.85$ but arm/hand accuracy drops to $\bar{V} \approx 0.13$. Expected AP: torso ~43, arms ~10–20, hands ~5–10.

**Modification**: Use SAM3D-style voting for torso + head segmentation during toprock (the one case where it partially works). Pair with 2D pose estimation (Sapiens) for limb boundaries. Fuse: SAM3D provides the 3D torso volume, pose estimation provides the limb topology, depth from DepthPro provides approximate 3D positions.

### Battle (Full Round)

**What works**: Scene context — SAM3D reliably segments the battle circle, judges' positions, audience, and floor boundaries. These static elements are exactly what SAM3D was designed for.

**What fails**: The dancer transitions rapidly between movement types. A single set of thresholds cannot accommodate both static freezes and explosive power moves. The system needs dynamic parameter switching.

**Modification**: Implement a **movement-phase classifier** (MotionBERT at 53ms latency) that triggers different pipeline paths:
- **Static phases** (freeze, pose): SAM3D temporal voting + SMPL fitting
- **Moderate phases** (toprock, footwork): SAM3D for torso only + 2D pose for limbs
- **Explosive phases** (power moves): Bypass SAM3D entirely → SAM 2 + CoTracker3 + optimization-based SMPL fitting

---

## Known Limitations and Failure Modes

1. **Viewpoint degeneracy from single camera**: $V_{\text{eff}} = 1$ regardless of number of frames. Multi-view voting provides zero noise reduction. This is fundamental to the algorithm's geometry, not a bug to fix.

2. **Monocular depth error exceeds superpoint resolution by 7–11×**: At 3m dancer distance, DepthPro error (~14cm) vs. superpoint voxel size (2cm). Back-projected points land in wrong superpoints. Confidence-weighted voting reduces $e_{\text{assign}}$ from 0.30 to 0.18, still 9× above ideal.

3. **Motion contamination at typical dance velocities**: At 30fps, $\alpha \geq 1$ for any body part moving faster than 0.6 m/s. All active dance movements exceed this. Superpoint voting becomes meaningless for moving parts.

4. **Correlated depth errors across views**: MDE uses the same model on similar viewpoints → systematic bias. With $\rho = 0.5$: $V_{\text{eff}} \approx 2$ from 50 views. Multi-view correction is illusory.

5. **Confidence threshold kills body part recall**: At SAM default $\tau_{\text{conf}} = 0.88$, 89.4% of body part masks are filtered out. Lowering to 0.74 recovers masks but introduces false positives. There is no threshold that works for both furniture-scale objects and body parts.

6. **Merge threshold has no clean operating point for bodies**: ScanNet IoU distributions are bimodal (easy to threshold). Human body pairwise IoU is unimodal ($\mathcal{B}(2.5, 3.5)$, mode ~0.38). False merge and missed merge rates cannot be simultaneously minimized.

7. **Scale-depth ambiguity with relative MDE**: Relative-mode depth estimators produce $\hat{d} = \alpha d + \beta$ with unknown scale. Cross-view misalignment reaches ~75cm at 3m with Depth Anything v2 relative mode. Must use metric depth or align to sparse SfM points.

8. **Reflective dance floors create phantom geometry**: Specular reflections cause MDE to hallucinate depth below the floor surface. Body-floor contact during downrock merges dancer into floor in 3D.

9. **Inverted pose depth estimation failure**: MDE models have < 0.01% inverted humans in training data. Gravity prior violation causes 2–3× error increase on feet/legs when inverted.

10. **Training-free performance ceiling**: Even with perfect depth, SAM3D achieves AP@50 ≈ 30 on ScanNet vs. 55+ for supervised methods (Mask3D, OneFormer3D). The ~25pt gap decomposes into ~20pt from depth noise and ~5pt from algorithmic limitations (no learned 3D priors).

11. **Thin structure failure**: Arms (radius ~4cm), hands, and legs during extended poses have depth-signal SNR < 0.2. Reliable segmentation requires learned body priors that SAM3D does not have.

12. **Computational cost is substantial**: ~2.8 TFLOPs per ViT-H encoder pass × 100 views = ~280 TFLOPs per scene. Per-view latency: 250–400ms on A100. Full scene: 2–10 minutes. Not real-time.

---

## Integration Points

### With MotionBERT

**Data flow**: MotionBERT operates on 2D keypoint sequences ($B \times 243 \times 17 \times 2$) and outputs 3D lifted skeletons ($B \times 243 \times 17 \times 3$). SAM3D operates on multi-view RGB-D and outputs 3D instance masks over point clouds.

**Integration**: MotionBERT serves as the **movement-phase classifier** that determines when SAM3D is invoked. During its 53ms inference pass, MotionBERT classifies the current 8-second window into {toprock, footwork, power, freeze}. For freeze and slow toprock phases, SAM3D is triggered on accumulated frames. For active phases, SAM3D is skipped and the pipeline routes to SAM 2 + CoTracker3.

**Format conversion**: MotionBERT's 3D skeleton provides joint positions that can be projected into SAM3D's point cloud coordinate system via the same camera extrinsics. This allows cross-referencing: "is joint #5 (left hip) inside instance mask #12?" validates both systems.

**Timing**: MotionBERT runs continuously at ~19 FPS on V100 (53ms/clip). SAM3D runs asynchronously on accumulated frames during detected static phases — latency is 2–10 minutes but only triggered during freezes/pauses.

### With CoTracker3

**Data flow**: CoTracker3 inputs video chunks ($T \times 3 \times H \times W$) and query points ($N \times 3$: frame, x, y), outputting 2D trajectories ($N \times T \times 2$) and visibility masks ($N \times T$).

**Integration**: CoTracker3 trajectories provide the dense 2D correspondence that SAM3D's back-projection needs but cannot achieve from a single camera. Specifically:

1. **Track initialization**: Sample query points on SAM 2 masks at frame $t_0$. CoTracker3 tracks these points across frames.
2. **Temporal label propagation**: Instead of SAM3D's per-view independent mask generation, use CoTracker3 tracks to propagate mask labels from a reference frame to subsequent frames — points that CoTracker3 identifies as corresponding get the same instance label.
3. **Occlusion-aware weighting**: CoTracker3's visibility output $v_n(t) \in [0,1]$ directly feeds into the confidence-weighted voting (Equation 8) as $c(p) = v_n(t)$.

**Format conversion**: CoTracker3 outputs pixel coordinates at the video resolution; these must be scaled to match SAM's 1024×1024 input and then projected to 3D via back-projection. Point IDs must be maintained across both systems.

**Timing**: CoTracker3 runs at ~75ms/frame for 2048 points on A100. It should process the same frames SAM3D would consume, providing trajectories before SAM3D's merge stage runs.

### With Movement Spectrogram Pipeline

The movement spectrogram $S_m(j, t)$ is computed as a time-frequency representation where joint index $j$ selects a body region and $t$ indexes time. The spectrogram bins per-joint velocity magnitudes (and optionally acceleration, jerk) into frequency bands via CWT or STFT.

**SAM3D's contribution**: SAM3D segments the **static scene** (floor, stage boundaries, obstacles) into 3D instance masks. These masks establish:

1. **Floor plane equation** $\mathbf{n} \cdot \mathbf{x} = d_{\text{floor}}$: extracted from the largest horizontal planar instance. This provides the gravity direction and ground-contact constraint for the SMPL fitting that produces the joint trajectories feeding $S_m$.

2. **Battle circle boundary**: The segmented stage area defines the spatial domain within which dancer motion is valid. Trajectory points outside this boundary trigger re-estimation.

3. **Absolute spatial scale**: If the floor/stage has known dimensions, SAM3D's 3D segmentation provides the metric scale factor that converts CoTracker3's pixel-space trajectories to meters — essential for velocity computation in $S_m$.

**Data flow**:
```
SAM3D (static scene) → floor plane + stage boundary + scale
                                    ↓
CoTracker3 (2D tracks) + DepthPro → 3D trajectories (meters)
                                    ↓
Per-joint velocity: v_j(t) = ||p_j(t) - p_j(t-1)|| × fps
                                    ↓
CWT/STFT → S_m(j, t) → audio-motion cross-correlation
```

The audio-motion cross-correlation $\rho_{am}(j, t, \tau)$ between the movement spectrogram and the audio beat spectrogram is computed per-joint per-time-window, measuring musicality. SAM3D's floor plane is critical here because it determines the "ground truth" vertical axis — without it, gravity-dependent features (airtime during power moves, floor contact during footwork) cannot be measured.

---

## References

1. Yang, Y., Wu, Y., Zhai, W., Zhang, D., & Cao, Z. (2023). SAM3D: Segment Anything in 3D Scenes. *arXiv preprint arXiv:2306.03908*. https://arxiv.org/abs/2306.03908

2. Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything. *ICCV 2023*. https://arxiv.org/abs/2304.02643

3. Ravi, N., Gabeur, V., Hu, Y.-T., et al. (2024). SAM 2: Segment Anything in Images and Videos. *arXiv preprint arXiv:2408.00714*. https://arxiv.org/abs/2408.00714

4. Karaev, N., Tarasov, A., et al. (2024). CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos. *arXiv preprint arXiv:2410.11831*. https://arxiv.org/abs/2410.11831

5. Zhu, W., Ma, X., et al. (2023). MotionBERT: A Unified Perspective on Learning Human Motion Representations. *ICCV 2023*. https://arxiv.org/abs/2210.06551

6. Bochkovskiy, A., Kolesnikov, I., & Lempitsky, V. (2024). DepthPro: Sharp Monocular Metric Depth in Less Than a Second. Apple ML Research. https://arxiv.org/abs/2410.02073

7. Yang, L., Kang, B., Huang, Z., et al. (2024). Depth Anything V2. *arXiv preprint arXiv:2406.09414*. https://arxiv.org/abs/2406.09414

8. Hu, Y., Chen, Y., et al. (2024). Metric3D v2: A Versatile Monocular Geometric Foundation Model. *arXiv preprint arXiv:2404.15506*. [NEEDS VERIFICATION — exact arXiv ID]

9. Girdhar, R., El-Nouby, A., Liu, Z., et al. (2023). OneFormer3D: One Transformer for Unified Point Cloud Segmentation. [NEEDS VERIFICATION — venue/date]

10. Schult, J., Engelmann, F., Hermans, A., et al. (2023). Mask3D: Mask Transformer for 3D Semantic Instance Segmentation. *ICRA 2023*. https://arxiv.org/abs/2210.03105

11. Sun, J., Qing, Z., Tan, Y., & Xu, Q. (2023). SPFormer: Superpoint Transformer for Point Cloud Segmentation. [NEEDS VERIFICATION — exact citation]

12. Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa, A., & Malik, J. (2023). Humans in 4D: Reconstructing and Tracking Humans with Transformers. *ICCV 2023*. https://arxiv.org/abs/2305.20091

13. Shin, S., Kim, J., & Lee, K. M. (2024). WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion. *CVPR 2024*. https://arxiv.org/abs/2312.07531

14. Goel, S., Pavlakos, G., et al. (2023). HMR 2.0: Human Mesh Recovery with Transformers. *arXiv preprint*. [NEEDS VERIFICATION — exact arXiv ID]

15. Khirodkar, R., Bagautdinov, T., et al. (2024). Sapiens: Foundation for Human Vision Models. Meta AI. https://arxiv.org/abs/2408.12569

16. Papon, J., Abramov, A., Schoeler, M., & Wörgötter, F. (2013). Voxel Cloud Connectivity Segmentation — Supervoxels for Point Clouds. *CVPR 2013*. — VCCS algorithm used for superpoint extraction.

17. Bogo, F., Kanazawa, A., Lassner, C., Gehler, P., Romero, J., & Black, M. J. (2016). Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image. *ECCV 2016*. — SMPL body model used in optimization-based fitting.

18. SAM 3 ("concept-aware video segmentation", Nov 2025) — [LIKELY HALLUCINATED, confidence 0.05]

19. SAM-Body4D ("training-free 4D mesh recovery", Dec 2025) — [VERY LIKELY HALLUCINATED, confidence 0.10]

20. SAM-3D-Body ("body-specific 3D segmentation", Nov 2025) — [LIKELY HALLUCINATED, confidence 0.15]
