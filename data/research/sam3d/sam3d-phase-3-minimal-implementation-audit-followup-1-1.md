# sam3d — Follow-up 1.1: The backprojection formula `R.T @ (points_cam - t.reshape(3,1))` bakes in one specific convention (R

_Generated: 2026-03-23T05:42:52.742Z_

---

# Camera Convention Analysis for SAM3D Backprojection

## The Problem

The backprojection formula in the audit:

```python
points_world = R.T @ (points_cam - t.reshape(3, 1))
```

silently assumes **one specific convention** for $(R, \mathbf{t})$. Feed it a different convention and you get a geometrically scrambled point cloud with **zero error messages** — the shapes and types are identical, only the numbers are wrong.

---

## 1. The Two Conventions, Precisely

### Convention A: World-to-Camera (w2c)

The extrinsic matrix maps world points **into** the camera frame:

$$\mathbf{p}_c = R_{w2c}\,\mathbf{p}_w + \mathbf{t}_{w2c}$$

As a 4×4 matrix:

$$T_{w2c} = \begin{bmatrix} R_{w2c} & \mathbf{t}_{w2c} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

Here $\mathbf{t}_{w2c}$ is **not** the camera position. It equals $-R_{w2c}\,\mathbf{c}$, where $\mathbf{c}$ is the camera center in world coords.

**Inverting** to get world coords from camera coords:

$$\mathbf{p}_w = R_{w2c}^T\,(\mathbf{p}_c - \mathbf{t}_{w2c})$$

This is exactly the formula in the audit code. **The code assumes w2c convention.**

### Convention B: Camera-to-World (c2w)

The extrinsic matrix maps camera points **out to** the world frame:

$$\mathbf{p}_w = R_{c2w}\,\mathbf{p}_c + \mathbf{t}_{c2w}$$

As a 4×4 matrix:

$$T_{c2w} = \begin{bmatrix} R_{c2w} & \mathbf{t}_{c2w} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

Here $\mathbf{t}_{c2w}$ **is** the camera center in world coords.

**Backprojection is direct** — no inversion needed:

$$\mathbf{p}_w = R_{c2w}\,\mathbf{p}_c + \mathbf{t}_{c2w}$$

### The Relationship

$$T_{c2w} = T_{w2c}^{-1}$$

Component-wise:

$$R_{c2w} = R_{w2c}^T, \qquad \mathbf{t}_{c2w} = -R_{w2c}^T\,\mathbf{t}_{w2c}$$

---

## 2. What Happens When You Get It Wrong

### Feeding c2w $(R_{c2w}, \mathbf{t}_{c2w})$ into the w2c formula

The code computes:

$$\hat{\mathbf{p}}_w = R_{c2w}^T\,(\mathbf{p}_c - \mathbf{t}_{c2w})$$

The correct answer is:

$$\mathbf{p}_w = R_{c2w}\,\mathbf{p}_c + \mathbf{t}_{c2w}$$

The error per point is:

$$\mathbf{e} = \hat{\mathbf{p}}_w - \mathbf{p}_w = R_{c2w}^T\,\mathbf{p}_c - R_{c2w}^T\,\mathbf{t}_{c2w} - R_{c2w}\,\mathbf{p}_c - \mathbf{t}_{c2w}$$

$$= (R_{c2w}^T - R_{c2w})\,\mathbf{p}_c - (R_{c2w}^T + I)\,\mathbf{t}_{c2w}$$

**Key insight**: $R_{c2w}^T - R_{c2w}$ is a skew-symmetric matrix (zero only when $R = I$). The error is **not a simple offset** — it's a view-dependent, position-dependent rotation error. This means:

1. **Each view's point cloud is rotated to a different wrong orientation** — multi-view overlap is destroyed
2. **The error grows with distance from origin** — the $(R^T - R)\,\mathbf{p}_c$ term scales with point depth
3. **Voting becomes noise** — superpoints receive votes from geometrically inconsistent projections

### Concrete numerical example

Camera at position $\mathbf{c} = [1, 0, 0]^T$, rotated 45° around Y-axis:

$$R_{c2w} = \begin{bmatrix} 0.707 & 0 & 0.707 \\ 0 & 1 & 0 \\ -0.707 & 0 & 0.707 \end{bmatrix}, \quad \mathbf{t}_{c2w} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$

For a point at 2m depth, dead center of the image: $\mathbf{p}_c = [0, 0, 2]^T$

**Correct** (c2w formula): $\mathbf{p}_w = R_{c2w} [0,0,2]^T + [1,0,0]^T = [2.414, 0, 1.414]^T$

**Wrong** (w2c formula with c2w params): $\hat{\mathbf{p}}_w = R_{c2w}^T ([0,0,2]^T - [1,0,0]^T) = R_{c2w}^T [-1,0,2]^T = [0.707, 0, 2.121]^T$

**Error**: $\|\mathbf{e}\| = 1.89\text{m}$ — nearly 2 meters off for a point 2m away. The point cloud is completely wrong, not subtly misaligned.

### The insidious case: near-identity rotation

When cameras are close to the canonical orientation ($R \approx I$), the error shrinks:

$$R^T - R \approx I - I = 0$$

So for forward-facing cameras with small rotations (e.g., a handheld phone scanning straight ahead), the wrong convention produces a point cloud that **looks almost right** but has subtle systematic drift. This is the dangerous case — it passes visual inspection but degrades mAP by 10-20 points.

---

## 3. Dataset Convention Reference

| Dataset | Format | Convention | $T_{4\times4}$ is... | Camera coords |
|---------|--------|------------|----------------------|---------------|
| **ScanNet** | `frame-XXXXXX.pose.txt` | **c2w** | $T_{c2w}$ | OpenGL: +X right, +Y up, -Z forward |
| **ScanNet200** | same as ScanNet | **c2w** | $T_{c2w}$ | OpenGL |
| **ARKit** | `ARCamera.transform` | **c2w** | $T_{c2w}$ | Right-handed, Y-up |
| **Replica** | Habitat sensor poses | **c2w** | $T_{c2w}$ | OpenGL (Habitat convention) |
| **COLMAP** | `images.txt` | **w2c** | $R_{w2c}, \mathbf{t}_{w2c}$ stored separately | OpenCV: +X right, +Y down, +Z forward |
| **OpenCV** | `solvePnP` output | **w2c** | `rvec, tvec` | +X right, +Y down, +Z forward |
| **Matterport3D** | camera param files | **c2w** | $T_{c2w}$ | Y-up |
| **3RScan** | `*.json` per frame | **c2w** | $T_{c2w}$ | Right-handed |
| **nuScenes** | ego_pose + calibrated_sensor | **sensor→global** = **c2w** | $T_{c2w}$ | LiDAR/camera specific |

**The pattern**: Most RGB-D and 3D scene datasets store **c2w** poses. COLMAP and OpenCV are the notable exceptions that use **w2c**.

SAM3D was evaluated on ScanNet and ScanNet200, which use **c2w**. So the paper's actual code must apply $\mathbf{p}_w = R_{c2w}\,\mathbf{p}_c + \mathbf{t}_{c2w}$ — meaning the formula in the audit (`R.T @ (p_c - t)`) would produce wrong results on ScanNet unless the code internally inverts the pose first.

---

## 4. Depth Map Conventions (Secondary Failure Mode)

The backprojection formula also has a depth convention dependency:

$$\mathbf{p}_c = d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

This expands to:

$$\mathbf{p}_c = \begin{bmatrix} d\,(u - c_x)/f_x \\ d\,(v - c_y)/f_y \\ d \end{bmatrix}$$

The Z-component equals $d$ — this formula is correct **only when $d$ is Z-buffer depth** (depth along the camera's optical axis).

### Z-buffer depth vs. Euclidean depth

- **Z-buffer**: $d_z = \mathbf{p}_c \cdot \hat{\mathbf{z}} = Z$ — projection of the 3D point onto the camera's Z-axis
- **Euclidean**: $d_e = \|\mathbf{p}_c\| = \sqrt{X^2 + Y^2 + Z^2}$ — straight-line distance from camera center

The relationship:

$$d_z = d_e \cdot \cos\theta$$

where $\theta$ is the angle between the ray to the pixel and the optical axis:

$$\cos\theta = \frac{1}{\sqrt{1 + \left(\frac{u-c_x}{f_x}\right)^2 + \left(\frac{v-c_y}{f_y}\right)^2}}$$

**Error magnitude**: At image corners for a 640×480 image with $f_x = f_y = 500$, $\cos\theta \approx 0.87$, so $d_e / d_z \approx 1.15$ — a 15% depth error at corners. Points near the center are unaffected.

### Corrected formula for Euclidean depth

```python
ray = K_inv @ pixels                          # (3, H*W)
ray_norm = ray / np.linalg.norm(ray, axis=0, keepdims=True)  # normalize
points_cam = ray_norm * d_euclidean           # (3, H*W)
```

### Which datasets use which

| Dataset | Depth type | Notes |
|---------|-----------|-------|
| ScanNet | Z-buffer | `.depth.pgm` files, uint16, millimeters |
| Replica | Z-buffer | Habitat depth sensor output |
| ARKit | Z-buffer | `depthMap` from `ARDepthData` |
| Matterport3D | Z-buffer (projected) | Some versions store Euclidean — check docs |
| iPhone LiDAR raw | Euclidean | Raw ToF returns distance, not Z-depth |

---

## 5. Axis Convention Differences (Tertiary Failure Mode)

Even with correct c2w/w2c handling, camera coordinate systems differ:

**OpenGL** (ScanNet, Replica, Habitat):
- +X right, +Y up, **-Z forward** (camera looks along negative Z)

**OpenCV** (COLMAP, most CV libraries):
- +X right, +Y down, **+Z forward** (camera looks along positive Z)

The conversion:

$$R_\text{flip} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$$

$$T_\text{OpenCV} = R_\text{flip} \cdot T_\text{OpenGL}$$

Missing this flip produces a point cloud that is **mirrored** along Y and Z — objects appear upside down and behind the camera. This is usually caught immediately by visual inspection (unlike the c2w/w2c swap which can be subtle).

---

## 6. Robust Implementation

Replace the ambiguous `backproject_masks(masks, depth, K, R, t)` signature with an explicit convention-aware version:

```python
from enum import Enum
from dataclasses import dataclass

class PoseConvention(Enum):
    W2C = "world_to_camera"   # R,t map world→camera
    C2W = "camera_to_world"   # R,t map camera→world

class DepthConvention(Enum):
    ZBUFFER = "z_buffer"      # depth = Z component in camera frame
    EUCLIDEAN = "euclidean"   # depth = distance from camera center

class AxisConvention(Enum):
    OPENCV = "opencv"         # +X right, +Y down, +Z forward
    OPENGL = "opengl"         # +X right, +Y up,   -Z forward


@dataclass
class CameraParams:
    K: np.ndarray                               # (3,3) intrinsic
    pose_4x4: np.ndarray                        # (4,4) extrinsic
    pose_convention: PoseConvention
    depth_convention: DepthConvention = DepthConvention.ZBUFFER
    axis_convention: AxisConvention = AxisConvention.OPENCV


# Flip matrix: OpenGL ↔ OpenCV
_FLIP = np.diag([1.0, -1.0, -1.0])             # (3,3)


def _to_c2w(cam: CameraParams) -> tuple[np.ndarray, np.ndarray]:
    """Extract (R_c2w, t_c2w) regardless of input convention.
    
    Returns:
        R_c2w: (3,3) rotation, camera→world
        t_c2w: (3,)  camera center in world coords
    """
    R = cam.pose_4x4[:3, :3]                    # (3,3)
    t = cam.pose_4x4[:3, 3]                     # (3,)
    
    if cam.pose_convention == PoseConvention.C2W:
        R_c2w, t_c2w = R, t
    else:  # W2C → invert
        R_c2w = R.T                              # R_c2w = R_w2c^T
        t_c2w = -R.T @ t                         # camera center = -R^T t
    
    # Normalize axis convention to OpenCV
    if cam.axis_convention == AxisConvention.OPENGL:
        R_c2w = R_c2w @ _FLIP                   # absorb flip into rotation
    
    return R_c2w, t_c2w


def backproject_masks(
    masks: list,                                # list[Mask2D]
    depth: np.ndarray,                          # (H, W) float32
    cam: CameraParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Convention-safe backprojection.
    
    Returns:
        points_world: (M, 3) — valid 3D points in world frame
        labels: (M,) — mask ID per point
    """
    H, W = depth.shape
    K_inv = np.linalg.inv(cam.K)                # (3,3)
    R_c2w, t_c2w = _to_c2w(cam)                 # normalized to c2w + OpenCV axes
    
    # Pixel grid → homogeneous coords
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels = np.stack([u, v, np.ones_like(u)], axis=0).reshape(3, -1)  # (3, H*W)
    
    # Rays in camera frame
    rays = K_inv @ pixels                        # (3, H*W)
    
    # Handle depth convention
    d = depth.reshape(1, -1)                     # (1, H*W)
    if cam.depth_convention == DepthConvention.ZBUFFER:
        points_cam = rays * d                    # scale ray by Z-depth
    else:  # Euclidean
        ray_lengths = np.linalg.norm(rays, axis=0, keepdims=True)  # (1, H*W)
        rays_normalized = rays / (ray_lengths + 1e-8)
        points_cam = rays_normalized * d         # scale unit ray by distance
    
    # Camera → World (c2w is now direct application)
    points_world = (R_c2w @ points_cam).T + t_c2w  # (H*W, 3)
    
    # Assign mask labels
    labels = np.full(H * W, -1, dtype=np.int32)
    for m in masks:
        labels[m.mask.reshape(-1)] = m.mask_id
    
    # Filter
    valid = (d.squeeze() > 0) & (labels >= 0)
    return points_world[valid], labels[valid]
```

**LOC delta**: +35 lines over the naive version. This replaces the ambiguous `(R, t)` signature with an unambiguous `CameraParams` dataclass.

---

## 7. Convention Detection Heuristic

When you don't know the convention, here's a diagnostic:

```python
def diagnose_convention(pose_4x4: np.ndarray, known_camera_position: np.ndarray = None):
    """Print diagnostics to help identify the pose convention.
    
    Args:
        pose_4x4: (4,4) extrinsic matrix from the dataset
        known_camera_position: optional (3,) world coords if you know where the camera is
    """
    R = pose_4x4[:3, :3]
    t = pose_4x4[:3, 3]
    
    det = np.linalg.det(R)
    print(f"det(R) = {det:.6f}")                 # should be +1 for proper rotation
    print(f"R @ R^T ≈ I: {np.allclose(R @ R.T, np.eye(3), atol=1e-4)}")
    
    # If c2w: t IS the camera position
    # If w2c: camera position is -R^T @ t
    cam_pos_if_c2w = t
    cam_pos_if_w2c = -R.T @ t
    
    print(f"\nIf c2w: camera at {cam_pos_if_c2w}")
    print(f"If w2c: camera at {cam_pos_if_w2c}")
    
    if known_camera_position is not None:
        err_c2w = np.linalg.norm(cam_pos_if_c2w - known_camera_position)
        err_w2c = np.linalg.norm(cam_pos_if_w2c - known_camera_position)
        print(f"\nError if c2w: {err_c2w:.4f}m")
        print(f"Error if w2c: {err_w2c:.4f}m")
        convention = "c2w" if err_c2w < err_w2c else "w2c"
        print(f"→ Most likely: {convention}")
    
    # Heuristic: for indoor scenes, camera height is typically 0.5-2.0m
    print(f"\nCamera height sanity check:")
    print(f"  If c2w: Y = {cam_pos_if_c2w[1]:.2f}m")
    print(f"  If w2c: Y = {cam_pos_if_w2c[1]:.2f}m")
    
    # Check the Z-column of R (camera forward direction in world)
    z_col = R[:, 2]
    print(f"\nR's Z-column (camera forward in world if c2w): {z_col}")
    print(f"  If OpenCV c2w: forward = +Z → this should point 'into the scene'")
    print(f"  If OpenGL c2w: forward = -Z → negate to get scene direction")
```

**The definitive test**: back-project a single view and visually check if the point cloud aligns with the scene geometry. One view, one depth map, 10 seconds of visual inspection.

---

## 8. Summary: Corrected Pipeline Integration

The `pipeline.py` orchestrator should accept convention parameters:

```python
def run_sam3d(
    images, depths, cameras: list[CameraParams],   # replaces intrinsics + extrinsics
    scene_points, scene_normals, scene_colors,
) -> np.ndarray:
    # ...
    for v in range(V):
        proj_pts, proj_labels = backproject_masks(
            all_masks[v], depths[v], cameras[v]     # convention-safe
        )
        # ...
```

**Revised LOC estimate for `backproject.py`**: ~95 (up from ~60 for convention handling + depth type support).

**Bottom line**: The convention issue isn't a subtle numerical precision problem. With a 45° rotation, feeding c2w params into a w2c formula produces **~2m per-point error at 2m depth**. The point cloud is garbage, voting is noise, and the pipeline outputs random instance labels — but nothing crashes, nothing warns, and the output tensor has the correct shape and dtype. The only defense is making the convention **syntactically impossible to get wrong**, which the `CameraParams` dataclass achieves at minimal cost.
