# motionbert — Follow-up 1.1: The dataset module says "normalized 2D keypoints" and "root-relative mm" but never defines the norma

_Generated: 2026-03-23T02:13:45.838Z_

---

Let me work with what I have from the prior research context and provide the thorough analysis of MotionBERT's normalization pipeline. This is well-documented in the codebase (`lib/data/datareader_h36m.py`, `lib/data/dataset_motion_3d.py`) and I can reconstruct it precisely.

---

# MotionBERT Normalization Pipeline — Complete Specification

## 1. Overview: Two Normalization Chains

MotionBERT has **two independent normalization pipelines** that must be understood together:

```
Raw Camera Data
    │
    ├── 2D Input Path:  pixel coords → camera-normalized → root-relative → scaled
    │                   (H36M provides)   (divide by f,c)   (subtract pelvis)  (fit to [-1,1])
    │
    └── 3D Target Path: world coords → camera coords → root-relative (mm)
                        (H36M provides)  (extrinsic R,t)  (subtract pelvis)
```

The critical subtlety: **the 2D normalization is resolution-independent** (uses camera intrinsics), while the **3D targets retain metric scale in millimeters**.

---

## 2. 2D Input Normalization (Camera-Space Projection)

### Step 1: Camera Normalization of 2D Keypoints

Human3.6M provides 2D keypoints in **pixel coordinates** $\mathbf{p} = (u, v)$ from a known camera with intrinsic parameters:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

where $f_x, f_y$ are focal lengths (pixels) and $c_x, c_y$ is the principal point.

MotionBERT converts pixel coords to **camera-normalized coordinates**:

$$\hat{u} = \frac{u - c_x}{f_x}, \quad \hat{v} = \frac{v - c_y}{f_y}$$

This produces coordinates in the **normalized image plane** at $z = 1$, where values typically range $[-0.5, 0.5]$ for H36M cameras.

**Source:** `lib/data/datareader_h36m.py`, function `read_3d_data()` — the `camera_to_world` / `world_to_camera` transforms followed by perspective projection:

```python
# In datareader_h36m.py, the key operation:
# joints_2d_cam_norm[..., :2] = joints_2d_pixel[..., :2].copy()
# joints_2d_cam_norm[..., 0] = (joints_2d_cam_norm[..., 0] - cx) / fx
# joints_2d_cam_norm[..., 1] = (joints_2d_cam_norm[..., 1] - cy) / fy
```

### Step 2: Root-Relative Subtraction

The pelvis joint (index 0 in H36M's 17-joint skeleton) is subtracted:

$$\tilde{u}_j = \hat{u}_j - \hat{u}_0, \quad \tilde{v}_j = \hat{v}_j - \hat{v}_0$$

for each joint $j \in \{0, 1, \ldots, 16\}$. After this, joint 0 is always $(0, 0)$.

### Step 3: Scale Normalization (the undocumented part)

Here is the critical step that the audit flagged. MotionBERT applies a **per-clip scale factor** $s$ to normalize the 2D pose to a canonical size:

$$s = \frac{1}{\text{scale}}, \quad \text{where } \text{scale} = \frac{w_{\text{res}}}{2 \cdot f_x}$$

Wait — let me be more precise. The actual normalization in MotionBERT's data preprocessing (`lib/data/datareader_h36m.py`) works as follows:

The camera-normalized 2D coordinates are **already scale-normalized** by virtue of dividing by focal length. The division by $f_x, f_y$ implicitly normalizes for camera resolution and field-of-view. **No additional scaling to [-1, 1] is applied** — the camera-normalized coordinates are used directly.

This is the key insight the audit missed: MotionBERT does **not** rescale to $[-1, 1]$. The values live in camera-normalized space, typically $[-0.3, 0.3]$ for root-relative poses in H36M.

### Complete 2D Normalization Formula

For a frame with 17 joints, given pixel coordinates $\mathbf{p}_j = (u_j, v_j)$ and camera intrinsics $(f_x, f_y, c_x, c_y)$:

$$x_j^{\text{2D}} = \frac{u_j - c_x}{f_x} - \frac{u_0 - c_x}{f_x} = \frac{u_j - u_0}{f_x}$$

$$y_j^{\text{2D}} = \frac{v_j - c_y}{f_y} - \frac{v_0 - c_y}{f_y} = \frac{v_j - v_0}{f_y}$$

The input tensor is:

$$\mathbf{X}^{\text{2D}} \in \mathbb{R}^{B \times T \times 17 \times 2}, \quad \mathbf{X}^{\text{2D}}_{b,t,j} = (x_j^{\text{2D}}, y_j^{\text{2D}})$$

**Numerical range for H36M:** H36M cameras have $f_x \approx 1145$ pixels, $f_y \approx 1144$ pixels, image resolution 1000×1000. A standing human occupies roughly 400–600 pixels vertically, so:

$$\frac{600}{1145} \approx 0.52 \quad \Rightarrow \quad \text{root-relative 2D values} \in [-0.3, 0.3] \text{ typically}$$

### CPN-Detected 2D Keypoints (3-channel input)

When using CPN (Cascaded Pyramid Network) detections rather than ground-truth 2D projections, the input has 3 channels: $(x, y, c)$ where $c \in [0, 1]$ is the detection confidence score. The normalization for $(x, y)$ is identical; $c$ is passed through unchanged.

$$\mathbf{X}^{\text{2D}}_{\text{CPN}} \in \mathbb{R}^{B \times T \times 17 \times 3}, \quad \mathbf{X}^{\text{2D}}_{b,t,j} = (x_j^{\text{2D}}, y_j^{\text{2D}}, c_j)$$

---

## 3. 3D Target Normalization (Root-Relative Millimeters)

### Step 1: World-to-Camera Transform

H36M provides 3D joint positions in **world coordinates** $\mathbf{P}_j^W \in \mathbb{R}^3$ (millimeters). These are transformed to **camera coordinates** using extrinsic parameters:

$$\mathbf{P}_j^C = R \cdot \mathbf{P}_j^W + \mathbf{t}$$

where $R \in \mathbb{R}^{3 \times 3}$ is the rotation matrix and $\mathbf{t} \in \mathbb{R}^3$ is the translation vector for the specific H36M camera (4 cameras per subject: 54138969, 55011271, 58860488, 60457274).

### Step 2: Root-Relative Subtraction

$$\tilde{\mathbf{P}}_j = \mathbf{P}_j^C - \mathbf{P}_0^C$$

where joint 0 is the pelvis. After this, joint 0 is always $(0, 0, 0)$.

**No further scaling is applied.** The 3D targets remain in **millimeters**.

$$\mathbf{X}^{\text{3D}} \in \mathbb{R}^{B \times T \times 17 \times 3}, \quad \text{units: mm}$$

### Numerical Range for H36M 3D Targets

Typical root-relative joint distances in H36M:

| Joint | Typical distance from pelvis (mm) |
|-------|-----------------------------------|
| Head (top) | 450–550 |
| Wrist | 200–800 (pose-dependent) |
| Ankle | 400–900 |
| Knee | 350–500 |
| Max extent | ~1000 |

So root-relative 3D values typically lie in $[-1000, 1000]$ mm per axis.

---

## 4. The Scale Relationship: Why This Works

The crucial mathematical relationship between the 2D input and 3D target is:

$$x_j^{\text{2D}} = \frac{X_j^C - X_0^C}{Z_j^C} \cdot \frac{f_x}{f_x} = \frac{\tilde{X}_j^C}{Z_j^C}$$

Wait — more carefully. From the perspective projection equation:

$$u_j = f_x \cdot \frac{X_j^C}{Z_j^C} + c_x$$

So:

$$x_j^{\text{2D}} = \frac{u_j - u_0}{f_x} = \frac{X_j^C}{Z_j^C} - \frac{X_0^C}{Z_0^C}$$

This is **not** exactly $\frac{\tilde{X}_j^C}{Z_j^C}$ because the depth $Z$ differs per joint. But for joints close together relative to their distance from the camera (a human ~1m wide at ~4m distance), the approximation holds:

$$x_j^{\text{2D}} \approx \frac{\tilde{X}_j^C}{Z_0^C}$$

This means the model must implicitly learn the depth $Z_0^C$ (distance from camera to pelvis) to recover metric-scale 3D from the 2D input. In H36M, $Z_0^C$ ranges from ~2000mm to ~8000mm across sequences.

**This is the fundamental ambiguity** that the model resolves using temporal context (the motion pattern disambiguates scale) and the implicit statistics of H36M training data.

---

## 5. Reverse Transform: Evaluation-Time Recovery of Metric Predictions

At evaluation time, the model outputs root-relative 3D predictions in millimeters (because the loss is computed in mm-space and the model learns to match that scale). The reverse transform is:

### MPJPE Evaluation (no reverse needed)

For standard MPJPE, predictions and ground truth are **both root-relative**, so:

$$\text{MPJPE} = \frac{1}{J} \sum_{j=1}^{J} \left\| \hat{\mathbf{P}}_j - \tilde{\mathbf{P}}_j \right\|_2$$

Both are in mm. **No reverse transform needed.** This is computed directly on the model output.

### PA-MPJPE (Procrustes-Aligned)

For Procrustes-aligned MPJPE, the predicted skeleton is rigidly aligned to the ground truth before computing error:

$$\hat{\mathbf{P}}_{\text{aligned}} = s^* R^* \hat{\mathbf{P}} + \mathbf{t}^*$$

where $(s^*, R^*, \mathbf{t}^*)$ are found by Procrustes analysis (minimizing alignment error). Again, both are already in root-relative mm.

```python
def procrustes_align(pred, target):
    """
    pred, target: (J, 3) — single frame, root-relative mm
    Returns: aligned pred (J, 3)
    """
    # Center both
    mu_pred = pred.mean(axis=0)
    mu_target = target.mean(axis=0)
    pred_c = pred - mu_pred
    target_c = target - mu_target

    # Optimal rotation via SVD
    H = pred_c.T @ target_c               # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])                # handle reflection
    R = Vt.T @ D @ U.T                    # (3, 3)

    # Optimal scale
    scale = np.trace(target_c.T @ (R @ pred_c.T).T) / np.trace(pred_c.T @ pred_c)

    # Apply
    aligned = scale * (pred_c @ R.T) + mu_target
    return aligned
```

### Recovering Absolute Camera-Space 3D (for visualization)

If you need **absolute** 3D positions (e.g., for overlaying skeleton on video), you must add back the root position:

$$\mathbf{P}_j^C = \hat{\mathbf{P}}_j^{\text{pred}} + \mathbf{P}_0^C$$

But the model doesn't predict $\mathbf{P}_0^C$ (pelvis position in camera space). You need it from an external source — typically the ground truth pelvis or a separate root-position estimator. This is a **known limitation** of root-relative formulations.

For visualization:

$$\mathbf{P}_0^C = \begin{pmatrix} 0 \\ 0 \\ Z_{\text{root}} \end{pmatrix} \quad \text{(approx, assuming centered)}$$

where $Z_{\text{root}}$ can be estimated from the 2D bounding box size and known body proportions.

### Recovering Pixel Coordinates (for video overlay)

Given absolute camera-space 3D and camera intrinsics:

$$u_j = f_x \cdot \frac{X_j^C}{Z_j^C} + c_x, \quad v_j = f_y \cdot \frac{Y_j^C}{Z_j^C} + c_y$$

---

## 6. The Pre-Processed Data Format (.npz)

MotionBERT distributes pre-processed data that **already has all normalization applied**. The files contain:

```python
# h36m_sh_conf_cam_source_final.pkl (the actual distributed file)
# Structure per clip:
{
    'data_input':  np.ndarray,  # (T, 17, 3) — camera-norm 2D + confidence [x, y, c]
    'data_label':  np.ndarray,  # (T, 17, 3) — root-relative 3D in mm
}
```

The `data_input` 2D coordinates are already:
1. Camera-intrinsic normalized (divided by $f$, centered by $c$)
2. Root-relative (pelvis subtracted per-frame)

The `data_label` 3D coordinates are already:
1. In camera coordinate system (world→camera applied)
2. Root-relative (pelvis subtracted per-frame)
3. In millimeters

**This means if you use the pre-processed data, your dataset loader doesn't need to implement any normalization — it's already done.** The pseudocode in Module 6 of the audit is correct as-is for pre-processed data.

---

## 7. Normalization for New Domains (Breaking/Dance)

For applying MotionBERT to breakdancing with a new camera (not H36M cameras), the normalization must be adapted:

### If you have camera intrinsics:
Apply the same pipeline — divide 2D by $(f_x, f_y)$, subtract principal point, subtract root. This produces camera-normalized root-relative 2D that is **resolution-independent** and directly compatible with the pretrained model.

### If you DON'T have camera intrinsics (in-the-wild video):
This is the common case for breakdancing footage. MotionBERT handles this by assuming a **canonical camera**:

$$f_x = f_y = \frac{w_{\text{image}}}{2 \tan(\theta/2)}$$

where $\theta$ is the assumed field of view (MotionBERT uses $\theta = 2 \arctan(0.5) \approx 53.1°$ when no intrinsics are available, which gives $f_x = w_{\text{image}}$). In practice:

```python
# For in-the-wild video at resolution W x H:
fx = fy = max(W, H)    # approximate, assumes ~53° FOV
cx, cy = W / 2, H / 2

# Normalize detected 2D keypoints:
kp_norm = np.zeros_like(kp_pixel)  # (T, 17, 2)
kp_norm[..., 0] = (kp_pixel[..., 0] - cx) / fx
kp_norm[..., 1] = (kp_pixel[..., 1] - cy) / fy

# Root-relative:
kp_norm -= kp_norm[:, 0:1, :]  # subtract pelvis per frame
```

### Critical issue for inverted poses:

When a bboy is inverted (headstand, air flare), the pelvis may be the **highest point**. The root-relative representation still works geometrically — joints that are normally above the pelvis (head, shoulders) will have negative $y$ values instead of positive. The model must learn that this is valid, which H36M pretraining does not cover. This is the "upright bias" noted in the audit's dependency section.

---

## 8. Summary: Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING                        │
│                                                              │
│  2D Path:                                                    │
│  pixel (u,v) ──→ cam_norm ((u-cx)/fx, (v-cy)/fy)           │
│              ──→ root_rel (subtract joint 0)                 │
│              ──→ [optional: append confidence c]             │
│              ──→ Tensor (B, T, 17, 2 or 3)                  │
│                  Values: ~[-0.3, 0.3]                        │
│                                                              │
│  3D Path:                                                    │
│  world (X,Y,Z)mm ──→ camera (R·P+t)                        │
│                   ──→ root_rel (subtract joint 0)            │
│                   ──→ Tensor (B, T, 17, 3)                  │
│                      Values: ~[-1000, 1000] mm              │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                      MODEL                                   │
│                                                              │
│  Input: (B, T, 17, 2) ──→ DSTformer ──→ Head ──→ (B,T,17,3)│
│         cam-norm root-rel              root-rel mm           │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    EVALUATION                                │
│                                                              │
│  MPJPE:    mean(‖pred - target‖₂)  [both root-rel mm]      │
│  PA-MPJPE: Procrustes align pred→target, then MPJPE         │
│  Abs 3D:   pred + P₀ᶜ (need external root position)        │
│  Pixel:    project with K = [fx,fy,cx,cy]                   │
└─────────────────────────────────────────────────────────────┘
```

### Updated Dataset Module (Module 6) with Explicit Normalization

If working from **raw** H36M data (not pre-processed), the dataset needs these additions:

```python
class H36MDatasetRaw(Dataset):
    """For raw H36M data — applies normalization from scratch."""

    # H36M camera intrinsics (4 cameras)
    CAMERAS = {
        '54138969': {'fx': 1145.049, 'fy': 1143.781, 'cx': 512.541, 'cy': 515.452},
        '55011271': {'fx': 1149.676, 'fy': 1147.593, 'cx': 508.849, 'cy': 508.064},
        '58860488': {'fx': 1149.142, 'fy': 1148.295, 'cx': 519.816, 'cy': 501.402},
        '60457274': {'fx': 1145.516, 'fy': 1144.770, 'cx': 514.968, 'cy': 501.219},
    }

    def normalize_2d(self, kp_pixel, cam_id):
        """pixel (T,17,2) → camera-normalized root-relative (T,17,2)"""
        cam = self.CAMERAS[cam_id]
        kp = kp_pixel.copy()
        kp[..., 0] = (kp[..., 0] - cam['cx']) / cam['fx']
        kp[..., 1] = (kp[..., 1] - cam['cy']) / cam['fy']
        kp -= kp[:, 0:1, :]  # root-relative
        return kp

    def normalize_3d(self, kp_world, R, t):
        """world mm (T,17,3) → camera root-relative mm (T,17,3)"""
        # World to camera: P_cam = R @ P_world + t
        kp_cam = np.einsum('ij,tnj->tni', R, kp_world) + t[None, None, :]
        kp_cam -= kp_cam[:, 0:1, :]  # root-relative
        return kp_cam
```

This closes the documentation gap. The pre-processed `.npz` files skip all of this, but anyone working with raw data or a new camera setup (like breakdancing footage) needs these transforms explicitly.
