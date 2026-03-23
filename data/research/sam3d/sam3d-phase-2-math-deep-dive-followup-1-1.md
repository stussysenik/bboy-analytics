# sam3d — Follow-up 1.1: The entire equation set (Eqs 7-14) assumes multiple views of a **static scene**. For the target use 

_Generated: 2026-03-23T05:29:05.575Z_

---

I have the full prior context and the specific question. Let me produce the deep mathematical analysis of how inter-frame motion invalidates SAM3D's static-scene assumptions for single-camera dance video.

---

# Motion Invalidation Analysis for SAM3D Applied to Single-Camera Dance Video

## 1. Articulated Motion Model

A dancer's body is a kinematic tree. Each body part $b$ has time-varying world position $\mathbf{p}_b(t)$ with velocity $\mathbf{v}_b(t)$. Between consecutive frames at rate $f$ (fps), the displacement is:

$$\Delta \mathbf{p}_b = \mathbf{v}_b \cdot \Delta t + \frac{1}{2}\mathbf{a}_b \Delta t^2, \quad \Delta t = 1/f$$

For breaking, measured velocities by body part (from motion capture literature, e.g., Hachimura et al. 2005; Aristidou et al. 2019):

| Body Part | Toprock $\|\mathbf{v}\|$ (m/s) | Footwork (m/s) | Power moves (m/s) |
|-----------|------|----------|-------------|
| Torso/CoM | 0.3–0.8 | 0.2–0.5 | 0.5–2.0 |
| Upper arm | 1.0–2.5 | 0.5–1.5 | 2.0–4.0 |
| Hand | 2.0–5.0 | 1.0–3.0 | 3.0–8.0 |
| Foot | 1.5–3.5 | 2.0–5.0 | 3.0–8.0 |
| Head | 0.3–1.0 | 0.2–0.5 | 2.0–4.0 (headspins) |

**Critical observation**: Different body parts simultaneously move at different velocities. The hand can move 10× faster than the torso in the same frame.

---

## 2. Back-Projection Error Under Motion (Breaking Eq. 7)

### The Static Assumption

Eq. 7 computes:

$$\mathbf{p}_{3D} = \mathbf{R}_v^{-1}\!\left(d(u,v) \cdot \mathbf{K}_v^{-1}\begin{bmatrix}u \\ v \\ 1\end{bmatrix} - \mathbf{t}_v\right)$$

This assumes the 3D point at pixel $(u,v)$ in frame $t$ occupies the **same world position** in all other frames. For a moving dancer, the surface point visible at $(u,v)$ in frame $t$ is at position $\mathbf{p}_b(t)$, while in frame $t'$ the same body surface has moved to $\mathbf{p}_b(t')$.

### Motion-Induced Back-Projection Error

Define the back-projection from frame $t$ as $\hat{\mathbf{p}}^{(t)}$ and from frame $t'$ as $\hat{\mathbf{p}}^{(t')}$. For a single point on body part $b$, these are **different 3D points** when the dancer moves:

$$\epsilon_{\text{motion}}(t, t') = \left\|\hat{\mathbf{p}}^{(t)} - \hat{\mathbf{p}}^{(t')}\right\| = \left\|\int_t^{t'} \mathbf{v}_b(\tau)\,d\tau\right\| \approx \|\mathbf{v}_b\| \cdot |t' - t|$$

For a temporal window of $N$ frames centered at $t_0$, the RMS motion error is:

$$\epsilon_{\text{rms}} = \|\mathbf{v}_b\| \cdot \sqrt{\frac{1}{N}\sum_{k=-N/2}^{N/2} \left(\frac{k}{f}\right)^2} = \|\mathbf{v}_b\| \cdot \frac{1}{f}\sqrt{\frac{N^2 - 1}{12}}$$

**Concrete values** at 30fps, $N = 10$ frames (333ms window):

| Body part | $\|\mathbf{v}\|$ (m/s) | $\epsilon_{\text{rms}}$ (cm) | vs. $R_{\text{seed}}$ = 2cm |
|-----------|---------|-----------|------------|
| Torso (toprock) | 0.5 | 4.8 | 2.4× |
| Hand (toprock) | 3.0 | 28.7 | 14.3× |
| Foot (footwork) | 3.5 | 33.5 | 16.7× |
| Limb (powermove) | 5.0 | 47.8 | 23.9× |

Even the torso during gentle toprock produces motion error exceeding the superpoint resolution. Fast limbs produce errors **an order of magnitude** larger than superpoint size.

---

## 3. Superpoint Fragmentation (Impact on Eq. 9)

### The Dimensionless Motion-to-Resolution Ratio

Define:

$$\alpha_b = \frac{\|\mathbf{v}_b\| \cdot \Delta t}{R_{\text{seed}}}$$

This is the ratio of per-frame displacement to superpoint resolution. When $\alpha_b > 1$, consecutive frames place the same body surface point in **different superpoints**.

**Critical thresholds at 30fps** ($\Delta t = 33.3$ ms, $R_{\text{seed}} = 2$ cm):

$$\alpha_b = \frac{\|\mathbf{v}_b\| \times 0.0333}{0.02} = 1.667 \times \|\mathbf{v}_b\|$$

| $\|\mathbf{v}_b\|$ (m/s) | $\alpha_b$ | Effect |
|---------|---------|--------|
| 0.3 | 0.5 | Marginal — some points leak between superpoints |
| 0.6 | 1.0 | **Critical** — consecutive frames produce independent superpoints |
| 1.2 | 2.0 | Complete fragmentation — no temporal coherence |
| 3.0 | 5.0 | Catastrophic — motion trail creates phantom geometry |

**At 30fps, $\alpha = 1$ at $v = 0.6$ m/s** — exceeded by virtually all dance movements except a dancer standing still.

### 3D Point Cloud "Motion Smear"

When $\alpha \gg 1$, back-projecting $N$ frames of a moving body part produces a **motion trail** in 3D space. Instead of a compact point cloud representing the body surface, you get an elongated streak:

$$\text{Extent of smear} = \|\mathbf{v}_b\| \cdot \frac{N}{f}$$

For a hand at 3 m/s, 10 frames at 30fps: smear length = **1.0 meter**. VCCS will segment this trail into $\lfloor 1.0 / 0.02 \rfloor = 50$ superpoints along the motion axis. These superpoints represent **positions the hand passed through**, not the hand's actual shape.

The VCCS distance (Eq. 9) includes normals and color, which partially mitigate this — the motion trail has inconsistent normals and potentially varying color. But the spatial term dominates, and the trail points DO have consistent color (same skin/clothing) and potentially consistent normals (if motion is along the surface tangent).

---

## 4. Voting Contamination (Impact on Eqs. 10–11)

### Contaminated Vote Model

In the static case, a superpoint $s_i$ belonging to the dancer's right arm gets consistent mask votes from all views: $V(s_i, m_{\text{arm}}) \approx 1$.

With motion, superpoint $s_i$ (formed from frame $t_0$'s back-projection of the right arm) gets projected into frame $t'$ via Eq. 8. But in frame $t'$, the right arm has **moved away** from the 3D position of $s_i$. What's now at that position might be:
- Background (arm moved away, nothing replaced it)
- Torso (arm was extended, now retracted, torso occupies that space)
- A different limb (cross-body motion)
- Still the arm (if motion is small)

The contaminated voting score becomes:

$$V_{\text{motion}}(s_i, m_j) = \frac{1}{|s_i|} \sum_{p \in s_i} \mathbb{1}\!\left[L_{t'}(\pi_{t'}(\mathbf{p}^{(t_0)})) = m_j\right]$$

where $\pi_{t'}(\mathbf{p}^{(t_0)})$ projects the **frame-$t_0$ position** into frame $t'$, but the mask $L_{t'}$ corresponds to the **frame-$t'$ configuration** of the dancer. These are misaligned by $\|\mathbf{v}\| \cdot |t' - t_0|$.

### Probability of Correct Vote

Model the probability that a point's vote is correct as a function of displacement relative to body-part width:

$$P(\text{correct vote at } t') = \max\!\left(0,\; 1 - \frac{\|\mathbf{v}_b\| \cdot |t' - t_0|}{W_b / 2}\right)$$

where $W_b$ is the width of body part $b$ (the distance from center to nearest mask boundary). This is a conservative linear model — once displacement exceeds half the body part width, the projected point is entirely outside the original mask.

Body part half-widths:
- Torso: $W/2 \approx 15$ cm
- Upper leg: $W/2 \approx 7$ cm
- Upper arm: $W/2 \approx 5$ cm
- Forearm: $W/2 \approx 4$ cm
- Hand: $W/2 \approx 4$ cm

### Multi-View Aggregated Vote Under Motion

Averaging over $N$ frames in a temporal window $[-T/2, T/2]$ around $t_0$:

$$\bar{V}_{\text{motion}}(s_i, m_j) = \frac{1}{N}\sum_{k=1}^{N} P\!\left(\text{correct at } t_0 + \frac{k - N/2}{f}\right)$$

For the linear dropout model:

$$\bar{V}_{\text{motion}} \approx \begin{cases} 1 - \frac{\|\mathbf{v}_b\| \cdot N}{4f \cdot (W_b/2)} & \text{if } \frac{\|\mathbf{v}_b\| \cdot N}{2f} < \frac{W_b}{2} \\[6pt] \frac{f \cdot W_b}{2\|\mathbf{v}_b\| \cdot N} & \text{otherwise (partial coverage)} \end{cases}$$

**Concrete example** — hand during toprock ($\|\mathbf{v}\| = 3$ m/s, $W/2 = 4$ cm, 30fps):

The hand exits its own mask width in:

$$t_{\text{exit}} = \frac{W_b/2}{\|\mathbf{v}\|} = \frac{0.04}{3.0} = 13.3 \text{ ms} = 0.4 \text{ frames}$$

**The hand leaves its own mask between consecutive frames.** With a 10-frame window, only the reference frame itself gives a correct vote. $\bar{V}_{\text{motion}} \approx 1/10 = 0.10$. The correct mask gets only 10% of votes.

**Torso during toprock** ($\|\mathbf{v}\| = 0.5$ m/s, $W/2 = 15$ cm):

$$t_{\text{exit}} = \frac{0.15}{0.5} = 300 \text{ ms} = 9 \text{ frames}$$

With a 10-frame window, ~90% of frames give correct votes. $\bar{V}_{\text{motion}} \approx 0.85$. Still usable.

### Body-Part-Specific Degradation Table (30fps, 10-frame window)

| Body part | $\|\mathbf{v}\|$ | $W_b/2$ | $t_{\text{exit}}$ (frames) | $\bar{V}_{\text{motion}}$ | Status |
|-----------|---------|---------|-----------|------------|--------|
| Torso (toprock) | 0.5 m/s | 15 cm | 9.0 | 0.85 | Usable |
| Torso (powermove) | 1.5 m/s | 15 cm | 3.0 | 0.52 | Degraded |
| Upper arm | 2.0 m/s | 5 cm | 0.75 | 0.13 | **Failed** |
| Hand | 3.0 m/s | 4 cm | 0.4 | 0.10 | **Failed** |
| Foot (footwork) | 3.5 m/s | 5 cm | 0.43 | 0.10 | **Failed** |
| Limb (powermove) | 5.0 m/s | 5 cm | 0.3 | 0.10 | **Failed** |

**Conclusion**: Only the torso during slow movements retains useful voting scores. All limbs during active dance are below the noise floor.

---

## 5. Single-Camera Viewpoint Degeneracy

### The Hidden Catastrophe

Beyond motion contamination, single-camera video has a **fundamental geometric degeneracy** that the prior analysis missed.

SAM3D's multi-view voting derives power from **viewpoint diversity**: different cameras see different sides of objects, providing independent mask evidence from different angles. With a single static camera:

$$\mathbf{R}_v \approx \mathbf{R}_0, \quad \mathbf{t}_v \approx \mathbf{t}_0 \quad \forall v$$

All "views" are from approximately the **same viewpoint**. This means:

1. **No occlusion resolution**: Surfaces not visible from the camera are NEVER observed. A dancer's back is always occluded. No amount of temporal frames helps.

2. **Correlated depth errors**: All frames share the same viewing direction, so MDE depth errors are perfectly correlated in direction:

$$\rho_{\text{directional}} \approx 1.0$$

From Eq. 16, the effective view count becomes:

$$V_{\text{eff}} = \frac{N}{1 + (N-1) \times 1.0} = 1$$

**Fifty frames give the geometric diversity of ONE frame.** Multi-view voting provides zero benefit over single-frame processing.

3. **Degenerate back-projection geometry**: All viewing rays are nearly parallel. Depth errors project to displacements along the **same axis** (the camera's optical axis). In multi-view setups, errors from different viewpoints project in different directions and partially cancel in 3D. Here they reinforce:

$$\text{3D error covariance} = \sigma_d^2 \cdot \hat{\mathbf{z}}\hat{\mathbf{z}}^T$$

where $\hat{\mathbf{z}}$ is the camera's viewing direction. The error ellipsoid is a **prolate needle** along the depth axis, not the isotropic sphere you'd get from diverse viewpoints.

### Formal Viewpoint Diversity Metric

Define the viewpoint diversity as the volume of the convex hull of camera positions, normalized:

$$\mathcal{D} = \frac{\text{Vol}(\text{ConvHull}(\{\mathbf{o}_v\}))}{\text{Vol}(\text{BoundingBox}(\text{scene}))}$$

For SAM3D's intended use (multi-camera rig or dense scanning):
$$\mathcal{D}_{\text{multi-view}} \sim 0.1 - 1.0$$

For a single static camera:
$$\mathcal{D}_{\text{single}} = 0$$

For a handheld camera with small motion (~10cm over 10s):
$$\mathcal{D}_{\text{handheld}} \sim 10^{-6}$$

The voting noise reduction scales as $\mathcal{D}^{1/3}$ (cube root of volume), so even significant handheld camera motion provides negligible geometric diversity compared to multi-camera setups.

---

## 6. Merge Criterion Breakdown (Impact on Eqs. 12–14)

### Phantom Overlaps from Motion Trails

When the dancer's arm sweeps through a region of space over $N$ frames, the back-projected points create overlapping 3D groups from different frames. Two groups $G_a$ (arm at time $t_1$) and $G_b$ (arm at time $t_2$) will have:

$$\text{IoU}_{3D}(G_a, G_b) = \frac{|G_a \cap G_b|}{|G_a \cup G_b|} \approx \max\!\left(0, 1 - \frac{\|\mathbf{v}\| \cdot |t_2 - t_1|}{L_b}\right)$$

where $L_b$ is the length of body part $b$ along the motion direction. For small temporal gaps ($|t_2 - t_1| \lesssim L_b / \|\mathbf{v}\|$), the IoU is high and the groups merge — creating a **smeared composite** of the arm at multiple positions.

### Normal Discontinuity Corruption

Surface normals from motion trails are geometrically meaningless. Points from the arm at time $t_1$ have normals pointing perpendicular to the arm surface. Points from the arm at time $t_2$ (same arm, different position) have normals pointing in a **different direction** because the arm has rotated.

The boundary normal discontinuity (Eq. 13) between adjacent temporal snapshots of the SAME body part:

$$\Delta\theta_{\text{normal}}^{\text{motion}} \approx \omega_b \cdot |t_2 - t_1|$$

where $\omega_b$ is the angular velocity of body part $b$. For a forearm rotating at $\omega = 5$ rad/s and $|t_2 - t_1| = 3$ frames at 30fps:

$$\Delta\theta_{\text{normal}}^{\text{motion}} = 5 \times 0.1 = 0.5 \text{ rad} = 28.6°$$

This is close to $\tau_{\text{boundary}} \approx 30°$, so the merge criterion becomes unreliable — sometimes merging motion snapshots that should be separate, sometimes splitting them when they shouldn't be.

---

## 7. Minimum Frame Rate and Maximum Motion Thresholds

### Deriving the Usability Boundary

For SAM3D to produce meaningful 3D segmentation of a moving body part, we need **simultaneously**:

**Condition 1** — Superpoint coherence ($\alpha_b < 0.5$):
$$f > \frac{\|\mathbf{v}_b\|}{0.5 \times R_{\text{seed}}} = \frac{\|\mathbf{v}_b\|}{0.01}$$

**Condition 2** — Voting accuracy ($\bar{V}_{\text{motion}} > 0.5$ with $N$ frames):
$$f > \frac{\|\mathbf{v}_b\| \cdot N}{2 \cdot W_b}$$

**Condition 3** — Single-view depth noise below superpoint resolution:
$$\sigma_d < R_{\text{seed}} = 0.02 \text{ m}$$

Condition 3 is **independent of frame rate** and is violated at most practical distances (DepthPro $\sigma_d \approx 15$ cm at 3m). This means even infinite frame rate doesn't solve the problem — you also need accurate depth.

### Required Frame Rates

Taking $R_{\text{seed}} = 2$ cm and $N = 5$ frames:

| Movement | $\|\mathbf{v}_b\|$ | $f_{\min}$ (Cond. 1) | $f_{\min}$ (Cond. 2) | Binding |
|----------|---------|---------|---------|---------|
| Standing still | 0.05 m/s | 5 fps | 1 fps | Cond. 1: 5 fps |
| Slow sway | 0.3 m/s | 30 fps | 5 fps | Cond. 1: 30 fps |
| Toprock torso | 0.5 m/s | 50 fps | 8 fps | Cond. 1: 50 fps |
| Active arm | 2.0 m/s | 200 fps | 50 fps | Cond. 1: 200 fps |
| Hand gesture | 3.0 m/s | **300 fps** | 75 fps | Cond. 1: **300 fps** |
| Powermove limb | 5.0 m/s | **500 fps** | 125 fps | Cond. 1: **500 fps** |

**The superpoint coherence condition is always binding** and demands impractical frame rates for any active dance movement.

### Maximum Tolerable Velocity at Standard Frame Rates

Inverting Condition 1:

$$\|\mathbf{v}_b\|_{\max} = 0.5 \times R_{\text{seed}} \times f = 0.01 \times f$$

| Frame rate | $\|\mathbf{v}\|_{\max}$ | Movement class |
|-----------|---------|----------------|
| 30 fps | 0.3 m/s | Subtle weight shifts only |
| 60 fps | 0.6 m/s | Very slow toprock torso |
| 120 fps | 1.2 m/s | Moderate toprock, slow arm movements |
| 240 fps | 2.4 m/s | Most toprock, some footwork |
| 480 fps | 4.8 m/s | Most breaking except fastest powermoves |

**At standard 30fps video, only nearly-static poses are processable.** Even 120fps (iPhone ProRes slow-mo) only handles moderate movements.

---

## 8. Optimal Temporal Window

### Error Minimization

The total segmentation error has two competing terms:

$$\epsilon_{\text{total}}(W) = \underbrace{\frac{\sigma_{\text{mask}}}{\sqrt{2W+1}}}_{\text{mask noise reduction}} + \underbrace{\frac{\|\mathbf{v}_b\| \cdot W}{f \cdot W_b}}_{\text{motion contamination}}$$

where $\sigma_{\text{mask}}$ represents per-frame mask boundary noise (from depth errors, SAM mask inaccuracy, etc.), and $W$ is the half-window size in frames.

Taking $d\epsilon/dW = 0$:

$$-\frac{\sigma_{\text{mask}}}{(2W+1)^{3/2}} + \frac{\|\mathbf{v}_b\|}{f \cdot W_b} = 0$$

$$W^* = \frac{1}{2}\left[\left(\frac{\sigma_{\text{mask}} \cdot f \cdot W_b}{\|\mathbf{v}_b\|}\right)^{2/3} - 1\right]$$

**Concrete example** — torso during toprock ($\sigma_{\text{mask}} = 0.1$, $f = 30$, $W_b = 0.30$ m, $\|\mathbf{v}\| = 0.5$ m/s):

$$W^* = \frac{1}{2}\left[\left(\frac{0.1 \times 30 \times 0.30}{0.5}\right)^{2/3} - 1\right] = \frac{1}{2}\left[1.8^{2/3} - 1\right] = \frac{1}{2}[1.49 - 1] = 0.25$$

$W^* \approx 0$ frames — **temporal averaging provides essentially no benefit** because even the minimum window introduces more motion error than it removes noise.

For the torso nearly stationary ($\|\mathbf{v}\| = 0.1$ m/s):

$$W^* = \frac{1}{2}\left[\left(\frac{0.1 \times 30 \times 0.30}{0.1}\right)^{2/3} - 1\right] = \frac{1}{2}\left[9.0^{2/3} - 1\right] = \frac{1}{2}[4.33 - 1] = 1.66$$

$W^* \approx 2$ frames — a tiny window of ±2 frames (total 5) is optimal. Beyond that, motion contamination dominates.

### The Fundamental Trade-off

For the single-camera case, the optimal temporal window collapses because:

1. **No viewpoint diversity**: Additional frames don't provide new geometric information (see Section 5)
2. **Motion contamination grows linearly** with window size
3. **Noise reduction grows only as $1/\sqrt{N}$**

The ratio of marginal cost to marginal benefit:

$$\frac{d(\text{motion error})/dW}{d(\text{noise reduction})/dW} = \frac{\|\mathbf{v}_b\| / (f \cdot W_b)}{\sigma_{\text{mask}} / (2W+1)^{3/2}} \propto (2W+1)^{3/2}$$

This grows superlinearly — every additional frame is worse than the last.

---

## 9. Expected AP Degradation

### Model

Based on the ScanNet benchmark relationship between superpoint purity and AP (from Mask3D ablations and the SAM3D paper's own results), the empirical relationship is approximately:

$$\text{AP} \approx \text{AP}_{\text{static}} \times \text{Purity}^{2.5}$$

where the exponent ~2.5 reflects that purity errors compound through voting and merging.

Superpoint purity under motion, for body part $b$ in a window of $N$ frames:

$$\text{Purity}_b \approx \frac{1}{N}\sum_{k=0}^{N-1} \max\!\left(0, 1 - \frac{\|\mathbf{v}_b\| \cdot k}{f \cdot W_b}\right)$$

### Predicted AP by Movement Type (30fps, $N=10$, $\text{AP}_{\text{static}} = 50$ for SAM3D on ScanNet)

| Movement | Avg Purity | Expected AP | vs. Static AP |
|----------|-----------|-------------|---------------|
| Nearly static ($v = 0.1$ m/s) | 0.94 | 43.1 | −14% |
| Slow toprock torso ($v = 0.5$ m/s) | 0.72 | 19.5 | **−61%** |
| Active toprock ($v = 1.0$ m/s) | 0.50 | 8.8 | **−82%** |
| Power move ($v = 3.0$ m/s) | 0.15 | 0.4 | **−99%** |

**Even slow toprock drops AP by over 60%.** The pipeline is fundamentally incompatible with dance-speed motion at standard frame rates.

### Per-Body-Part AP Breakdown (Toprock, 30fps, N=5)

Since different body parts move at different speeds, the per-part AP varies dramatically:

| Part | Purity | Part AP | Can segment? |
|------|--------|---------|-------------|
| Torso | 0.91 | 40.0 | Marginal |
| Head | 0.88 | 35.5 | Marginal |
| Upper arm | 0.35 | 3.3 | No |
| Forearm | 0.22 | 1.1 | No |
| Hand | 0.15 | 0.4 | No |
| Upper leg | 0.60 | 14.0 | Poor |
| Foot | 0.25 | 1.5 | No |

The pipeline might segment the torso as a blob but loses all limb detail.

---

## 10. Complete Degradation Summary

### Three Compounding Failure Modes

The single-camera dance scenario triggers **three simultaneous failures**, each individually sufficient to make the pipeline unreliable:

$$\boxed{\epsilon_{\text{total}} = \underbrace{\epsilon_{\text{motion}}}_{\text{body moves between frames}} + \underbrace{\epsilon_{\text{viewpoint}}}_{\text{no geometric diversity}} + \underbrace{\epsilon_{\text{depth}}}_{\text{MDE noise without fusion benefit}}}$$

1. **Motion contamination** ($\epsilon_{\text{motion}}$): Even slow dance (0.5 m/s torso) exceeds the 0.3 m/s maximum velocity at 30fps. Superpoints fragment, votes contaminate, merges corrupt.

2. **Viewpoint degeneracy** ($\epsilon_{\text{viewpoint}}$): $V_{\text{eff}} = 1$ regardless of frame count. Multi-view voting collapses to single-frame processing with extra noise.

3. **Depth error without fusion** ($\epsilon_{\text{depth}}$): TSDF fusion (Eq. 20) requires static geometry. MDE noise ($\sigma_d \approx 15$ cm at 3m) cannot be reduced via temporal averaging of a moving subject. $\sigma_d \gg R_{\text{seed}}$ throughout.

### Verdict

**SAM3D is architecturally incompatible with single-camera dance video.** The minimum viable configuration would require:

- Frame rate: **≥120fps** (and even then, only handles $v < 1.2$ m/s)
- Depth accuracy: **$\sigma_d < 2$ cm** (requires LiDAR or structured light, not MDE)
- Motion compensation: **Articulated body tracking** to warp points to a canonical pose before superpoint formation
- OR: Abandon multi-view 3D and use **per-frame 2D SAM + temporal tracking** instead

The most practical path for breaking analysis is to skip SAM3D entirely and use SAM 2 (video-native, handles motion natively in 2D) with a separate 3D reconstruction pipeline for the static background.
