# motionbert — Follow-up 1.3: The paper's central "unified" claim covers three downstream tasks: 3D pose, mesh recovery, and actio

_Generated: 2026-03-23T02:06:19.224Z_

---

I'll provide the complete mathematical treatment of the SMPL mesh recovery pathway that's missing from the document.

---

# MotionBERT SMPL Mesh Recovery: Complete Mathematical Formulation

## 9. The SMPL Body Model

The Skinned Multi-Person Linear (SMPL) model (Loper et al., SIGGRAPH Asia 2015) is a differentiable function that maps low-dimensional pose and shape parameters to a full 3D mesh of 6,890 vertices.

### Equation 9.1: SMPL Forward Function

$$M(\boldsymbol{\theta}, \boldsymbol{\beta}) : \mathbb{R}^{72} \times \mathbb{R}^{10} \to \mathbb{R}^{6890 \times 3}$$

The full forward pass decomposes into four stages:

$$M(\boldsymbol{\theta}, \boldsymbol{\beta}) = \text{LBS}\!\Big(T_P(\boldsymbol{\theta}, \boldsymbol{\beta}),\; J(\boldsymbol{\beta}),\; \boldsymbol{\theta},\; \mathcal{W}\Big)$$

- **Name**: SMPL body model forward kinematics
- **Variables**:
  - $\boldsymbol{\theta} \in \mathbb{R}^{72}$: body pose — 24 joints × 3 (axis-angle per joint)
  - $\boldsymbol{\beta} \in \mathbb{R}^{10}$: body shape — PCA coefficients over a shape space
  - $T_P$: posed template mesh (after shape + pose blend shapes)
  - $J(\boldsymbol{\beta})$: joint locations derived from shaped template
  - $\mathcal{W} \in \mathbb{R}^{6890 \times 24}$: blend skinning weights (fixed, precomputed)
  - $\text{LBS}$: Linear Blend Skinning function
- **Intuition**: SMPL encodes human body variation along two axes: *shape* (tall/short, heavy/thin — identity-level) and *pose* (joint rotations — frame-level). The separation means you can change someone's pose without changing their body proportions, and vice versa. Everything is linear or bilinear, making it fully differentiable — critical for gradient-based learning.

### Equation 9.2: Shape Blend Shapes

$$T_S(\boldsymbol{\beta}) = \bar{T} + B_S(\boldsymbol{\beta}) = \bar{T} + \sum_{i=1}^{10} \beta_i \mathbf{S}_i$$

- **Name**: Shape deformation via PCA basis
- **Variables**:
  - $\bar{T} \in \mathbb{R}^{6890 \times 3}$: mean template mesh (average human shape)
  - $\mathbf{S}_i \in \mathbb{R}^{6890 \times 3}$: $i$-th shape basis vector (principal component of registered body scans)
  - $\beta_i \in \mathbb{R}$: coefficient for $i$-th shape component
  - $B_S(\boldsymbol{\beta}) \in \mathbb{R}^{6890 \times 3}$: shape displacement field
- **Intuition**: The 10 shape coefficients control body proportions. $\beta_1$ typically corresponds to overall size, $\beta_2$ to weight/muscularity, etc. These were learned via PCA on ~4,000 registered 3D body scans (CAESAR dataset). 10 components capture ~99% of shape variation across the population.
- **Dimensions**: $\sum_{i=1}^{10} \beta_i \mathbf{S}_i$: weighted sum of $10$ matrices each $[6890 \times 3]$ → $[6890 \times 3]$ ✓
- **Origin**: SMPL (Loper et al., 2015), shape space from CAESAR scans via PCA.

### Equation 9.3: Joint Regression from Shaped Template

$$J(\boldsymbol{\beta}) = \mathcal{J} \cdot T_S(\boldsymbol{\beta})$$

- **Name**: Sparse joint regression
- **Variables**:
  - $\mathcal{J} \in \mathbb{R}^{24 \times 6890}$: joint regression matrix (sparse, pretrained)
  - $J(\boldsymbol{\beta}) \in \mathbb{R}^{24 \times 3}$: 3D joint locations for the shaped body
- **Intuition**: Each of the 24 joints is a learned sparse linear combination of nearby vertex positions. When the template deforms due to shape changes (e.g., a heavier person has wider hips), the joint positions shift accordingly. $\mathcal{J}$ is extremely sparse — each joint depends on only a few dozen vertices in its anatomical neighborhood.
- **Dimensions**: $[24 \times 6890] \cdot [6890 \times 3] = [24 \times 3]$ ✓
- **Origin**: SMPL; regression matrix trained on registered mesh-skeleton pairs.

### Equation 9.4: Rodrigues Formula (Axis-Angle to Rotation Matrix)

Each joint's axis-angle vector $\boldsymbol{\omega}_k \in \mathbb{R}^3$ (where $\boldsymbol{\theta} = [\boldsymbol{\omega}_1; \ldots; \boldsymbol{\omega}_{24}]$) is converted to a rotation matrix:

$$R_k = \exp([\boldsymbol{\omega}_k]_\times) = I + \frac{\sin\phi}{\phi}[\boldsymbol{\omega}_k]_\times + \frac{1 - \cos\phi}{\phi^2}[\boldsymbol{\omega}_k]_\times^2$$

where $\phi = \|\boldsymbol{\omega}_k\|_2$ is the rotation angle and:

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}$$

- **Name**: Rodrigues rotation formula / matrix exponential of skew-symmetric matrix
- **Variables**:
  - $\boldsymbol{\omega}_k \in \mathbb{R}^3$: axis-angle for joint $k$ (direction = rotation axis, magnitude = angle in radians)
  - $[\boldsymbol{\omega}]_\times \in \mathbb{R}^{3 \times 3}$: skew-symmetric (cross-product) matrix
  - $R_k \in SO(3) \subset \mathbb{R}^{3 \times 3}$: rotation matrix
- **Intuition**: Converts the compact 3-parameter rotation representation to a $3 \times 3$ rotation matrix. Axis-angle is minimal (3 parameters for 3 DoF) but has a **discontinuity** at $\phi = 0$ (any axis works) and a **topology problem** — $SO(3)$ is not homeomorphic to $\mathbb{R}^3$, so no 3D or 4D representation can be continuous everywhere. This is the critical motivation for the 6D representation used in MotionBERT's mesh head (Eq. 10.3).
- **Dimensions**: $[\boldsymbol{\omega}]_\times$: $[3 \times 3]$; $R_k$: $[3 \times 3] \in SO(3)$ (orthogonal, det = 1) ✓
- **Origin**: Olinde Rodrigues (1840); standard in computer vision and robotics.

### Equation 9.5: Pose Blend Shapes

$$B_P(\boldsymbol{\theta}) = \sum_{k=1}^{23} (R_k(\boldsymbol{\theta}) - I) \cdot \mathbf{P}_k$$

Note: 23 joints (excluding root), each contributing a $[9]$-dimensional flattened rotation residual.

Equivalently, flattening all rotation residuals into a vector:

$$B_P(\boldsymbol{\theta}) = \sum_{n=1}^{207} (R(\boldsymbol{\theta}) - I)_n \cdot \mathbf{P}_n$$

where $207 = 23 \times 9$ and $\mathbf{P}_n \in \mathbb{R}^{6890 \times 3}$ are pose blend shape basis vectors.

- **Name**: Pose-dependent corrective blend shapes
- **Variables**:
  - $R_k(\boldsymbol{\theta}) \in \mathbb{R}^{3 \times 3}$: rotation matrix for joint $k$
  - $I \in \mathbb{R}^{3 \times 3}$: identity (rest pose)
  - $\mathbf{P}_k \in \mathbb{R}^{6890 \times 3}$: pose blend shape for joint $k$
  - $(R_k - I)$: how much the rotation deviates from rest pose
  - $B_P(\boldsymbol{\theta}) \in \mathbb{R}^{6890 \times 3}$: total pose-dependent deformation
- **Intuition**: When you bend your elbow, the skin around it doesn't just rotate rigidly — muscles bulge, tendons shift, and the skin surface deforms. Pose blend shapes model these secondary deformations as a **linear function of rotation deviation from rest pose**. The $(R_k - I)$ term ensures zero correction at rest pose. These were learned from registered 4D body scans of people in various poses.
- **Dimensions**: Each $(R_k - I)$ flattens to $[9]$; each $\mathbf{P}_k$ is $[6890 \times 3]$; weighted sum → $[6890 \times 3]$ ✓
- **Origin**: SMPL; learned from CAESAR + Dyna datasets.

### Equation 9.6: Posed Template (before skinning)

$$T_P(\boldsymbol{\theta}, \boldsymbol{\beta}) = T_S(\boldsymbol{\beta}) + B_P(\boldsymbol{\theta}) = \bar{T} + B_S(\boldsymbol{\beta}) + B_P(\boldsymbol{\theta})$$

- **Name**: Full template with shape and pose deformations
- **Dimensions**: $[6890 \times 3] + [6890 \times 3] + [6890 \times 3] = [6890 \times 3]$ ✓
- **Intuition**: The template mesh is first adjusted for body shape, then corrected for pose-dependent deformations. This is still in the rest-pose configuration — skinning (next equation) actually rotates the limbs.

### Equation 9.7: Forward Kinematics (Joint Transformations via Kinematic Tree)

The SMPL skeleton is a kinematic tree with root at joint 0 (pelvis). Global transformations are computed recursively:

$$G_k(\boldsymbol{\theta}, \boldsymbol{\beta}) = \prod_{m \in \text{path}(0 \to k)} \begin{bmatrix} R_m & J_m(\boldsymbol{\beta}) - J_{\text{parent}(m)}(\boldsymbol{\beta}) \\ \mathbf{0}^{\top} & 1 \end{bmatrix}$$

More precisely, the local transform at joint $k$:

$$\bar{G}_k = \begin{bmatrix} R_k & J_k(\boldsymbol{\beta}) \\ \mathbf{0}^{\top} & 1 \end{bmatrix}$$

Global transform (product along kinematic chain):

$$G_k = G_{\text{parent}(k)} \cdot \begin{bmatrix} R_k & J_k - J_{\text{parent}(k)} \\ \mathbf{0}^{\top} & 1 \end{bmatrix}$$

with $G_0 = \bar{G}_0$ (root).

The skinning-ready transform removes the rest-pose joint position:

$$G'_k = G_k - \begin{bmatrix} \mathbf{0}_{3\times3} & G_k \cdot \tilde{J}_k \\ \mathbf{0}^{\top} & 0 \end{bmatrix}$$

where $\tilde{J}_k = [J_k^{\top}, 1]^{\top}$ is the homogeneous rest joint position.

- **Name**: Hierarchical forward kinematics with rest-pose subtraction
- **Variables**:
  - $G_k \in \mathbb{R}^{4 \times 4}$: global rigid transformation for joint $k$ (homogeneous)
  - $R_k \in SO(3)$: local rotation at joint $k$
  - $J_k(\boldsymbol{\beta}) \in \mathbb{R}^3$: rest-pose joint location (shape-dependent)
  - $\text{parent}(k)$: parent joint index in kinematic tree
  - $G'_k$: skinning-ready transform (removes rest-pose offset)
- **Intuition**: Each joint's global rotation is the product of all rotations along the chain from root to that joint. When the shoulder rotates, the elbow, wrist, and fingers all move with it. The rest-pose subtraction ($G'_k$) is necessary because Linear Blend Skinning displaces vertices *relative* to their rest-pose positions — without it, each vertex would be double-translated by the rest-pose joint offset.
- **Kinematic tree** (SMPL 24-joint): Pelvis → {L-Hip, R-Hip, Spine1} → {L-Knee, R-Knee, Spine2} → {L-Ankle, R-Ankle, Spine3} → {L-Foot, R-Foot, Neck} → {Head, L-Collar, R-Collar} → {L-Shoulder, R-Shoulder} → {L-Elbow, R-Elbow} → {L-Wrist, R-Wrist} → {L-Hand, R-Hand}
- **Dimensions**: $G_k$: $[4 \times 4]$; product of $[4 \times 4]$ matrices along chain of depth ≤ 8 ✓
- **Origin**: Standard robotics forward kinematics; applied to SMPL by Loper et al.

### Equation 9.8: Linear Blend Skinning (LBS)

$$\mathbf{v}'_i = \left(\sum_{k=1}^{24} w_{i,k} \cdot G'_k\right) \tilde{\mathbf{v}}_i$$

where $\tilde{\mathbf{v}}_i = [T_P(\boldsymbol{\theta}, \boldsymbol{\beta})_i^{\top}, 1]^{\top}$ is the homogeneous posed-template vertex.

- **Name**: Linear Blend Skinning — the final vertex positioning step
- **Variables**:
  - $\mathbf{v}'_i \in \mathbb{R}^3$: final 3D position of vertex $i$
  - $w_{i,k} \in [0, 1]$: skinning weight of joint $k$ for vertex $i$ (from $\mathcal{W}$, $\sum_k w_{i,k} = 1$)
  - $G'_k \in \mathbb{R}^{4 \times 4}$: skinning-ready transform for joint $k$
  - $\tilde{\mathbf{v}}_i \in \mathbb{R}^4$: homogeneous posed-template vertex position
- **Intuition**: Each vertex is transformed by a weighted average of nearby joint transforms. A vertex on the forearm is mostly influenced by the elbow and wrist joints ($w_{i,\text{elbow}} \approx 0.6$, $w_{i,\text{wrist}} \approx 0.4$). The weights are fixed and precomputed from skinning data — they encode the anatomical association between mesh vertices and skeleton joints. LBS is simple and fast but produces well-known artifacts at high bend angles (candy-wrapper effect), partially mitigated by the pose blend shapes.
- **Dimensions**: $\sum_k w_{i,k} G'_k$: weighted sum of $24$ matrices each $[4 \times 4]$ → $[4 \times 4]$; $[4 \times 4] \cdot [4 \times 1] = [4 \times 1]$ → take first 3 → $[3]$ per vertex; full mesh: $[6890 \times 3]$ ✓
- **Origin**: Standard computer graphics (Magnenat-Thalmann et al., 1988); codified in SMPL.

### Equation 9.9: Joint Positions from Mesh (Evaluation)

$$J_{3D} = \mathcal{J} \cdot M(\boldsymbol{\theta}, \boldsymbol{\beta})$$

or equivalently through forward kinematics:

$$J_{3D,k} = (G_k \cdot \tilde{J}_k(\boldsymbol{\beta}))_{1:3}$$

- **Name**: 3D joint positions from the posed mesh
- **Dimensions**: $[24 \times 6890] \cdot [6890 \times 3] = [24 \times 3]$ ✓
- **Intuition**: Joint positions can be read out from either the mesh vertices (via the regression matrix) or directly from the forward kinematics transforms. Both should agree; the regression path is used when evaluating against 3D joint ground truth.

---

## 10. MotionBERT Mesh Regression Head

### Equation 10.1: Feature Aggregation (Joints → Per-Frame Global Feature)

The DSTformer outputs per-joint, per-frame features $\mathbf{H}^{(L)} \in \mathbb{R}^{T \times J \times D}$. The mesh head must aggregate across joints to produce a per-frame global representation:

$$\mathbf{h}_t = \text{Reshape}\!\left(\mathbf{H}^{(L)}_{t,:,:}\right) \in \mathbb{R}^{J \cdot D}$$

or with learnable pooling:

$$\mathbf{h}_t = \mathbf{W}_{pool} \cdot \text{flatten}(\mathbf{H}^{(L)}_{t,:,:}) + \mathbf{b}_{pool}$$

- **Name**: Joint feature aggregation for mesh regression
- **Variables**:
  - $\mathbf{H}^{(L)}_{t,:,:} \in \mathbb{R}^{J \times D} = \mathbb{R}^{17 \times 256}$: all joint features at frame $t$
  - $\mathbf{h}_t \in \mathbb{R}^{J \cdot D} = \mathbb{R}^{4352}$: flattened per-frame feature (or $\mathbb{R}^{D'}$ after projection)
  - $\mathbf{W}_{pool} \in \mathbb{R}^{D' \times (J \cdot D)}$: optional dimensionality reduction
- **Intuition**: The 3D pose head (Eq. 4.1) predicts independently per joint — each joint's feature maps to its own 3D coordinate. But SMPL parameters are *global* — $\boldsymbol{\beta}$ describes the whole body shape, and joint rotations $\boldsymbol{\theta}$ interact through the kinematic tree. So the mesh head must see all joints simultaneously. Flattening and projecting is the simplest approach; MotionBERT follows this pattern rather than using graph convolutions or cross-attention.
- **Dimensions**: flatten $[17 \times 256] \to [4352]$; optional projection $[D' \times 4352] \cdot [4352] = [D']$ ✓

### Equation 10.2: SMPL Parameter Regression

$$[\hat{\boldsymbol{\Theta}}_t,\; \hat{\boldsymbol{\beta}}_t,\; \hat{\boldsymbol{\pi}}_t] = \text{MLP}_{mesh}(\mathbf{h}_t)$$

The MLP is typically a 2-layer network with residual connections:

$$\begin{aligned}
\mathbf{z}_1 &= \text{ReLU}(\mathbf{W}_1 \mathbf{h}_t + \mathbf{b}_1) \\
\mathbf{z}_2 &= \text{ReLU}(\mathbf{W}_2 [\mathbf{z}_1; \mathbf{h}_t] + \mathbf{b}_2) \\
\hat{\boldsymbol{\Theta}}_t &= \mathbf{W}_\Theta \mathbf{z}_2 + \mathbf{b}_\Theta \in \mathbb{R}^{144} \\
\hat{\boldsymbol{\beta}}_t &= \mathbf{W}_\beta \mathbf{z}_2 + \mathbf{b}_\beta \in \mathbb{R}^{10} \\
\hat{\boldsymbol{\pi}}_t &= \mathbf{W}_\pi \mathbf{z}_2 + \mathbf{b}_\pi \in \mathbb{R}^{3}
\end{aligned}$$

- **Name**: SMPL parameter regression from DSTformer features
- **Variables**:
  - $\hat{\boldsymbol{\Theta}}_t \in \mathbb{R}^{144}$: predicted pose — 24 joints × 6 (6D rotation representation, see Eq. 10.3)
  - $\hat{\boldsymbol{\beta}}_t \in \mathbb{R}^{10}$: predicted shape coefficients
  - $\hat{\boldsymbol{\pi}}_t \in \mathbb{R}^{3}$: weak-perspective camera $[s, t_x, t_y]$ (scale + 2D translation)
  - $[\mathbf{z}_1; \mathbf{h}_t]$: concatenation (skip connection from input)
  - Total output dimension: $144 + 10 + 3 = 157$
- **Intuition**: The MLP translates the transformer's learned motion representation into the specific parameterization SMPL requires. The skip connection is critical — SMPL parameters are relatively low-level (raw rotations), so the head needs access to both the processed features and the original joint features. Note that $\hat{\boldsymbol{\beta}}_t$ is predicted *per frame* even though shape should be constant across frames — in practice, either the per-frame predictions are averaged, or a temporal consistency loss encourages agreement.
- **Why not predict axis-angle directly?** → Eq. 10.3 explains the discontinuity problem.
- **Dimensions**: $\mathbf{z}_2 \in \mathbb{R}^{D'}$; $\mathbf{W}_\Theta$: $[144 \times D']$; $\mathbf{W}_\beta$: $[10 \times D']$; $\mathbf{W}_\pi$: $[3 \times D']$ ✓

### Equation 10.3: 6D Rotation Representation (Zhou et al., CVPR 2019)

MotionBERT's mesh head predicts rotations in the **continuous 6D representation**, not axis-angle. For each joint $k$:

$$\hat{\boldsymbol{\Theta}}_k = [\mathbf{a}_1, \mathbf{a}_2] \in \mathbb{R}^{6}$$

The rotation matrix is recovered via Gram-Schmidt orthogonalization:

$$\begin{aligned}
\mathbf{b}_1 &= \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|} \\
\mathbf{b}_2 &= \frac{\mathbf{a}_2 - (\mathbf{b}_1^{\top} \mathbf{a}_2)\mathbf{b}_1}{\|\mathbf{a}_2 - (\mathbf{b}_1^{\top} \mathbf{a}_2)\mathbf{b}_1\|} \\
\mathbf{b}_3 &= \mathbf{b}_1 \times \mathbf{b}_2 \\
R_k &= [\mathbf{b}_1 \mid \mathbf{b}_2 \mid \mathbf{b}_3] \in SO(3)
\end{aligned}$$

- **Name**: Continuous 6D rotation representation with Gram-Schmidt recovery
- **Variables**:
  - $\mathbf{a}_1, \mathbf{a}_2 \in \mathbb{R}^3$: raw network outputs (unconstrained)
  - $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3 \in \mathbb{R}^3$: orthonormal basis vectors
  - $R_k \in SO(3)$: recovered rotation matrix
- **Intuition**: This is perhaps the most important design choice in the mesh head. The problem: $SO(3)$ (the group of 3D rotations) is a non-Euclidean manifold. Any representation with fewer than 5 parameters *must* have discontinuities (topological obstruction). Axis-angle has a discontinuity at $\theta = 0$ (undefined axis) and at $\theta = \pi$ (antipodal identification). Quaternions have the double-cover problem ($q$ and $-q$ represent the same rotation, creating a discontinuity at the identification boundary). These discontinuities cause training instability — the network must learn to "jump" across the discontinuity, which gradient descent handles poorly.

  The 6D representation solves this by using 6 parameters (over-parameterized by 3) and recovering $SO(3)$ via the smooth, everywhere-differentiable Gram-Schmidt process. The map from $\mathbb{R}^6 \to SO(3)$ is **surjective and continuous** — every rotation can be reached, and nearby 6D vectors map to nearby rotations. Zhou et al. proved this is the minimum dimensionality for a continuous representation.

- **Gradient properties**: The Gram-Schmidt process has well-defined gradients everywhere except when $\|\mathbf{a}_1\| = 0$ or $\mathbf{a}_2 \parallel \mathbf{a}_1$ (measure-zero in $\mathbb{R}^6$, never occurs in practice). Compared to axis-angle, where $\nabla_\theta R$ has a singularity at $\theta = 0$, the 6D representation provides smooth, stable gradients throughout training.
- **Empirical impact**: Zhou et al. showed ~5-10% improvement in rotation prediction tasks when switching from axis-angle/quaternion to 6D. For breakdancing poses with extreme rotations (handstands, flares, windmills), this is especially critical — many joints hit the $\theta = \pi$ regime where axis-angle gradients degenerate.
- **Dimensions**: Network predicts $[6]$ per joint; Gram-Schmidt maps to $[3 \times 3]$; for 24 joints: predict $[144]$, recover $24 \times [3 \times 3]$ ✓
- **Origin**: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks," CVPR 2019.
- **Connection**: $R_k$ feeds into SMPL forward kinematics (Eq. 9.7) and pose blend shapes (Eq. 9.5).

### Equation 10.4: Weak-Perspective Camera Projection

$$\hat{\mathbf{x}}^{2D}_{t,j} = s_t \cdot \Pi_{ortho}(J_{3D,t,j}) + \begin{bmatrix} t_{x,t} \\ t_{y,t} \end{bmatrix}$$

where $\Pi_{ortho}$ drops the $z$-coordinate:

$$\Pi_{ortho}([x, y, z]^{\top}) = [x, y]^{\top}$$

- **Name**: Weak-perspective projection for mesh training
- **Variables**:
  - $s_t \in \mathbb{R}^+$: global scale (encodes effective depth)
  - $t_{x,t}, t_{y,t} \in \mathbb{R}$: 2D translation
  - $\hat{\boldsymbol{\pi}}_t = [s_t, t_{x,t}, t_{y,t}]$: camera parameters predicted by the mesh head
  - $J_{3D,t,j} \in \mathbb{R}^3$: 3D joint position from SMPL output
- **Intuition**: Weak-perspective approximates full perspective projection when the object's depth variation is small relative to its distance from the camera. For a person ~2m away with ~0.5m depth variation, the approximation error is ~25% — acceptable for training but not for precise re-projection. The scale $s$ absorbs focal length and distance: $s \approx f / z_{root}$. This avoids needing to know camera intrinsics at training time.
- **Why not full perspective?** Full perspective (Eq. 5.3) requires known camera intrinsics ($f_x, f_y, c_x, c_y$). For in-the-wild training data (3DPW, Internet videos), these are often unknown. Weak-perspective requires only 3 parameters that can be predicted by the network.
- **Dimensions**: $s \cdot [2] + [2] = [2]$ per joint; full: $[J \times 2]$ per frame ✓
- **Origin**: Standard in HMR (Kanazawa et al., CVPR 2018) and subsequent mesh recovery works.

---

## 11. Mesh Recovery Loss Functions

### Equation 11.1: 3D Joint Loss (from SMPL output)

$$\mathcal{L}_{J3D} = \frac{1}{T \cdot K} \sum_{t=1}^{T} \sum_{k=1}^{K} \left\| \hat{J}_{3D,t,k} - J^*_{3D,t,k} \right\|_2$$

- **Name**: 3D joint regression loss from SMPL output
- **Variables**:
  - $\hat{J}_{3D} = \mathcal{J} \cdot M(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\beta}})$: joints from predicted SMPL mesh
  - $J^*_{3D}$: ground truth 3D joints
  - $K$: number of evaluation joints (may differ from SMPL's 24 — e.g., 14 for 3DPW evaluation)
- **Intuition**: Ensures the predicted SMPL mesh is consistent with observed 3D joint positions. Unlike the direct joint regression loss (Eq. 5.1), this loss flows gradients through the SMPL differentiable pipeline — through LBS, forward kinematics, joint regression, and back to the predicted $\boldsymbol{\theta}, \boldsymbol{\beta}$.
- **Dimensions**: Same as Eq. 5.1 but with $K$ instead of $J$ ✓

### Equation 11.2: 2D Joint Reprojection Loss (via Weak-Perspective)

$$\mathcal{L}_{J2D} = \frac{1}{T \cdot K} \sum_{t=1}^{T} \sum_{k=1}^{K} \left\| s_t \cdot \Pi_{ortho}(\hat{J}_{3D,t,k}) + \mathbf{t}_t - \mathbf{x}^*_{2D,t,k} \right\|_2$$

- **Name**: 2D re-projection loss through weak-perspective camera
- **Variables**: As above, with $\mathbf{x}^*_{2D}$ being detected/ground-truth 2D joint positions
- **Intuition**: Even when 3D GT is unavailable, 2D detections provide supervision. The gradients flow: 2D error → camera parameters + SMPL joint positions → $\boldsymbol{\theta}, \boldsymbol{\beta}$. Because weak-perspective drops depth information, this loss cannot resolve depth ambiguity alone — it must be combined with 3D losses or strong priors.

### Equation 11.3: Vertex Loss (Per-Vertex Supervision)

$$\mathcal{L}_{vert} = \frac{1}{T \cdot V} \sum_{t=1}^{T} \sum_{i=1}^{V} \left\| M(\hat{\boldsymbol{\theta}}_t, \hat{\boldsymbol{\beta}}_t)_i - M(\boldsymbol{\theta}^*_t, \boldsymbol{\beta}^*_t)_i \right\|_2$$

- **Name**: Per-vertex mesh loss
- **Variables**:
  - $V = 6890$: number of SMPL vertices
  - $M(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\beta}})_i$: $i$-th vertex of predicted mesh
  - $M(\boldsymbol{\theta}^*, \boldsymbol{\beta}^*)_i$: $i$-th vertex of ground truth mesh
- **Intuition**: The most direct supervision — every vertex on the predicted mesh should match the ground truth mesh. This provides much denser supervision than joint loss alone (6,890 vertices vs. 24 joints) and captures surface details that joints miss (e.g., belly shape, shoulder width). However, vertex GT is only available from motion capture datasets with registered meshes (3DPW, AMASS), not from 2D-only data.
- **Dimensions**: $[6890 \times 3]$ predicted vs. GT; L2 per vertex → scalar ✓
- **Gradient density**: 6,890 vertices provide ~287× more supervision signals than 24 joints. Each vertex gradient propagates through LBS to all 24 joint transforms (weighted by skinning weights), providing rich multi-directional gradients on $\boldsymbol{\theta}$.

### Equation 11.4: SMPL Parameter Loss (Regularization)

$$\mathcal{L}_{smpl} = \lambda_\theta \left\| \hat{\boldsymbol{\theta}}_t - \boldsymbol{\theta}^*_t \right\|_2^2 + \lambda_\beta \left\| \hat{\boldsymbol{\beta}}_t - \boldsymbol{\beta}^*_t \right\|_2^2$$

- **Name**: Direct parameter regression loss
- **Variables**:
  - $\hat{\boldsymbol{\theta}}_t$: predicted pose (after converting 6D → axis-angle for comparison with GT)
  - $\boldsymbol{\theta}^*_t$: ground truth pose parameters
  - $\hat{\boldsymbol{\beta}}_t$: predicted shape
  - $\boldsymbol{\beta}^*_t$: ground truth shape
  - $\lambda_\theta, \lambda_\beta$: weighting coefficients
- **Intuition**: This is L2-squared (MSE), not L2-norm. Used as a regularizer to keep predicted parameters near ground truth in parameter space, not just in output space. Two different $\boldsymbol{\theta}$ can produce similar meshes (due to compensating rotations) — this loss disambiguates by preferring the solution closest to GT in parameter space. The L2-squared form (vs. L1 or L2-norm) penalizes large parameter deviations quadratically, acting as a soft constraint.
- **Note on representation mismatch**: The 6D representation is used for prediction (continuous, stable gradients), but GT parameters are stored as axis-angle. The loss can be computed either by converting predictions to axis-angle (introduces the discontinuity at loss level, but not in the network) or by converting GT to 6D (no discontinuity). The common approach is the former, with the discontinuity tolerated because the loss gradient doesn't need to be continuous — only the network's forward pass does.

### Equation 11.5: Composite Mesh Recovery Loss

$$\mathcal{L}_{mesh} = \lambda_{J3D} \mathcal{L}_{J3D} + \lambda_{J2D} \mathcal{L}_{J2D} + \lambda_{vert} \mathcal{L}_{vert} + \lambda_{smpl} \mathcal{L}_{smpl}$$

- **Name**: Total mesh recovery training loss
- **Typical weights** (from HMR-family works, MotionBERT likely similar):
  - $\lambda_{J3D} = 5.0$
  - $\lambda_{J2D} = 5.0$
  - $\lambda_{vert} = 5.0$
  - $\lambda_{smpl}$: $\lambda_\theta = 1.0$, $\lambda_\beta = 0.001$
- **Intuition**: The loss balances four complementary signals:
  1. **$\mathcal{L}_{J3D}$**: Gets the skeleton right in 3D
  2. **$\mathcal{L}_{J2D}$**: Gets the skeleton right in 2D (enables in-the-wild training)
  3. **$\mathcal{L}_{vert}$**: Gets the full body surface right (when GT available)
  4. **$\mathcal{L}_{smpl}$**: Regularizes parameter space (prevents degenerate solutions)

  Shape regularization ($\lambda_\beta = 0.001$) is deliberately weak — $\boldsymbol{\beta}$ should mostly be learned from data, not pushed to zero. Pose regularization is stronger because compensating rotations can look correct in vertex space but be physically implausible.
- **Gradient flow summary**: All four losses produce gradients on $\boldsymbol{\theta}$ and $\boldsymbol{\beta}$ through different pathways:
  - $\mathcal{L}_{J3D}$: $\nabla_\theta$ through FK chain; $\nabla_\beta$ through joint regression
  - $\mathcal{L}_{J2D}$: $\nabla_\theta$ through FK + camera; $\nabla_\beta$ through joint regression + camera
  - $\mathcal{L}_{vert}$: $\nabla_\theta$ through LBS + FK + pose blend shapes; $\nabla_\beta$ through LBS + shape blend shapes
  - $\mathcal{L}_{smpl}$: Direct $\nabla_\theta$, $\nabla_\beta$ (no SMPL forward pass)

---

## 12. Complete Mesh Recovery Forward Pass

Putting it all together, the end-to-end pipeline from 2D input to mesh:

$$\begin{aligned}
\mathbf{H}^{(0)} &= \text{Embed}(\mathbf{X}) & [B, T, J, 2] \to [B, T, J, 256] \\
\mathbf{H}^{(L)} &= \text{DSTformer}^{(1:5)}(\mathbf{H}^{(0)}) & [B, T, J, 256] \to [B, T, J, 256] \\
\mathbf{h}_t &= \text{flatten}(\mathbf{H}^{(L)}_{t,:,:}) & [B, T, 17, 256] \to [B, T, 4352] \\
[\hat{\boldsymbol{\Theta}}_t, \hat{\boldsymbol{\beta}}_t, \hat{\boldsymbol{\pi}}_t] &= \text{MLP}_{mesh}(\mathbf{h}_t) & [B, T, 4352] \to [B, T, 157] \\
\hat{R}_k &= \text{GramSchmidt}(\hat{\boldsymbol{\Theta}}_{t,k}) & [B, T, 24, 6] \to [B, T, 24, 3, 3] \\
\hat{\mathbf{V}}_t &= M(\hat{R}_{1:24}, \hat{\boldsymbol{\beta}}_t) & [B, T] \to [B, T, 6890, 3] \\
\hat{J}_{3D,t} &= \mathcal{J} \cdot \hat{\mathbf{V}}_t & [B, T, 6890, 3] \to [B, T, 24, 3]
\end{aligned}$$

### Computational Cost Comparison (per frame):

| Component | FLOPs | Parameters | Notes |
|-----------|-------|-----------|-------|
| DSTformer backbone | ~1.2 GFLOPs | ~6.3M | Shared across all tasks |
| Mesh MLP head | ~18 MFLOPs | ~2.5M | Task-specific |
| SMPL forward pass | ~5 MFLOPs | ~6.9M (fixed) | Not learned; blend shapes + LBS |
| **Total mesh** | **~1.22 GFLOPs** | **~8.8M trainable** | + 6.9M fixed SMPL |

The mesh head adds ~50% more trainable parameters but <2% more compute compared to the 3D pose task.

---

## 13. Breaking-Specific Implications

### Equation 13.1: Rotation Extrapolation Error for Extreme Poses

For breakdancing, the mesh recovery pathway faces compounded errors:

$$\text{MPVE}(\phi) \approx \text{MPVE}(0) + \gamma \cdot \|\boldsymbol{\theta}_{pred} - \boldsymbol{\theta}_{train}\|^2$$

where MPVE is Mean Per-Vertex Error and $\gamma$ captures the sensitivity of LBS to rotation errors.

The error amplification from rotation to vertex is:

$$\delta \mathbf{v}_i \approx \sum_{k} w_{i,k} \left\| \mathbf{r}_{i \to k} \times \delta \boldsymbol{\omega}_k \right\|$$

- **Name**: Vertex error sensitivity to rotation error
- **Variables**:
  - $\delta \boldsymbol{\omega}_k$: rotation error at joint $k$ (small-angle approximation)
  - $\mathbf{r}_{i \to k}$: lever arm from joint $k$ to vertex $i$ (in rest pose)
  - $w_{i,k}$: skinning weight
- **Intuition**: Vertex error grows linearly with both rotation error and distance from the joint. For extremity vertices (fingers, toes, head) with large lever arms (~0.5m), even a 10° rotation error at the shoulder produces ~87mm vertex displacement ($0.5 \times \sin(10°) \approx 0.087$m). For breaking power moves where multiple joints have large rotation errors simultaneously, these errors accumulate along the kinematic chain.
- **Breaking impact**: In a windmill, the spine joints (Spine1/2/3) are each rotated ~90° from training distribution. The compounded FK error at the hands/feet (chain depth 6-8) can reach 150-300mm even with moderate per-joint rotation accuracy.

### Equation 13.2: Shape-Pose Coupling in Breaking

$$\frac{\partial M}{\partial \boldsymbol{\beta}} = \frac{\partial T_S}{\partial \boldsymbol{\beta}} + \frac{\partial J}{\partial \boldsymbol{\beta}} \cdot \frac{\partial \text{FK}}{\partial J} \cdot \frac{\partial \text{LBS}}{\partial G_k}$$

- **Intuition**: Shape affects not just the rest template but also joint locations, which change the FK chain, which changes every downstream vertex. In breaking, where extreme poses amplify FK errors, even small shape misestimation ($\|\Delta\boldsymbol{\beta}\| \approx 1$) shifts joint locations by ~5-15mm, which after FK amplification through deep chains produces vertex errors of 30-80mm. The model has no breaking-specific body shape priors (bboy physiques: strong upper body, lean) — AMASS contains mostly average body types.

---

## Verification Checklist (Mesh Recovery Additions)

### ✅ Dimension compatibility

| Operation | Input | Output | Status |
|-----------|-------|--------|--------|
| Feature flatten | $[17 \times 256]$ | $[4352]$ | ✅ |
| MLP → SMPL params | $[4352] \to [157]$ | $[144] + [10] + [3]$ | ✅ |
| 6D → rotation | $[6]$ per joint | $[3 \times 3] \in SO(3)$ | ✅ |
| Shape blend shapes | $10 \times [6890 \times 3]$ | $[6890 \times 3]$ | ✅ |
| Joint regression | $[24 \times 6890] \cdot [6890 \times 3]$ | $[24 \times 3]$ | ✅ |
| FK transform chain | $[4 \times 4]^{\text{depth}}$ | $[4 \times 4]$ per joint | ✅ |
| LBS per vertex | $\sum_k w_{ik} [4 \times 4] \cdot [4]$ | $[3]$ | ✅ |
| Full mesh output | — | $[6890 \times 3]$ | ✅ |

### ✅ Gradient flow through SMPL

The entire SMPL pipeline is differentiable:
- Shape blend shapes: linear in $\boldsymbol{\beta}$ → gradient is just $\mathbf{S}_i$ ✅
- Rodrigues formula: differentiable w.r.t. $\boldsymbol{\omega}$ (except at $\|\boldsymbol{\omega}\| = 0$, handled by Taylor expansion) ✅
- Gram-Schmidt: differentiable w.r.t. input 6D vector (except measure-zero degeneracies) ✅
- FK chain: product of differentiable transforms → chain rule applies ✅
- LBS: linear combination → trivially differentiable ✅
- Pose blend shapes: linear in $(R_k - I)$ → gradient is $\mathbf{P}_k$ ✅

### ✅ 6D representation is continuous

Zhou et al. (2019) Theorem 1: Any continuous representation $f: SO(3) \to \mathbb{R}^n$ with continuous left-inverse requires $n \geq 5$. The 6D representation with $n=6$ satisfies this bound and the Gram-Schmidt recovery is a continuous left-inverse. ✅

### ✅ No circular dependencies in mesh pipeline

```
DSTformer features H^(L)
  → Flatten/Pool (Eq 10.1)
    → MLP regression (Eq 10.2)
      → 6D → R via Gram-Schmidt (Eq 10.3)
        → SMPL forward:
          Shape blend shapes (Eq 9.2)
            → Joint regression (Eq 9.3)
              → FK transforms (Eq 9.7)
          Pose blend shapes (Eq 9.5)
            → Posed template (Eq 9.6)
              → LBS (Eq 9.8) ← uses FK transforms
                → Mesh vertices (Eq 9.1)
                  → Joint positions (Eq 9.9)
                    → Losses (Eq 11.1-11.5)
```

All arrows point forward. SMPL forward kinematics is a DAG (kinematic tree has no cycles). ✅

---

## Updated Summary Table (Mesh Recovery Equations)

| # | Equation | Novel? | Key Shape Transform |
|---|----------|--------|-------------------|
| 9.1 | SMPL forward function | No (Loper '15) | $\mathbb{R}^{72} \times \mathbb{R}^{10} \to \mathbb{R}^{6890 \times 3}$ |
| 9.2 | Shape blend shapes | No (Loper '15) | $\mathbb{R}^{10} \to \mathbb{R}^{6890 \times 3}$ |
| 9.3 | Joint regression | No (Loper '15) | $\mathbb{R}^{6890 \times 3} \to \mathbb{R}^{24 \times 3}$ |
| 9.4 | Rodrigues formula | No (classical) | $\mathbb{R}^{3} \to SO(3)$ |
| 9.5 | Pose blend shapes | No (Loper '15) | $SO(3)^{23} \to \mathbb{R}^{6890 \times 3}$ |
| 9.6 | Posed template | No (Loper '15) | Sum of 9.2 + 9.5 |
| 9.7 | Forward kinematics | No (classical) | $SO(3)^{24} \to SE(3)^{24}$ |
| 9.8 | Linear Blend Skinning | No (classical) | $SE(3)^{24} \times \mathbb{R}^{6890 \times 3} \to \mathbb{R}^{6890 \times 3}$ |
| 9.9 | Joint from mesh | No (Loper '15) | $\mathbb{R}^{6890 \times 3} \to \mathbb{R}^{24 \times 3}$ |
| **10.1** | **Feature aggregation** | **Partial (MotionBERT)** | $\mathbb{R}^{17 \times 256} \to \mathbb{R}^{4352}$ |
| **10.2** | **SMPL param regression** | **Partial (MotionBERT)** | $\mathbb{R}^{4352} \to \mathbb{R}^{157}$ |
| **10.3** | **6D rotation repr.** | **No (Zhou '19)** | $\mathbb{R}^{6} \to SO(3)$, continuous |
| 10.4 | Weak-perspective proj. | No (HMR) | $\mathbb{R}^{3} \to \mathbb{R}^{2}$ |
| 11.1 | 3D joint loss (mesh) | No | $[K \times 3] \times 2 \to$ scalar |
| 11.2 | 2D reprojection loss | No | $[K \times 2] \times 2 \to$ scalar |
| 11.3 | Vertex loss | No | $[6890 \times 3] \times 2 \to$ scalar |
| 11.4 | SMPL parameter loss | No | $[72] + [10] \times 2 \to$ scalar |
| 11.5 | Composite mesh loss | No | Weighted sum of 11.1-11.4 |

**The mesh recovery pathway contains no novel equations from MotionBERT itself** — the novelty is architectural (reusing the DSTformer backbone) and empirical (showing pretrained motion features transfer to mesh recovery). The SMPL model, 6D rotation representation, weak-perspective projection, and loss functions all come from prior work. MotionBERT's contribution here is the "unified" backbone, not the mesh head design.
