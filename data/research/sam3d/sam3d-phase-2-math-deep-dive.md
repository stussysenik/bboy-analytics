# sam3d — Math Deep Dive

_Generated: 2026-03-23T05:24:04.397Z_

---

# SAM3D — Complete Equation Extraction & Verification

SAM3D is **training-free** — zero trainable parameters. This fundamentally changes which equation categories apply: there are **no loss functions**, **no pretraining objectives**, and attention is inherited from frozen SAM. The paper's equations are geometric (projection, voting, merging) rather than learned.

---

## 1. SAM ViT-H Image Encoder (Frozen)

### Eq. 1 — Patch Embedding

$$\mathbf{z}_0 = \text{PatchEmbed}(I_v) + \mathbf{E}_{pos}, \quad \mathbf{z}_0 \in \mathbb{R}^{N \times D}$$

- **Name**: Patch tokenization + positional encoding
- **Variables**:
  - $I_v \in \mathbb{R}^{3 \times 1024 \times 1024}$ — input image for view $v$ (RGB, SAM's fixed resolution)
  - $\text{PatchEmbed}$: Conv2d with kernel $16 \times 16$, stride 16 → $N = 64 \times 64 = 4096$ patches
  - $D = 1280$ — ViT-H embedding dimension
  - $\mathbf{E}_{pos} \in \mathbb{R}^{4096 \times 1280}$ — absolute 2D positional embeddings
- **Intuition**: Chops the image into 16×16 pixel patches, projects each to a 1280-dim vector, and adds position information so the model knows where each patch came from spatially.
- **Dimensions**: $(3, 1024, 1024) \xrightarrow{\text{Conv2d}} (1280, 64, 64) \xrightarrow{\text{reshape}} (4096, 1280)$ ✓
- **Origin**: Standard ViT (Dosovitskiy et al., 2020). Nothing novel here.
- **Connection**: $\mathbf{z}_0$ feeds into the 32-block transformer encoder.

### Eq. 2 — Windowed Self-Attention (28 of 32 blocks)

$$\text{Attn}_w(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + \mathbf{B}_{rel}\right)V$$

where for each window $w$ of size $14 \times 14 = 196$ tokens:

$$Q = \mathbf{z}W_Q, \quad K = \mathbf{z}W_K, \quad V = \mathbf{z}W_V$$

- **Name**: Windowed multi-head self-attention with relative position bias
- **Variables**:
  - $\mathbf{z} \in \mathbb{R}^{196 \times 1280}$ — tokens within one window
  - $W_Q, W_K, W_V \in \mathbb{R}^{1280 \times 1280}$ — projection matrices (split across 16 heads, so per-head $d_k = 80$)
  - $\mathbf{B}_{rel} \in \mathbb{R}^{196 \times 196}$ — relative position bias (learned, per head)
  - $d_k = 80$ — per-head dimension ($1280 / 16$ heads)
- **Intuition**: Each 14×14 window of patches attends only to other patches in the same window, reducing quadratic cost from $O(4096^2)$ to $O(196^2) \times \lceil 4096/196 \rceil$. The relative position bias encodes spatial relationships without absolute coordinates.
- **Dimensions**:
  - $Q, K \in \mathbb{R}^{196 \times 80}$ per head → $QK^T \in \mathbb{R}^{196 \times 196}$ ✓
  - $\text{softmax}(\cdot) \in \mathbb{R}^{196 \times 196}$, rows sum to 1 ✓
  - Output: $\mathbb{R}^{196 \times 80}$ per head → concatenate 16 heads → $\mathbb{R}^{196 \times 1280}$ ✓
- **Origin**: Window attention from Swin Transformer (Liu et al., 2021); relative position bias from Shaw et al. (2018). Standard in SAM.
- **Connection**: 28 windowed blocks process local features; 4 global blocks (every 6th) process full 4096-token sequence.

### Eq. 3 — Global Self-Attention (4 of 32 blocks)

$$\text{Attn}_g(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + \mathbf{B}_{rel}\right)V, \quad Q, K, V \in \mathbb{R}^{4096 \times 80}$$

- **Name**: Full global self-attention (blocks 6, 12, 18, 24 — every 6th)
- **Variables**: Same as Eq. 2 but over the entire $4096$-token sequence
- **Intuition**: Every 6 blocks, the model does full-sequence attention to propagate information across distant image regions. This is what lets SAM understand global scene structure.
- **Dimensions**: $QK^T \in \mathbb{R}^{4096 \times 4096}$ — this is the expensive operation (~123.5 GFLOPs vs ~82.6 GFLOPs for windowed) ✓
- **Origin**: Standard ViT attention. The hybrid windowed/global schedule is from SAM (Kirillov et al., 2023).
- **Connection**: Final block outputs $\mathbf{z}_{32} \in \mathbb{R}^{4096 \times 1280}$, which is reshaped and projected to the image embedding.

### Eq. 4 — Image Embedding Output

$$\mathbf{F}_v = \text{Neck}(\text{reshape}(\mathbf{z}_{32})) \in \mathbb{R}^{256 \times 64 \times 64}$$

- **Name**: Encoder output (image embedding)
- **Variables**:
  - $\mathbf{z}_{32} \in \mathbb{R}^{4096 \times 1280}$ → reshaped to $(1280, 64, 64)$
  - Neck: $1\times1$ conv ($1280 \to 256$) + LayerNorm + $3\times3$ conv ($256 \to 256$) + LayerNorm
  - $\mathbf{F}_v$ — the per-view image embedding
- **Intuition**: Compresses the 1280-dim tokens down to 256-dim spatial feature maps. This is the representation SAM's mask decoder conditions on.
- **Dimensions**: $(1280, 64, 64) \xrightarrow{1\times1} (256, 64, 64) \xrightarrow{3\times3} (256, 64, 64)$ ✓
- **Origin**: Standard SAM neck architecture.
- **Connection**: $\mathbf{F}_v$ is the input to the mask decoder along with prompt embeddings.

### FLOPs Verification

$$\text{FLOPs}_{\text{encoder}} = 28 \times \underbrace{82.6\text{G}}_{\text{windowed}} + 4 \times \underbrace{123.5\text{G}}_{\text{global}} \approx 2{,}812.8\text{G} \approx \mathbf{2.8\text{ TFLOPs}}$$

Per windowed block: attention is $O(n_{windows} \times 196^2 \times d_k \times h)$ where $n_{windows} = \lceil 64/14 \rceil^2 \approx 25$. Per global block: $O(4096^2 \times d_k \times h)$. The architecture survey's correction from 370 GFLOPs to 2.8 TFLOPs is **verified** — the original survey missed the MLP FLOPs (which double the attention-only count) and likely confused per-head with total dimensions.

---

## 2. SAM Mask Decoder

### Eq. 5 — Grid Prompt Encoding

$$\mathbf{P} = \{(u_i, v_i)\}_{i=1}^{4096}, \quad u_i = 16i_x + 8, \quad v_i = 16i_y + 8$$

where $i_x, i_y \in \{0, 1, \ldots, 63\}$

- **Name**: Uniform grid point prompts (SAM3D's "automatic" mode)
- **Variables**:
  - $(u_i, v_i)$ — pixel coordinates of each prompt point
  - $16$ — stride matching patch size; $+8$ centers each prompt in its patch
- **Intuition**: SAM3D uses SAM in "segment everything" mode — a 64×64 grid of point prompts, one per image patch. Each prompt asks "what object is at this location?" The mask decoder generates one or more candidate masks per prompt.
- **Dimensions**: 4096 points → each encoded to $\mathbb{R}^{256}$ by the prompt encoder via positional encoding + learned foreground/background token ✓
- **Origin**: SAM's automatic mask generation mode (Kirillov et al., 2023). SAM3D doesn't innovate here.
- **Connection**: Prompt embeddings + image embedding → mask decoder → candidate masks.

### Eq. 6 — Mask Prediction & Filtering

$$\hat{M}_v^k = \sigma(\mathbf{F}_v^T \cdot \mathbf{e}_k) \in \mathbb{R}^{H \times W}, \quad \text{IoU}_k = f_{\text{IoU}}(\mathbf{e}_k)$$

$$\mathcal{M}_v = \{\hat{M}_v^k \mid \text{IoU}_k > \tau_{\text{conf}}\}, \quad \text{then NMS with } \tau_{\text{NMS}}$$

- **Name**: Mask generation with confidence filtering and NMS
- **Variables**:
  - $\mathbf{F}_v \in \mathbb{R}^{256 \times 64 \times 64}$ — image embedding
  - $\mathbf{e}_k \in \mathbb{R}^{256}$ — per-mask output token from the decoder
  - $\sigma$ — sigmoid activation
  - $\tau_{\text{conf}} = 0.88$ — SAM's default `pred_iou_thresh` (the survey incorrectly stated 0.7)
  - $\tau_{\text{NMS}} = 0.7$ — SAM's `box_nms_thresh` (the survey confused this with the confidence threshold)
  - $K_v = |\mathcal{M}_v|$ — number of surviving masks for view $v$
- **Intuition**: Each output token is dot-producted with the spatial feature map to produce a mask logit map, passed through sigmoid to get per-pixel probabilities. Low-confidence masks are dropped, then overlapping masks are suppressed by NMS. This produces a clean set of non-overlapping 2D instance masks per view.
- **Dimensions**: $\mathbf{F}_v^T \cdot \mathbf{e}_k$: $(64 \times 64 \times 256) \cdot (256) \to (64 \times 64)$, upsampled to $(H \times W)$ ✓
- **Origin**: Standard SAM decoder (Kirillov et al., 2023).
- **Connection**: $\mathcal{M}_v$ (filtered mask set per view) feeds into Stage 2 back-projection.

---

## 3. 2D→3D Back-Projection (SAM3D's Core Geometric Equation)

### Eq. 7 — Pinhole Back-Projection

$$\mathbf{p}_{3D} = \mathbf{R}_v^{-1}\!\left(d(u,v) \cdot \mathbf{K}_v^{-1}\begin{bmatrix}u \\ v \\ 1\end{bmatrix} - \mathbf{t}_v\right)$$

- **Name**: 2D pixel → 3D world point transformation
- **Variables**:
  - $(u, v)$ — pixel coordinates, unitless integers
  - $d(u,v) \in \mathbb{R}^+$ — depth at pixel $(u,v)$ in meters
  - $\mathbf{K}_v \in \mathbb{R}^{3 \times 3}$ — camera intrinsic matrix for view $v$:
    $$\mathbf{K}_v = \begin{bmatrix}f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{bmatrix}$$
    where $f_x, f_y$ are focal lengths (pixels), $(c_x, c_y)$ is the principal point
  - $\mathbf{R}_v \in SO(3) \subset \mathbb{R}^{3 \times 3}$ — rotation matrix (world → camera), orthogonal so $\mathbf{R}_v^{-1} = \mathbf{R}_v^T$
  - $\mathbf{t}_v \in \mathbb{R}^3$ — translation vector (world → camera), in meters
  - $\mathbf{p}_{3D} \in \mathbb{R}^3$ — reconstructed 3D point in world coordinates, in meters
- **Intuition**: This reverses the camera projection. First, $\mathbf{K}_v^{-1}[u, v, 1]^T$ converts pixel coords to a normalized ray direction in camera space. Multiplying by depth $d$ places the point along that ray at the correct distance. Then $\mathbf{R}_v^{-1}(\cdot - \mathbf{t}_v)$ transforms from camera coordinates to world coordinates. **Depth is multiplicative** — any error in $d$ scales linearly into 3D position error.
- **Dimensions**:
  - $\mathbf{K}_v^{-1} \in \mathbb{R}^{3\times3}$, $[u,v,1]^T \in \mathbb{R}^3$ → product $\in \mathbb{R}^3$ ✓
  - Scalar $d$ times $\mathbb{R}^3$ → $\mathbb{R}^3$ ✓
  - Subtract $\mathbf{t}_v \in \mathbb{R}^3$ → $\mathbb{R}^3$ ✓
  - $\mathbf{R}_v^{-1} \in \mathbb{R}^{3\times3}$ times $\mathbb{R}^3$ → $\mathbf{p}_{3D} \in \mathbb{R}^3$ ✓
- **Origin**: Standard pinhole camera model (Hartley & Zisserman, *Multiple View Geometry*, 2003). Not novel.
- **Connection**: Each pixel with mask label $m_j$ and valid depth gets a 3D point $\mathbf{p}_{3D}$ labeled with $m_j$. The collection of labeled 3D points feeds into superpoint voting.

### Eq. 8 — Mask-to-Point Label Assignment

$$L_v(p) = \arg\max_{m_j \in \mathcal{M}_v} \hat{M}_v^j(\pi_v(p))$$

where $\pi_v(p)$ is the projection of 3D point $p$ onto view $v$:

$$\pi_v(p) = \mathbf{K}_v \left(\mathbf{R}_v \mathbf{p}_{3D} + \mathbf{t}_v\right) \bigg/ z$$

(dividing by the $z$-component for perspective normalization)

- **Name**: View-specific label assignment for a 3D point
- **Variables**:
  - $L_v(p)$ — the mask ID assigned to point $p$ from view $v$
  - $\pi_v: \mathbb{R}^3 \to \mathbb{R}^2$ — 3D-to-2D projection
  - $\hat{M}_v^j(u,v) \in [0,1]$ — mask $j$'s probability at pixel $(u,v)$
- **Intuition**: For each 3D point visible in view $v$, project it to the image plane and check which mask covers that pixel. The point inherits that mask's label. If a point falls in an overlap region, it takes the mask with the highest confidence.
- **Dimensions**: $\mathbf{R}_v(3\times3) \cdot \mathbf{p}_{3D}(3) + \mathbf{t}_v(3) \to \mathbb{R}^3 \xrightarrow{/z} \mathbb{R}^2$ ✓
- **Origin**: Standard multi-view label transfer. Not novel.
- **Connection**: Labels $L_v(p)$ for all views accumulate into the superpoint voting equation.

---

## 4. Superpoint Generation (VCCS)

### Eq. 9 — Voxel Cloud Connectivity Segmentation Distance

$$D_{\text{VCCS}}(p_i, p_j) = \sqrt{w_s \left\|\frac{\mathbf{x}_i - \mathbf{x}_j}{R_{\text{seed}}}\right\|^2 + w_c \left\|\frac{\mathbf{c}_i - \mathbf{c}_j}{m}\right\|^2 + w_n (1 - \mathbf{n}_i \cdot \mathbf{n}_j)}$$

- **Name**: VCCS supervoxel distance metric
- **Variables**:
  - $\mathbf{x}_i, \mathbf{x}_j \in \mathbb{R}^3$ — 3D point positions (meters)
  - $\mathbf{c}_i, \mathbf{c}_j \in \mathbb{R}^3$ — RGB color values $\in [0,1]$
  - $\mathbf{n}_i, \mathbf{n}_j \in \mathbb{R}^3$ — unit surface normals, $\|\mathbf{n}\| = 1$
  - $R_{\text{seed}}$ — seed resolution (~2cm for SAM3D), controls spatial scale
  - $m$ — color normalization factor
  - $w_s, w_c, w_n$ — relative weights for spatial, color, normal terms (typically $w_s = 1.0, w_c = 0.4, w_n = 0.9$)
- **Intuition**: Groups nearby 3D points into "superpoints" — coherent local clusters that share similar position, color, and surface orientation. The normal term $(1 - \mathbf{n}_i \cdot \mathbf{n}_j)$ is 0 when normals are parallel (same surface) and approaches 2 when opposed (different surfaces). This ensures superpoints don't straddle geometric boundaries.
- **Dimensions**: Each squared term is dimensionless (normalized by $R_{\text{seed}}$ and $m$), sum is dimensionless, output is a scalar distance ✓
- **Origin**: Papon et al., "Voxel Cloud Connectivity Segmentation" (CVPR 2013). Not novel to SAM3D.
- **Connection**: Produces superpoints $\{s_i\}$ — groups of 3D points that serve as the atomic units for voting and merging.

---

## 5. Multi-View Superpoint Voting (SAM3D's Key Contribution)

### Eq. 10 — Voting Score

$$V(s_i, m_j) = \frac{1}{|s_i|}\sum_{p \in s_i} \mathbb{1}[L_v(p) = m_j]$$

- **Name**: Superpoint-mask affinity via majority voting
- **Variables**:
  - $s_i$ — the $i$-th superpoint (a set of 3D points)
  - $|s_i|$ — number of points in superpoint $i$
  - $m_j$ — the $j$-th 2D mask (from any view)
  - $L_v(p)$ — the mask label assigned to point $p$ from view $v$ (Eq. 8)
  - $\mathbb{1}[\cdot]$ — indicator function (1 if true, 0 if false)
  - $V(s_i, m_j) \in [0, 1]$ — fraction of points in $s_i$ that were assigned mask $m_j$
- **Intuition**: For each superpoint, count what fraction of its points were labeled by each mask. If 80% of a superpoint's points were assigned mask $m_j$ from some view, then $V(s_i, m_j) = 0.8$. This creates a soft assignment between superpoints and masks. The key insight: **averaging over points within a superpoint** reduces noise from individual mis-projections, and **aggregating across views** reduces per-view errors — assuming views are independent (which breaks down with correlated depth errors).
- **Dimensions**: Scalar sum over a finite set, divided by set cardinality → scalar $\in [0,1]$ ✓
- **Origin**: **Novel to SAM3D.** The specific formulation of superpoint-level voting for lifting 2D masks to 3D is this paper's primary contribution. The concept of voting is standard, but applying it to SAM masks + geometric superpoints is new.
- **Connection**: Voting scores determine which mask each superpoint belongs to. Superpoints with the same winning mask are grouped into 3D instance candidates for merging.

### Eq. 11 — Multi-View Aggregated Voting

$$V_{\text{agg}}(s_i, m_j) = \frac{1}{|\mathcal{V}(s_i)|}\sum_{v \in \mathcal{V}(s_i)} V_v(s_i, m_j)$$

where $\mathcal{V}(s_i) = \{v \mid s_i \text{ is visible in view } v\}$

- **Name**: Cross-view vote aggregation
- **Variables**:
  - $\mathcal{V}(s_i)$ — set of views in which superpoint $s_i$ is visible
  - $V_v(s_i, m_j)$ — per-view voting score (Eq. 10 applied to a single view)
- **Intuition**: Averages the voting score across all views that can see a given superpoint. A superpoint visible from 30 views gets 30 independent "opinions" on what mask it belongs to. This is where multi-view consistency provides robustness — **in theory**. When depth errors are correlated across views (same MDE bias, adjacent frames), $V_{eff}$ drops dramatically (see Eq. 16).
- **Dimensions**: Average of scalars → scalar ✓
- **Origin**: Novel aggregation strategy in SAM3D.
- **Connection**: The mask with highest aggregated vote wins each superpoint. Superpoints sharing the same winning mask form initial 3D instance groups, which then undergo boundary-aware merging.

---

## 6. Boundary-Aware Merging

### Eq. 12 — 3D IoU Between Instance Groups

$$\text{IoU}_{3D}(G_a, G_b) = \frac{|G_a \cap G_b|}{|G_a \cup G_b|}$$

where groups are sets of superpoints, and intersection/union are computed over their constituent 3D points.

- **Name**: 3D Intersection-over-Union for superpoint groups
- **Variables**:
  - $G_a, G_b$ — two candidate 3D instance groups (sets of superpoints)
  - $|G_a \cap G_b|$ — number of shared 3D points (points that appear in both groups due to overlapping mask assignments from different views)
  - $|G_a \cup G_b|$ — total unique points across both groups
- **Intuition**: Measures how much two candidate instances overlap in 3D space. When different views produce slightly different masks for the same object, the back-projected 3D groups will partially overlap. High IoU means they're probably the same instance and should be merged.
- **Dimensions**: Ratio of cardinalities → scalar $\in [0,1]$ ✓
- **Origin**: Standard IoU metric (Jaccard, 1912). Application to 3D superpoint groups is SAM3D-specific.
- **Connection**: IoU exceeding $\tau_{\text{merge}}$ is a necessary condition for merging (Eq. 14).

### Eq. 13 — Normal Discontinuity at Group Boundary

$$\Delta\theta_{\text{normal}}(G_a, G_b) = \arccos\!\left(\frac{1}{|B_{ab}|}\sum_{(p_i, p_j) \in B_{ab}} \mathbf{n}_i \cdot \mathbf{n}_j\right)$$

where $B_{ab}$ is the set of neighboring point pairs along the boundary of $G_a$ and $G_b$.

- **Name**: Average surface normal angle change at the group boundary
- **Variables**:
  - $B_{ab}$ — boundary point pairs: $\{(p_i, p_j) \mid p_i \in G_a, p_j \in G_b, \|p_i - p_j\| < r_{\text{neighbor}}\}$
  - $\mathbf{n}_i, \mathbf{n}_j \in \mathbb{R}^3$ — unit surface normals, $\|\mathbf{n}\| = 1$
  - $\Delta\theta_{\text{normal}} \in [0°, 180°]$ — average angular discontinuity
- **Intuition**: A sharp change in surface normal at a boundary indicates a geometric edge (corner of a table, edge of a wall). Two groups separated by a sharp edge are likely different objects, even if their IoU is high. Smooth boundaries (same flat surface) suggest they should be merged. **This is the "boundary-aware" in the paper title.**
- **Dimensions**: $\mathbf{n}_i \cdot \mathbf{n}_j \in [-1, 1]$, average is scalar, $\arccos$ maps to $[0°, 180°]$ ✓
- **Origin**: Normal discontinuity is standard in point cloud segmentation (Rusu et al., 2009). SAM3D's specific integration with IoU-based merging is novel.
- **Connection**: Normal discontinuity below $\tau_{\text{boundary}}$ is the second condition for merging.

### Eq. 14 — Merge Decision

$$\text{merge}(G_a, G_b) = \begin{cases}\text{True} & \text{if } \text{IoU}_{3D}(G_a, G_b) > \tau_{\text{merge}} \;\wedge\; \Delta\theta_{\text{normal}}(G_a, G_b) < \tau_{\text{boundary}} \\ \text{False} & \text{otherwise}\end{cases}$$

- **Name**: Boundary-aware merge criterion
- **Variables**:
  - $\tau_{\text{merge}} \approx 0.5$ — IoU threshold for merging (plausible range 0.35–0.65)
  - $\tau_{\text{boundary}} \approx 30°$ — normal discontinuity threshold (plausible range 20°–40°)
- **Intuition**: Two candidate instance groups are merged if and only if they (a) overlap substantially in 3D space AND (b) there's no sharp geometric boundary between them. This prevents merging a coffee mug sitting on a table with the table itself, even if some views produced masks covering both. **Both conditions must hold** — high IoU alone isn't sufficient.
- **Dimensions**: Boolean output from two scalar comparisons ✓
- **Origin**: The joint IoU + normal criterion is **novel to SAM3D**. Either condition alone is standard.
- **Connection**: Iterative merging produces the final 3D instance segmentation. The merge process runs until no more pairs satisfy the criterion.

---

## 7. Evaluation Metrics

### Eq. 15a — 3D Instance Segmentation AP (Average Precision)

$$\text{AP}@\tau = \frac{1}{|\mathcal{C}|}\sum_{c \in \mathcal{C}} \text{AP}_c(\tau)$$

where for each class $c$:

$$\text{AP}_c(\tau) = \int_0^1 p_c(r) \, dr$$

with precision $p_c$ at recall $r$ computed over predicted instances sorted by confidence, and a predicted instance is a true positive if:

$$\text{IoU}_{3D}(\hat{G}, G^*) > \tau$$

- **Name**: Mean Average Precision at IoU threshold $\tau$
- **Variables**:
  - $\tau \in \{0.25, 0.50\}$ — IoU thresholds used in the paper
  - $\mathcal{C}$ — set of semantic classes (ScanNet has 18 classes)
  - $\hat{G}$ — predicted 3D instance, $G^*$ — ground truth instance
  - $p_c(r)$ — precision at recall level $r$ for class $c$
- **Intuition**: For each predicted 3D instance, check if it overlaps sufficiently with a ground truth instance. AP@25 is lenient (25% overlap counts), AP@50 is strict (50% overlap). Averaging over classes accounts for class imbalance.
- **Dimensions**: Integral of precision (dimensionless) over recall (dimensionless) → scalar $\in [0, 100]$ (reported as percentage) ✓
- **Origin**: Standard COCO-style AP adapted for 3D (Dai et al., ScanNet benchmark, 2017).
- **Connection**: Primary comparison metric against Mask3D and other baselines.

### Eq. 15b — Point-Level IoU (for Superpoint Purity)

$$\text{Purity}(s_i) = \max_{c} \frac{|\{p \in s_i \mid \text{gt}(p) = c\}|}{|s_i|}$$

- **Name**: Superpoint label purity
- **Variables**:
  - $s_i$ — a superpoint
  - $\text{gt}(p)$ — ground truth instance label for point $p$
  - $c$ — candidate instance label
- **Intuition**: What fraction of a superpoint's points belong to its dominant ground truth instance? Purity of 0.95 means 5% of points have wrong labels. Depth errors reduce purity because mis-projected points end up in wrong superpoints.
- **Dimensions**: Count ratio → scalar $\in [0, 1]$ ✓
- **Origin**: Standard clustering metric.
- **Connection**: Directly affects downstream AP — impure superpoints produce noisy instance boundaries.

---

## 8. Depth Error Propagation (Analysis Equations)

These equations are from the architecture survey's analysis, not the SAM3D paper itself, but they are critical for understanding SAM3D's limitations.

### Eq. 16 — Effective Independent Views Under Correlated Errors

$$V_{\text{eff}} = \frac{V}{1 + (V-1)\rho}$$

- **Name**: Effective view count with error correlation
- **Variables**:
  - $V$ — number of physical views
  - $\rho \in [0, 1]$ — pairwise depth error correlation coefficient
  - $V_{\text{eff}}$ — effective number of independent views for noise reduction
- **Intuition**: Multi-view averaging theoretically reduces noise by $\sqrt{V}$, but only if views are independent. When depth errors are correlated (same MDE systematically underestimates depth on textureless surfaces), the benefit collapses. With $\rho = 0.5$ and $V = 50$ views: $V_{\text{eff}} = 50 / (1 + 49 \times 0.5) \approx 1.96$. **50 views give the noise reduction of ~2 independent measurements.** This is devastating for SAM3D's multi-view voting strategy.
- **Dimensions**: Unitless ratio → unitless ✓
- **Origin**: Standard formula from statistics (effective sample size under equicorrelation). Not novel.
- **Connection**: Explains why SAM3D's multi-view voting doesn't compensate for depth estimation errors.

### Eq. 17 — Mask Assignment Error Rate

$$e_{\text{assign}} \approx 1 - \Phi\!\left(\frac{d_{\text{boundary}}}{\sigma_d}\right)$$

- **Name**: Probability of assigning a boundary point to the wrong mask
- **Variables**:
  - $d_{\text{boundary}}$ — distance from a point to the nearest mask boundary (meters)
  - $\sigma_d$ — standard deviation of depth error (meters), e.g., ~15cm for DepthPro at 3m
  - $\Phi(\cdot)$ — standard normal CDF
  - $e_{\text{assign}} \in [0, 0.5]$ — assignment error probability
- **Intuition**: A point near a mask boundary (e.g., hand near hip, 5cm gap) can get displaced across the boundary by depth error. The CDF gives the probability that a Gaussian perturbation exceeds the boundary distance. For $d_{\text{boundary}} = 5\text{cm}$, $\sigma_d = 15\text{cm}$: $e_{\text{assign}} = 1 - \Phi(0.33) \approx 0.37$ — **37% of boundary points get wrong labels**.
- **Dimensions**: Meters / meters → dimensionless argument to CDF → probability ✓
- **Origin**: Standard Gaussian error analysis. Applied to SAM3D's specific failure mode by the survey.
- **Connection**: Directly degrades superpoint purity, which degrades AP.

### Eq. 18 — Edge Depth Error Amplification

$$\|\Delta\mathbf{p}_{3D}\|_{\text{edge}} = |\epsilon_d| \cdot \sec\!\left(\frac{\theta}{2}\right)$$

- **Name**: Depth error amplification at image edges
- **Variables**:
  - $\epsilon_d$ — depth error (meters)
  - $\theta$ — camera field of view (radians)
  - $\sec(\theta/2) = 1/\cos(\theta/2)$ — amplification factor
- **Intuition**: Near the image center, a depth error of $\epsilon_d$ meters causes approximately $\epsilon_d$ meters of 3D displacement (1:1). At image edges, the ray direction is angled, so the same depth error projects to a larger lateral displacement. For iPhone 14 Pro (FOV ≈ 75°): $\sec(37.5°) \approx 1.26$, so **26% amplification at edges**.
- **Dimensions**: Meters × dimensionless → meters ✓
- **Origin**: Follows from the geometry of perspective projection. Standard.
- **Connection**: Means boundary points near image edges are even more likely to be misassigned.

---

## 9. Confidence-Weighted Voting (Mitigation Strategy)

### Eq. 19 — Weighted Superpoint Vote

$$V_w(s_i, m_j) = \frac{\sum_{p \in s_i} c(p) \cdot \mathbb{1}[L_v(p) = m_j]}{\sum_{p \in s_i} c(p)}$$

- **Name**: Confidence-weighted superpoint voting
- **Variables**:
  - $c(p) \in [0, 1]$ — depth confidence for point $p$ (e.g., from MDE uncertainty output or inverse of predicted variance)
  - All other variables as in Eq. 10
- **Intuition**: Instead of equal-weight voting, downweight points with uncertain depth. A point with 90% depth confidence gets 9× the influence of a point with 10% confidence. This reduces $e_{\text{assign}}$ from ~0.30 to ~0.18 because high-uncertainty boundary points (the ones most likely misassigned) contribute less.
- **Dimensions**: Weighted sum of {0,1} → scalar $\in [0,1]$ ✓
- **Origin**: **Proposed in the survey analysis** as a mitigation for SAM3D. Standard weighted voting concept.
- **Connection**: Drop-in replacement for Eq. 10 when depth is estimated rather than ground truth.

---

## 10. TSDF Fusion (Background Depth Refinement)

### Eq. 20 — Truncated Signed Distance Function Fusion

$$F(\mathbf{x}) = \frac{\sum_v w_v \cdot f_v(\mathbf{x})}{\sum_v w_v}$$

where the per-view TSDF value:

$$f_v(\mathbf{x}) = \text{clamp}\!\left(\frac{d_v(\pi_v(\mathbf{x})) - \|\mathbf{x} - \mathbf{o}_v\|}{\delta}, -1, 1\right)$$

- **Name**: Multi-view TSDF integration for 3D volume reconstruction
- **Variables**:
  - $\mathbf{x} \in \mathbb{R}^3$ — a 3D voxel center
  - $w_v$ — per-view weight (e.g., $1/\sigma_d^2$ for depth-confidence weighting)
  - $d_v(\pi_v(\mathbf{x}))$ — observed depth at the projection of $\mathbf{x}$ into view $v$
  - $\|\mathbf{x} - \mathbf{o}_v\|$ — distance from voxel to camera center $\mathbf{o}_v$
  - $\delta$ — truncation distance (e.g., 4cm)
  - $F(\mathbf{x}) \in [-1, 1]$ — fused signed distance (negative = behind surface, positive = in front)
- **Intuition**: Fuses noisy per-frame depth maps into a clean 3D volume. The zero-crossing of $F(\mathbf{x})$ defines the surface. With 50 frames and $V_{\text{eff}} = 10$, RMSE reduces by ~$\sqrt{10} \approx 3.2\times$ (21cm → 6.5cm). **Only works for static geometry** — a moving dancer invalidates the averaging assumption.
- **Dimensions**: Meters in numerator, meters in denominator of the clamp argument → dimensionless. Weighted average of dimensionless values → dimensionless ✓
- **Origin**: Curless & Levoy (1996). Completely standard. Not part of SAM3D but proposed as augmentation.
- **Connection**: Improves background scene geometry, which indirectly helps dancer segmentation by providing better context.

---

## Complete Forward Pass

Combining everything, SAM3D's full pipeline is:

$$\boxed{\{I_v, d_v, \mathbf{K}_v, \mathbf{R}_v, \mathbf{t}_v\}_{v=1}^V \xrightarrow{\text{Eq. 1-4}} \{\mathbf{F}_v\} \xrightarrow{\text{Eq. 5-6}} \{\mathcal{M}_v\} \xrightarrow{\text{Eq. 7-8}} \{(p, L_v(p))\} \xrightarrow{\text{Eq. 9}} \{s_i\} \xrightarrow{\text{Eq. 10-11}} V_{\text{agg}} \xrightarrow{\text{Eq. 12-14}} \{G^*_k\}}$$

In one expression:

$$f(\{I_v, d_v, \mathbf{K}_v, [\mathbf{R}_v|\mathbf{t}_v]\}_{v=1}^V) = \text{Merge}\!\left(\text{Vote}\!\left(\text{VCCS}\!\left(\bigcup_v \text{BackProj}(d_v, \mathbf{K}_v, \mathbf{R}_v, \mathbf{t}_v, \text{SAM}(I_v))\right)\right)\right)$$

**Inputs**: $V$ RGB images + $V$ depth maps + $V$ camera poses
**Output**: Set of 3D instance masks $\{G^*_k\}_{k=1}^K$ (each a set of 3D points)
**Trainable parameters**: **0**

---

## Verification Checklist

- [x] **All dimensions compatible in matrix operations**: $\mathbf{K}^{-1}(3\times3) \cdot (3\times1) \to (3\times1)$; $\mathbf{R}^{-1}(3\times3) \cdot (3\times1) \to (3\times1)$. All verified above.
- [x] **Loss gradients push in correct direction**: **N/A — no loss functions.** SAM3D is training-free. All "learning" was done during SAM's original training (frozen).
- [x] **Attention scores sum to 1**: Softmax in Eq. 2-3 normalizes over the key dimension. Each row of the attention matrix sums to 1. ✓
- [x] **Implicit assumptions stated**:
  - Depth maps are metric (not relative/affine) — violated by monocular relative estimators
  - Camera intrinsics and extrinsics are accurate — violated by uncalibrated setups
  - Points are visible in at least one view — occluded regions have no votes
  - Superpoints don't straddle instance boundaries — violated at thin structures
  - Multi-view errors are approximately independent — violated by systematic MDE bias ($\rho \approx 0.5$)
- [x] **No circular dependencies**: The pipeline is strictly sequential: encode → decode → project → cluster → vote → merge. No equation references a downstream result.

---

## Equations NOT Present (Due to Training-Free Design)

| Category | Status | Reason |
|----------|--------|--------|
| **Loss functions** | None | Zero trainable parameters — no optimization |
| **Pretraining objective** | None (SAM's was masked image modeling, but that's frozen) | SAM3D adds no learning |
| **Gradient updates** | None | Inference-only pipeline |
| **Learned positional encodings** | Frozen from SAM | Not modified |
| **MPJPE / P-MPJPE** | Not applicable | SAM3D does instance segmentation, not pose estimation |

This is the defining characteristic of SAM3D: it is a **geometric algorithm** that uses a frozen neural network as a subroutine. Every equation unique to SAM3D (Eq. 10, 11, 14) is combinatorial/geometric, not differentiable.
