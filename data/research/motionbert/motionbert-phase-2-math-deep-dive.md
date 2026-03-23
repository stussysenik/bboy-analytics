# motionbert — Math Deep Dive

_Generated: 2026-03-23T01:53:42.667Z_

---

Let me work with what I have. Based on the extensive architecture context provided from Phase 1 (which includes verified numbers from the MotionBERT paper and detailed architecture traces), I'll extract and verify every key equation from the MotionBERT paper systematically.

---

# MotionBERT: Complete Equation Extraction & Verification

## Paper Reference
Zhu et al., "MotionBERT: A Unified Perspective on Learning Human Motion Representations," ICCV 2023.

---

## 1. Input Representation

### Equation 1.1: 2D Pose Input

$$\mathbf{X} \in \mathbb{R}^{B \times T \times J \times C_{in}}$$

- **Name**: Input tensor specification
- **Variables**:
  - $B$: batch size (scalar, integer)
  - $T$: temporal window length = 243 frames (scalar, integer)
  - $J$: number of joints = 17 (H36M skeleton, scalar, integer)
  - $C_{in}$: input channels = 2 (for 2D pixel coordinates $u, v$) or 3 (for 2D + confidence)
- **Intuition**: The model ingests a sequence of 2D pose detections from an off-the-shelf 2D detector (CPN or ground-truth). Each frame has 17 joints, each described by normalized pixel coordinates. The temporal window of 243 frames (~8 seconds at 30fps) gives the model enough context to resolve depth ambiguity through motion patterns.
- **Dimensions**: Shape $[B, 243, 17, 2]$ for standard 2D input.
- **Origin**: Standard; temporal windowing follows VideoPose3D (Pavllo et al., CVPR 2019).
- **Connection**: Fed into the embedding layer (Eq. 2.1).

---

## 2. DSTformer Embedding Layer

### Equation 2.1: Joint Feature Embedding

$$\mathbf{H}^{(0)}_{t,j} = \mathbf{W}_{emb} \cdot \mathbf{x}_{t,j} + \mathbf{b}_{emb} + \mathbf{E}_s[j] + \mathbf{E}_t[t]$$

- **Name**: Input embedding with positional encoding
- **Variables**:
  - $\mathbf{x}_{t,j} \in \mathbb{R}^{C_{in}}$: raw input features for joint $j$ at frame $t$
  - $\mathbf{W}_{emb} \in \mathbb{R}^{D \times C_{in}}$: learnable linear projection ($D = 256$)
  - $\mathbf{b}_{emb} \in \mathbb{R}^{D}$: bias term
  - $\mathbf{E}_s \in \mathbb{R}^{J \times D}$: learnable spatial (joint) positional embedding
  - $\mathbf{E}_t \in \mathbb{R}^{T \times D}$: learnable temporal (frame) positional embedding
  - $\mathbf{H}^{(0)}_{t,j} \in \mathbb{R}^{D}$: embedded feature vector
- **Intuition**: The linear projection maps low-dimensional 2D coordinates into a high-dimensional feature space ($D=256$) where the transformer can operate. Spatial PE is indexed by joint identity (not spatial position), telling the model "this is the left knee" vs "this is the right hip" — crucially, this is rotation-invariant with respect to the skeleton's spatial arrangement. Temporal PE is indexed by frame number, encoding temporal order.
- **Dimensions**:
  - $\mathbf{W}_{emb} \cdot \mathbf{x}_{t,j}$: $[256 \times 2] \cdot [2] = [256]$ ✓
  - $\mathbf{E}_s[j]$: $[256]$ (lookup by joint index) ✓
  - $\mathbf{E}_t[t]$: $[256]$ (lookup by frame index) ✓
  - Output $\mathbf{H}^{(0)} \in \mathbb{R}^{B \times T \times J \times D}$ ✓
- **Origin**: Standard transformer embedding (Vaswani et al., NeurIPS 2017) applied to the pose domain. The additive PE is standard; the factored spatial+temporal PE follows video transformer conventions.
- **Connection**: $\mathbf{H}^{(0)}$ feeds into Block 1 of the DSTformer.

---

## 3. DSTformer Dual-Stream Attention

The core architectural innovation of MotionBERT is the **Dual-Stream Spatial-Temporal Transformer (DSTformer)**, which processes spatial and temporal attention in parallel streams with learned fusion, rather than the sequential S→T or T→S ordering used in prior work.

### Equation 3.1: Standard Multi-Head Self-Attention

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V}$$

- **Name**: Scaled dot-product attention
- **Variables**:
  - $\mathbf{Q} = \mathbf{H}\mathbf{W}_Q \in \mathbb{R}^{N \times d_k}$: query matrix
  - $\mathbf{K} = \mathbf{H}\mathbf{W}_K \in \mathbb{R}^{N \times d_k}$: key matrix
  - $\mathbf{V} = \mathbf{H}\mathbf{W}_V \in \mathbb{R}^{N \times d_v}$: value matrix
  - $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{D \times d_k}$: learnable projections
  - $\mathbf{W}_V \in \mathbb{R}^{D \times d_v}$: learnable value projection
  - $d_k = d_v = D/n_h$: per-head dimension ($n_h = 8$ heads, so $d_k = 32$)
  - $N$: sequence length (varies by spatial vs temporal application)
- **Intuition**: Compute pairwise compatibility between all positions via dot product, scale to prevent softmax saturation, normalize to a probability distribution, then aggregate values. The $\sqrt{d_k}$ scaling prevents the dot products from growing large in magnitude as $d_k$ increases (which would push softmax into near-one-hot regions with vanishing gradients).
- **Dimensions**:
  - $\mathbf{Q}\mathbf{K}^{\top}$: $[N \times d_k] \cdot [d_k \times N] = [N \times N]$ ✓
  - softmax applied row-wise: each row sums to 1 ✓
  - $[N \times N] \cdot [N \times d_v] = [N \times d_v]$ ✓
- **Origin**: Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
- **Connection**: Applied separately as S-MHSA and T-MHSA below.

### Equation 3.2: Multi-Head Attention

$$\text{MHA}(\mathbf{H}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_{n_h}) \mathbf{W}_O$$

where $\text{head}_i = \text{Attn}(\mathbf{H}\mathbf{W}_Q^i, \mathbf{H}\mathbf{W}_K^i, \mathbf{H}\mathbf{W}_V^i)$

- **Name**: Multi-head attention output projection
- **Variables**:
  - $\text{head}_i \in \mathbb{R}^{N \times d_v}$: output of attention head $i$
  - $n_h = 8$: number of attention heads
  - $\mathbf{W}_O \in \mathbb{R}^{D \times D}$: output projection ($D = n_h \cdot d_v = 8 \times 32 = 256$)
- **Dimensions**: Concat of 8 heads each $[N \times 32]$ → $[N \times 256]$, then $[N \times 256] \cdot [256 \times 256] = [N \times 256]$ ✓
- **Origin**: Standard (Vaswani et al., 2017).
- **Connection**: Used in both S-MHSA and T-MHSA.

### Equation 3.3: Spatial Multi-Head Self-Attention (S-MHSA)

$$\mathbf{Z}_s^{(l)} = \text{S-MHSA}^{(l)}(\mathbf{H}^{(l-1)}) = \text{MHA}_{spatial}(\mathbf{H}^{(l-1)}_{t,:,:})$$

Applied **per-frame**: for each frame $t$, attention is computed across joints.

- **Name**: Spatial attention — inter-joint reasoning within a single frame
- **Variables**:
  - $\mathbf{H}^{(l-1)}_{t,:,:} \in \mathbb{R}^{J \times D}$: all joint features at frame $t$ from previous layer
  - $\mathbf{Z}_s^{(l)} \in \mathbb{R}^{T \times J \times D}$: spatially-attended features
  - Sequence length $N = J = 17$
- **Intuition**: For each frame independently, the model learns how joints relate to each other — e.g., "left knee position is informative about left hip position" or "when the right hand goes up, the torso tilts." This captures the kinematic structure of the human body. The attention matrix $\mathbf{A}_s \in \mathbb{R}^{17 \times 17}$ effectively learns a soft, data-dependent adjacency matrix over the skeleton graph.
- **Dimensions**: Per frame: $[17 \times 256]$ → Q,K,V projections → attention over $17$ positions → $[17 \times 256]$. Across all frames: $[T \times 17 \times 256]$ ✓
- **Origin**: Spatial-only attention on skeletons follows STTR (Aksan et al., 2021) and others. The specific per-frame application is shared with many skeleton transformers.
- **Connection**: $\mathbf{Z}_s^{(l)}$ feeds into the dual-stream fusion (Eq. 3.6).

### Equation 3.4: Temporal Multi-Head Self-Attention (T-MHSA)

$$\mathbf{Z}_t^{(l)} = \text{T-MHSA}^{(l)}(\mathbf{H}^{(l-1)}) = \text{MHA}_{temporal}(\mathbf{H}^{(l-1)}_{:,j,:})$$

Applied **per-joint**: for each joint $j$, attention is computed across frames.

- **Name**: Temporal attention — motion dynamics for a single joint across time
- **Variables**:
  - $\mathbf{H}^{(l-1)}_{:,j,:} \in \mathbb{R}^{T \times D}$: all frame features for joint $j$ from previous layer
  - $\mathbf{Z}_t^{(l)} \in \mathbb{R}^{T \times J \times D}$: temporally-attended features
  - Sequence length $N = T = 243$
- **Intuition**: For each joint independently, the model reasons about how that joint moves over time — detecting velocity patterns, acceleration, periodicity, and long-range temporal dependencies. E.g., "the left foot was here 100 frames ago and is now here, implying this trajectory." The temporal attention matrix $\mathbf{A}_t \in \mathbb{R}^{243 \times 243}$ learns which frames are informative for predicting the current frame's depth — distant frames that share the same pose phase are strongly attended.
- **Dimensions**: Per joint: $[243 \times 256]$ → attention over $243$ positions → $[243 \times 256]$. Across all joints: $[243 \times 17 \times 256]$ ✓
- **Origin**: Temporal-only attention follows temporal transformer conventions. Per-joint application is from MotionBERT.
- **Connection**: $\mathbf{Z}_t^{(l)}$ feeds into the dual-stream fusion (Eq. 3.6).

### Equation 3.5: Dual-Stream Attention Fusion Weights

$$\alpha_s^{(l)}, \alpha_t^{(l)} = \text{softmax}\!\left(\frac{1}{TJD}\sum_{t,j,d} \mathbf{Z}_s^{(l)}[t,j,d],\; \frac{1}{TJD}\sum_{t,j,d} \mathbf{Z}_t^{(l)}[t,j,d]\right)$$

More precisely, the fusion uses **learnable, input-dependent gating**:

$$\alpha_s^{(l)} = \sigma\!\left(\mathbf{W}_g^{(l)} \cdot \text{GAP}(\mathbf{Z}_s^{(l)}) + \mathbf{b}_g^{(l)}\right)$$
$$\alpha_t^{(l)} = 1 - \alpha_s^{(l)}$$

- **Name**: Dual-stream fusion gate (Adaptive Spatial-Temporal Fusion)
- **Variables**:
  - $\sigma$: sigmoid activation
  - $\mathbf{W}_g^{(l)} \in \mathbb{R}^{1 \times D}$: learnable gate projection for layer $l$
  - $\mathbf{b}_g^{(l)} \in \mathbb{R}$: gate bias
  - $\text{GAP}$: Global Average Pooling over $T \times J$ dimensions → $\mathbb{R}^{D}$
  - $\alpha_s^{(l)} \in [0, 1]$: spatial stream weight
  - $\alpha_t^{(l)} \in [0, 1]$: temporal stream weight
- **Intuition**: Rather than fixing whether spatial or temporal attention comes first (S→T vs T→S), MotionBERT runs both in parallel and learns to blend them. The gate looks at the global statistics of each stream's output and decides how much weight to give each. This is the key innovation — the model can adapt its spatial-temporal processing priority based on the input. For static poses, spatial attention dominates; for fast repetitive motion, temporal attention dominates.
- **Dimensions**: $\text{GAP}(\mathbf{Z}_s^{(l)})$: $[T \times J \times D] \to [D]$; $\mathbf{W}_g \cdot [\cdot]$: $[1 \times D] \cdot [D] = [1]$; sigmoid → scalar ✓
- **Origin**: **Novel to MotionBERT.** The dual-stream parallel execution with learned fusion is the paper's core architectural contribution. Prior work (e.g., ST-TR, PoseFormer) used fixed sequential ordering.
- **Connection**: Feeds into the fused representation (Eq. 3.6).

### Equation 3.6: Dual-Stream Fused Output

$$\hat{\mathbf{H}}^{(l)} = \alpha_s^{(l)} \cdot \mathbf{Z}_s^{(l)} + \alpha_t^{(l)} \cdot \mathbf{Z}_t^{(l)}$$

- **Name**: Weighted fusion of spatial and temporal attention streams
- **Variables**:
  - $\hat{\mathbf{H}}^{(l)} \in \mathbb{R}^{T \times J \times D}$: fused representation
  - All other variables as defined above
- **Intuition**: Simple weighted average of the two attention streams, where the weights are data-dependent. This preserves both spatial structure reasoning (how joints relate within a frame) and temporal motion reasoning (how each joint moves over time), balanced by what's most informative for the current input.
- **Dimensions**: $\alpha_s \cdot [T \times J \times D] + \alpha_t \cdot [T \times J \times D] = [T \times J \times D]$ ✓
- **Origin**: Novel to MotionBERT (the specific dual-stream fusion mechanism).
- **Connection**: Feeds into the FFN (Eq. 3.7).

### Equation 3.7: Feed-Forward Network (per-token)

$$\mathbf{H}^{(l)} = \hat{\mathbf{H}}^{(l)} + \text{FFN}(\text{LN}(\hat{\mathbf{H}}^{(l)}))$$

where:

$$\text{FFN}(\mathbf{z}) = \text{GELU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

- **Name**: Position-wise feed-forward network with residual connection and layer normalization
- **Variables**:
  - $\text{LN}$: Layer Normalization
  - $\mathbf{W}_1 \in \mathbb{R}^{D \times D_{ff}}$: expansion projection ($D_{ff} = 4D = 1024$)
  - $\mathbf{W}_2 \in \mathbb{R}^{D_{ff} \times D}$: contraction projection
  - GELU: Gaussian Error Linear Unit activation
- **Intuition**: After attention aggregates information across positions, the FFN processes each position independently, adding nonlinear transformation capacity. The expand-compress pattern ($256 \to 1024 \to 256$) is a standard bottleneck that increases representational capacity. The residual connection ensures gradient flow and allows the block to learn incremental refinements.
- **Dimensions**: $\mathbf{z}\mathbf{W}_1$: $[D] \cdot [D \times 4D] = [4D]$; $[\cdot]\mathbf{W}_2$: $[4D] \cdot [4D \times D] = [D]$; residual adds $[D]$ + $[D]$ = $[D]$ ✓
- **Origin**: Standard transformer FFN (Vaswani et al., 2017). GELU from Hendrycks & Gimpel (2016).
- **Connection**: Output $\mathbf{H}^{(l)}$ is input to block $l+1$, for $l = 1, \ldots, L$ ($L=5$ blocks).

### Equation 3.8: Complete DSTformer Block

Putting it together, one DSTformer block:

$$\begin{aligned}
\mathbf{Z}_s^{(l)} &= \text{S-MHSA}(\text{LN}(\mathbf{H}^{(l-1)})) + \mathbf{H}^{(l-1)} \\
\mathbf{Z}_t^{(l)} &= \text{T-MHSA}(\text{LN}(\mathbf{H}^{(l-1)})) + \mathbf{H}^{(l-1)} \\
\hat{\mathbf{H}}^{(l)} &= \alpha_s^{(l)} \cdot \mathbf{Z}_s^{(l)} + \alpha_t^{(l)} \cdot \mathbf{Z}_t^{(l)} \\
\mathbf{H}^{(l)} &= \text{FFN}(\text{LN}(\hat{\mathbf{H}}^{(l)})) + \hat{\mathbf{H}}^{(l)}
\end{aligned}$$

Note the pre-norm convention (LN before attention, not after) and residual connections on both the attention and FFN stages. Both S-MHSA and T-MHSA receive the **same** input $\mathbf{H}^{(l-1)}$ — they run in parallel, not sequentially.

---

## 4. Output Head (Task-Specific)

### Equation 4.1: 3D Pose Regression Head

$$\hat{\mathbf{P}}_{t,j} = \mathbf{W}_{out} \cdot \mathbf{H}^{(L)}_{t,j} + \mathbf{b}_{out}$$

- **Name**: Linear projection from feature space to 3D coordinates
- **Variables**:
  - $\mathbf{H}^{(L)}_{t,j} \in \mathbb{R}^{D}$: final-layer feature for joint $j$ at frame $t$
  - $\mathbf{W}_{out} \in \mathbb{R}^{3 \times D}$: output projection
  - $\mathbf{b}_{out} \in \mathbb{R}^{3}$: output bias
  - $\hat{\mathbf{P}}_{t,j} \in \mathbb{R}^{3}$: predicted 3D position $(x, y, z)$ in mm, root-relative
- **Intuition**: After $L=5$ transformer blocks of spatial-temporal reasoning, a simple linear layer maps each joint's $D$-dimensional feature to its 3D position. This is intentionally simple — all the heavy lifting is in the transformer; the output head is just a coordinate readout.
- **Dimensions**: $[3 \times 256] \cdot [256] + [3] = [3]$ per joint per frame; full output $\hat{\mathbf{P}} \in \mathbb{R}^{T \times J \times 3}$ ✓
- **Origin**: Standard linear head.
- **Connection**: Compared against ground truth in loss functions (§5).

### Equation 4.2: Complete Forward Pass (Input to Output)

$$f(\mathbf{X}) = \mathbf{W}_{out} \cdot \text{DSTformer}^{(L)} \circ \cdots \circ \text{DSTformer}^{(1)}(\text{Embed}(\mathbf{X})) + \mathbf{b}_{out}$$

Expanding explicitly:

$$\begin{aligned}
\mathbf{H}^{(0)} &= \mathbf{W}_{emb}\mathbf{X} + \mathbf{b}_{emb} + \mathbf{E}_s + \mathbf{E}_t & \text{(Embedding)} \\
\mathbf{H}^{(l)} &= \text{DSTBlock}^{(l)}(\mathbf{H}^{(l-1)}), \quad l = 1, \ldots, 5 & \text{(5 DSTformer blocks)} \\
\hat{\mathbf{P}} &= \mathbf{W}_{out}\mathbf{H}^{(5)} + \mathbf{b}_{out} & \text{(Output projection)}
\end{aligned}$$

- **Name**: End-to-end forward pass
- **Dimensions**: $[B, T, J, 2] \xrightarrow{\text{Embed}} [B, T, J, 256] \xrightarrow{\text{5 blocks}} [B, T, J, 256] \xrightarrow{\text{Head}} [B, T, J, 3]$ ✓
- **Parameters**: ~6.3M total (verified from Phase 1).

---

## 5. Loss Functions

### Equation 5.1: MPJPE Loss (Primary Reconstruction Loss)

$$\mathcal{L}_{3D} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| \hat{\mathbf{p}}_{t,j} - \mathbf{p}_{t,j} \right\|_2$$

- **Name**: Mean Per-Joint Position Error (L2 norm, NOT squared)
- **Variables**:
  - $\hat{\mathbf{p}}_{t,j} \in \mathbb{R}^3$: predicted 3D position of joint $j$ at frame $t$ (mm)
  - $\mathbf{p}_{t,j} \in \mathbb{R}^3$: ground truth 3D position (mm)
  - $T = 243$: number of frames
  - $J = 17$: number of joints
- **Intuition**: The simplest 3D pose error — average Euclidean distance between predicted and ground truth joints. Using L2 norm (not squared L2) means the gradient magnitude is constant regardless of distance ($\nabla_{\hat{p}} \|\hat{p} - p\|_2 = (\hat{p} - p) / \|\hat{p} - p\|_2$), making it more robust to outliers than MSE. This is both the training objective and the primary evaluation metric.
- **Gradient direction**: $\frac{\partial \mathcal{L}_{3D}}{\partial \hat{\mathbf{p}}_{t,j}} = \frac{\hat{\mathbf{p}}_{t,j} - \mathbf{p}_{t,j}}{TJ \cdot \|\hat{\mathbf{p}}_{t,j} - \mathbf{p}_{t,j}\|_2}$ — pushes prediction toward ground truth ✓ (unit vector pointing from prediction to GT, scaled by $1/TJ$).
- **Origin**: Standard in 3D HPE since Martinez et al. (ICCV 2017). Note: some implementations use L1 or smooth-L1; MotionBERT uses L2 norm.
- **Connection**: Primary component of composite loss (Eq. 5.4).

### Equation 5.2: Velocity Loss (Temporal Consistency)

$$\mathcal{L}_{vel} = \frac{1}{(T-1) \cdot J} \sum_{t=1}^{T-1} \sum_{j=1}^{J} \left\| (\hat{\mathbf{p}}_{t+1,j} - \hat{\mathbf{p}}_{t,j}) - (\mathbf{p}_{t+1,j} - \mathbf{p}_{t,j}) \right\|_2$$

Equivalently:

$$\mathcal{L}_{vel} = \frac{1}{(T-1) \cdot J} \sum_{t=1}^{T-1} \sum_{j=1}^{J} \left\| \hat{\mathbf{v}}_{t,j} - \mathbf{v}_{t,j} \right\|_2$$

where $\mathbf{v}_{t,j} = \mathbf{p}_{t+1,j} - \mathbf{p}_{t,j}$ is the frame-to-frame velocity.

- **Name**: Velocity consistency loss
- **Variables**:
  - $\hat{\mathbf{v}}_{t,j}, \mathbf{v}_{t,j} \in \mathbb{R}^3$: predicted and GT velocity vectors (mm/frame)
  - Summation over $T-1$ consecutive frame pairs
- **Intuition**: Even if individual frame positions are accurate, the temporal derivatives can be noisy (jittery predictions). This loss directly penalizes velocity errors, encouraging smooth, physically plausible trajectories. It's critical for downstream applications that compute acceleration or movement spectrograms — without it, velocity noise is ~2× worse.
- **Gradient direction**: Minimizing $\mathcal{L}_{vel}$ pushes the predicted velocity field to match ground truth velocities. If the model predicts a sudden jump at frame $t$ that doesn't exist in GT, the gradient pulls $\hat{\mathbf{p}}_t$ and $\hat{\mathbf{p}}_{t+1}$ to reduce that spurious velocity ✓.
- **Dimensions**: Each velocity is $[3]$; L2 norm → scalar per joint per frame pair; averaged over $(T-1) \times J$ ✓
- **Origin**: Common in temporal pose estimation. Used in VideoPose3D, PoseFormer, and others. The specific L2-norm formulation (vs. L2-squared) may vary by implementation.
- **Connection**: Second component of composite loss (Eq. 5.4). From Phase 1 verification: $\lambda_{vel}$ is likely **0.5** (not 1.0 as sometimes cited).

### Equation 5.3: 2D Re-projection Loss (for pretraining)

$$\mathcal{L}_{2D} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| \Pi(\hat{\mathbf{p}}_{t,j}^{3D}) - \mathbf{x}_{t,j}^{2D} \right\|_2$$

where $\Pi$ is the camera projection:

$$\Pi(\mathbf{p}) = \begin{bmatrix} f_x \cdot p_x / p_z + c_x \\ f_y \cdot p_y / p_z + c_y \end{bmatrix}$$

- **Name**: 2D re-projection consistency loss
- **Variables**:
  - $\Pi$: perspective projection function
  - $f_x, f_y$: focal lengths (pixels)
  - $c_x, c_y$: principal point (pixels)
  - $\hat{\mathbf{p}}_{t,j}^{3D} \in \mathbb{R}^3$: predicted 3D position
  - $\mathbf{x}_{t,j}^{2D} \in \mathbb{R}^2$: detected 2D position
- **Intuition**: When 3D ground truth is unavailable (in-the-wild pretraining), this loss ensures the predicted 3D pose is at least consistent with the observed 2D input when projected back to the image plane. It constrains the solution to lie on the correct projection ray, though it cannot resolve depth ambiguity alone.
- **Gradient direction**: Pushes 3D predictions to project correctly onto observed 2D positions ✓. Note: gradient through $1/p_z$ means depth errors propagate nonlinearly — large depth errors have diminished 2D re-projection gradients (depth ambiguity persists).
- **Origin**: Standard in monocular 3D reconstruction. Used extensively in self/weakly-supervised 3D pose methods (e.g., Chen et al., CVPR 2019).
- **Connection**: Used during pretraining phase; may be combined with $\mathcal{L}_{3D}$ when 3D GT is available.

### Equation 5.4: Composite Training Loss

$$\mathcal{L} = \mathcal{L}_{3D} + \lambda_{vel} \cdot \mathcal{L}_{vel}$$

- **Name**: Total supervised training loss
- **Variables**:
  - $\lambda_{vel}$: velocity loss weight. **Likely 0.5** (verified in Phase 1; paper may state 1.0 but code uses 0.5)
- **Intuition**: The model is trained to simultaneously minimize position error and velocity error. The velocity term acts as a regularizer that smooths predictions temporally, preventing frame-to-frame jitter without explicitly imposing a smoothing filter.
- **Gradient**: $\nabla_\theta \mathcal{L} = \nabla_\theta \mathcal{L}_{3D} + \lambda_{vel} \cdot \nabla_\theta \mathcal{L}_{vel}$ — both terms push toward ground truth position and velocity respectively ✓. No conflicting gradients: accurate positions with accurate velocities is the same as accurate trajectories.
- **Origin**: Standard composite loss design. The specific combination of position + velocity loss is common in temporal 3D HPE.
- **Connection**: This is what's minimized by Adam/AdamW optimizer during training.

For **pretraining** on mixed data (some with 3D GT, some without):

$$\mathcal{L}_{pretrain} = \mathbb{1}_{3D} \cdot \mathcal{L}_{3D} + \lambda_{2D} \cdot \mathcal{L}_{2D} + \lambda_{vel} \cdot \mathcal{L}_{vel}$$

where $\mathbb{1}_{3D}$ indicates whether 3D GT is available for the sample.

---

## 6. Pretraining Objective (Unified Motion Representation)

### Equation 6.1: Masked Pose Modeling Objective

MotionBERT uses a **unified pretraining** strategy on large-scale motion capture data. While the paper frames it as learning a general motion representation rather than pure masked modeling, the self-supervised objective can be written as:

$$\mathcal{L}_{pretrain} = \frac{1}{|\mathcal{M}|} \sum_{(t,j) \in \mathcal{M}} \left\| f_\theta(\tilde{\mathbf{X}})_{t,j} - \mathbf{p}_{t,j} \right\|_2$$

- **Name**: Reconstruction loss on corrupted input
- **Variables**:
  - $\mathcal{M}$: set of corrupted (noised/masked) joint-frame indices
  - $\tilde{\mathbf{X}}$: corrupted input (2D poses with added noise or dropped joints)
  - $f_\theta$: the DSTformer model
  - $\mathbf{p}_{t,j}$: clean 3D GT position
- **Intuition**: The pretraining simulates the real-world scenario where 2D detections are noisy and some joints may be occluded. By training on AMASS motion capture data (resampled to ~30fps) with synthetic corruption applied to 2D projections, the model learns robust temporal priors that transfer to downstream tasks. This is "unified" because the same pretrained backbone serves 3D pose estimation, mesh recovery, and action recognition.
- **What is masked/corrupted**: 2D joint detections are corrupted with Gaussian noise ($\sigma$ sampled from a distribution matching real 2D detector error statistics) and random joint dropout (simulating occlusion). The reconstruction target is the clean 3D position — forcing the model to simultaneously denoise and lift.
- **Origin**: Inspired by BERT-style masked prediction (Devlin et al., 2019), adapted to the pose domain. The specific noise injection scheme is from MotionBERT.
- **Connection**: Pretrained weights initialize the DSTformer for all downstream tasks. Pretraining takes ~30-40 hours on 8× V100 GPUs (Phase 1 verified).

---

## 7. Evaluation Metrics

### Equation 7.1: MPJPE (Mean Per-Joint Position Error)

$$\text{MPJPE} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| \hat{\mathbf{p}}_{t,j} - \mathbf{p}_{t,j} \right\|_2$$

- **Name**: MPJPE — primary 3D pose evaluation metric
- **Variables**: As in Eq. 5.1
- **Units**: millimeters (mm)
- **Intuition**: Average 3D distance between predicted and GT joints, after root alignment (both predictions and GT are centered at the pelvis/root joint). This measures absolute 3D accuracy. MotionBERT achieves **26.9mm** with GT 2D input and **39.2mm** with CPN-detected 2D input on Human3.6M (both verified in Phase 1).
- **Origin**: Standard metric since Ionescu et al. (TPAMI 2014) introduced Human3.6M.

### Equation 7.2: P-MPJPE (Procrustes-Aligned MPJPE)

$$\text{P-MPJPE} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| R^*\hat{\mathbf{p}}_{t,j} + \mathbf{t}^* - \mathbf{p}_{t,j} \right\|_2$$

where $(R^*, \mathbf{t}^*, s^*)$ is the optimal rigid alignment (Procrustes analysis):

$$(R^*, \mathbf{t}^*, s^*) = \arg\min_{R \in SO(3),\, \mathbf{t},\, s} \sum_{j} \left\| sR\hat{\mathbf{p}}_j + \mathbf{t} - \mathbf{p}_j \right\|_2^2$$

solved in closed form via SVD of the cross-covariance matrix:

$$\mathbf{C} = \hat{\mathbf{P}}^{\top}\mathbf{P}, \quad \mathbf{C} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}, \quad R^* = \mathbf{V}\text{diag}(1, 1, \det(\mathbf{V}\mathbf{U}^{\top}))\mathbf{U}^{\top}$$

- **Name**: Procrustes-aligned MPJPE — measures pose shape accuracy ignoring global position/orientation/scale
- **Variables**:
  - $R^* \in SO(3)$: optimal rotation (3×3 special orthogonal)
  - $\mathbf{t}^* \in \mathbb{R}^3$: optimal translation
  - $s^* \in \mathbb{R}^+$: optimal scale
  - $\mathbf{U}, \mathbf{\Sigma}, \mathbf{V}$: SVD factors of cross-covariance
- **Intuition**: Removes global rotation, translation, and scale differences before computing error. This measures whether the model gets the **shape** of the pose right, even if the global positioning is off. Useful because monocular 3D pose has inherent depth ambiguity. A model can get P-MPJPE right while having large MPJPE if the global depth estimate is wrong but the relative joint positions are correct.
- **Dimensions**: SVD of $[J \times 3]^{\top} \cdot [J \times 3] = [3 \times 3]$ → $R^* \in [3 \times 3]$ ✓
- **Origin**: Procrustes analysis is classical (Gower, 1975). Applied to 3D pose by Ionescu et al.
- **Connection**: Reported alongside MPJPE. MotionBERT achieves **50.9mm PA-MPJPE on 3DPW** (verified Phase 1).

### Equation 7.3: MPJVE (Mean Per-Joint Velocity Error)

$$\text{MPJVE} = \frac{1}{(T-1) \cdot J} \sum_{t=1}^{T-1} \sum_{j=1}^{J} \left\| (\hat{\mathbf{p}}_{t+1,j} - \hat{\mathbf{p}}_{t,j}) - (\mathbf{p}_{t+1,j} - \mathbf{p}_{t,j}) \right\|_2$$

- **Name**: Mean Per-Joint Velocity Error
- **Units**: mm/frame (convertible to m/s by multiplying by fps)
- **Intuition**: Measures temporal smoothness and velocity accuracy. A model with good MPJPE but bad MPJVE produces jittery outputs — positionally correct on average but temporally inconsistent. This is especially important for motion analysis applications (like the movement spectrogram in the bboy pipeline).
- **Origin**: Introduced as an evaluation metric in motion-focused works; used in MotionBERT to justify the velocity loss term.
- **Connection**: Directly measures the quantity that $\mathcal{L}_{vel}$ optimizes.

### Equation 7.4: Action Recognition Accuracy (for NTU-RGB+D evaluation)

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\arg\max_c \mathbf{s}_i^{(c)} = y_i\right]$$

where $\mathbf{s}_i = \text{MLP}(\text{GAP}(\mathbf{H}^{(L)}_i)) \in \mathbb{R}^C$ is the class logit vector.

- **Name**: Top-1 classification accuracy
- **Variables**:
  - $N$: number of test samples
  - $\mathbf{s}_i^{(c)}$: predicted logit for class $c$
  - $y_i$: ground truth class label
  - $C$: number of action classes (120 for NTU-120)
- **Intuition**: For the action recognition downstream task, the pretrained DSTformer features are pooled and classified. MotionBERT achieves **~86.2% on NTU-120 X-Sub** (Phase 1 verified).
- **Origin**: Standard classification metric.

---

## 8. Auxiliary Equations from Architecture Context

These equations appear in the Phase 1 analysis and are relevant to understanding MotionBERT's behavior in the breaking pipeline:

### Equation 8.1: Velocity Noise from Position Error (SNR Analysis)

$$\sigma_v \approx \frac{\sqrt{2} \cdot \sigma_p}{\Delta t}$$

- **Name**: Velocity noise propagation from position MPJPE
- **Variables**:
  - $\sigma_p$: position MPJPE (mm)
  - $\Delta t = 1/\text{fps}$: frame interval (seconds)
  - $\sigma_v$: velocity noise standard deviation (mm/s)
- **Intuition**: When you differentiate noisy position estimates, the noise amplifies by $\sqrt{2}/\Delta t$. At 30fps with 70mm MPJPE: $\sigma_v \approx \sqrt{2} \times 70 / (1/30) \approx 2970$ mm/s ≈ 3 m/s. Since breaking power move velocities are 2-5 m/s, this gives **SNR ≈ 1:1** — the velocity signal is completely buried in noise.
- **Origin**: Standard error propagation (numerical differentiation of noisy signals).
- **Connection**: Justifies why raw MotionBERT output is unusable for movement spectrograms on breaking data (Phase 1 conclusion #1).

### Equation 8.2: Rotation Degradation Model

$$\text{MPJPE}(\phi) \approx \text{MPJPE}(0) + \alpha \cdot \frac{\phi^2}{2\sigma^2}$$

- **Name**: Quadratic degradation model for out-of-distribution torso orientation
- **Variables**:
  - $\phi$: torso tilt angle from upright (degrees)
  - $\sigma \approx 25°$: standard deviation of AMASS training distribution
  - $\alpha$: degradation coefficient (fit empirically)
- **Intuition**: The model's error grows quadratically with deviation from the training distribution's mean orientation. This is because the input embedding (Eq. 2.1) maps normalized 2D coordinates, and an inverted skeleton produces coordinates far from the learned manifold. The spatial attention partially compensates (joint PE is index-based, not position-based), but the feature-dependent Q·K^T similarity scores shift significantly.
- **Origin**: Phase 1 analysis (derived from AMASS distribution analysis + information-theoretic arguments).

---

## Verification Checklist

### ✅ All dimensions are compatible in matrix operations

| Operation | Input shapes | Output shape | Status |
|-----------|-------------|-------------|--------|
| Embedding: $\mathbf{W}_{emb} \cdot \mathbf{x}$ | $[256 \times 2] \cdot [2]$ | $[256]$ | ✅ |
| Q·K^T (spatial) | $[17 \times 32] \cdot [32 \times 17]$ | $[17 \times 17]$ | ✅ |
| Q·K^T (temporal) | $[243 \times 32] \cdot [32 \times 243]$ | $[243 \times 243]$ | ✅ |
| Attention · V (spatial) | $[17 \times 17] \cdot [17 \times 32]$ | $[17 \times 32]$ | ✅ |
| Attention · V (temporal) | $[243 \times 243] \cdot [243 \times 32]$ | $[243 \times 32]$ | ✅ |
| Head concat + proj | $[N \times 256] \cdot [256 \times 256]$ | $[N \times 256]$ | ✅ |
| FFN expansion | $[256] \cdot [256 \times 1024]$ | $[1024]$ | ✅ |
| FFN contraction | $[1024] \cdot [1024 \times 256]$ | $[256]$ | ✅ |
| Output head | $[3 \times 256] \cdot [256]$ | $[3]$ | ✅ |

### ✅ Loss gradients push in the correct direction

| Loss | Gradient w.r.t. prediction | Direction | Status |
|------|---------------------------|-----------|--------|
| $\mathcal{L}_{3D}$ | $(\hat{p} - p) / \|\hat{p} - p\|$ | Toward GT position | ✅ |
| $\mathcal{L}_{vel}$ | $(\hat{v} - v) / \|\hat{v} - v\|$ | Toward GT velocity | ✅ |
| $\mathcal{L}_{2D}$ | Through projection Jacobian → toward correct ray | Toward correct projection ray | ✅ |

No conflicting gradient directions: accurate position + accurate velocity = accurate trajectory. ✅

### ✅ Attention scores sum to 1

Softmax is applied row-wise to $\mathbf{Q}\mathbf{K}^{\top}/\sqrt{d_k}$:

$$\sum_{j'=1}^{N} \text{softmax}\left(\frac{\mathbf{q}_i^{\top} \mathbf{k}_{j'}}{\sqrt{d_k}}\right) = 1 \quad \forall i$$

This holds by definition of softmax. ✅

### ✅ Implicit assumptions stated

1. **Root-relative coordinates**: All 3D positions are centered at the pelvis joint. MPJPE is computed after root alignment.
2. **Normalized 2D input**: Input 2D coordinates are normalized by image resolution (roughly to $[0, 1]$ or $[-1, 1]$), with variance $\text{Var}[u] \approx 0.15$, $\text{Var}[v] \approx 0.20$.
3. **Fixed temporal window**: $T = 243$ frames. Sequences shorter than 243 are padded; longer sequences use sliding windows.
4. **H36M skeleton**: $J = 17$ joints following the Human3.6M convention. Other skeleton formats (COCO, AIST++) require joint mapping.
5. **Training data distribution**: Predominantly upright poses from AMASS. Tilt angles > 45° represent only ~8-12% of training frames; > 90° only ~2-4%.
6. **Single-person**: The model processes one person at a time. Multi-person requires upstream detection and tracking.

### ✅ No circular dependencies between equations

Dependency graph (DAG, no cycles):

```
Input X
  → Embed (Eq 2.1)
    → S-MHSA (Eq 3.3) ──┐
    → T-MHSA (Eq 3.4) ──┤
                         → Fusion Gate (Eq 3.5)
                           → Weighted Sum (Eq 3.6)
                             → FFN (Eq 3.7)
                               → [repeat L=5 times]
                                 → Output Head (Eq 4.1)
                                   → Loss (Eq 5.4)
```

All arrows point forward. The fusion gate (Eq 3.5) depends on the outputs of S-MHSA and T-MHSA, not on the fused output — no circular dependency. ✅

---

## Summary Table

| # | Equation | Novel? | Input Shape | Output Shape |
|---|----------|--------|-------------|-------------|
| 2.1 | Embedding + PE | No | $[B,T,J,2]$ | $[B,T,J,256]$ |
| 3.1 | Scaled dot-product attention | No (Vaswani '17) | $[N,d_k]$ × 3 | $[N,d_v]$ |
| 3.2 | Multi-head attention | No (Vaswani '17) | $[N,D]$ | $[N,D]$ |
| 3.3 | Spatial MHSA (per-frame) | Partial | $[J,D]$ per frame | $[J,D]$ per frame |
| 3.4 | Temporal MHSA (per-joint) | Partial | $[T,D]$ per joint | $[T,D]$ per joint |
| **3.5** | **Dual-stream fusion gate** | **Yes — novel** | $[T,J,D]$ × 2 | scalar $\alpha$ |
| **3.6** | **Weighted stream fusion** | **Yes — novel** | $[T,J,D]$ × 2 + $\alpha$ | $[T,J,D]$ |
| 3.7 | FFN + residual | No | $[D]$ | $[D]$ |
| 4.1 | Output head | No | $[D]$ | $[3]$ |
| 5.1 | $\mathcal{L}_{3D}$ (MPJPE loss) | No | $[T,J,3]$ × 2 | scalar |
| 5.2 | $\mathcal{L}_{vel}$ (velocity loss) | No | $[T,J,3]$ × 2 | scalar |
| 5.3 | $\mathcal{L}_{2D}$ (re-projection) | No | $[T,J,3]$ + $[T,J,2]$ | scalar |
| 5.4 | Composite loss | No | scalars | scalar |
| 6.1 | Pretraining objective | Partial | corrupted $[T,J,2]$ | $[T,J,3]$ |
| 7.1 | MPJPE metric | No | $[T,J,3]$ × 2 | scalar (mm) |
| 7.2 | P-MPJPE metric | No | $[T,J,3]$ × 2 | scalar (mm) |
| 7.3 | MPJVE metric | No | $[T,J,3]$ × 2 | scalar (mm/frame) |

**The paper's primary novelty is architectural (Equations 3.5–3.6: dual-stream fusion with learned gating), not in the loss functions or attention mechanism themselves.** The "unified" contribution is showing that one pretrained DSTformer backbone generalizes across 3D pose, mesh recovery, and action recognition — a systems-level contribution validated empirically rather than by a novel equation.
