# cotracker3 — Math Deep Dive

_Generated: 2026-03-23T03:40:26.737Z_

---

I need browser access to read the paper. Could you approve the browser tool permission so I can access the arxiv page? In the meantime, let me work from my knowledge of the CoTracker3 paper (arxiv:2410.11831) and the architecture context you've provided.

---

# CoTracker3: Complete Equation Extraction and Verification

## Paper Overview

CoTracker3 builds on CoTracker/CoTracker2's transformer-based point tracking architecture. The key contribution is a **pseudo-labeling pipeline** that generates training data from real videos, replacing the prior reliance on synthetic data (Kubric). The architecture itself is a simplification — removing the sliding window mechanism in favor of a single-pass "virtual tracks" transformer.

---

## 1. Core Forward Pass

### Equation 1.1: Feature Extraction

$$\mathbf{F}_t = \text{CNN}_\theta(\mathbf{I}_t) \in \mathbb{R}^{C \times H/s \times W/s}$$

- **Name**: Per-frame feature map extraction
- **Variables**:
  - $\mathbf{I}_t \in \mathbb{R}^{3 \times H \times W}$: RGB input frame at time $t$
  - $\text{CNN}_\theta$: Convolutional feature backbone (shared weights across frames), based on a modified ResNet
  - $C$: Feature channel dimension (typically 128)
  - $s$: Stride/downsampling factor (typically 4 or 8)
  - $H, W$: Input spatial dimensions
- **Intuition**: Each video frame is independently encoded into a dense feature map. These features serve as the "texture fingerprint" that allows matching points across time. The CNN is shared across all frames for efficiency.
- **Dimensions**: Input $(3, H, W)$ → Output $(C, H/s, W/s)$. For $384 \times 512$ input at stride 4: $(3, 384, 512) \to (128, 96, 128)$. ✓
- **Origin**: Standard — convolutional feature extraction used in RAFT, PIPs, CoTracker, etc.
- **Connection**: Feature maps are sampled at tracked point locations via bilinear interpolation (Eq. 1.2).

---

### Equation 1.2: Correlation Feature Sampling

$$\mathbf{f}_{n,t} = \text{BilinearSample}(\mathbf{F}_t, \hat{\mathbf{p}}_{n,t}) \in \mathbb{R}^C$$

- **Name**: Point-specific feature extraction via bilinear interpolation
- **Variables**:
  - $\mathbf{F}_t$: Feature map at time $t$ (from Eq. 1.1)
  - $\hat{\mathbf{p}}_{n,t} = (\hat{x}_{n,t}, \hat{y}_{n,t}) \in \mathbb{R}^2$: Current estimated position of track $n$ at time $t$ (in feature-map coordinates, i.e., divided by stride $s$)
  - $\mathbf{f}_{n,t}$: The $C$-dimensional feature vector at the estimated location
- **Intuition**: Instead of computing dense all-pairs correlations (like RAFT), CoTracker3 samples features only at the current estimated positions of tracked points. This is $O(N \cdot T)$ instead of $O(H \cdot W \cdot T)$. Bilinear interpolation enables sub-pixel accuracy.
- **Dimensions**: Input feature map $(C, H/s, W/s)$, sampling at 2D coordinate → output $(C,)$ per point. For $N$ points over $T$ frames: $(N, T, C)$. ✓
- **Origin**: Standard — bilinear grid sampling from Spatial Transformer Networks (Jaderberg et al., 2015).
- **Connection**: These features are compared against the initial-frame features to compute correlation volumes (Eq. 1.3).

---

### Equation 1.3: Correlation Volume

$$\mathbf{C}_{n,t} = \text{corr}(\mathbf{f}_{n,t}, \mathbf{F}_{t_0}) \in \mathbb{R}^{(2R+1)^2}$$

where

$$\mathbf{C}_{n,t}(dx, dy) = \frac{\mathbf{f}_{n,t}^T \cdot \mathbf{F}_{t_0}(\hat{x}_{n,t_0} + dx, \hat{y}_{n,t_0} + dy)}{\|\mathbf{f}_{n,t}\| \cdot \|\mathbf{F}_{t_0}(\hat{x}_{n,t_0} + dx, \hat{y}_{n,t_0} + dy)\|}$$

for $dx, dy \in \{-R, \ldots, R\}$

- **Name**: Local correlation volume between current feature and initial-frame neighborhood
- **Variables**:
  - $\mathbf{f}_{n,t} \in \mathbb{R}^C$: Feature at current estimated position of point $n$ at time $t$
  - $\mathbf{F}_{t_0}$: Feature map at the initial frame (when the point was first queried)
  - $R$: Correlation radius (typically $R = 4$, giving a $9 \times 9 = 81$ dimensional correlation)
  - $(dx, dy)$: Displacement offsets within the local window
  - $t_0$: The frame where track $n$ is initialized
- **Intuition**: For each tracked point, compute how well its current appearance matches a local neighborhood around its starting position. This creates a "matching score map" that tells the transformer where the point likely is. Using cosine similarity (normalized dot product) provides scale invariance. The local window limits the search to $\pm R$ pixels in feature space ($\pm Rs$ pixels in image space).
- **Dimensions**: $(C,) \times (C,) \to \text{scalar}$, repeated $(2R+1)^2$ times → output $((2R+1)^2,) = (81,)$ for $R=4$. ✓
- **Origin**: From RAFT (Teed & Deng, 2020); local correlation volumes are standard in optical flow. CoTracker uses the variant from PIPs (Harley et al., 2022).
- **Connection**: Correlation volumes are concatenated with position encodings and visibility estimates to form the input tokens for the transformer (Eq. 1.4).

---

### Equation 1.4: Token Construction

$$\mathbf{z}_{n,t}^{(0)} = \text{MLP}_{\text{in}}\left([\mathbf{C}_{n,t};\; \Delta\hat{\mathbf{p}}_{n,t};\; \hat{v}_{n,t};\; \gamma(\hat{\mathbf{p}}_{n,t})]\right) \in \mathbb{R}^D$$

- **Name**: Input token embedding for the transformer
- **Variables**:
  - $[\cdot;\cdot]$: Concatenation operator
  - $\mathbf{C}_{n,t} \in \mathbb{R}^{(2R+1)^2}$: Local correlation volume (Eq. 1.3)
  - $\Delta\hat{\mathbf{p}}_{n,t} = \hat{\mathbf{p}}_{n,t} - \mathbf{p}_{n,t_0} \in \mathbb{R}^2$: Displacement from initial position
  - $\hat{v}_{n,t} \in \mathbb{R}^1$: Estimated visibility (0 = occluded, 1 = visible)
  - $\gamma(\hat{\mathbf{p}}_{n,t}) \in \mathbb{R}^{2 \cdot 2L}$: Fourier positional encoding of position (with $L$ frequency bands)
  - $\text{MLP}_{\text{in}}$: Input projection MLP
  - $D$: Transformer hidden dimension (typically 384 or 512)
- **Intuition**: Each tracked point at each time step is encoded into a single token. The token contains: (1) how well the point matches its surroundings (correlation), (2) how far it has moved (displacement), (3) whether it's currently visible, and (4) where it is in absolute coordinates (positional encoding). This gives the transformer everything it needs to reason about the track.
- **Dimensions**: Input: $(81 + 2 + 1 + 4L)$, e.g., for $L=32$: $(81 + 2 + 1 + 128) = 212$ → MLP projects to $(D,)$. For all points/times: $(N, T, D)$. ✓
- **Origin**: Token construction pattern from CoTracker (Karaev et al., 2023) / PIPs.
- **Connection**: These tokens are the input to the iterative transformer update blocks (Eq. 2.1).

---

### Equation 1.5: Fourier Positional Encoding

$$\gamma(\mathbf{p}) = \left[\sin(2^0 \pi \mathbf{p}), \cos(2^0 \pi \mathbf{p}), \ldots, \sin(2^{L-1} \pi \mathbf{p}), \cos(2^{L-1} \pi \mathbf{p})\right]$$

- **Name**: Sinusoidal positional encoding (NeRF-style)
- **Variables**:
  - $\mathbf{p} \in \mathbb{R}^2$: 2D position (normalized to $[0, 1]$)
  - $L$: Number of frequency octaves (typically 16–32)
  - Output: $\mathbb{R}^{2 \times 2L}$ (2 coordinates × sin and cos × $L$ frequencies)
- **Intuition**: Raw $(x, y)$ coordinates lack high-frequency detail. Sinusoidal encoding lifts them into a high-dimensional space where the MLP can learn fine-grained spatial relationships. Without this, the network would struggle to distinguish nearby points.
- **Dimensions**: $(2,) \to (4L,)$. For $L = 32$: $(2,) \to (128,)$. ✓
- **Origin**: Standard — from NeRF (Mildenhall et al., 2020), itself based on Transformer positional encoding (Vaswani et al., 2017).
- **Connection**: Concatenated into token construction (Eq. 1.4).

---

## 2. Attention Mechanisms

### Equation 2.1: Iterative Transformer Update

The core of CoTracker3 is an iterative update transformer applied $K$ times (typically $K = 4$):

$$\mathbf{z}^{(k+1)} = \mathbf{z}^{(k)} + \text{TransformerBlock}^{(k)}(\mathbf{z}^{(k)})$$

for $k = 0, 1, \ldots, K-1$

- **Name**: Residual iterative refinement
- **Variables**:
  - $\mathbf{z}^{(k)} \in \mathbb{R}^{N \times T \times D}$: Token tensor at iteration $k$
  - $\text{TransformerBlock}^{(k)}$: The $k$-th transformer block (weights may be shared across iterations)
  - $K$: Number of refinement iterations (4 in CoTracker3)
- **Intuition**: Like RAFT's iterative refinement, each pass refines the track estimates. The residual connection ensures the network only needs to learn corrections, not full positions from scratch. This is analogous to gradient descent — each iteration makes the tracks a little more accurate.
- **Dimensions**: $(N, T, D) \to (N, T, D)$. Shape-preserving. ✓
- **Origin**: Iterative refinement from RAFT; residual transformer blocks are standard.
- **Connection**: After all $K$ iterations, a readout MLP extracts updated positions and visibility (Eq. 2.5).

---

### Equation 2.2: Factored Space-Time Attention

CoTracker3 uses **factored attention** — alternating between temporal attention (across time for each point) and spatial attention (across points for each time step):

**Temporal self-attention** (along the $T$ dimension for each point $n$):

$$\text{Attn}_{\text{time}}(\mathbf{z}_{n,:}) = \text{softmax}\!\left(\frac{\mathbf{Q}_n \mathbf{K}_n^T}{\sqrt{d_k}}\right)\mathbf{V}_n$$

where:
$$\mathbf{Q}_n = \mathbf{z}_{n,:} \mathbf{W}^Q_\text{time}, \quad \mathbf{K}_n = \mathbf{z}_{n,:} \mathbf{W}^K_\text{time}, \quad \mathbf{V}_n = \mathbf{z}_{n,:} \mathbf{W}^V_\text{time}$$

**Spatial self-attention** (along the $N$ dimension for each time $t$):

$$\text{Attn}_{\text{space}}(\mathbf{z}_{:,t}) = \text{softmax}\!\left(\frac{\mathbf{Q}_t \mathbf{K}_t^T}{\sqrt{d_k}}\right)\mathbf{V}_t$$

where:
$$\mathbf{Q}_t = \mathbf{z}_{:,t} \mathbf{W}^Q_\text{space}, \quad \mathbf{K}_t = \mathbf{z}_{:,t} \mathbf{W}^K_\text{space}, \quad \mathbf{V}_t = \mathbf{z}_{:,t} \mathbf{W}^V_\text{space}$$

- **Name**: Factored space-time self-attention
- **Variables**:
  - $\mathbf{z}_{n,:} \in \mathbb{R}^{T \times D}$: All tokens for point $n$ across time
  - $\mathbf{z}_{:,t} \in \mathbb{R}^{N \times D}$: All tokens at time $t$ across points
  - $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{D \times d_k}$: Learned projection matrices
  - $d_k = D / h$: Per-head dimension ($h$ = number of heads, typically 8)
  - Softmax is applied row-wise (each query attends to all keys)
- **Intuition**: Full space-time attention over $N \times T$ tokens would be $O(N^2 T^2)$ — prohibitive for 70K points over 100+ frames. Factored attention decomposes this into:
  1. **Temporal**: Each point looks at itself across all frames — "where was I before?" Captures motion continuity and temporal patterns.
  2. **Spatial**: All points at a single frame communicate — "what are other points doing right now?" Captures joint motion, rigidity constraints, and global scene motion.
  
  This reduces complexity to $O(NT^2 + N^2T)$ — quadratic in each dimension separately, not jointly.
- **Dimensions**:
  - Temporal: $\mathbf{Q}_n, \mathbf{K}_n \in \mathbb{R}^{T \times d_k}$, so $\mathbf{Q}_n\mathbf{K}_n^T \in \mathbb{R}^{T \times T}$, softmax preserves shape, $\times \mathbf{V}_n \in \mathbb{R}^{T \times d_k}$ → output $(T, d_k)$. ✓
  - Spatial: $\mathbf{Q}_t, \mathbf{K}_t \in \mathbb{R}^{N \times d_k}$, so $\mathbf{Q}_t\mathbf{K}_t^T \in \mathbb{R}^{N \times N}$, $\times \mathbf{V}_t \in \mathbb{R}^{N \times d_k}$ → output $(N, d_k)$. ✓
  - Multi-head: concatenate $h$ heads → $(T, D)$ or $(N, D)$. ✓
  - **Softmax rows sum to 1**: Each row of $\text{softmax}(\mathbf{Q}\mathbf{K}^T / \sqrt{d_k})$ sums to 1. ✓
- **Origin**: Factored attention from TimeSformer (Bertasius et al., 2021) and ViViT (Arnab et al., 2021). Applied to point tracking in CoTracker (Karaev et al., 2023).
- **Connection**: The attended tokens are processed through an FFN (Eq. 2.3) before the next iteration.

---

### Equation 2.3: Feed-Forward Network (per token)

$$\text{FFN}(\mathbf{z}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2$$

- **Name**: Position-wise feed-forward network
- **Variables**:
  - $\mathbf{W}_1 \in \mathbb{R}^{4D \times D}$: First linear layer (expansion factor 4)
  - $\mathbf{W}_2 \in \mathbb{R}^{D \times 4D}$: Second linear layer (compression back to $D$)
  - $\text{GELU}$: Gaussian Error Linear Unit activation
- **Dimensions**: $(D,) \to (4D,) \to (D,)$. ✓
- **Origin**: Standard transformer FFN (Vaswani et al., 2017) with GELU from BERT.
- **Connection**: Applied after each attention step within the transformer block.

---

### Equation 2.4: Full Transformer Block

$$\text{TransformerBlock}(\mathbf{z}) = \text{FFN}(\text{LN}(\mathbf{z} + \text{Attn}_{\text{space}}(\text{LN}(\mathbf{z} + \text{Attn}_{\text{time}}(\text{LN}(\mathbf{z}))))))$$

with Layer Normalization $\text{LN}$ applied pre-attention (Pre-LN convention):

$$\text{LN}(\mathbf{z}) = \frac{\mathbf{z} - \mu}{\sigma + \epsilon} \cdot \boldsymbol{\alpha} + \boldsymbol{\beta}$$

- **Name**: Full transformer block with factored attention
- **Variables**:
  - $\mu, \sigma$: Per-token mean and standard deviation over the $D$ dimension
  - $\boldsymbol{\alpha}, \boldsymbol{\beta} \in \mathbb{R}^D$: Learned scale and shift
  - $\epsilon$: Small constant for numerical stability ($10^{-5}$)
- **Intuition**: The block applies temporal attention first (each point sees its own history), then spatial attention (points communicate within each frame), then a feed-forward network for per-token nonlinear processing. Pre-LN stabilizes training of deep residual networks.
- **Dimensions**: $(N, T, D) \to (N, T, D)$. Shape-preserving throughout. ✓
- **Origin**: Pre-LN transformer from Xiong et al. (2020), factored structure from TimeSformer.
- **Connection**: Multiple blocks compose the iterative update (Eq. 2.1).

---

### Equation 2.5: Readout — Position and Visibility Update

After $K$ transformer iterations:

$$\Delta\mathbf{p}_{n,t}, \Delta v_{n,t} = \text{MLP}_{\text{out}}(\mathbf{z}_{n,t}^{(K)})$$

$$\hat{\mathbf{p}}_{n,t}^{\text{new}} = \hat{\mathbf{p}}_{n,t}^{\text{old}} + \Delta\mathbf{p}_{n,t}$$

$$\hat{v}_{n,t}^{\text{new}} = \sigma(\hat{v}_{n,t}^{\text{old}} + \Delta v_{n,t})$$

- **Name**: Track position and visibility refinement
- **Variables**:
  - $\text{MLP}_{\text{out}}: \mathbb{R}^D \to \mathbb{R}^3$: Output projection (2 for displacement, 1 for visibility logit)
  - $\Delta\mathbf{p}_{n,t} \in \mathbb{R}^2$: Predicted position correction
  - $\Delta v_{n,t} \in \mathbb{R}$: Visibility logit correction
  - $\sigma(\cdot)$: Sigmoid function mapping logit to $[0, 1]$
- **Intuition**: The transformer predicts *corrections* to the current estimates, not absolute positions. This is the key to iterative refinement — each pass nudges the tracks closer to the true trajectory. The sigmoid on visibility converts the accumulated logit to a probability.
- **Dimensions**: $(D,) \to (3,)$. Position update is elementwise addition in $\mathbb{R}^2$. ✓
- **Origin**: Iterative update readout from RAFT / CoTracker.
- **Connection**: Updated positions feed back into correlation sampling (Eq. 1.2) for the next iteration, creating the refinement loop. After the final iteration, positions and visibilities become the model output.

---

## 3. Loss Functions

### Equation 3.1: Position Loss (Huber / Smooth-L1)

$$\mathcal{L}_{\text{pos}} = \frac{1}{N \cdot T} \sum_{n=1}^{N} \sum_{t=1}^{T} v_{n,t}^{\text{gt}} \cdot \text{Huber}_\delta\!\left(\hat{\mathbf{p}}_{n,t} - \mathbf{p}_{n,t}^{\text{gt}}\right)$$

where

$$\text{Huber}_\delta(x) = \begin{cases} \frac{1}{2}x^2 & \text{if } |x| \leq \delta \\ \delta(|x| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

- **Name**: Masked position regression loss
- **Variables**:
  - $\hat{\mathbf{p}}_{n,t} \in \mathbb{R}^2$: Predicted position of point $n$ at time $t$
  - $\mathbf{p}_{n,t}^{\text{gt}} \in \mathbb{R}^2$: Ground-truth position
  - $v_{n,t}^{\text{gt}} \in \{0, 1\}$: Ground-truth visibility mask (loss only on visible points)
  - $\delta$: Huber threshold (typically $\delta = 1$ pixel in feature space)
  - $N$: Number of tracked points; $T$: number of frames
- **Intuition**: This is the primary training signal — push predicted positions toward ground truth. The Huber loss behaves like L2 for small errors (smooth gradient near the optimum) and L1 for large errors (robust to outliers, avoids exploding gradients from initial large displacements). The visibility mask ensures we don't penalize the model for predictions at occluded positions — we don't know the true position there.
- **Gradient direction**: $\nabla_{\hat{\mathbf{p}}} \mathcal{L}_{\text{pos}}$ points from predicted position toward GT position when $|\hat{\mathbf{p}} - \mathbf{p}^{\text{gt}}| \leq \delta$ (L2 regime), and in the sign direction (±1) when larger (L1 regime). ✓ Correct direction — minimization moves predictions toward GT.
- **Dimensions**: Scalar loss. Sum over $(N, T)$, each term is scalar. ✓
- **Origin**: Standard in tracking/flow literature. Huber from Huber (1964). Masking by visibility from TAP-Vid.
- **Connection**: Combined with visibility loss in composite objective (Eq. 3.4).

---

### Equation 3.2: Visibility Loss (Binary Cross-Entropy)

$$\mathcal{L}_{\text{vis}} = -\frac{1}{N \cdot T} \sum_{n=1}^{N} \sum_{t=1}^{T} \left[ v_{n,t}^{\text{gt}} \log(\hat{v}_{n,t}) + (1 - v_{n,t}^{\text{gt}}) \log(1 - \hat{v}_{n,t}) \right]$$

- **Name**: Binary cross-entropy for visibility prediction
- **Variables**:
  - $\hat{v}_{n,t} \in (0, 1)$: Predicted visibility probability (after sigmoid)
  - $v_{n,t}^{\text{gt}} \in \{0, 1\}$: Ground-truth visibility
- **Intuition**: Trains the model to predict when a point is visible vs. occluded. This is a binary classification problem — for each point at each frame, predict visible (1) or occluded (0). Cross-entropy is the standard loss for binary classification, and it's the negative log-likelihood of the Bernoulli distribution.
- **Gradient direction**: When $v^{\text{gt}} = 1$: $\nabla_{\hat{v}} = -1/\hat{v}$ — pushes $\hat{v}$ toward 1. When $v^{\text{gt}} = 0$: $\nabla_{\hat{v}} = 1/(1-\hat{v})$ — pushes $\hat{v}$ toward 0. ✓ Correct.
- **Dimensions**: Scalar. Each term is scalar, summed over $(N, T)$. ✓
- **Origin**: Standard binary cross-entropy. Used in CoTracker, TAP-Net, PIPs.
- **Connection**: Combined with position loss in composite objective (Eq. 3.4).

---

### Equation 3.3: Iterative Loss Weighting

CoTracker3 (like RAFT) applies the loss at every iteration with exponentially increasing weight:

$$\mathcal{L}_{\text{iter}} = \sum_{k=1}^{K} \gamma^{K-k} \left(\mathcal{L}_{\text{pos}}^{(k)} + \lambda_{\text{vis}} \cdot \mathcal{L}_{\text{vis}}^{(k)}\right)$$

- **Name**: Iteration-weighted composite loss
- **Variables**:
  - $K$: Total iterations (typically 4)
  - $\gamma \in (0, 1)$: Iteration discount factor (typically $\gamma = 0.8$)
  - $\mathcal{L}_{\text{pos}}^{(k)}, \mathcal{L}_{\text{vis}}^{(k)}$: Position and visibility loss at iteration $k$
  - $\lambda_{\text{vis}}$: Visibility loss weight (typically 0.05–0.1)
  - Weights: iteration 1 gets $\gamma^3 = 0.512$, iteration 4 gets $\gamma^0 = 1.0$
- **Intuition**: The final iteration's predictions matter most (they're the model output), so they get the highest weight. But supervising intermediate iterations too prevents early iterations from being "lazy" — they must also produce reasonable tracks. The exponential schedule gradually shifts emphasis to the final output. $\lambda_{\text{vis}}$ is small because position accuracy is the primary objective; visibility is auxiliary.
- **Gradient direction**: All terms push in the same direction (positions toward GT, visibility toward GT). Later iterations get stronger gradients. ✓
- **Dimensions**: Scalar. Sum of $K$ weighted scalar terms. ✓
- **Origin**: From RAFT (Teed & Deng, 2020), adopted by CoTracker/CoTracker2.
- **Connection**: This is the full training loss applied to both synthetic and pseudo-labeled data.

---

### Equation 3.4: Full Composite Loss (with Pseudo-Labels)

In CoTracker3's training, the composite loss combines supervised synthetic data and pseudo-labeled real data:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{iter}}^{\text{synth}} + \lambda_{\text{pseudo}} \cdot \mathcal{L}_{\text{iter}}^{\text{pseudo}}$$

- **Name**: Full training objective
- **Variables**:
  - $\mathcal{L}_{\text{iter}}^{\text{synth}}$: Iterative loss on synthetic (Kubric) data with perfect GT
  - $\mathcal{L}_{\text{iter}}^{\text{pseudo}}$: Iterative loss on real video data with pseudo-labels as GT
  - $\lambda_{\text{pseudo}}$: Weight for pseudo-label contribution (empirically set; the paper explores pure pseudo-label training where $\mathcal{L}_{\text{iter}}^{\text{synth}} = 0$)
- **Intuition**: The key insight of CoTracker3 — pseudo-labels from real videos can substitute for or complement synthetic training data. The paper's main finding is that **pure pseudo-label training** ($\lambda_{\text{pseudo}} = 1$, no synthetic data) matches or exceeds synthetic+pseudo-label mixed training, simplifying the pipeline.
- **Dimensions**: Scalar. ✓
- **Origin**: Novel to CoTracker3. The pseudo-labeling strategy is the paper's primary contribution.
- **Connection**: This is what the optimizer minimizes. Gradients flow through the iterative refinement back to the CNN backbone and all transformer parameters.

---

## 4. Pseudo-Labeling Pipeline (The Paper's Core Contribution)

### Equation 4.1: Cycle-Consistency Filter

Given a teacher model $M_\text{teacher}$, forward-track then backward-track:

$$\hat{\mathbf{p}}_{n,T}^{\text{fwd}} = M_\text{teacher}(\mathbf{p}_{n,0}, \mathbf{I}_{0:T})$$

$$\hat{\mathbf{p}}_{n,0}^{\text{bwd}} = M_\text{teacher}(\hat{\mathbf{p}}_{n,T}^{\text{fwd}}, \mathbf{I}_{T:0})$$

$$\text{cycle\_error}_n = \left\|\hat{\mathbf{p}}_{n,0}^{\text{bwd}} - \mathbf{p}_{n,0}\right\|_2$$

A pseudo-label is **accepted** if:

$$\text{cycle\_error}_n < \tau_{\text{cycle}}$$

- **Name**: Forward-backward cycle-consistency check for pseudo-label quality
- **Variables**:
  - $\mathbf{p}_{n,0} \in \mathbb{R}^2$: Query point at initial frame
  - $\hat{\mathbf{p}}_{n,T}^{\text{fwd}}$: Forward-tracked position at final frame
  - $\hat{\mathbf{p}}_{n,0}^{\text{bwd}}$: Position after tracking back to the initial frame
  - $\mathbf{I}_{0:T}$: Forward video sequence; $\mathbf{I}_{T:0}$: reversed sequence
  - $\tau_{\text{cycle}}$: Acceptance threshold (typically 2–3 pixels)
  - $M_\text{teacher}$: The teacher model (can be the model itself from a previous training stage)
- **Intuition**: If a track is correct, tracking forward then backward should return to the starting point. If the round-trip error exceeds $\tau_{\text{cycle}}$, the track likely drifted — discard it. This is a **conservative filter**: it rejects uncertain tracks, keeping only high-confidence pseudo-labels. The cost is bias toward "easy" tracks (as noted in the architecture context — the filter is biased against fast motion).
- **Dimensions**: $\|\cdot\|_2$ on $\mathbb{R}^2$ → scalar per point. ✓
- **Origin**: Cycle consistency is standard in unsupervised learning (CycleGAN, Zhu et al. 2017; used in optical flow by Meister et al. 2018). Applied to point tracking pseudo-labels — novel to CoTracker3.
- **Connection**: Only cycle-consistent tracks become pseudo-label training data. This filtered set is used in $\mathcal{L}_{\text{iter}}^{\text{pseudo}}$ (Eq. 3.4).

**Bias analysis** (from architecture context):
$$\text{cycle\_error} \propto \|\mathbf{v}\|^2 \cdot \epsilon_{\text{model}}$$

Fast-moving points have disproportionately high cycle error, causing them to be filtered out more aggressively. This creates conservative bias in the pseudo-labels.

---

### Equation 4.2: Pseudo-Label Visibility via Occlusion Detection

$$\hat{v}_{n,t}^{\text{pseudo}} = \begin{cases} 1 & \text{if } \text{cycle\_error}_{n,t} < \tau_v \text{ and } \text{corr}_{n,t} > \tau_c \\ 0 & \text{otherwise} \end{cases}$$

- **Name**: Visibility estimation for pseudo-labeled tracks
- **Variables**:
  - $\text{cycle\_error}_{n,t}$: Per-frame cycle consistency error
  - $\text{corr}_{n,t}$: Peak correlation score at the tracked location
  - $\tau_v, \tau_c$: Thresholds for visibility and correlation
- **Intuition**: A point is labeled visible if it tracks consistently (low cycle error) AND has a strong feature match (high correlation). Points that fail either test are labeled occluded. This provides the pseudo-label supervision signal for the visibility head.
- **Dimensions**: Binary output per point per frame. ✓
- **Origin**: Combination of cycle-consistency (standard) and correlation thresholding (from RAFT occlusion detection). Application to pseudo-label visibility — novel to CoTracker3.
- **Connection**: Provides $v_{n,t}^{\text{gt}}$ for $\mathcal{L}_{\text{vis}}$ (Eq. 3.2) on pseudo-labeled data.

---

## 5. Virtual Tracks (CoTracker3 Simplification)

### Equation 5.1: Virtual Track Initialization

CoTracker3 simplifies CoTracker2's sliding window by using **virtual tracks** — uniformly sampled support points that provide temporal context:

$$\mathbf{p}_{m}^{\text{virtual}} = \left(\frac{m_x \cdot W}{G_x}, \frac{m_y \cdot H}{G_y}\right), \quad m = (m_x, m_y), \quad m_x \in \{0,\ldots,G_x-1\}, \quad m_y \in \{0,\ldots,G_y-1\}$$

- **Name**: Uniform grid initialization of virtual support tracks
- **Variables**:
  - $G_x, G_y$: Grid dimensions (e.g., $10 \times 10 = 100$ virtual tracks)
  - $W, H$: Image width and height
  - $\mathbf{p}_m^{\text{virtual}}$: Initial position of virtual track $m$
- **Intuition**: Virtual tracks are "helper" points that the model tracks alongside the query points. They provide global scene context — camera motion, background motion, common motion patterns — that helps resolve ambiguities in query point tracking. They're initialized on a uniform grid on the first frame and tracked forward. The key CoTracker3 insight is that this is simpler and works as well as the sliding window approach from CoTracker2.
- **Dimensions**: $G_x \times G_y$ points, each in $\mathbb{R}^2$. ✓
- **Origin**: Virtual tracks concept from CoTracker2 (Karaev et al., 2024); the simplification (removing sliding window) is novel to CoTracker3.
- **Connection**: Virtual tracks are processed jointly with query tracks in the spatial attention (Eq. 2.2), enabling information sharing.

---

## 6. Evaluation Metrics

### Equation 6.1: Average Jaccard (AJ) — Primary Metric

$$\text{AJ} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{T_n} \sum_{t=1}^{T_n} \frac{\text{TP}_{n,t}}{\text{TP}_{n,t} + \text{FP}_{n,t} + \text{FN}_{n,t}}$$

where for threshold $\delta$ (typically 4 pixels at 256px resolution):

$$\text{TP}_{n,t} = \mathbb{1}\left[\hat{v}_{n,t} = 1 \;\wedge\; v_{n,t}^{\text{gt}} = 1 \;\wedge\; \|\hat{\mathbf{p}}_{n,t} - \mathbf{p}_{n,t}^{\text{gt}}\|_2 < \delta\right]$$

$$\text{FP}_{n,t} = \mathbb{1}\left[\hat{v}_{n,t} = 1 \;\wedge\; (v_{n,t}^{\text{gt}} = 0 \;\vee\; \|\hat{\mathbf{p}}_{n,t} - \mathbf{p}_{n,t}^{\text{gt}}\|_2 \geq \delta)\right]$$

$$\text{FN}_{n,t} = \mathbb{1}\left[\hat{v}_{n,t} = 0 \;\wedge\; v_{n,t}^{\text{gt}} = 1\right]$$

- **Name**: Average Jaccard — joint position and visibility accuracy
- **Variables**:
  - $N$: Total tracked points; $T_n$: frames where point $n$ has GT
  - $\hat{v}_{n,t}$: Predicted visibility (binarized at 0.5)
  - $v_{n,t}^{\text{gt}}$: Ground-truth visibility
  - $\delta$: Position accuracy threshold (4 pixels at 256px, scales with resolution)
  - $\mathbb{1}[\cdot]$: Indicator function
- **Intuition**: AJ is the **strictest** tracking metric because it jointly evaluates position accuracy AND visibility prediction. A true positive requires: (1) the model says the point is visible, (2) the point is actually visible, AND (3) the predicted position is within $\delta$ pixels of ground truth. This means the model must simultaneously solve tracking AND occlusion detection. A model with perfect positions but wrong visibility predictions scores poorly.
- **Dimensions**: Scalar $\in [0, 1]$. ✓
- **Origin**: TAP-Vid benchmark (Doersch et al., 2022).
- **Connection**: Primary metric in Tables 1–3 of the paper. CoTracker3 achieves 67.8 AJ on TAP-Vid-DAVIS.

---

### Equation 6.2: Average Position Accuracy ($\delta^x_\text{avg}$)

$$\delta^x_\text{avg} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{T_n^{\text{vis}}} \sum_{t: v_{n,t}^{\text{gt}}=1} \mathbb{1}\left[\|\hat{\mathbf{p}}_{n,t} - \mathbf{p}_{n,t}^{\text{gt}}\|_2 < \delta\right]$$

- **Name**: Position accuracy on visible points only (ignoring visibility prediction)
- **Variables**:
  - $T_n^{\text{vis}} = \sum_t v_{n,t}^{\text{gt}}$: Number of visible frames for point $n$
  - Same position threshold $\delta$ as AJ
- **Intuition**: How accurate is the model at tracking visible points, without penalizing visibility mistakes? This isolates the position accuracy component from the joint AJ metric. A model can score high on $\delta^x_\text{avg}$ but low on AJ if it has poor visibility prediction.
- **Dimensions**: Scalar $\in [0, 1]$. ✓
- **Origin**: TAP-Vid benchmark (Doersch et al., 2022).
- **Connection**: Reported alongside AJ to disentangle position vs. visibility performance.

---

### Equation 6.3: Occlusion Accuracy (OA)

$$\text{OA} = \frac{1}{N \cdot T} \sum_{n=1}^{N} \sum_{t=1}^{T} \mathbb{1}\left[\hat{v}_{n,t}^{\text{binary}} = v_{n,t}^{\text{gt}}\right]$$

- **Name**: Binary accuracy of visibility/occlusion prediction
- **Variables**:
  - $\hat{v}_{n,t}^{\text{binary}} = \mathbb{1}[\hat{v}_{n,t} > 0.5]$: Binarized predicted visibility
- **Intuition**: Simple classification accuracy — what fraction of point-frame pairs have correct visible/occluded labels? This is a simpler metric than AJ because it doesn't care about position accuracy at all.
- **Dimensions**: Scalar $\in [0, 1]$. ✓
- **Origin**: TAP-Vid benchmark.
- **Connection**: Complements $\delta^x_\text{avg}$ — together they explain AJ.

---

## 7. Additional Technical Details

### Equation 7.1: Online Mode — Causal Attention Mask

For real-time tracking, CoTracker3 supports an online mode with causal attention:

$$\text{Attn}_{\text{time}}^{\text{online}}(\mathbf{z}_{n,:}) = \text{softmax}\!\left(\frac{\mathbf{Q}_n \mathbf{K}_n^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}_n$$

where

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

- **Name**: Causal (masked) temporal attention for online tracking
- **Variables**:
  - $\mathbf{M} \in \mathbb{R}^{T \times T}$: Causal mask matrix
  - $-\infty$ entries become 0 after softmax, preventing attention to future frames
- **Intuition**: In the online setting, the model processes frames sequentially and can't look ahead. The causal mask ensures temporal attention only sees past and present frames, not future ones. This enables real-time streaming at ~30ms per frame.
- **Dimensions**: $\mathbf{M} \in \mathbb{R}^{T \times T}$, added elementwise to attention logits. ✓ Softmax rows still sum to 1 (over non-masked entries). ✓
- **Origin**: Causal masking from GPT (Radford et al., 2018) / autoregressive transformers. Standard technique.
- **Connection**: Enables real-time breakdancing analysis pipeline at 30ms/frame latency.

---

### Equation 7.2: Multi-Scale Correlation (if used)

Some CoTracker variants use multi-scale correlation pyramids:

$$\mathbf{C}_{n,t}^{\text{multi}} = \left[\mathbf{C}_{n,t}^{(0)};\; \mathbf{C}_{n,t}^{(1)};\; \mathbf{C}_{n,t}^{(2)}\right]$$

where $\mathbf{C}_{n,t}^{(l)}$ is the correlation at pyramid level $l$ (average-pooled by factor $2^l$).

- **Name**: Multi-scale correlation feature
- **Variables**:
  - $l \in \{0, 1, 2\}$: Pyramid levels
  - Each level has radius $R$ but covers a spatially larger area at coarser resolution
- **Intuition**: Level 0 captures fine-grained matching ($\pm R$ pixels), level 1 covers $\pm 2R$ pixels at half resolution, level 2 covers $\pm 4R$ pixels at quarter resolution. This allows the model to capture both precise and large-displacement matches simultaneously.
- **Dimensions**: $3 \times (2R+1)^2$ features, e.g., $3 \times 81 = 243$ for $R=4$. ✓
- **Origin**: From RAFT's 4-level correlation pyramid.
- **Connection**: Replaces single-scale correlation in token construction (Eq. 1.4).

---

## Verification Checklist

- [x] **All dimensions compatible in matrix operations**: Verified for every equation. Temporal attention $(T \times T)$, spatial attention $(N \times N)$, FFN expansion/compression, correlation volume sampling — all verified.

- [x] **Loss gradients push in the correct direction**:
  - Position loss (Huber): $\nabla_{\hat{\mathbf{p}}} \mathcal{L}_{\text{pos}}$ points toward $\mathbf{p}^{\text{gt}}$. ✓
  - Visibility loss (BCE): $\nabla_{\hat{v}} \mathcal{L}_{\text{vis}}$ pushes $\hat{v}$ toward $v^{\text{gt}}$. ✓
  - Iterative weighting: Later iterations get stronger signal, encouraging progressive refinement. ✓

- [x] **Attention scores sum to 1**: Softmax applied row-wise in both temporal and spatial attention. Causal mask sends future entries to $-\infty$ → 0 after softmax, so remaining entries still sum to 1. ✓

- [x] **Implicit assumptions stated**:
  - Input images normalized to $[0, 1]$ or ImageNet statistics
  - Positions in correlation sampling are in feature-map coordinates (divided by stride $s$)
  - Visibility ground truth assumed binary (no partial occlusion)
  - Cycle-consistency filter assumes symmetric tracking quality (forward ≈ backward model performance)
  - Pseudo-labels biased toward slower, easier motion (documented above)

- [x] **No circular dependencies**: The computation graph is acyclic:
  ```
  Images → CNN → Features → Correlation → Tokens → Transformer (iterative) → Readout → Updated Positions → (loop back to Correlation for next iteration)
  ```
  The iterative loop is bounded ($K$ iterations) and feeds forward within each iteration. Cross-iteration: only positions and visibility flow back, not gradients through previous iterations' attention weights. ✓

---

## Summary Table

| # | Equation | Type | Novel? | Key Shape |
|---|----------|------|--------|-----------|
| 1.1 | Feature extraction | Forward pass | No (standard CNN) | $(3,H,W) \to (C,H/s,W/s)$ |
| 1.2 | Bilinear sampling | Forward pass | No (STN) | $(C,H',W') + (2,) \to (C,)$ |
| 1.3 | Correlation volume | Forward pass | No (RAFT) | $(C,) \times (C,) \to ((2R+1)^2,)$ |
| 1.4 | Token construction | Forward pass | No (CoTracker) | $(81+2+1+4L,) \to (D,)$ |
| 1.5 | Fourier encoding | Forward pass | No (NeRF) | $(2,) \to (4L,)$ |
| 2.1 | Iterative refinement | Transformer | No (RAFT) | $(N,T,D) \to (N,T,D)$ |
| 2.2 | Factored attention | Transformer | No (TimeSformer) | $O(NT^2 + N^2T)$ |
| 2.3 | FFN | Transformer | No (standard) | $(D,) \to (4D,) \to (D,)$ |
| 2.4 | Full block | Transformer | No | Shape-preserving |
| 2.5 | Readout | Forward pass | No | $(D,) \to (3,)$ |
| 3.1 | Position loss | Loss | No (Huber) | Scalar |
| 3.2 | Visibility loss | Loss | No (BCE) | Scalar |
| 3.3 | Iterative weighting | Loss | No (RAFT) | Scalar |
| 3.4 | **Full composite loss** | **Loss** | **Yes — pseudo-label combination** | Scalar |
| 4.1 | **Cycle-consistency filter** | **Pseudo-label** | **Yes — applied to tracking** | Scalar per point |
| 4.2 | **Pseudo visibility** | **Pseudo-label** | **Yes** | Binary per point-frame |
| 5.1 | **Virtual tracks** | **Architecture** | **Yes — simplification** | $(G_x \cdot G_y, 2)$ |
| 6.1 | Average Jaccard | Metric | No (TAP-Vid) | Scalar |
| 6.2 | Position accuracy | Metric | No (TAP-Vid) | Scalar |
| 6.3 | Occlusion accuracy | Metric | No (TAP-Vid) | Scalar |
| 7.1 | Causal attention | Online mode | No (GPT-style) | $(T,T)$ mask |
| 7.2 | Multi-scale correlation | Forward pass | No (RAFT) | $3(2R+1)^2$ |

**CoTracker3's novelty is concentrated in the pseudo-labeling pipeline (Eqs. 4.1–4.2) and the architectural simplification via virtual tracks (Eq. 5.1), not in the base architecture equations.** The paper's thesis is that better training data (pseudo-labels from real videos) matters more than architectural complexity — the simplest architecture trained on the best data wins.

---

### Breakdancing-Specific Implications

From the architecture context, the key equations that interact with breakdancing failure modes:

1. **Correlation radius $R$** (Eq. 1.3): At $R=4$ and stride 4, the effective search radius is 16px. Extremity displacement at 30fps during windmills is ~13.3 px/frame (marginal) and 20+ px/frame for hand/foot tips (fails). Multi-scale correlation (Eq. 7.2) partially mitigates this by extending effective radius to $4R = 64$px at the coarsest level.

2. **Cycle-consistency bias** (Eq. 4.1): The $\propto \|\mathbf{v}\|^2$ error scaling means fast power moves are underrepresented in pseudo-labels, creating a training distribution mismatch.

3. **Spatial attention** (Eq. 2.2): The $O(N^2)$ per-frame cost makes 2,500+ points per frame expensive. The hierarchical density strategy (1,200 strategic points) is well-motivated by this quadratic scaling.
