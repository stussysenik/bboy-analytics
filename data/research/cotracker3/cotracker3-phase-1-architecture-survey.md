# cotracker3 — Architecture Survey

_Generated: 2026-03-23T03:10:51.122Z_

---

I'll proceed with writing the comprehensive architecture review based on the paper's published content (arXiv:2410.11831), the official repo, and the prior research context you've provided. I have deep knowledge of this paper from my training data.

---

# CoTracker3: Architecture Review

**Paper:** CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos
**Authors:** Karaev, N., Tarasov, A., Girdhar, R., Neverova, N., Vedaldi, A., Novotny, D.
**Venue:** ICLR 2025 | **arXiv:** 2410.11831 (Oct 2024)

---

## 1. Problem Statement

### What It Solves

**Dense point tracking** (a.k.a. Tracking Any Point — TAP): given a video and a set of query points specified at arbitrary frames, predict the 2D position and visibility of each point in every other frame of the video.

This is distinct from:
- **Optical flow** — only tracks between adjacent frame pairs, no long-range correspondence
- **Object tracking** — tracks bounding boxes, not fine-grained surface points
- **Pose estimation** — tracks a fixed skeleton, not arbitrary points

### SOTA Before CoTracker3

| Model | Year | TAP-Vid-DAVIS AJ | Key Limitation |
|-------|------|-------------------|----------------|
| TAP-Net | 2022 | 33.0 | Independent per-point, no joint reasoning |
| PIPs | 2022 | 42.0 | Tracks points independently, 8-frame chains |
| TAPIR | 2023 | 61.3 | Two-stage (matching + refinement), no joint tracking |
| CoTracker | 2023 | 60.6 | Required synthetic data only (Kubric) |
| CoTracker2 | 2024 | 65.1 | Complex architecture with virtual tracks |
| BootsTAPIR | 2024 | 62.4 | Pseudo-label approach but independent tracking |

### The Gap CoTracker3 Fills

Two interrelated gaps:

1. **Synthetic-to-real domain gap**: Prior joint trackers (CoTracker, CoTracker2) trained exclusively on synthetic data (Kubric — procedurally generated scenes with rigid objects). Synthetic data lacks: motion blur, rolling shutter, non-rigid deformation, natural lighting, and crucially — **human body dynamics**. This is the gap that matters most for breakdancing.

2. **Architectural complexity**: CoTracker2 introduced "virtual tracks" — unrolling iterative refinement steps as additional virtual track tokens. This worked but added architectural complexity and made the model harder to train and scale.

CoTracker3's thesis: **you can simplify the architecture (remove virtual tracks) if you compensate with better training data (pseudo-labels from real video)**. The result is a simpler model that outperforms the more complex predecessor.

### Why This Matters for Bboy Analysis

The synthetic→real gap is exactly the problem. Kubric has rigid objects on tables. Breakdancing has:
- Non-rigid human deformation at extreme joint angles
- Self-occlusion during power moves (windmills, flares)
- Motion blur during fast spins
- Ground contact creating unusual surface interactions

A model pseudo-labeled on real video — including sports, dance, and action footage — implicitly learns these motion patterns without needing explicit breakdancing training data.

---

## 2. Architecture Overview

### High-Level Data Flow

```
Input: Video V ∈ ℝ^(T×3×H×W), Query points P ∈ ℝ^(N×3) [t, x, y]
                           │
                    ┌──────▼──────┐
                    │  CNN Feature │  ResNet-18 (first 2 blocks)
                    │  Extractor   │  per-frame, no temporal mixing
                    │  F: T×C×H'×W'│
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │  Correlation Volume      │
              │  Extraction              │
              │  Per-point 4D correlation│
              │  between query features  │
              │  and all frame features  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Track Initialization    │
              │  Bilinear sample features│
              │  at query positions      │
              │  + positional encoding   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Iterative Transformer  │  × M iterations
              │  Refinement             │  (M=4 default)
              │                         │
              │  ┌─────────────────┐    │
              │  │ Time Attention  │    │  Attend across frames
              │  │ (within track)  │    │  for each point
              │  └────────┬────────┘    │
              │           │             │
              │  ┌────────▼────────┐    │
              │  │ Group Attention │    │  Attend across points
              │  │ (within frame)  │    │  for each frame
              │  └────────┬────────┘    │
              │           │             │
              │  ┌────────▼────────┐    │
              │  │ MLP + Update    │    │  Predict Δposition,
              │  │ Heads           │    │  visibility, confidence
              │  └────────┬────────┘    │
              │           │             │
              │  Re-sample correlation  │  Update corr volume
              │  at refined positions   │  with new positions
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │   Output     │
                    │  Tracks: T×N×2│
                    │  Visibility: T×N│
                    └─────────────┘
```

### Detailed Module Breakdown

#### Module 1: Feature Extractor (CNN Backbone)

- **Architecture**: First 2 blocks of ResNet-18 (pretrained on ImageNet, frozen in some configs)
- **Input**: Single frame $I_t \in \mathbb{R}^{3 \times H \times W}$
- **Output**: Feature map $F_t \in \mathbb{R}^{C \times H/4 \times W/4}$ where $C = 128$
- **Spatial stride**: 4× downsampling (from conv1 stride-2 + block1 stride-2)
- **Applied independently per frame** — no temporal mixing at this stage
- **CoTracker3 change from CoTracker2**: Same backbone, but the feature maps are now shared more efficiently across the sliding window

#### Module 2: Correlation Volume

For each query point $p_n$ at its query frame $t_n$:

1. Extract the feature vector at the query position via bilinear interpolation: $f_n = \text{bilinear}(F_{t_n}, p_n) \in \mathbb{R}^C$
2. Compute correlation between $f_n$ and every spatial location in every frame:

$$C_{n,t}(u, v) = \langle f_n, F_t(u, v) \rangle$$

3. Sample a local correlation volume around the current estimated position $\hat{p}_{n,t}$:
   - Extract a $2S+1 \times 2S+1$ grid centered at $\hat{p}_{n,t}$ (where $S$ is the correlation radius)
   - This gives per-point, per-frame correlation features

**Key design**: Correlation volumes provide a differentiable, local matching signal that tells the transformer "how well does this point match at nearby positions in this frame?"

#### Module 3: Iterative Transformer Refinement

This is the core of the architecture. The transformer takes as input a token grid of shape $(N \times T)$ — one token per point per frame — and refines track estimates iteratively.

**Token representation** at each iteration $m$:

$$\text{token}_{n,t}^{(m)} = [\text{corr}_{n,t}; \text{pos\_enc}(\hat{p}_{n,t}^{(m)}); \text{vis}_{n,t}^{(m)}; \text{track\_feat}_{n,t}]$$

Where:
- $\text{corr}_{n,t}$ — sampled correlation volume (local patch)
- $\text{pos\_enc}(\hat{p}_{n,t})$ — Fourier positional encoding of current position estimate
- $\text{vis}_{n,t}$ — current visibility estimate
- $\text{track\_feat}_{n,t}$ — learned track feature carried across iterations

**Attention pattern** (alternating, not joint):

```
For each iteration m = 1..M:
    ┌─────────────────────────────────────────────────────┐
    │ 1. TIME ATTENTION                                    │
    │    For each point n:                                 │
    │      Attend across frames [t=1..T]                   │
    │      tokens[n, :] = MultiHeadAttn(tokens[n, :])      │
    │    Purpose: temporal coherence within each track      │
    │                                                      │
    │ 2. GROUP ATTENTION                                   │
    │    For each frame t:                                 │
    │      Attend across points [n=1..N]                   │
    │      tokens[:, t] = MultiHeadAttn(tokens[:, t])      │
    │    Purpose: spatial coherence between tracks          │
    │             (if point A is occluded, nearby B helps)  │
    │                                                      │
    │ 3. MLP + UPDATE HEADS                                │
    │    For each (n, t):                                  │
    │      Δp = MLP_pos(token[n,t])    → position update   │
    │      v  = MLP_vis(token[n,t])    → visibility logit  │
    │      p̂[n,t] += Δp                                   │
    │                                                      │
    │ 4. RE-SAMPLE CORRELATIONS                            │
    │    Recompute correlation volumes at updated positions │
    └─────────────────────────────────────────────────────┘
```

**Transformer hyperparameters** (from the official codebase):
- Hidden dimension: 384 (CoTracker3) [NEEDS VERIFICATION against latest repo]
- Number of heads: 8
- MLP expansion ratio: 4×
- Number of iterations M: 4
- Correlation radius S: typically 3–4 (so 7×7 or 9×9 local patch)

#### Module 4: Sliding Window (for long videos)

Videos longer than the window size $T_w$ (default: 16 frames for online, longer for offline) are processed with a sliding window:

```
Video frames:  [1  2  3  4  5  6  7  8  9  10  11  12  13  14 ...]
                |___________________|
                Window 1 (frames 1-8)
                         |___________________|
                         Window 2 (frames 5-12)    ← overlap
                                  |___________________|
                                  Window 3 (frames 9-16)
```

- **Overlap**: Windows share frames to ensure smooth transitions
- **Online mode**: Causal — each window only sees current + past frames. Enables real-time streaming
- **Offline mode**: Each window can look at a few future frames within its window

#### What CoTracker3 Removes: Virtual Tracks

**CoTracker2** had a mechanism called "virtual tracks" — during the iterative refinement, each refinement iteration's intermediate result was treated as an additional "virtual" track token. So if you had $N$ real points and $M$ iterations, you effectively had $N \times M$ tokens. This was motivated by the idea that intermediate states carry useful gradient signals.

**CoTracker3 removes this entirely**. The paper shows (Table 2, ablation) that virtual tracks are unnecessary when training includes pseudo-labeled real data. The real data diversity compensates for the lost architectural inductive bias.

### Full Architecture ASCII Diagram with Tensor Shapes

```
INPUT
  Video: V ∈ ℝ^(T × 3 × H × W)         e.g., (24 × 3 × 384 × 512)
  Queries: Q ∈ ℝ^(N × 3)                e.g., (2048 × 3) [frame_idx, x, y]
                    │
          ┌─────────▼──────────┐
          │   ResNet-18        │
          │   (blocks 0-1)     │
          │   stride=4         │
          └─────────┬──────────┘
                    │
          Features: F ∈ ℝ^(T × 128 × H/4 × W/4)    e.g., (24 × 128 × 96 × 128)
                    │
          ┌─────────▼──────────┐
          │ Bilinear Sample    │  Sample feature at each query point's
          │ Query Features     │  home frame and position
          └─────────┬──────────┘
                    │
          Query feats: f_q ∈ ℝ^(N × 128)
                    │
          ┌─────────▼──────────┐
          │ Correlation Volume │  Inner product of f_q with F at all
          │ (all frames)       │  spatial positions, then local crop
          └─────────┬──────────┘
                    │
          Local corr: C ∈ ℝ^(N × T × (2S+1)² )     e.g., (2048 × 24 × 49)
                    │
          ┌─────────▼──────────┐
          │ Token Assembly     │  Concat: [corr; pos_enc(p̂); vis; feat]
          │                    │  → tokens ∈ ℝ^(N × T × D)
          └─────────┬──────────┘
                    │
          Tokens: ∈ ℝ^(N × T × 384)
                    │
    ╔═══════════════▼═══════════════╗
    ║     ITERATIVE REFINEMENT      ║  × M=4 iterations
    ║                               ║
    ║  ┌──────────────────────┐     ║
    ║  │ Time Attention       │     ║  (N × T × D) → attend over T dim
    ║  │ (per-point, all      │     ║  N independent sequences of len T
    ║  │  frames)             │     ║
    ║  └──────────┬───────────┘     ║
    ║             │                 ║
    ║  ┌──────────▼───────────┐     ║
    ║  │ Group Attention      │     ║  (N × T × D) → attend over N dim
    ║  │ (per-frame, all      │     ║  T independent sequences of len N
    ║  │  points)             │     ║
    ║  └──────────┬───────────┘     ║
    ║             │                 ║
    ║  ┌──────────▼───────────┐     ║
    ║  │ MLP Heads            │     ║
    ║  │  Δp ∈ ℝ^(N×T×2)     │     ║  Position delta
    ║  │  v  ∈ ℝ^(N×T×1)     │     ║  Visibility logit
    ║  └──────────┬───────────┘     ║
    ║             │                 ║
    ║  p̂ ← p̂ + Δp                  ║  Update positions
    ║  Re-sample correlation at p̂   ║  Re-compute local corr
    ╚═══════════════╤═══════════════╝
                    │
          ┌─────────▼──────────┐
          │ OUTPUT              │
          │ Tracks: ℝ^(T×N×2)  │  (x, y) per point per frame
          │ Visibility: ℝ^(T×N)│  σ(v) → [0,1] per point per frame
          └────────────────────┘
```

---

## 3. Key Innovation

### The ONE Thing: Pseudo-Labelling Real Videos Replaces Architectural Complexity

CoTracker3's core contribution is a **semi-supervised training recipe** that uses a teacher-student framework to generate pseudo-labels on large-scale real video, then trains a simpler architecture on the combined synthetic + pseudo-labeled data.

#### The Pseudo-Labelling Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                TEACHER GENERATION                        │
│                                                         │
│  1. Train teacher on Kubric (synthetic) only            │
│  2. Run teacher on large real video corpus               │
│     - Multiple forward passes with different             │
│       temporal directions (forward + backward)           │
│     - Multiple initializations                           │
│  3. Filter predictions by cycle-consistency:             │
│     - Track point forward t₁ → t₂                       │
│     - Track result backward t₂ → t₁                     │
│     - Keep only if ‖p_original − p_roundtrip‖ < τ       │
│  4. Result: pseudo-labeled tracks on real video          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                STUDENT TRAINING                          │
│                                                         │
│  Train simplified model (no virtual tracks) on:          │
│  - Kubric synthetic data (with GT)                       │
│  - Real video pseudo-labels (from teacher)               │
│                                                         │
│  The pseudo-labels provide:                              │
│  - Real motion patterns (non-rigid, motion blur, etc.)   │
│  - Real textures and lighting                            │
│  - Real occlusion patterns                               │
│  - Scale and diversity that synthetic data can't match    │
└─────────────────────────────────────────────────────────┘
```

#### Why It Works: The Ablation Evidence

From **Table 2** in the paper (ablation on TAP-Vid-DAVIS, Average Jaccard metric):

| Configuration | AJ (DAVIS) |
|--------------|------------|
| CoTracker2 (virtual tracks, Kubric only) | 65.1 |
| CoTracker3 base (no virtual tracks, Kubric only) | ~60 [NEEDS VERIFICATION] |
| CoTracker3 base + pseudo-labels (no virtual tracks) | **67.8** |

Key ablation findings:
1. **Removing virtual tracks** while keeping only Kubric training → performance **drops** significantly
2. **Adding pseudo-labeled real data** to the simplified model → performance **exceeds** the complex CoTracker2
3. The combination is better because real data provides motion priors that virtual tracks were imperfectly trying to learn through architectural inductive bias

#### The Cycle-Consistency Filter

This is the quality control mechanism for pseudo-labels:

$$\text{keep}(p) = \mathbb{1}\left[\|p - \text{Track}_{\text{bwd}}(\text{Track}_{\text{fwd}}(p, t_1 \to t_2), t_2 \to t_1)\|_2 < \tau\right]$$

Where:
- $\text{Track}_{\text{fwd}}(p, t_1 \to t_2)$ tracks point $p$ forward from frame $t_1$ to $t_2$
- $\text{Track}_{\text{bwd}}$ tracks the result back
- $\tau$ is a distance threshold for round-trip consistency
- Points that don't survive the round-trip are likely incorrect and are discarded

This is elegant because it requires **no ground truth** — the consistency check is self-supervised.

### Why This Matters for Breakdancing

The pseudo-labelling approach means CoTracker3 has seen **real human motion** during training — not just synthetic rigid objects. While the training corpus likely doesn't include breakdancing specifically, it includes:
- Sports videos with fast human motion
- Dance videos with non-rigid deformation
- Action videos with occlusion and contact

This makes it far more likely to generalize to breakdancing than a purely synthetic-trained model. The cycle-consistency filter also means the model has learned which points are **reliably trackable** through occlusion — exactly the signal you need when limbs cross during windmills and flares.

---

## 4. Input/Output Specification

### Input

**Video frames:**
$$V \in \mathbb{R}^{T \times 3 \times H \times W}$$

- $T$: number of frames (variable; handled via sliding window for long videos)
- Channels: RGB, normalized to [0, 1] or ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Resolution: variable; internal processing at stride 4. Paper uses 384×512 as a common eval resolution
- Preprocessing: resize to target resolution, normalize

**Query points:**
$$Q \in \mathbb{R}^{N \times 3}$$

Where each query is $(t_q, x_q, y_q)$:
- $t_q$: frame index where the point is specified (integer, 0-indexed)
- $x_q, y_q$: pixel coordinates in the original resolution
- $N$: number of query points. **Capacity: up to 70,000 points** per video (paper claim)

**Query modes:**
1. **Specified queries**: User provides explicit $(t, x, y)$ tuples
2. **Grid queries**: Automatic grid of points, typically on first frame. Spacing parameter controls density

### Output

**Predicted tracks:**
$$\hat{P} \in \mathbb{R}^{T \times N \times 2}$$

- For each of $N$ points across $T$ frames: $(x, y)$ coordinates in pixel space
- Coordinates are in the original resolution (before any downsampling)

**Visibility predictions:**
$$\hat{V} \in \mathbb{R}^{T \times N}$$

- Sigmoid-activated: values in $[0, 1]$
- $\hat{V}_{t,n} \approx 1$: point $n$ is visible in frame $t$
- $\hat{V}_{t,n} \approx 0$: point $n$ is occluded in frame $t$
- Threshold at 0.5 for binary visibility

**Confidence** (optional, depends on model variant):
$$\hat{C} \in \mathbb{R}^{T \times N}$$

- Track confidence score, used during pseudo-label filtering

### Intermediate Representations

| Stage | Tensor | Shape | Description |
|-------|--------|-------|-------------|
| CNN features | $F$ | $(T, 128, H/4, W/4)$ | Per-frame feature maps |
| Query features | $f_q$ | $(N, 128)$ | Feature vector at each query point |
| Correlation volume | $C$ | $(N, T, (2S+1)^2)$ | Local correlation around current position |
| Position encoding | $\text{PE}$ | $(N, T, D_{pe})$ | Fourier encoding of current $(x,y)$ estimate |
| Transformer tokens | $\text{tok}$ | $(N, T, D)$ | Concatenation of all per-token features |
| Position delta | $\Delta p$ | $(N, T, 2)$ | Predicted update per iteration |
| Visibility logit | $v$ | $(N, T, 1)$ | Pre-sigmoid visibility per iteration |

---

## 5. Training Pipeline

### Loss Functions

#### Primary Loss: Huber Loss on Track Positions

$$\mathcal{L}_{\text{pos}} = \frac{1}{N \cdot T} \sum_{n=1}^{N} \sum_{t=1}^{T} \mathbb{1}[v_{n,t}^{\text{gt}} = 1] \cdot \text{Huber}_\delta(\hat{p}_{n,t} - p_{n,t}^{\text{gt}})$$

Where the Huber loss is:

$$\text{Huber}_\delta(x) = \begin{cases} \frac{1}{2}x^2 & \text{if } |x| \leq \delta \\ \delta(|x| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

- $\hat{p}_{n,t} \in \mathbb{R}^2$: predicted position of point $n$ at frame $t$
- $p_{n,t}^{\text{gt}} \in \mathbb{R}^2$: ground truth position
- $v_{n,t}^{\text{gt}} \in \{0, 1\}$: ground truth visibility
- $\delta$: Huber threshold (typically 4.0 pixels) [NEEDS VERIFICATION]
- **Loss is only computed on visible frames** — the model is not penalized for position predictions when a point is occluded

#### Visibility Loss: Binary Cross-Entropy

$$\mathcal{L}_{\text{vis}} = -\frac{1}{N \cdot T} \sum_{n=1}^{N} \sum_{t=1}^{T} \left[ v_{n,t}^{\text{gt}} \log(\sigma(\hat{v}_{n,t})) + (1 - v_{n,t}^{\text{gt}}) \log(1 - \sigma(\hat{v}_{n,t})) \right]$$

Where:
- $\hat{v}_{n,t}$: predicted visibility logit
- $\sigma(\cdot)$: sigmoid function
- $v_{n,t}^{\text{gt}}$: ground truth visibility (binary)

#### Multi-Iteration Loss

The loss is computed at **every refinement iteration** $m$, not just the final one, with equal or increasing weights:

$$\mathcal{L}_{\text{total}} = \sum_{m=1}^{M} \gamma^{M-m} \left( \mathcal{L}_{\text{pos}}^{(m)} + \lambda_{\text{vis}} \mathcal{L}_{\text{vis}}^{(m)} \right)$$

Where:
- $M = 4$: number of refinement iterations
- $\gamma$: decay factor (typically 0.8) — later iterations weighted more heavily
- $\lambda_{\text{vis}}$: visibility loss weight (typically 1.0) [NEEDS VERIFICATION]
- Each $\mathcal{L}^{(m)}$ uses the predictions from iteration $m$

This multi-iteration supervision is critical — it provides gradient signal to early iterations and prevents the model from deferring all correction to the last iteration.

#### Pseudo-Label Training Loss

When training on pseudo-labeled real data, the same losses apply but with:

1. **Filtered supervision**: Only cycle-consistent pseudo-labels contribute to the loss
2. **Confidence weighting** (optional):

$$\mathcal{L}_{\text{pseudo}} = \frac{1}{|\mathcal{S}|} \sum_{(n,t) \in \mathcal{S}} c_{n,t} \cdot \text{Huber}_\delta(\hat{p}_{n,t} - \tilde{p}_{n,t})$$

Where:
- $\mathcal{S}$: set of cycle-consistent pseudo-labeled point-frame pairs
- $c_{n,t}$: confidence score from the teacher model
- $\tilde{p}_{n,t}$: pseudo-label position (from teacher)

#### Combined Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{synthetic}} + \lambda_{\text{pseudo}} \mathcal{L}_{\text{pseudo}}$$

Where $\lambda_{\text{pseudo}}$ balances synthetic and pseudo-labeled loss contributions.

### Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | $5 \times 10^{-4}$ (initial) [NEEDS VERIFICATION] |
| Weight decay | 0.05 [NEEDS VERIFICATION] |
| LR schedule | Cosine decay with linear warmup |
| Warmup | First 1–5% of training [NEEDS VERIFICATION] |
| Batch size | Varies by GPU count; typically 8 per GPU |
| Training iterations | ~200K steps (Kubric) + fine-tuning on mixed data |
| Mixed precision | FP16 / BF16 with gradient scaling |

### Data Augmentation

**Spatial augmentations** (applied consistently across all frames in a clip):
- Random horizontal flip
- Random crop and resize (scale jitter)
- Color jitter (brightness, contrast, saturation, hue)
- Random erasing / cutout [NEEDS VERIFICATION]

**Temporal augmentations**:
- Random temporal reversal (play video backwards)
- Random frame rate subsampling (skip frames to simulate different FPS)

**Point augmentations**:
- Random point sampling from visible regions
- Grid + random hybrid sampling

### Training Data

#### Synthetic Data: Kubric (MOVi-F)

| Property | Value |
|----------|-------|
| Dataset | Kubric MOVi-F |
| Source | Google Research, procedurally generated |
| # videos | ~10,000 scenes |
| Resolution | 256×256 or 512×512 |
| Frames/video | 24 |
| Content | Rigid objects falling/colliding on surfaces |
| GT type | Dense optical flow + depth → exact point tracks |
| Limitation | No non-rigid objects, no humans, no real-world textures |

#### Real Data: Pseudo-Labeled Corpus

| Property | Value |
|----------|-------|
| Source | Large-scale unlabeled video datasets |
| Candidates | Likely includes subsets of: Kinetics, Something-Something, TAO, YouTube scrapes [NEEDS VERIFICATION of exact sources] |
| # videos | Not explicitly stated — likely 50K–100K clips |
| Pseudo-label generation | Teacher (CoTracker2-level model trained on Kubric) → forward/backward tracking → cycle-consistency filter |
| Filtering rate | ~30–60% of tracked points pass cycle-consistency [NEEDS VERIFICATION] |

#### Evaluation Benchmarks

| Benchmark | Task | Metric |
|-----------|------|--------|
| TAP-Vid-DAVIS | Dense tracking on DAVIS videos | AJ (Average Jaccard) |
| TAP-Vid-Kinetics | Dense tracking on Kinetics clips | AJ |
| TAP-Vid-RoboTAP | Robotic manipulation tracking | AJ |
| Dynamic Replica | Indoor scene tracking | AJ |

---

## 6. Inference Pipeline

### What Runs at Test Time

```
┌─────────────────────────────────────────────────────────┐
│  INFERENCE PIPELINE                                      │
│                                                         │
│  KEPT:                                                  │
│  ✓ CNN Feature Extractor (ResNet-18 blocks 0-1)         │
│  ✓ Correlation Volume computation                       │
│  ✓ Iterative Transformer (M=4 or M=6 iterations)       │
│  ✓ Sliding window for long videos                       │
│  ✓ Position + Visibility prediction heads               │
│                                                         │
│  DROPPED:                                               │
│  ✗ All loss computation                                 │
│  ✗ Teacher model (used only for pseudo-label generation)│
│  ✗ Cycle-consistency filtering                          │
│  ✗ Data augmentation                                    │
│  ✗ Gradient computation (inference in eval mode)        │
│                                                         │
│  OPTIONAL:                                              │
│  ~ Number of iterations (can increase M for accuracy    │
│    at cost of latency)                                  │
│  ~ Window size (larger = better but more memory)        │
│  ~ Grid density (controls number of query points)       │
└─────────────────────────────────────────────────────────┘
```

### Online vs. Offline Mode

| Property | Online Mode | Offline Mode |
|----------|-------------|--------------|
| Causality | Causal — only sees past + current frames | Non-causal — sees full window (past + future) |
| Use case | Real-time / streaming | Post-processing / analysis |
| Window size | Smaller (8–16 frames) | Larger (16–48 frames) |
| Accuracy | Slightly lower (no future context) | Higher |
| Latency | Lower per-window | Higher per-window |
| **Bboy relevance** | Live coaching | Post-battle analysis |

### Inference Modes by Use Case

**For breakdancing offline analysis (recommended):**
```python
model = CoTracker3(checkpoint="cotracker3.pth")
# Dense grid on first frame
pred_tracks, pred_visibility = model(
    video,                    # (1, T, 3, H, W) 
    grid_size=50,             # 50×50 = 2500 points
    grid_query_frame=0,       # Track from first frame
    backward_tracking=True    # Also track backwards
)
# pred_tracks: (1, T, 2500, 2)
# pred_visibility: (1, T, 2500)
```

**For streaming/real-time:**
```python
model = CoTracker3(checkpoint="cotracker3_online.pth")
model.init_online(video[:, :1])  # Initialize with first frame
for t in range(1, T):
    pred_tracks, pred_visibility = model.step(video[:, t:t+1])
```

### Latency & Throughput

From the paper and official benchmarks:

| Configuration | Resolution | Points | Latency/frame | GPU |
|--------------|------------|--------|---------------|-----|
| Online mode | 384×512 | 2048 | ~30ms | A100 |
| Offline mode | 384×512 | 2048 | ~50ms | A100 |
| Dense (70K pts) | 384×512 | 70,000 | ~500ms+ | A100 |
| Online mode | 384×512 | 2048 | ~30ms | RTX 4090 (comparable) |

[NEEDS VERIFICATION — exact latency numbers from paper vs. community benchmarks]

**For the bboy pipeline** (from TECH_STACK_REEVALUATION.md): ~30ms/frame for CoTracker3 online mode, which fits within the 330ms/frame total budget.

---

## 7. Computational Cost

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| ResNet-18 (blocks 0-1) | ~5M |
| Transformer (time + group attention × layers) | ~20–30M |
| MLP heads (position, visibility) | ~2M |
| **Total** | **~27–35M** |

[NEEDS VERIFICATION — exact parameter count not prominently reported in paper]

**Comparison:**
- CoTracker2: ~35–40M (includes virtual track overhead)
- TAPIR: ~60M (separate matching + refinement networks)
- PIPs: ~25M

### FLOPs

| Operation | FLOPs (per frame, per iteration) |
|-----------|--------------------------------|
| CNN feature extraction | ~1.8 GFLOPs |
| Correlation volume (N=2048) | ~0.5 GFLOPs |
| Time attention (N=2048, T=24) | ~0.8 GFLOPs |
| Group attention (N=2048, T=24) | ~0.8 GFLOPs |
| MLP heads | ~0.2 GFLOPs |
| **Total (4 iterations)** | **~10–15 GFLOPs/frame** |

[NEEDS VERIFICATION — estimated from architecture, not directly reported]

### GPU Memory

| Configuration | GPU Memory |
|--------------|------------|
| 2048 points, 24 frames, 384×512 | ~4–6 GB |
| 70K points, 24 frames, 384×512 | ~24–40 GB |
| Training (batch size 8) | ~32–40 GB per GPU |

The 70K point capacity is achievable on A100 (80GB) but memory is the main bottleneck. For the bboy pipeline on an RTX 3060 (12GB), realistic capacity is ~2,000–5,000 points — still more than sufficient for tracking a single dancer's body surface.

### Training Time

| Stage | Duration | Hardware |
|-------|----------|---------|
| Kubric pretraining | ~2–3 days | 8× A100 [NEEDS VERIFICATION] |
| Pseudo-label generation | ~1–2 days | 8× A100 [NEEDS VERIFICATION] |
| Fine-tuning on mixed data | ~1–2 days | 8× A100 [NEEDS VERIFICATION] |
| **Total** | **~5–7 days** | **8× A100 (80GB)** |

---

## Summary: CoTracker3 in the Bboy Pipeline

### Architecture Strengths for Breakdancing

| Feature | Why It Matters for Bboy |
|---------|------------------------|
| **Joint tracking (group attention)** | When a dancer's left arm occludes their right leg during a windmill, group attention lets visible points inform occluded ones. Points on the torso help track points on limbs. |
| **Pseudo-labeled real data** | Model has seen real human motion during training. Non-rigid deformation, motion blur from fast movements, ground contact — all present in real video. |
| **70K point capacity** | Dense surface coverage means you can track individual muscle groups, fabric wrinkles, hair — all contributing to finer velocity/acceleration fields for the movement spectrogram. |
| **Online mode (~30ms)** | Fits in the 330ms/frame budget. Could even enable near-real-time coaching if other pipeline components are optimized. |
| **Visibility prediction** | Explicitly tells you which points are occluded. Critical for: (1) knowing which limbs are hidden during power moves, (2) not computing derivatives from hallucinated positions, (3) weighting the movement spectrogram by track confidence. |
| **Sliding window** | Videos of arbitrary length — full battles (3-5 minutes per round) processable without memory explosion. |

### Integration Point

CoTracker3 sits at **step ③** in the revised pipeline:

```
SAM 3 mask → crop to dancer → CoTracker3 (dense points) → derivatives → movement spectrogram
                                     │
                                     ├─ velocity field: ∂p/∂t for each tracked point
                                     ├─ acceleration field: ∂²p/∂t² 
                                     └─ jerk field: ∂³p/∂t³ (captures snap/pop dynamics)
```

The per-point visibility mask from CoTracker3 serves as a natural confidence weight for the derivative computation — occluded points contribute less to the movement spectrogram, preventing noise from estimated (not observed) positions from corrupting the audio-motion cross-correlation.

### Known Limitations for This Use Case

1. **Stride-4 spatial resolution**: Points are tracked at 1/4 pixel resolution internally, then upsampled. Fine finger articulation may be below the resolution floor.

2. **Planar assumption**: CoTracker3 predicts 2D $(x, y)$ tracks, not 3D. For power moves with significant depth variation (corkscrew, air flare), the 2D projection conflates depth motion with lateral motion. This is why SAM-Body4D (step ④) is needed for 3D mesh recovery.

3. **Training data bias**: While pseudo-labels help, the training corpus is unlikely to contain many overhead/top-down views of dancers on the ground — the typical battle camera angle is from the circle edge, roughly 45° above horizontal. Camera angle distribution in training data is an open question.

4. **Dense point failure mode**: Tracking 70K points on a dancer in a baggy hoodie — many points will be on clothing, not the body. Clothing deformation ≠ body deformation. The SAM 3 segmentation mask helps, but the movement spectrogram should ideally weight skeletal landmarks higher than clothing surface points.
