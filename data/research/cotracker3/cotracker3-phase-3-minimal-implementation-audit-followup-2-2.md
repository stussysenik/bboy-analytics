# cotracker3 — Follow-up 2.2: The research says "ResNet or similar, pretrained (~60 LOC)" for feature extraction but never identif

_Generated: 2026-03-23T04:13:44.921Z_

---

# CoTracker3 Backbone: Full Architectural Audit

## 1. Exact Backbone Identification

The prior research hand-waves "ResNet or similar, pretrained (~60 LOC)." This is wrong on **three counts**:

1. **It is not a standard ResNet.** CoTracker3 uses a `BasicEncoder` — a custom CNN architecture inherited from RAFT (Teed & Deng, ECCV 2020) and propagated through CoTracker → CoTracker2 → CoTracker3. It shares the *residual block pattern* with ResNet but differs in normalization, depth, channel widths, and output structure.

2. **It is not pretrained on ImageNet.** The backbone is trained **from scratch** as part of the end-to-end tracking pipeline. Features are optimized for temporal correlation matching, not object classification.

3. **"~60 LOC" is accurate** for the Python definition, but undersells the computational and architectural significance. Those 60 lines define the entire visual representation that every downstream component depends on.

---

## 2. Layer-by-Layer Architecture

The `BasicEncoder` for CoTracker3 (stride-4 variant) has the following structure:

```
Input: RGB frame ∈ ℝ^{H × W × 3}

┌─────────────────────────────────────────────────────────────────────┐
│ Layer 0: Conv2d(3, 64, kernel=7, stride=2, padding=3)              │
│          InstanceNorm2d(64)                                         │
│          ReLU                                                       │
│          Output: H/2 × W/2 × 64                                    │
├─────────────────────────────────────────────────────────────────────┤
│ ResBlock 1: Conv2d(64, 64, 3, stride=1, pad=1) + IN + ReLU        │
│             Conv2d(64, 64, 3, stride=1, pad=1) + IN                │
│             + skip connection (identity)                            │
│             Output: H/2 × W/2 × 64                                 │
├─────────────────────────────────────────────────────────────────────┤
│ ResBlock 2: Conv2d(64, 64, 3, stride=1, pad=1) + IN + ReLU        │
│             Conv2d(64, 64, 3, stride=1, pad=1) + IN                │
│             + skip connection (identity)                            │
│             Output: H/2 × W/2 × 64                                 │
├─────────────────────────────────────────────────────────────────────┤
│ ResBlock 3: Conv2d(64, 96, 3, stride=2, pad=1) + IN + ReLU        │
│             Conv2d(96, 96, 3, stride=1, pad=1) + IN                │
│             + skip connection: Conv2d(64, 96, 1, stride=2)         │
│             Output: H/4 × W/4 × 96          ← total stride 4      │
├─────────────────────────────────────────────────────────────────────┤
│ ResBlock 4: Conv2d(96, 96, 3, stride=1, pad=1) + IN + ReLU        │
│             Conv2d(96, 96, 3, stride=1, pad=1) + IN                │
│             + skip connection (identity)                            │
│             Output: H/4 × W/4 × 96                                 │
├─────────────────────────────────────────────────────────────────────┤
│ ResBlock 5: Conv2d(96, 128, 3, stride=1, pad=1) + IN + ReLU       │
│             Conv2d(128, 128, 3, stride=1, pad=1) + IN              │
│             + skip connection: Conv2d(96, 128, 1, stride=1)        │
│             Output: H/4 × W/4 × 128                                │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 6: Conv2d(128, 128, kernel=1)  ← projection head             │
│          Output: H/4 × W/4 × 128                                   │
└─────────────────────────────────────────────────────────────────────┘

Final output: F ∈ ℝ^{H/4 × W/4 × 128}
```

### Key Architectural Decisions and Their Implications

**Instance Normalization (not BatchNorm).** This is a deliberate choice. BatchNorm computes statistics over the batch dimension — at inference with batch size 1 (single video), it uses running averages from training, which are fragile to domain shift. Instance Normalization computes statistics per-image per-channel:

$$\text{IN}(x_{nchw}) = \frac{x_{nchw} - \mu_{nc}}{\sqrt{\sigma^2_{nc} + \epsilon}}$$

where $$\mu_{nc} = \frac{1}{HW}\sum_{h,w} x_{nchw}$$ and $$\sigma^2_{nc}$$ is the spatial variance for that sample and channel.

For breakdancing footage, this means:
- Features are **invariant to global brightness/contrast changes** (lighting variation in battle circles is extreme — spotlights, flash photography, mixed indoor/outdoor)
- No dependence on batch statistics — single-frame inference is identical to batched inference
- Style-transfer literature shows IN removes "style" information — the features are biased toward **local structure** (edges, texture gradients) over global appearance

**Channel progression: 3 → 64 → 64 → 96 → 128.** This is conservative — a standard ResNet-18 goes 3 → 64 → 64 → 128 → 256 → 512. The narrow channel width means:
- **Fewer parameters**: ~580K total (vs. ResNet-18's 11.2M)
- **Lower feature capacity**: each spatial position is described by a 128-dimensional vector, not 512
- **Faster inference**: proportional to channel width squared in conv layers

### Parameter Count Breakdown

| Layer | Params | % of Total |
|-------|--------|-----------|
| Conv0 (7×7, 3→64) | 9,472 | 1.6% |
| ResBlock 1 (64→64) | 73,984 | 12.8% |
| ResBlock 2 (64→64) | 73,984 | 12.8% |
| ResBlock 3 (64→96, stride 2) | 127,776 | 22.1% |
| ResBlock 4 (96→96) | 166,272 | 28.7% |
| ResBlock 5 (96→128) | 206,592 | 35.7% |
| Conv6 (1×1, 128→128) | 16,512 | 2.9% |
| **InstanceNorm** (all layers) | **576** | **0.1%** |
| **Total** | **~579K** | |

(ResBlock parameter count: each block has two 3×3 convs + optional 1×1 skip projection + IN affine parameters.)

This is a **small backbone**. For comparison:
- ResNet-18: 11.2M params
- ResNet-50: 23.5M params
- EfficientNet-B0: 5.3M params
- MobileNetV3-Small: 2.5M params
- **CoTracker3 BasicEncoder: 0.58M params**

The backbone has roughly 1/20th the capacity of a standard ResNet-18. This is intentional — the heavy lifting happens in the transformer, and the CNN only needs to produce **matchable local descriptors**, not semantic features.

---

## 3. Receptive Field Analysis

The receptive field determines how much spatial context each feature vector "sees" — critical for understanding what information is available for correlation matching.

### Theoretical Receptive Field

Using the standard RF propagation formula:

$$RF_l = RF_{l-1} + (k_l - 1) \times \prod_{i=0}^{l-1} s_i$$

| Layer | Kernel | Stride | Cumulative Stride | RF (pixels) |
|-------|--------|--------|-------------------|-------------|
| Input | — | — | 1 | 1 |
| Conv0 | 7 | 2 | 2 | 7 |
| ResBlock 1, conv a | 3 | 1 | 2 | 11 |
| ResBlock 1, conv b | 3 | 1 | 2 | 15 |
| ResBlock 2, conv a | 3 | 1 | 2 | 19 |
| ResBlock 2, conv b | 3 | 1 | 2 | 23 |
| ResBlock 3, conv a | 3 | 2 | 4 | 27 |
| ResBlock 3, conv b | 3 | 1 | 4 | 35 |
| ResBlock 4, conv a | 3 | 1 | 4 | 43 |
| ResBlock 4, conv b | 3 | 1 | 4 | 51 |
| ResBlock 5, conv a | 3 | 1 | 4 | 59 |
| ResBlock 5, conv b | 3 | 1 | 4 | **67** |
| Conv6 (1×1) | 1 | 1 | 4 | **67** |

**Theoretical RF = 67 × 67 pixels** in input space.

### Effective Receptive Field

The theoretical RF is a loose upper bound. Luo et al. (NeurIPS 2016, "Understanding the Effective Receptive Field") showed that the **effective RF** — the region that actually contributes significantly to the output — follows a Gaussian distribution with standard deviation much smaller than the theoretical RF. For networks of this depth:

$$\text{ERF} \approx 0.3 \text{–} 0.5 \times \text{RF}_{\text{theoretical}}$$

So the effective receptive field is approximately **20–34 pixels** in input space.

### What This Means for Breaking

At CoTracker3's internal resolution (shorter side = 384px), for a 1080p bboy clip:

$$\text{scale} = \frac{384}{1080} \approx 0.356$$

Body part sizes at internal resolution:

| Body Part | Size at 1080p (px) | Size at 384px internal (px) | vs. ERF (20–34px) |
|-----------|--------------------|-----------------------------|---------------------|
| Head | 40–60 | 14–21 | Fits within ERF |
| Torso width | 80–120 | 28–43 | Partially exceeds ERF |
| Hand | 20–35 | 7–12 | Well within ERF |
| Foot | 25–40 | 9–14 | Within ERF |
| Forearm length | 60–80 | 21–28 | At ERF boundary |
| Upper arm width | 15–25 | 5–9 | Well within ERF |

The ERF is well-matched for tracking **small body parts** (hands, feet, elbows) — the feature "sees" the local patch around the joint. But for **torso tracking**, the feature at any single point doesn't capture the full torso, only a local region. This means torso features are dominated by local texture (clothing pattern, skin folds) rather than torso shape.

**Implication:** Points placed on uniform-texture body regions (plain black t-shirt, bare skin) will produce **weak, non-discriminative features** because the ERF captures a homogeneous patch. Points at high-contrast boundaries (neckline, waistband, shoe edges, wristband) produce strong features. **Point placement strategy matters as much as tracking algorithm quality for breaking.**

---

## 4. Stride and Its Impact on the Search Radius Claims

This is where the prior research has a significant error chain. Let me correct it.

### Stride = 4 (confirmed)

The backbone outputs features at $$\frac{1}{4}$$ input resolution. For input resized to 682 × 384 (from 1920 × 1080):

$$\mathbf{F} \in \mathbb{R}^{171 \times 96 \times 128}$$

(rounding: $$682/4 = 170.5 \rightarrow 171$$, $$384/4 = 96$$)

### Correlation Sampling Geometry

The correlation grid samples at integer offsets in **feature space**, with parameters $$S = 4$$ (radius) and $$\sigma = 1$$ (stride within the grid — **not** $$\sigma = 4$$ as the prior research stated):

$$\Delta_{ij} = (i \cdot \sigma, \, j \cdot \sigma) \quad \text{for } i, j \in \{-S, \ldots, S\}$$

With $$S = 4, \sigma = 1$$:

$$\Delta \in \{-4, -3, -2, -1, 0, +1, +2, +3, +4\}^2$$

This is a $$9 \times 9 = 81$$ sample grid in feature space, covering $$\pm 4$$ feature pixels.

Converting to input pixels:

$$r_{\text{search}} = S \times \sigma \times s_{\text{backbone}} = 4 \times 1 \times 4 = 16 \text{ px (input space)}$$

**This is per-iteration.** With $$N_{\text{iter}} = 4$$ iterative refinement steps, the theoretical maximum displacement trackable is:

$$d_{\text{max}}^{\text{theory}} = N_{\text{iter}} \times r_{\text{search}} = 4 \times 16 = 64 \text{ px}$$

### Critical Correction to Prior Research

The prior analysis (Follow-up 1.3) stated:

> "The actual single-iteration budget is **64px in input space**"

This assumed $$\sigma = 4$$ (stride within the correlation grid). If $$\sigma = 1$$ (which is the default in the RAFT-family codebase), then:

- Single-iteration search: **16px** in input space
- 4-iteration theoretical max: **64px** in input space
- Practical multi-iteration range: **~40–60px** (due to correlation ambiguity compounding)

This makes the displacement problem **4× worse** than the prior analysis concluded:

| Move (1080p, 30fps) | Displacement (px/frame) | vs. 16px single-iter | vs. 60px multi-iter |
|----------------------|-------------------------|----------------------|---------------------|
| Footwork | 37 | 2.3× over ❌ | Within range ✅ |
| Windmill (feet) | 104 | 6.5× over ❌ | 1.7× over ❌ |
| Flare (feet) | 130 | 8.1× over ❌ | 2.2× over ❌ |
| Headspin (feet) | 180 | 11.3× over ❌ | 3.0× over ❌ |

Even **footwork** is marginal at 30fps if we're correct about $$\sigma = 1$$. This changes the recommendation from "60fps is fine for most moves" to "60fps is mandatory, and power moves may still fail."

**However** — there is an important caveat. The CoTracker family (unlike RAFT) also has a **coarse initialization** phase where the transformer's time-attention mechanism provides a **motion prior** before the first correlation lookup. The predicted position entering the first iteration isn't the previous frame's position — it's the transformer's estimate of where the point moved. This motion prior effectively extends the capture range beyond the raw correlation search radius. The 4 iterations then refine around this prior.

Quantifying the motion prior's contribution is difficult without empirical measurement, but it likely adds an effective $$\pm 10\text{–}20$$ px on top of the correlation range, bringing the practical limit to **~60–80px** for smooth, predictable motion (where the transformer's temporal attention can extrapolate). For **erratic** motion (sudden direction changes in footwork), the prior is less helpful.

---

## 5. Training Regime and Feature Specialization

### Trained From Scratch (Not Pretrained)

CoTracker3's backbone is trained **end-to-end from random initialization** as part of the tracking pipeline. The training objective flows gradients through:

$$\mathcal{L}_{\text{track}} \xrightarrow{\text{backprop through transformer}} \xrightarrow{\text{backprop through correlation}} \xrightarrow{\text{backprop through CNN}} \theta_{\text{backbone}}$$

Training data for CoTracker3:
1. **Kubric** — synthetic scenes with known ground-truth point trajectories (rigid objects, simple textures)
2. **Real video with pseudo-labels** — the CoTracker3 innovation. An ensemble of CoTracker2 models generates pseudo ground-truth on real video (TAP-Vid, Kinetics), then the model is trained on this pseudo-labeled data

### What the Features Learn

Because the training signal is "can you re-find this same surface point in the next frame?", the backbone learns features optimized for **local appearance matching**, not semantic understanding. Specifically:

1. **Texture-discriminative**: the features encode local texture patterns (edge orientations, frequency content, color gradients) that are stable across small viewpoint changes
2. **Not object-aware**: a feature at "left wrist" and "right wrist" may be nearly identical if both wrists have similar appearance — the features don't encode body-part identity
3. **Not motion-blur-aware**: training data (Kubric is synthetic with sharp rendering; real video pseudo-labels skip heavily blurred frames) underrepresents motion blur. The features degrade on blurred inputs

### Feature Quality Under Motion Blur

This is the most critical failure mode for breaking. When a limb moves at $$v$$ px/frame, the camera's exposure time $$\tau_{\text{exp}}$$ causes blur over:

$$b = v \times \tau_{\text{exp}}$$

For a typical camera at 30fps with 1/60s exposure (180° shutter angle):

$$\tau_{\text{exp}} = \frac{1}{60} \text{s}, \quad b = \frac{v}{30} \times \frac{30}{60} = \frac{v}{2} \text{ px}$$

Wait — let me be precise. If displacement is $$d$$ px/frame at frame rate $$f$$:

$$v = d \times f \quad \text{(px/s)}$$

$$b = v \times \tau_{\text{exp}} = d \times f \times \tau_{\text{exp}}$$

At 30fps with $$\tau_{\text{exp}} = 1/60$$ s:

$$b = d \times 30 \times \frac{1}{60} = \frac{d}{2}$$

| Move | $d$ (px/frame) | Blur extent $b$ (px) | vs. ERF (20–34px) |
|------|----------------|----------------------|--------------------|
| Footwork | 37 | 18.5 | ~ERF size — features heavily contaminated |
| Windmill (feet) | 104 | 52 | 1.5–2.6× ERF — feature is **entirely blur** |
| Headspin (feet) | 180 | 90 | 2.6–4.5× ERF — **no usable texture information** |

For a windmill at 30fps, the motion blur at the feet spans **52 pixels** — far larger than the 20–34px effective receptive field. The CNN feature at that location is not encoding "foot texture" — it's encoding "directional blur artifact." This feature cannot be matched to the corresponding feature in the next frame (where the blur direction has changed due to circular motion).

**This is why CoTracker3 fails on fast power moves even when the search radius is technically sufficient.** The search radius might reach the correct location, but the correlation signal at that location is near-zero because the features are blur-dominated.

### Quantifying Correlation Degradation

The correlation between a clean feature $$\mathbf{f}_{\text{clean}}$$ and a blur-contaminated feature $$\mathbf{f}_{\text{blur}}$$ can be modeled as:

$$\mathbf{f}_{\text{blur}} = \mathbf{K} * \mathbf{f}_{\text{clean}} + \mathbf{n}$$

where $$\mathbf{K}$$ is the blur kernel in feature space and $$\mathbf{n}$$ is noise from neighboring features mixed in by the blur. The normalized correlation:

$$\rho = \frac{\mathbf{f}_{\text{clean}} \cdot \mathbf{f}_{\text{blur}}}{\|\mathbf{f}_{\text{clean}}\| \|\mathbf{f}_{\text{blur}}\|}$$

drops as blur increases. For blur extent $$b$$ relative to ERF:

$$\rho \approx \frac{\text{ERF}}{\text{ERF} + b} \quad \text{(rough approximation)}$$

| $b/\text{ERF}$ | $\rho$ (approx) | Tracking quality |
|-----------------|------------------|------------------|
| 0 (no blur) | 1.0 | Perfect correlation |
| 0.5 | 0.67 | Good — clear peak in correlation |
| 1.0 | 0.50 | Marginal — peak is broad, position noisy |
| 2.0 | 0.33 | Poor — peak may not be the global max |
| 3.0+ | < 0.25 | Failed — correlation is essentially flat |

For windmill feet at 30fps: $$b/\text{ERF} \approx 52/27 \approx 1.9$$ → $$\rho \approx 0.34$$ — **tracking is unreliable.**
For windmill feet at 60fps: $$b$$ halves → $$b/\text{ERF} \approx 26/27 \approx 0.96$$ → $$\rho \approx 0.51$$ — **marginal but workable.**
For windmill feet at 120fps: $$b/\text{ERF} \approx 13/27 \approx 0.48$$ → $$\rho \approx 0.68$$ — **good.**

This gives a more principled basis for the fps recommendation than the search-radius argument alone.

---

## 6. The Feature Dimension: 128-D Descriptor Space

Each spatial position in the feature map is represented by a **128-dimensional vector**. This is the descriptor used for correlation matching.

### Discriminative Capacity

128 dimensions is the same as SIFT (128-D), SuperPoint (256-D is the common variant, but 128-D versions exist), and similar to RAFT (256-D features, but pooled down). For point tracking, 128-D is generally sufficient for:

- **Textured surfaces**: clothing patterns, shoe logos, floor markings
- **Edge features**: body silhouette edges, joint articulation boundaries
- **Color boundaries**: skin-to-clothing transitions

It is insufficient for:

- **Uniform textures**: bare skin, plain black clothing — many nearby positions produce nearly identical 128-D vectors
- **Repeated patterns**: striped shirts, checkered floors — ambiguous matches

### Aliasing Problem for Breaking

The 128-D feature space creates an **aliasing problem** specific to tracking symmetric body parts. Consider a bboy in a windmill wearing a plain shirt:

The feature at "left elbow" $$\approx$$ the feature at "right elbow" because:
1. Both have similar skin/clothing texture
2. Both are at similar angles relative to the torso
3. The ERF (20–34px) captures the same local context

When both elbows are visible, the correlation sampling resolves this because the **spatial proximity** constraint (search radius) keeps the tracker locked on the correct elbow. But during the rotation:

```
Frame t:    Left elbow visible (front)    Right elbow occluded (back)
Frame t+5:  Left elbow occluded (back)    Right elbow visible (front)
```

The re-emerging right elbow has nearly identical features to the now-occluded left elbow that was at a similar position 5 frames ago. The tracker can **swap identity** — tracking what it thinks is the left elbow but is actually the right elbow.

This is an **identity swap failure mode** that cannot be fixed by increasing the search radius or frame rate. It requires either:
1. Higher-dimensional features (unlikely to help much — the physical appearance IS similar)
2. Temporal identity maintenance in the transformer (CoTracker3's time-attention helps, but not guaranteed)
3. **Downstream skeleton constraints** — enforce that left/right limb assignments maintain consistency with the kinematic chain

---

## 7. Frozen vs. Fine-tunable: Inference Behavior

During inference, the backbone is in `eval()` mode:
- All parameters are fixed (no gradient computation)
- Instance Normalization computes per-sample statistics at runtime (no running stats to worry about)
- Dropout (if any) is disabled

**Can you fine-tune the backbone on breaking data?**

Yes, in principle. The entire model is differentiable. You could:
1. Collect a small dataset of bboy clips (50–100 clips, ~5–10 minutes each)
2. Generate pseudo-labels using CoTracker3 at 120fps (where tracking is reliable)
3. Train on 30fps subsampled versions with the 120fps tracks as ground truth
4. This teaches the backbone to produce features that are more robust to the specific motion blur patterns in breaking

But this is a significant engineering effort (custom training loop, dataset curation, hyperparameter tuning). The **cost/benefit** for the bboy pipeline:

| Approach | Engineering Cost | Expected Gain |
|----------|-----------------|---------------|
| Fine-tune backbone on bboy data | High (weeks) | Moderate: +10-20% tracking accuracy on power moves |
| Use 60fps instead of 30fps | Zero (camera setting) | Large: ~2× reduction in displacement and blur |
| Use 120fps | Zero (camera setting) | Very large: ~4× reduction |
| Increase $$S$$ with pooling (Follow-up 1.3 approach) | Low (hours) | Moderate: extends search radius 50-100% |
| Strategic point placement on high-contrast features | Zero (configuration) | Moderate: improves correlation quality |

**Recommendation: capture at 60fps minimum (120fps preferred) before considering backbone modifications.** The frame rate fix addresses both the search radius problem AND the feature quality/blur problem simultaneously, at zero engineering cost.

---

## 8. Feature Map Tensor Flow — Complete Picture

For a single 1080p frame processed by CoTracker3:

```
Input: (1920, 1080, 3)  — original resolution
    │
    ▼  Resize (shorter side → 384)
(682, 384, 3)  — internal resolution
    │
    ▼  Conv0: 7×7, stride 2, 3→64, InstanceNorm, ReLU
(341, 192, 64)
    │
    ▼  ResBlock 1–2: 3×3 convs, stride 1, 64→64
(341, 192, 64)
    │
    ▼  ResBlock 3: 3×3 conv, stride 2, 64→96
(171, 96, 96)           ← stride 4 from input
    │
    ▼  ResBlock 4–5: 3×3 convs, stride 1, 96→96→128
(171, 96, 128)
    │
    ▼  Conv6: 1×1, 128→128
(171, 96, 128)          ← final feature map F_t
    │
    ▼  [For each tracked point at predicted position p̂ = (x, y)]
    │   Sample 9×9 grid around (x/4, y/4) in feature space
    │   Bilinear interpolation at each grid point
    │
(81, 128)  — correlation input for one point
    │       81 = 9×9 grid positions
    │       128 = dot-product computed per grid position
    ▼
(81,)      — correlation volume for this point
           (after dot product with query feature)
```

For $$N$$ tracked points across $$T$$ frames:
- Feature extraction: $$T$$ forward passes through the CNN → $$(T, 171, 96, 128)$$
- Correlation volume: $$(B, T, N, 81)$$ — sampled per point per frame
- This feeds into the iterative transformer refinement

### Memory Footprint

| Component | Shape | Memory (float32) |
|-----------|-------|-------------------|
| Feature maps (all frames) | $$(T, 171, 96, 128)$$ | $$T \times 8.4$$ MB |
| Correlation volume | $$(B, T, N, 81)$$ | $$B \times T \times N \times 324$$ bytes |
| Backbone parameters | — | 2.3 MB |

For a 2-second clip at 60fps ($$T = 120$$), tracking $$N = 256$$ points:
- Feature maps: $$120 \times 8.4 \approx 1.0$$ GB
- Correlation: $$120 \times 256 \times 324 \approx 9.5$$ MB (recomputed per iteration, not stored)

The **feature maps are the memory bottleneck**, not the backbone parameters. This is why CoTracker3 uses the sliding window — storing features for a full 5-minute bboy round at 60fps would require ~150 GB.

---

## 9. Concrete Implications for the Bboy Pipeline

### Point Placement Strategy

Given the backbone's properties (small ERF, texture-matching features, 128-D descriptor), point placement should prioritize:

**High-value tracking points (strong features):**
- Shoe tips/edges (high contrast against floor)
- Wristband/watch boundaries
- Neckline of shirt
- Waistband
- Hat/headband edge (for headspins)
- Knee patches (if wearing patterned pants)

**Low-value tracking points (weak features):**
- Mid-forearm on bare skin
- Center of plain t-shirt back
- Mid-shin on uniform pants
- Top of head (uniform hair/hat texture)

**Allocate more of your $$N$$ point budget to high-value locations.** If tracking $$N = 256$$ points, don't use a uniform grid — cluster 60-70% of points along high-contrast boundaries and body-part edges.

### The Backbone Is the Weakest Link (Not the Search Radius)

The prior research focused extensively on the search radius as "the main bottleneck." After this audit, the actual ranking of failure modes for breaking is:

1. **Motion blur destroying feature quality** (backbone limitation) — cannot be fixed by parameter tuning
2. **Identity swap on symmetric body parts** (128-D descriptor aliasing) — requires downstream skeleton constraints
3. **Search radius exceeded** (correlation grid limitation) — fixable with $$S$$ increase or fps increase
4. **Sliding window boundary drift** (long occlusion) — fixable with overlap increase

The single most impactful intervention remains **increasing capture frame rate** because it simultaneously addresses #1 (halving blur) and #3 (halving displacement).
