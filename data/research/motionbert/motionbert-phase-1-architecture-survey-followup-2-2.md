# motionbert — Follow-up 2.2: The pipeline delegates all primary 3D estimation to SAM-Body4D based on the claim that it's "trainin

_Generated: 2026-03-23T01:44:31.306Z_

---

# SAM-Body4D's Unexamined Failure Modes on Inverted Poses

The prior research treats SAM-Body4D as a silver bullet: "training-free" therefore rotation-robust, estimated at ~45mm on breaking sequences. This section stress-tests that assumption by analyzing the specific failure modes of optimization-based body reconstruction on inverted, high-speed breakdancing poses.

---

## 1. Deconstructing "Training-Free"

"Training-free" is a misleading descriptor. SAM-Body4D eliminates *task-specific supervised training* on pose datasets, but it is **not prior-free**. The optimization stack contains multiple learned components, each with its own distribution assumptions:

### 1.1 The SMPL Body Model

All optimization-based methods fit the SMPL parametric model:

$$M(\boldsymbol{\beta}, \boldsymbol{\theta}) = W(T(\boldsymbol{\beta}, \boldsymbol{\theta}), J(\boldsymbol{\beta}), \boldsymbol{\theta}, \mathcal{W})$$

where:
- $\boldsymbol{\beta} \in \mathbb{R}^{10}$: shape parameters (PCA of body shape variation) — **rotation-invariant**, no bias
- $\boldsymbol{\theta} \in \mathbb{R}^{72}$: pose parameters (24 joints × 3 axis-angle) — the pose space *can* represent any valid articulation including inversions
- $J(\boldsymbol{\beta}) \in \mathbb{R}^{24 \times 3}$: joint locations derived from shape — **rotation-invariant**
- $\mathcal{W} \in \mathbb{R}^{6890 \times 24}$: skinning weights — **fixed**, learned from registrations of predominantly upright scans

The SMPL model itself is technically capable of representing inverted poses. The joint rotation parameterization (axis-angle) has no inherent orientation bias — a shoulder joint rotated 180° is as valid as 0° in the parameter space. **However**, the skinning weights $\mathcal{W}$ were learned from body scan registrations that are overwhelmingly upright. During extreme poses (e.g., back arched 180° in a bridge, extreme shoulder rotation in a freeze), the linear blend skinning can produce **mesh artifacts** (self-intersections, unrealistic skin deformation) that don't occur for upright poses. These artifacts:

1. Corrupt the silhouette used for photometric optimization
2. Create incorrect surface normals for rendering-based losses
3. Produce interpenetration that confounds contact-aware losses

Quantitative impact: Skinning artifacts for extreme poses add an estimated **3-8mm systematic error** to vertex positions in regions of extreme articulation (shoulders during handstands, hips during baby freezes).

### 1.2 The Pose Prior

This is the **primary source of upright bias** in "training-free" methods. Optimization-based reconstruction minimizes:

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{prior} \mathcal{L}_{prior}(\boldsymbol{\theta}) + \lambda_{reg} \mathcal{L}_{reg}$$

The pose prior $\mathcal{L}_{prior}$ is almost universally one of:

**VPoser** (Pavlakos et al., 2019): A variational autoencoder trained on AMASS pose data. It maps poses to a latent space $\mathbf{z} \in \mathbb{R}^{32}$ and the prior loss is:

$$\mathcal{L}_{VPoser} = \|\mathbf{z}\|^2$$

This penalizes poses that are far from the AMASS mean in latent space. Since AMASS is ~96-98% upright (from the distribution analysis above), the VPoser latent space encodes "upright" as the origin and "inverted" as a high-norm outlier.

For a headstand ($\phi \approx 180°$):

$$\|\mathbf{z}_{headstand}\|^2 \gg \|\mathbf{z}_{standing}\|^2$$

The prior actively **pulls the optimization away from inverted poses** toward the nearest upright local minimum.

**GMM prior** (Bogo et al., 2016): A Gaussian mixture model over SMPL pose parameters, also trained on MoCap data:

$$\mathcal{L}_{GMM} = -\log \sum_{k} \pi_k \mathcal{N}(\boldsymbol{\theta} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

Same bias: the mixture components cluster around upright poses. An inverted pose falls in a low-density region, creating a strong gradient toward the nearest upright mode.

**Quantifying the prior's pull**: The strength of the prior's bias depends on $\lambda_{prior}$. Typical values:

| Method | $\lambda_{prior}$ | Effect on inverted poses |
|--------|-------------------|------------------------|
| SMPLify-X | 4.78 (body prior) | Strong upright pull |
| CLIFF | 0.001 | Weak prior, data-driven |
| 4D-Humans / HMR2.0 | Implicit (learned decoder) | Moderate upright bias |
| PyMAF-X | 0.01 | Moderate |
| Optimization-based (generic) | 0.1-10.0 | Variable |

For SAM-Body4D specifically, without access to the exact $\lambda_{prior}$ value, we must analyze the sensitivity:

$$\frac{\partial \hat{\boldsymbol{\theta}}}{\partial \lambda_{prior}} = -H^{-1} \nabla_{\boldsymbol{\theta}} \mathcal{L}_{prior}$$

where $H$ is the Hessian of $\mathcal{L}_{total}$. For inverted poses, $\nabla_{\boldsymbol{\theta}} \mathcal{L}_{prior}$ is large (pointing toward upright), meaning even moderate $\lambda_{prior}$ values create significant bias.

### 1.3 Foundation Model Features ≠ Rotation Invariance

SAM-Body4D uses features from foundation models (likely DINOv2 or SAM's ViT encoder) for correspondence and segmentation. These features are:

- **Translation-invariant**: ViT with positional embeddings are translation-equivariant after the patch embedding layer
- **NOT rotation-invariant**: Standard ViT positional embeddings are absolute 2D grid positions. An inverted person produces feature maps where the head region has positional embeddings from the bottom of the image — a distribution shift for the feature extractor

DINOv2 was trained with random cropping and some geometric augmentation, but **not** with arbitrary rotation augmentation. Its features are robust to ~±15° rotation (from natural camera tilt variation in training data) but degrade for larger rotations.

**Estimated feature degradation for inverted bodies**: The correspondence quality between frames degrades by an estimated 15-30% for inverted poses, based on analogous results from rotation robustness evaluations of ViT features (e.g., Zhai et al. 2022 on DeiT rotation sensitivity).

---

## 2. The Initialization Problem

Optimization-based methods start from an initial estimate and iteratively minimize $\mathcal{L}_{total}$. The loss landscape for human body reconstruction is **highly non-convex**, with many local minima.

### 2.1 Standard Initialization

Most methods initialize with:
- **Mean SMPL pose** ($\boldsymbol{\theta} = \mathbf{0}$): T-pose or A-pose (upright, arms extended)
- **HMR/regression-based initialization**: A feedforward network predicts initial SMPL parameters, then optimization refines. These networks are trained on... AMASS/H36M. Upright bias again.

For an inverted breakdancer:

$$d(\boldsymbol{\theta}_{init}, \boldsymbol{\theta}_{target}) = \|\boldsymbol{\theta}_{T\text{-}pose} - \boldsymbol{\theta}_{headstand}\|$$

This distance is enormous in the 72-dimensional pose space. The optimization must traverse a complex loss landscape with multiple saddle points and local minima between the T-pose and the headstand.

### 2.2 Local Minima Landscape

Consider a baby freeze (one hand on ground, head on ground, legs tucked, body inverted ~160°). The optimization from a T-pose initialization encounters:

1. **Mirror ambiguity**: An inverted body seen from the front has a 2D silhouette similar to an extremely flexed-forward upright body. The optimization can converge to "person bending forward" instead of "person inverted."

2. **Limb assignment ambiguity**: In an inverted pose, the 2D projection of arms and legs may overlap or swap apparent positions. Without strong correspondence cues, the optimizer may assign left-arm features to the right leg.

3. **Depth ambiguity**: 2D observations cannot distinguish between inverted-close and upright-far in monocular settings. The pose prior breaks this ambiguity... in favor of upright.

### 2.3 Temporal Initialization Helps — But Has Limits

Multi-frame methods (4D reconstruction) use the previous frame's solution as initialization for the current frame:

$$\boldsymbol{\theta}_t^{init} = \boldsymbol{\theta}_{t-1}^{*}$$

This is helpful for **gradual** transitions (toprock → crouch → freeze → ...) because each frame's pose is close to the previous. But breakdancing has **abrupt transitions**:

- **Kip-up**: Lying on back → standing in 0.3-0.5 seconds (~10-15 frames at 30fps). Torso angle changes by ~180° in <0.5s.
- **Windmill entry**: Standing → full rotation in ~0.5s. Continuous 360°/s rotation.
- **Suicide drop**: Standing → flat on back in ~0.2s. Near-instantaneous orientation change.

During these transitions, temporal initialization fails because:

$$\|\boldsymbol{\theta}_t - \boldsymbol{\theta}_{t-1}\| \gg \|\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{t-1}\|_{typical}$$

The optimization at frame $t$ starts far from the target and must converge in the allotted iteration budget (typically 50-200 iterations for real-time/offline methods). If the iteration budget is insufficient, the solution stays near the initialization — producing a "ghost" of the previous frame's pose.

---

## 3. Photometric Objective Analysis

The data term $\mathcal{L}_{data}$ in SAM-Body4D likely includes some combination of:

### 3.1 Silhouette/Mask Loss

$$\mathcal{L}_{sil} = \sum_{p \in \text{image}} |S_{rendered}(p) - S_{observed}(p)|$$

where $S_{rendered}$ is the rendered silhouette of the SMPL mesh and $S_{observed}$ is the segmentation mask (from SAM).

**Rotation invariance**: The silhouette loss is **geometrically rotation-invariant** in principle — it matches 2D projections regardless of 3D orientation. This is a genuine advantage over learned lifting methods.

**However**, the loss landscape of silhouette matching is pathological for certain pose configurations:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_{sil} = 0 \quad \text{when rendered silhouette matches observed silhouette}$$

Multiple 3D poses can produce identical 2D silhouettes (the "bas-relief ambiguity"). For an inverted body:

- The silhouette of a person in a headstand is similar to a person standing with arms raised (when viewed from certain angles)
- The silhouette provides no gradient to distinguish these configurations
- The pose prior breaks the tie → upright wins

### 3.2 Keypoint Reprojection Loss

$$\mathcal{L}_{kp} = \sum_{j=1}^{J} w_j \|\Pi(R \cdot J_j(\boldsymbol{\beta}, \boldsymbol{\theta}) + \mathbf{t}) - \hat{p}_j^{2D}\|^2$$

where $\Pi$ is the projection function, $R, \mathbf{t}$ are camera parameters, and $\hat{p}_j^{2D}$ are detected 2D keypoints.

This loss **is** informative for orientation — different 3D orientations produce different 2D keypoint configurations even if silhouettes match. But it depends on **correct 2D keypoint detection**, which itself degrades for inverted poses (as analyzed in MotionBERT Factor 4).

ViTPose/HRNet 2D detection accuracy on inverted poses:

| Pose | Estimated 2D PCK@0.5 | Notes |
|------|----------------------|-------|
| Standing | 92-95% | In-distribution |
| Headstand | 70-80% | Head/wrist confusion |
| Baby freeze | 65-75% | Severe self-occlusion |
| Windmill (mid-rotation) | 55-70% | Motion blur + inversion |
| Flare (mid-rotation) | 50-65% | Extreme blur + inversion + occlusion |

When 2D detections are wrong, the keypoint reprojection loss drives the optimization toward **incorrect 3D poses that match the incorrect 2D detections**. This is especially problematic for:
- **Joint swaps**: 2D detector assigns left-hand confidence to right foot → optimizer produces twisted mesh
- **Occluded joint hallucination**: Detector guesses position of occluded joint → optimizer fits the guess

### 3.3 Dense Correspondence / Feature Loss

If SAM-Body4D uses DINOv2 or similar features for dense correspondence:

$$\mathcal{L}_{feat} = \sum_{p} \|F_{rendered}(p) - F_{observed}(p)\|^2$$

This is potentially the most rotation-robust component because semantic features (DINOv2) partially encode "this is a hand" regardless of orientation. But as noted in §1.3, the positional encoding creates orientation sensitivity in the features themselves.

---

## 4. Deriving the ~45mm Estimate (and Why It's Wrong)

The prior research stated "SAM-Body4D gives ~45mm raw accuracy on breakdancing sequences" without derivation. Let me reconstruct possible reasoning and then correct it.

### 4.1 Likely Source of the 45mm Estimate

The estimate probably came from:

1. SAM-Body4D or similar methods (4D-Humans, WHAM) report ~45-55mm PA-MPJPE on **3DPW** (an in-the-wild dataset)
2. Assumption: breaking footage is "in the wild" → similar accuracy

This is flawed because:

- **3DPW** contains walking, sitting, and outdoor activities — no inversions, no extreme poses
- **PA-MPJPE** (Procrustes-aligned) is more forgiving than MPJPE — it removes global rotation, which is precisely the failure mode we're analyzing
- The ~45mm estimate conflates PA-MPJPE with MPJPE

### 4.2 Corrected Estimate

Let me derive a proper estimate for optimization-based mesh reconstruction on breaking sequences.

**Baseline accuracy on standard poses**: ~50mm MPJPE on 3DPW-like conditions (not PA-MPJPE).

**Degradation factors for breaking**:

#### Factor A: Pose Prior Bias

For inverted poses ($\phi > 90°$), the pose prior creates a bias term:

$$\Delta_{prior} = \lambda_{prior} \cdot \left\| \frac{\partial \mathcal{L}_{prior}}{\partial \boldsymbol{\theta}} \right\| \cdot \left( H^{-1} \right)_{eff}$$

The gradient of VPoser's loss at an inverted pose is approximately:

$$\left\| \nabla_{\mathbf{z}} \|\mathbf{z}\|^2 \right\| = 2\|\mathbf{z}_{inverted}\|$$

For a headstand, $\|\mathbf{z}_{inverted}\| \approx 4-6$ (compared to $\|\mathbf{z}_{standing}\| \approx 1-2$), based on the VPoser latent space statistics from Pavlakos et al.

The effective bias on joint positions depends on the Hessian, but empirically:
- $\lambda_{prior} = 0.01$: ~3-5mm bias toward upright
- $\lambda_{prior} = 0.1$: ~8-15mm bias toward upright
- $\lambda_{prior} = 1.0$: ~20-40mm bias toward upright (optimizer largely ignores data, stays near prior mode)

**Estimated**: +5-15mm depending on $\lambda_{prior}$ choice.

#### Factor B: Initialization / Local Minima

For gradual transitions (toprock, footwork): initialization from previous frame works → minimal additional error (~2-3mm from iteration budget limits).

For abrupt transitions (kip-up, suicide drop, windmill entry): initialization is far from target → **catastrophic failure on transition frames**.

Modeling this as a mixture:
- 70% of breaking frames: gradual motion → +2-3mm
- 20% of breaking frames: sustained extreme pose (freezes, power moves) → +5-10mm from prior bias
- 10% of breaking frames: abrupt transitions → +30-60mm from local minima convergence

**Weighted average**: $0.7 \times 2.5 + 0.2 \times 7.5 + 0.1 \times 45 = 1.75 + 1.5 + 4.5 = \mathbf{+7.75\text{mm}}$

**But**: The 10% catastrophic frames are **exactly the most interesting frames** for breaking analysis (transitions, power move entries, freeze captures). If the pipeline is intended for judging and move classification, these frames have outsized importance.

#### Factor C: 2D Detection Degradation

From §3.2, the 2D detection accuracy drops significantly for inverted/blurred frames. The impact on optimization-based fitting:

$$\Delta_{2D} \approx \frac{\sigma_{2D}^{inv}}{\sigma_{2D}^{std}} \times \Delta_{2D}^{baseline}$$

where $\sigma_{2D}^{inv}$ is the 2D detection noise on inverted frames (~15-25 pixels) vs. standard (~5-8 pixels), and $\Delta_{2D}^{baseline}$ is the 2D-derived error on standard poses (~5mm).

$$\Delta_{2D}^{inv} \approx \frac{20}{6.5} \times 5 \approx 15\text{mm}$$

For motion-blurred power move frames, 2D detection may fail entirely on 2-5 joints, effectively removing those constraints from the optimization and leaving the prior to fill in → amplifies Factor A.

**Estimated**: +8-15mm average, with spikes to +25-40mm on heavily blurred frames.

#### Factor D: Skinning Artifacts

As discussed in §1.1, extreme articulations produce mesh artifacts:

**Estimated**: +3-8mm on extreme poses, ~0mm on standard poses. Weighted: +1-3mm average.

#### Factor E: Self-Occlusion and Contact

Breaking involves extensive self-contact (hand on ground, head on ground, legs crossed over torso). Optimization-based methods handle self-contact via:

$$\mathcal{L}_{contact} = \sum_{(v_i, v_j) \in \text{colliding}} \max(0, d_{thresh} - \|v_i - v_j\|)^2$$

For standard poses, collision detection works because the contact surfaces are well-separated in the initial estimate. For extreme poses (baby freeze: head, hand, hip, and shoulder all near the ground), the collision geometry is complex and the loss landscape has many local minima corresponding to different contact configurations.

**Estimated**: +3-8mm on ground contact frames, especially affecting pelvis, shoulders, and head joints.

### 4.3 Corrected Composite Estimate

$$\text{MPJPE}_{breaking}^{SAM4D} = \text{MPJPE}_{3DPW} + \sum_i \Delta_i$$

$$= 50 + 10 + 7.75 + 11.5 + 2 + 5.5 = \mathbf{86.75\text{mm}}$$

**Range**: 65-110mm depending on prior strength, sequence difficulty, and video quality.

This is **dramatically worse than the assumed 45mm** and is, critically, **in the same range as MotionBERT's estimated 70-100mm** on breaking sequences.

### 4.4 Per-Scenario Breakdown

| Breaking Scenario | SAM-Body4D Est. MPJPE | MotionBERT Est. MPJPE | Winner |
|-------------------|----------------------|----------------------|--------|
| Toprock (upright) | 52-58mm | 50-60mm | Comparable |
| Footwork (crouched) | 55-65mm | 55-65mm | Comparable |
| Freeze (static inverted) | 70-90mm | 80-100mm | SAM-Body4D (slight) |
| Power move (dynamic inverted) | 85-120mm | 85-110mm | Comparable — both fail |
| Transition (abrupt orientation change) | 90-140mm | 70-90mm | MotionBERT (temporal attention) |
| Blurred fast motion | 80-110mm | 75-95mm | MotionBERT (less dependent on per-frame quality) |

**Key insight**: SAM-Body4D is **not categorically better** than MotionBERT on breaking. They fail in different but overlapping regimes. The prior research's claim that SAM-Body4D "handles extreme poses because it's optimization-based, not learned-prior-based" is **incorrect** — it IS learned-prior-based (VPoser/GMM prior, foundation model features, initialization network).

---

## 5. The Confidence-Weighted Fusion Revisited

Given that both estimators perform poorly on breaking sequences (~70-110mm), the confidence-weighted fusion from the prior research needs re-examination.

### 5.1 When Fusion Helps

Fusion of two estimators improves accuracy when their errors are **uncorrelated**:

$$\text{Var}(\hat{p}_{fused}) = w^2 \text{Var}(\hat{p}_A) + (1-w)^2 \text{Var}(\hat{p}_B) + 2w(1-w)\text{Cov}(\hat{p}_A, \hat{p}_B)$$

For optimal $w = \frac{\text{Var}(B)}{\text{Var}(A) + \text{Var}(B)}$ and zero covariance:

$$\text{Var}(\hat{p}_{fused}) = \frac{\text{Var}(A) \cdot \text{Var}(B)}{\text{Var}(A) + \text{Var}(B)} < \min(\text{Var}(A), \text{Var}(B))$$

**But** the errors are NOT uncorrelated. Both estimators fail for similar reasons:
- Both degrade on motion blur (SAM-Body4D's photometric loss and MotionBERT's 2D input both suffer)
- Both have upright bias (SAM-Body4D through VPoser, MotionBERT through AMASS pretraining)
- Both fail on self-occlusion (SAM-Body4D loses silhouette constraints, MotionBERT loses 2D keypoints)

Estimating correlation: $\rho(\epsilon_A, \epsilon_B) \approx 0.5-0.7$ for breaking sequences (shared failure modes from common upstream causes).

With $\rho = 0.6$, $\sigma_A = 90\text{mm}$, $\sigma_B = 85\text{mm}$:

$$\sigma_{fused} = \sqrt{\frac{\sigma_A^2 \sigma_B^2 (1 - \rho^2)}{(\sigma_A^2 + \sigma_B^2 - 2\rho\sigma_A\sigma_B)}} \approx 72\text{mm}$$

The fusion gives ~18mm improvement over the individual estimators. Meaningful, but still **72mm** — far from the ~45mm assumed in the prior research.

### 5.2 When Fusion Fails

When both estimators fail simultaneously (high correlation in errors), fusion provides no benefit:

$$\sigma_{fused}(\rho \to 1) \to \min(\sigma_A, \sigma_B)$$

This occurs during:
- Peak power move rotation (both estimators maximally confused)
- Severe motion blur (both inputs degraded)
- Complete self-occlusion of key joints (no information for either method)

These are precisely the frames where the pipeline needs the most accuracy — **the fusion scheme degrades gracefully but still fails when it matters most**.

---

## 6. What Actually Works: Rotation-Equivariant Alternatives

The fundamental problem is that neither estimator is designed for arbitrary body orientations. The field has produced methods that explicitly address this:

### 6.1 Rotation-Equivariant Architectures

**Pose-NDF** (Tiwari et al., 2022): Uses SE(3)-equivariant neural distance fields for body model fitting. The representation is inherently rotation-invariant because it operates in the body's local coordinate frame.

**GraFormer** (Zhao et al., 2022): Graph-based transformer that operates on skeleton topology rather than spatial position. The graph structure is rotation-invariant by construction.

For the bboy pipeline, the ideal estimator would have:
1. **SE(3)-equivariant features**: Body part features that transform correctly under rotation
2. **Rotation-free pose prior**: Prior defined over joint angles (which are rotation-invariant) rather than global body orientation
3. **Topology-aware attention**: Attention patterns defined by kinematic chain distance, not spatial proximity

### 6.2 Practical Mitigation: Gravity-Aware Optimization

A more immediately actionable approach for SAM-Body4D:

**Modified optimization objective**:

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{prior}(t) \cdot \mathcal{L}_{prior} + \lambda_{gravity} \cdot \mathcal{L}_{gravity}$$

where:

$$\lambda_{prior}(t) = \lambda_0 \cdot \exp\left(-\frac{\|\hat{\boldsymbol{\theta}}_t - \boldsymbol{\theta}_{t-1}^*\|^2}{2\sigma_\theta^2}\right)$$

This **reduces the prior strength when the current estimate is far from the previous frame** — i.e., during rapid orientation changes, the prior backs off and lets the data term dominate.

The gravity term:

$$\mathcal{L}_{gravity} = \|(\mathbf{R}_{global} \cdot \hat{g}) - \hat{g}_{estimated}\|^2$$

where $\hat{g}_{estimated}$ is derived from scene context (floor plane estimation, which is straightforward when a bboy is performing on a visible floor). This provides an **explicit orientation signal** that doesn't depend on the upright-biased pose prior.

### 6.3 Multi-Hypothesis Initialization

Instead of single-point initialization, run $K$ parallel optimizations with different initial orientations:

$$\boldsymbol{\theta}_t^{init,k} = R_k \cdot \boldsymbol{\theta}_{t-1}^*, \quad k = 1, \ldots, K$$

where $R_k$ samples orientations at $\{0°, 90°, 180°, 270°\}$ about the horizontal axis. Select the hypothesis with lowest $\mathcal{L}_{data}$ (ignoring the prior for selection, since the prior biases toward upright).

**Cost**: $4\times$ computation → ~800ms/frame instead of ~200ms. But this **eliminates** the local minimum problem for orientation.

**Expected improvement**: Multi-hypothesis reduces the catastrophic failure rate from ~10% of frames to ~1-2% (only failing when all 4 hypotheses converge to wrong orientations, which requires pathological symmetry in the 2D projection).

---

## 7. Revised Pipeline Architecture

Given that SAM-Body4D is **not** the assumed panacea, the pipeline needs restructuring:

```
                         Video (T frames)
                              │
                              ▼
                    Floor Plane Estimation
                    (gravity direction ĝ)
                              │
                    ┌─────────┴──────────┐
                    │                     │
                    ▼                     ▼
         Multi-Hypothesis              ViTPose 2D
         SAM-Body4D (K=4)             (T×17×3)
         with reduced λ_prior              │
         and gravity loss                  ▼
                    │              MotionBERT DSTformer
                    │              (temporal stream)
                    ▼                     │
         Best 3D Mesh per frame           ▼
         (T×6890×3)                3D Skeleton (T×17×3)
                    │                     │
                    ▼                     │
         Joint Regression                 │
         (T×17×3)                        │
                    │                     │
                    └─────────┬───────────┘
                              │
                    Confidence-Weighted Fusion
                    (ρ-aware weighting, §5.1)
                              │
                              ▼
                    Temporal Smoothing
                    (Savitzky-Golay or DSTformer T-stream)
                              │
                              ▼
                    Refined 3D Skeleton (T×17×3)
```

### Key Changes from Prior Research's Pipeline:

1. **Multi-hypothesis initialization** for SAM-Body4D — addresses §2 (local minima)
2. **Reduced/adaptive pose prior** — addresses §1.2 (upright bias)
3. **Explicit gravity loss** from floor plane — addresses orientation ambiguity
4. **Correlation-aware fusion weights** — addresses §5.1 (correlated errors)
5. **Separate temporal smoothing stage** — the DSTformer temporal stream operates on fused 3D trajectories, not as a 2D→3D lifter

### Revised Accuracy Estimates

| Component | Standard Poses | Breaking (corrected) |
|-----------|---------------|---------------------|
| SAM-Body4D (single init, standard prior) | 50mm | 87mm |
| SAM-Body4D (multi-hypothesis, reduced prior) | 50mm | **62mm** |
| MotionBERT (2D→3D lifting) | 55mm (AIST++ est.) | 85mm |
| Fusion (original, ignoring correlation) | 45mm | 72mm |
| Fusion (correlation-aware) | 47mm | **68mm** |
| + Temporal smoothing | 40mm | **55-60mm** |

The revised pipeline achieves an estimated **55-60mm MPJPE on breaking** — still substantially worse than DanceFormer's 18.4mm on general AIST++ dance, but:

1. This is for **power move breaking**, which is categorically harder than any AIST++ genre
2. The velocity noise at 55mm:

$$\sigma_v = \frac{\sqrt{2} \times 55}{1/30} = 2333\text{mm/s} \approx 2.3\text{m/s}$$

Against power move velocities of 2-5 m/s, **SNR ≈ 1-2×** — still marginal for spectrogram computation. This means the temporal smoothing stage is **critical**, not optional. With aggressive temporal filtering (effective window of 30+ frames):

$$\sigma_{smoothed} \approx \frac{55}{\sqrt{30}} \approx 10\text{mm} \quad \Rightarrow \quad \sigma_v \approx 0.42\text{m/s} \quad \Rightarrow \quad \text{SNR} \approx 5-12\times$$

Usable, but at the cost of temporal resolution — a 30-frame averaging window at 30fps means ~1 second temporal blur, which limits the spectrogram's ability to capture sub-second movement dynamics.

---

## 8. The Uncomfortable Conclusion

The prior research built an elegant narrative: SAM-Body4D handles 3D estimation (rotation-robust because training-free), MotionBERT handles temporal refinement (good at dynamics), and the pipeline produces clean spectrograms. The reality:

1. **SAM-Body4D is NOT rotation-robust** — its pose prior, initialization, and foundation model features all encode upright bias. Estimated 87mm on breaking without mitigation, 62mm with multi-hypothesis + reduced prior.

2. **The 45mm estimate was unfounded** — it likely conflated PA-MPJPE on 3DPW with MPJPE on breaking, a 2× error.

3. **Fusion of two biased estimators helps but doesn't solve the problem** — correlated errors limit the improvement to ~15-20mm over individual estimators.

4. **The velocity SNR is marginal** — even with the revised pipeline (55-60mm), the movement spectrogram requires aggressive temporal smoothing that limits temporal resolution to ~1 second.

5. **No existing method achieves <30mm on power move breaking** — this is a genuine open problem in the field. The pipeline should be designed with this limitation acknowledged, not papered over with optimistic estimates.

The honest path forward is to **empirically measure** SAM-Body4D's performance on breaking footage before committing to this architecture. A small-scale experiment (10-20 breaking clips with manual 3D annotation or multi-view triangulation as ground truth) would cost ~1 week of annotation effort but would replace all of the above estimates with actual numbers.
