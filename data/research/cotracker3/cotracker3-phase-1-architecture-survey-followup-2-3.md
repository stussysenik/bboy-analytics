# cotracker3 — Follow-up 2.3: The research proposes 4 concrete experiments (category-stratified TAP-Vid-Kinetics, synthetic windmi

_Generated: 2026-03-23T03:34:01.600Z_

---

# CoTracker3 — Follow-up 1.4: The Calibration Gap

**The Epistemic Integrity Problem: Theoretical Predictions Without Ground-Truth Anchors**

---

## 0. The Problem Statement

The prior research (Follow-ups 1.1–1.3) produced a detailed quantitative framework predicting CoTracker3's performance on breakdancing content. These numbers now permeate the architecture:

| Claim | Value | Basis |
|-------|-------|-------|
| Power move AJ | 35–50 | Extrapolation from TAP-Vid-DAVIS performance + occlusion rate estimates |
| Velocity noise (SG-filtered) | ~22 px/s | Propagation from assumed $\sigma_{\text{track}} \approx 2$ px |
| Acceleration noise (SG-filtered) | ~800 px/s² | Same chain |
| Jerk noise (SG-filtered) | ~15K px/s³ | Same chain |
| Point survival under cycle-consistency | 50–70% for power moves | Theoretical, from occlusion frequency estimates |
| RAFT chain-linking AJ | ~45 | Interpolated from old benchmarks, not measured on bboy |
| Training data human motion fraction | 30–50% | Dataset composition estimate, not verified against repo |

**Every one of these numbers is a prediction built on other predictions.** The entire quantitative framework rests on zero empirically measured values from the target domain. This is a house of cards — and the research acknowledges this implicitly by proposing experiments, then doesn't execute the simplest one.

This analysis examines: (1) how far off the predictions could be, (2) what a single calibration experiment would reveal, and (3) a concrete executable protocol for Experiment 1.

---

## 1. The Cascading Uncertainty Problem

### 1.1 Dependency Graph of Predictions

The numerical claims form a directed acyclic graph where each prediction depends on upstream assumptions:

```
[σ_track ≈ 2 px]  ←── ASSUMED, not measured on bboy
        │
        ├──→ [velocity noise = σ_track / (√2·Δt)]
        │           │
        │           ├──→ [acceleration noise = √6·σ_track / Δt²]
        │           │           │
        │           │           └──→ [jerk noise = √20·σ_track / Δt³]
        │           │
        │           └──→ [SG-filtered velocity ≈ 22 px/s]
        │                       │
        │                       └──→ [movement spectrogram SNR estimate]
        │
        ├──→ [AJ = f(σ_track, occlusion_rate)]
        │           │
        │           └──→ [per-category AJ predictions: toprock 65-70, power 35-45]
        │
        └──→ [hybrid vs. tracking-only decision]

[occlusion_rate ≈ 30-60%]  ←── ASSUMED from body geometry, not measured
        │
        ├──→ [point survival rate estimates]
        │
        ├──→ [AJ predictions (combined with σ_track)]
        │
        └──→ [cycle-consistency filter survival rates]

[training_data_composition ≈ 30-50% human]  ←── ESTIMATED, not verified
        │
        └──→ [generalization argument for bboy domain]
```

### 1.2 Quantifying the Uncertainty Cascade

The root assumption $\sigma_{\text{track}} \approx 2$ px is taken from CoTracker3's aggregate benchmark performance on TAP-Vid-DAVIS. But TAP-Vid-DAVIS contains mostly walking humans, animals, and vehicles — not breakdancing. The actual per-frame tracking error on breakdancing could be anywhere in a wide range:

**Best case** (toprock, well-lit, high-res): $\sigma_{\text{track}} \approx 1.0$ px
**Expected case** (mixed moves): $\sigma_{\text{track}} \approx 2.0$–$4.0$ px
**Worst case** (fast power moves, motion blur): $\sigma_{\text{track}} \approx 5.0$–$10.0$ px

This 10× range in the root assumption propagates through every downstream prediction:

$$\sigma_v = \frac{\sigma_{\text{track}}}{\sqrt{2} \cdot \Delta t}$$

| $\sigma_{\text{track}}$ | $\sigma_v$ (raw) | $\sigma_v$ (SG, est.) | $\sigma_a$ (SG, est.) | $\sigma_j$ (SG, est.) |
|---|---|---|---|---|
| 1.0 px | 21 px/s | ~11 px/s | ~400 px/s² | ~7.5K px/s³ |
| 2.0 px | 42 px/s | ~22 px/s | ~800 px/s² | ~15K px/s³ |
| 5.0 px | 106 px/s | ~55 px/s | ~2,000 px/s² | ~37K px/s³ |
| 10.0 px | 212 px/s | ~110 px/s | ~4,000 px/s² | ~75K px/s³ |

The jerk noise spans a **10× range** ($7.5$K to $75$K px/s³). Whether jerk is usable for detecting movement "hits" depends on where in this range the true value falls. At $\sigma_j = 75$K px/s³, the jerk signal from a freeze (estimated peak jerk of ~$50$K–$100$K px/s³) would have SNR $\lesssim 1.3$:

$$\text{SNR}_{\text{jerk}} = \frac{j_{\text{signal}}}{\sigma_j} = \frac{75\text{K}}{75\text{K}} = 1.0 \quad (\text{worst case})$$

vs. the optimistic prediction:

$$\text{SNR}_{\text{jerk}} = \frac{75\text{K}}{15\text{K}} = 5.0 \quad (\text{assumed case})$$

**A factor of 5× difference in SNR determines whether the jerk channel of the movement spectrogram is usable at all.**

### 1.3 The Occlusion Rate Uncertainty

The occlusion rate estimates ($30$–$60$% for power moves) are derived from geometric reasoning about body surface visibility during rotation. But occlusion rates affect both:

1. **AJ directly** — AJ penalizes both position error and visibility prediction error
2. **The effective number of usable tracks** — if $60$% of points are occluded, you have $40$%N usable tracks, and the spatial density of the movement field drops proportionally

The research assumes these occlusion rates based on "a simplified cylindrical body model rotating about its long axis." But breakdancers are not cylinders:

- **Arms extend and retract** during power moves, creating time-varying occlusion geometry
- **Baggy clothing** changes the effective body surface area and creates false texture matches
- **The ground plane** occludes differently than free-space rotation (contact with floor during windmills)
- **Other body parts create self-occlusion** in ways that depend on the specific move, not just rotation angle

The actual occlusion dynamics are **move-specific** and cannot be reduced to a single percentage per category without empirical measurement.

---

## 2. What the Theoretical Predictions Actually Rest On

### 2.1 The TAP-Vid-DAVIS → Bboy Extrapolation

The AJ predictions are fundamentally based on this reasoning:

$$\text{AJ}_{\text{bboy,category}} = \text{AJ}_{\text{DAVIS}} \times \underbrace{f(\text{motion speed, occlusion, blur})}_{\text{degradation factors}}$$

Where the degradation factors are estimated analytically. The problem is that this assumes **linear degradation** with respect to each factor, and that the factors are **independent**. In reality:

**Non-linear degradation**: Tracking algorithms exhibit phase transitions, not smooth degradation. A tracker that works well at 20 px/frame motion may catastrophically fail at 40 px/frame (when the displacement exceeds the correlation radius). The 2× increase in motion speed could cause a >10× increase in error if it crosses a critical threshold.

**Factor interaction**: Motion blur and occlusion interact multiplicatively, not additively. A point that is simultaneously blurred AND about to be occluded is far harder to track than one that is blurred OR about to be occluded. The tracker can't use appearance matching (blurred) or temporal extrapolation (about to disappear).

**Distribution shift**: The degradation factors are calibrated against CoTracker3's behavior on the TAP-Vid distribution. The model may have different sensitivity to speed/occlusion on out-of-distribution poses (inverted body, unusual joint angles) because the feature representations are less robust outside the training distribution.

### 2.2 The Noise Propagation Assumption

The noise analysis assumes $\epsilon_n(t) \sim \mathcal{N}(0, \sigma_{\text{track}}^2)$ — that tracking errors are:

1. **Gaussian-distributed**: Real tracking errors are heavy-tailed. When a tracker loses a point, the error jumps to many tens of pixels before (potentially) recovering. The distribution is better modeled as a mixture:

$$\epsilon_n(t) \sim (1-p_{\text{loss}}) \cdot \mathcal{N}(0, \sigma_{\text{small}}^2) + p_{\text{loss}} \cdot \mathcal{N}(0, \sigma_{\text{large}}^2)$$

Where $\sigma_{\text{small}} \approx 1$–$2$ px, $\sigma_{\text{large}} \approx 20$–$50$ px, and $p_{\text{loss}} \approx 0.01$–$0.10$.

The heavy tail changes the noise propagation fundamentally:

$$\text{Var}[\hat{v}_n(t)] = \frac{(1-p_{\text{loss}})\sigma_{\text{small}}^2 + p_{\text{loss}}\sigma_{\text{large}}^2}{2\Delta t^2}$$

With $p_{\text{loss}} = 0.05$, $\sigma_{\text{large}} = 30$ px:

$$\text{Var}[\hat{v}_n(t)] = \frac{0.95 \times 4 + 0.05 \times 900}{2 \times (1/30)^2} = \frac{3.8 + 45}{2/900} = \frac{48.8}{0.00222} \approx 22,000$$

$$\sigma_v \approx 148 \text{ px/s}$$

Compare to the Gaussian-only prediction of $42$ px/s. **The heavy tail increases velocity noise by 3.5×.** Savitzky-Golay filtering partially mitigates this (it's equivalent to a local polynomial fit, which is robust to occasional outliers), but the effect on acceleration and jerk is even more severe because differentiation amplifies outliers.

2. **Temporally independent**: Real tracking errors are temporally correlated. When the tracker starts losing a point, errors persist over multiple frames before either recovering or being flagged as lost. Autocorrelation in $\epsilon_n(t)$ means the finite-difference noise analysis (which assumes independent errors at each frame) underestimates low-frequency noise and overestimates high-frequency noise.

3. **Spatially homogeneous**: The analysis assumes $\sigma_{\text{track}}$ is the same for all points. In reality, points on joints, extremities, and occluded regions have higher error than points on large, textured body surfaces. The movement spectrogram needs **joint-proximal points** most (for kinematic analysis), which are exactly the points with the worst tracking.

---

## 3. Experiment 1: Category-Stratified TAP-Vid-Kinetics

### 3.1 Why This Experiment Is the Minimum Viable Calibration

Experiment 1 doesn't require any annotation. It requires:
1. The existing CoTracker3 model (public checkpoint)
2. The existing TAP-Vid-Kinetics benchmark (public)
3. Running evaluation with one additional grouping variable: the Kinetics action class label

TAP-Vid-Kinetics already has ground-truth point annotations on videos from Kinetics-700. Kinetics-700 includes action classes like:

- "breakdancing" (direct match)
- "robot dancing" (similar mechanical isolation)
- "capoeira" (ground-level acrobatics)
- "gymnastics tumbling" (rotational body motion)
- "headbanging" (fast head motion)
- "salsa spin" (rotational dance)
- "spinning poi" (fast rotational tracking)
- "cartwheel" (body inversion)
- "backflip" (fast rotation + inversion)
- "side kick" (fast limb motion)

**This benchmark already contains breakdancing videos with ground-truth point annotations.** Reporting AJ per action class would give us the first empirical measurement of CoTracker3's performance on motion that overlaps with our target domain.

### 3.2 Concrete Executable Protocol

**Step 1: Obtain TAP-Vid-Kinetics**

TAP-Vid (published by DeepMind, Doersch et al. 2022) is available at `google-deepmind/tapnet`. The Kinetics split contains 1,189 annotated videos (as of the last published version), each with 5–20 ground-truth point tracks annotated by humans.

The annotations include: $(x_t, y_t, v_t)$ — position and visibility for each annotated point at each frame.

**Step 2: Map Videos to Kinetics Action Classes**

Each TAP-Vid-Kinetics video retains its Kinetics-700 video ID, which maps to an action class label. This mapping is available in the Kinetics-700 metadata.

Define motion category bins:

$$\mathcal{C}_{\text{dance}} = \{\text{breakdancing, robot dancing, salsa spin, ...}\}$$
$$\mathcal{C}_{\text{acrobatic}} = \{\text{gymnastics tumbling, cartwheel, backflip, capoeira, ...}\}$$
$$\mathcal{C}_{\text{fast-limb}} = \{\text{side kick, punching bag, headbanging, ...}\}$$
$$\mathcal{C}_{\text{slow-human}} = \{\text{walking the dog, tai chi, stretching, ...}\}$$
$$\mathcal{C}_{\text{non-human}} = \{\text{driving, sailing, flying kite, ...}\}$$

**Step 3: Run CoTracker3 Evaluation**

For each video $v_i$ with ground-truth tracks $\{(\mathbf{p}_n^{\text{GT}}(t), v_n^{\text{GT}}(t))\}_{n=1}^{N_i}$:

1. Run CoTracker3 in evaluation mode: query the same initial points as the ground truth
2. Obtain predicted tracks $\{(\hat{\mathbf{p}}_n(t), \hat{v}_n(t))\}_{n=1}^{N_i}$
3. Compute standard metrics:

**Average Jaccard (AJ)**:

$$\text{AJ} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{T} \sum_{t=1}^{T} \frac{\mathbb{1}[\|\hat{\mathbf{p}}_n(t) - \mathbf{p}_n^{\text{GT}}(t)\|_2 < \delta] \cdot \mathbb{1}[\hat{v}_n(t) = v_n^{\text{GT}}(t)]}{\mathbb{1}[\|\hat{\mathbf{p}}_n(t) - \mathbf{p}_n^{\text{GT}}(t)\|_2 < \delta] + \mathbb{1}[\hat{v}_n(t) = v_n^{\text{GT}}(t)] - \mathbb{1}[\|\hat{\mathbf{p}}_n(t) - \mathbf{p}_n^{\text{GT}}(t)\|_2 < \delta] \cdot \mathbb{1}[\hat{v}_n(t) = v_n^{\text{GT}}(t)]}$$

With threshold $\delta$ typically computed as a fraction of image diagonal (TAP-Vid uses multiple thresholds).

**Position accuracy $\langle \delta^x_\text{avg} \rangle$** — mean per-frame position error for visible points:

$$\langle \delta^x_\text{avg} \rangle = \frac{1}{\sum_{n,t} v_n^{\text{GT}}(t)} \sum_{n,t} v_n^{\text{GT}}(t) \cdot \|\hat{\mathbf{p}}_n(t) - \mathbf{p}_n^{\text{GT}}(t)\|_2$$

**Occlusion accuracy (OA)** — classification accuracy for visibility:

$$\text{OA} = \frac{1}{NT} \sum_{n,t} \mathbb{1}[\hat{v}_n(t) = v_n^{\text{GT}}(t)]$$

**Step 4: Report Per-Category**

Compute each metric grouped by $\mathcal{C}_k$:

$$\text{AJ}_k = \frac{1}{|\{v_i : v_i \in \mathcal{C}_k\}|} \sum_{v_i \in \mathcal{C}_k} \text{AJ}(v_i)$$

**Step 5: Compute Derivative-Specific Metrics (Novel Contribution)**

This is what existing benchmarks do NOT report, and what the bboy pipeline specifically needs:

For each ground-truth track, compute ground-truth velocity via central differences on GT positions:

$$\mathbf{v}_n^{\text{GT}}(t) = \frac{\mathbf{p}_n^{\text{GT}}(t+1) - \mathbf{p}_n^{\text{GT}}(t-1)}{2\Delta t}$$

And predicted velocity from tracked positions:

$$\hat{\mathbf{v}}_n(t) = \frac{\hat{\mathbf{p}}_n(t+1) - \hat{\mathbf{p}}_n(t-1)}{2\Delta t}$$

**Velocity error** (in px/s):

$$\epsilon_v(k) = \frac{1}{|\mathcal{C}_k|} \sum_{v_i \in \mathcal{C}_k} \frac{1}{N_i T_i} \sum_{n,t} \|\hat{\mathbf{v}}_n(t) - \mathbf{v}_n^{\text{GT}}(t)\|_2$$

Repeat for acceleration and jerk. Also compute with Savitzky-Golay preprocessing:

$$\epsilon_v^{\text{SG}}(k), \quad \epsilon_a^{\text{SG}}(k), \quad \epsilon_j^{\text{SG}}(k)$$

This gives us the **first empirical measurement of derivative accuracy** on human motion, stratified by motion type. It directly validates or falsifies the noise predictions from Follow-up 1.2.

### 3.3 What the Results Would Tell Us

**Scenario A: AJ on dance/acrobatic classes ≈ 55–65**

This would validate the theoretical predictions and confirm that CoTracker3 generalizes reasonably to human motion with moderate occlusion. The derivative noise estimates would hold approximately, and the movement spectrogram architecture would be sound as designed.

**Scenario B: AJ on dance/acrobatic classes ≈ 35–45**

This would indicate a larger domain gap than predicted. The derivative noise estimates from Follow-up 1.2 would be underestimates. Specifically:

- $\sigma_{\text{track}}$ would need to be revised upward from $2$ px to $4$–$6$ px
- Jerk channel of the spectrogram would be borderline unusable ($\text{SNR} < 2$)
- The hybrid approach (CoTracker3 + RAFT velocity refinement) would go from "optional enhancement" to "essential"
- Power move tracking would need the multi-view or temporal super-resolution mitigations from Follow-up 1.1

**Scenario C: AJ on dance/acrobatic ≈ 20–35**

This would indicate a fundamental problem. The predictions from Follow-up 1.1–1.3 would be systematically overoptimistic, and the architecture would need rethinking:

- Dense point tracking as the primary motion representation might not work for power moves at all
- Alternative: segment-level tracking (SAM 3 masks tracked frame-to-frame) instead of point-level
- Alternative: learned motion descriptors (pose estimation → kinematic model) instead of appearance-based tracking
- The movement spectrogram would need to operate on pose keypoints (17–25 points from MoveNet/ViTPose) rather than dense surface tracks (2000+ points from CoTracker3)

**Scenario D: Large variance within dance/acrobatic (AJ range 25–75)**

This would indicate that the category-level predictions are too coarse. The real performance depends on sub-category factors:

- Speed of motion within the clip
- Camera angle and distance
- Video quality (resolution, compression, frame rate)
- Specific move type within the category

This scenario would require Experiment 3 (manual annotation on actual bboy footage) to get actionable numbers.

### 3.4 Computational Cost of Experiment 1

This is critical — the research presented this experiment as "zero annotation labor," which is correct, but didn't estimate the computational cost:

**TAP-Vid-Kinetics**: ~1,189 videos, average ~250 frames, 256×256 resolution
**CoTracker3 inference**: ~30ms per frame on RTX 4090

$$\text{Total time} = 1189 \times 250 \times 0.03 \text{s} \approx 8,918 \text{s} \approx 2.5 \text{ hours on one RTX 4090}$$

This is trivially achievable. The evaluation metrics computation adds negligible time.

**What's needed**:
- 1 GPU (any NVIDIA GPU with ≥8GB VRAM — CoTracker3 is efficient)
- ~3 hours of compute time
- ~50GB of disk for TAP-Vid-Kinetics data
- Standard Python environment with PyTorch + CoTracker3 repo

**There is no legitimate reason this experiment hasn't been run.** It provides maximum information per unit effort of any possible experiment.

---

## 4. The Error Budget Framework

Even without running Experiment 1, we can formalize the uncertainty in a way that makes the downstream consequences explicit. This is what the research should have done instead of presenting point estimates as if they were calibrated.

### 4.1 Per-Point Position Error Distribution

Model the true per-frame tracking error as a **mixture of modes**:

$$\epsilon_n(t) \sim \pi_1 \cdot \mathcal{N}(0, \sigma_1^2) + \pi_2 \cdot \text{Laplace}(0, b_2) + \pi_3 \cdot \text{Uniform}(-R, R)$$

Where:
- Mode 1 ($\pi_1 \approx 0.80$–$0.95$): Successful tracking, sub-pixel to few-pixel error
- Mode 2 ($\pi_2 \approx 0.04$–$0.15$): Drift/slip, moderate error with heavy tails  
- Mode 3 ($\pi_3 \approx 0.01$–$0.05$): Track loss, error uniformly distributed over image region

The mixture weights $(\pi_1, \pi_2, \pi_3)$ are **motion-type-dependent**:

| Motion Type | $\pi_1$ (good) | $\pi_2$ (drift) | $\pi_3$ (lost) | $\sigma_1$ | $b_2$ | $R$ |
|---|---|---|---|---|---|---|
| Toprock | 0.92 | 0.06 | 0.02 | 1.5 | 4.0 | 50 |
| Footwork | 0.85 | 0.10 | 0.05 | 2.0 | 6.0 | 80 |
| Freeze | 0.95 | 0.04 | 0.01 | 1.0 | 3.0 | 40 |
| Power (slow) | 0.70 | 0.20 | 0.10 | 3.0 | 10.0 | 120 |
| Power (fast) | 0.55 | 0.25 | 0.20 | 4.0 | 15.0 | 150 |
| Air moves | 0.45 | 0.30 | 0.25 | 5.0 | 20.0 | 200 |

**These numbers are estimates with wide uncertainty bands** — but the framework is correct and can be calibrated empirically from Experiment 1.

### 4.2 Impact on the Movement Spectrogram

The movement spectrogram computes a time-frequency representation where the "frequency" axis represents motion periodicity. Its usability depends on the **signal-to-noise ratio in each derivative channel**:

**Signal magnitudes** (estimated for typical breakdancing at 1080p, 30fps):

| Derivative | Toprock Signal | Power Move Signal | Freeze Signal |
|---|---|---|---|
| Velocity $\|\mathbf{v}\|$ | 50–200 px/s | 200–800 px/s | 0–20 px/s |
| Acceleration $\|\mathbf{a}\|$ | 500–2,000 px/s² | 2,000–15,000 px/s² | 0–500 px/s² (at entry/exit) |
| Jerk $\|\mathbf{j}\|$ | 5K–20K px/s³ | 20K–100K px/s³ | 50K–200K px/s³ (at freeze "hit") |

**SNR under different $\sigma_{\text{track}}$ assumptions** (SG-filtered, 4th order, window=7):

For **velocity channel** (most critical for beat-alignment):

| $\sigma_{\text{track}}$ | Noise (SG) | SNR (toprock) | SNR (power) |
|---|---|---|---|
| 1.5 px | ~16 px/s | 6.3–12.5 | 12.5–50.0 |
| 3.0 px | ~32 px/s | 1.6–6.3 | 6.3–25.0 |
| 6.0 px | ~64 px/s | 0.8–3.1 | 3.1–12.5 |

For **jerk channel** (critical for freeze/"hit" detection):

| $\sigma_{\text{track}}$ | Noise (SG) | SNR (freeze hit) | SNR (power move) |
|---|---|---|---|
| 1.5 px | ~10K px/s³ | 5.0–20.0 | 2.0–10.0 |
| 3.0 px | ~20K px/s³ | 2.5–10.0 | 1.0–5.0 |
| 6.0 px | ~40K px/s³ | 1.3–5.0 | 0.5–2.5 |

**The threshold for usability is SNR ≥ 3** (standard signal detection criterion). From the tables:

- **Velocity channel**: Usable for most motion if $\sigma_{\text{track}} \leq 3$ px. Marginal for toprock if $\sigma_{\text{track}} > 4$ px.
- **Jerk channel**: Usable for freeze detection if $\sigma_{\text{track}} \leq 3$ px. **Unusable for power moves if $\sigma_{\text{track}} > 4$ px.**

This means the entire "jerk-based freeze detection" feature hangs on whether the true tracking error is 2–3 px (as assumed) or 4–6 px (plausible for out-of-distribution motion). **A single empirical measurement would resolve this.**

---

## 5. What's Actually Knowable vs. What's Speculative

### 5.1 Classification of Every Quantitative Claim

| Claim | Status | How to Verify | Effort |
|---|---|---|---|
| CoTracker3 AJ on TAP-Vid-DAVIS = 67.8 | **Known** (published, reproduced) | — | — |
| CoTracker3 AJ on breakdancing ≈ 35–50 | **Speculative** (extrapolated) | Experiment 1 (per-category TAP-Vid-K) | 3 hours GPU |
| $\sigma_{\text{track}} \approx 2$ px on bboy | **Speculative** (assumed from aggregate) | Experiment 1 (measure $\langle\delta^x_\text{avg}\rangle$ per category) | Same run |
| SG filter reduces noise by ~2× | **Well-established** (classical result) | — | — |
| Velocity noise after SG ≈ 22 px/s | **Depends on** $\sigma_{\text{track}}$ assumption | Experiment 1 + derivative analysis | +1 hour post-processing |
| Jerk is usable for freeze detection | **Speculative** (depends on $\sigma_{\text{track}} \leq 3$) | Experiment 4 (post-processing ablation on real data) | Needs Experiment 1 first |
| Optical flow worse than tracking for $\geq 2$nd-order | **Theoretical but sound** (follows from Lagrangian/Eulerian analysis) | Experiment 1 with RAFT baseline comparison | +3 hours GPU |
| Occlusion rate 30–60% for power moves | **Speculative** (geometric estimate) | Experiment 1 (occlusion accuracy metric) or Experiment 3 | 3 hours or 20+ person-hours |
| Training data is 30–50% human motion | **Speculative** (dataset composition estimate) | Check `facebookresearch/co-tracker` repo | 30 minutes |
| RAFT chain-linking AJ ≈ 45 | **Stale** (from older benchmarks, not on bboy) | Experiment 1 with RAFT baseline | +3 hours GPU |
| Hybrid approach provides ~30% velocity improvement | **Speculative** (not measured) | Experiment 4 (ablation on real data) | Needs Experiments 1 & 2 first |

### 5.2 The Minimum Viable Verification Plan

If exactly one thing is done, it should be:

**Run CoTracker3 on TAP-Vid-Kinetics with per-action-class AJ reporting.**

This single experiment:
- Takes ~3 hours on one GPU
- Requires zero annotation
- Directly measures $\text{AJ}_k$ for motion categories relevant to breakdancing
- Provides $\langle\delta^x_\text{avg}\rangle_k$ which calibrates $\sigma_{\text{track}}$ per motion type
- Provides occlusion accuracy per motion type
- Can be extended to measure derivative accuracy with ~1 hour of additional post-processing code

The cascading effect of this one measurement:

```
[Measured AJ on dance/acrobatic/martial arts categories]
    │
    ├──→ Calibrates σ_track → validates or invalidates ALL derivative noise estimates
    │
    ├──→ Calibrates occlusion rate → validates or invalidates point survival predictions
    │
    ├──→ Empirical AJ by category → validates or invalidates the per-move-type AJ table
    │
    └──→ Determines whether Experiments 2-4 are worth running at all
        │
        ├──→ If AJ > 55: Architecture is sound, proceed with implementation
        ├──→ If AJ 40-55: Architecture needs the hybrid enhancement
        └──→ If AJ < 40: Architecture may need fundamental rethinking
```

---

## 6. Why the Gap Exists (Meta-Analysis)

The research produced 4 follow-ups (1.1–1.3 plus this one) of increasing analytical depth, each building on the previous. But it never ran a single experiment. This pattern — **infinite analysis, zero measurement** — is a known failure mode in research planning. The causes:

1. **Analysis is cheaper than experiments**: Writing equations is faster than setting up a codebase, downloading data, and debugging GPU code. Each follow-up question can be answered analytically in an hour; running Experiment 1 requires environmental setup that might take half a day.

2. **Precision illusion**: Numbers like "42 px/s" and "4,400 px/s²" look precise and create confidence. But they're derived quantities with inherited uncertainty. The precision of the computation masks the imprecision of the inputs.

3. **Cascading theoretical questions**: Each analysis naturally raises follow-up questions ("but what about the Eulerian vs. Lagrangian distinction?"), creating a depth-first exploration of theoretical space that never bottoms out into measurement.

The fix is to recognize that **the marginal value of additional analysis is now near zero without empirical calibration**. The theoretical framework is complete and internally consistent. What it lacks is a single anchor to reality. Further analysis without that anchor is intellectual momentum, not progress.

---

## 7. Concrete Next Step

**Stop analyzing. Run Experiment 1.**

Specifically:

```python
# Pseudocode for the minimum viable experiment
# ~100 lines of actual code on top of existing CoTracker3 eval

from cotracker.evaluation import evaluate_tap_vid
from kinetics_metadata import get_action_class  # map video_id → class

results = evaluate_tap_vid(
    model="cotracker3",
    dataset="kinetics",
    metrics=["AJ", "delta_avg", "OA"]
)

# Group by motion category
categories = {
    "dance": ["breakdancing", "robot_dancing", "salsa_spin", ...],
    "acrobatic": ["gymnastics_tumbling", "cartwheel", "backflip", ...],
    "fast_limb": ["side_kick", "punching_bag", ...],
    "slow_human": ["walking_the_dog", "tai_chi", ...],
    "non_human": ["driving_car", "sailing", ...]
}

for cat_name, classes in categories.items():
    cat_results = [r for r in results if get_action_class(r.video_id) in classes]
    print(f"{cat_name}: AJ={mean_aj(cat_results):.1f}, "
          f"δ_avg={mean_delta(cat_results):.2f}px, "
          f"OA={mean_oa(cat_results):.3f}")
    
    # NOVEL: derivative accuracy
    for r in cat_results:
        v_gt = central_diff(r.gt_positions) / dt
        v_pred = central_diff(r.pred_positions) / dt
        v_pred_sg = savgol_diff(r.pred_positions, order=1, window=7, polyorder=4) / dt
        r.velocity_error = norm(v_pred - v_gt).mean()
        r.velocity_error_sg = norm(v_pred_sg - v_gt).mean()
        # repeat for acceleration, jerk
    
    print(f"  v_error: {mean_vel_error(cat_results):.1f} px/s "
          f"(SG: {mean_vel_error_sg(cat_results):.1f} px/s)")
```

The output of this script would look something like:

```
dance:      AJ=52.3, δ_avg=3.41px, OA=0.847
  v_error: 68.2 px/s (SG: 38.1 px/s)
  a_error: 2140 px/s² (SG: 1050 px/s²)
acrobatic:  AJ=44.1, δ_avg=4.87px, OA=0.791
  v_error: 94.5 px/s (SG: 51.2 px/s)
  ...
```

Those numbers — **real, measured, not predicted** — would validate, revise, or falsify the entire theoretical framework in one afternoon.

---

*The theoretical analysis is complete. The error bounds are formalized. The experimental protocol is specified. The only remaining step is to execute. Every additional minute spent on theoretical analysis instead of running Experiment 1 has negative marginal value.*
