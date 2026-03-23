# cotracker3 — Follow-up 1.2: The breakdancing section identifies that 20+ px/frame extremity displacement exceeds the effective s

_Generated: 2026-03-23T03:48:32.925Z_

---

I'll dive deep into the three identified gaps around CoTracker3's search radius limitations for breakdancing. This requires building a proper kinematic model and propagating the failure analysis through the architecture equations.

---

# CoTracker3 Search Radius Failure Analysis for Breakdancing

## Kinematic Model Setup

Before analyzing failure fractions, I need to formalize the physical model. Model a power move (windmill) as rigid-body rotation about the upper back:

$$\mathbf{p}_j(t) = \mathbf{c} + r_j \begin{bmatrix} \cos(\omega t + \phi_j) \\ \sin(\omega t + \phi_j) \end{bmatrix}$$

where $j$ indexes body parts, $\mathbf{c}$ is the rotation center, $r_j$ is the radial distance from the axis, and $\omega = 2\pi / T_{\text{rot}}$.

**Per-frame displacement** (consecutive frames at framerate $f$):

$$d_j^{\text{frame}} = r_j \omega / f \cdot S = \frac{2\pi r_j S}{T_{\text{rot}} f}$$

where $S$ is the pixel scale factor (px/m), derived from dancer height occupying ~65% of frame height:

$$S = \frac{0.65 \cdot H_{\text{px}}}{1.75 \text{ m}}$$

| Resolution | $H_{\text{px}}$ | $S$ (px/m) |
|------------|----------|-----------|
| 384×512    | 384      | 143       |
| 720p       | 720      | 267       |
| 1080p      | 1080     | 401       |

**Anatomical distances** for a 175cm bboy during a windmill (rotation center ≈ upper back):

| Body Part | $r_j$ (m) | COCO Keypoints | Count |
|-----------|-----------|----------------|-------|
| Shoulders | 0.15      | L/R Shoulder   | 2     |
| Head      | 0.35      | Nose, Eyes, Ears | 5   |
| Hips      | 0.40      | L/R Hip        | 2     |
| Elbows    | 0.45      | L/R Elbow      | 2     |
| Wrists    | 0.65      | L/R Wrist      | 2     |
| Knees     | 0.75      | L/R Knee       | 2     |
| Ankles    | 0.95      | L/R Ankle      | 2     |

Total: 17 COCO keypoints. Rotation period $T_{\text{rot}} = 1.0$s (competitive level).

**Total displacement from query position** (chord distance across $t$ frames):

$$d_j^{\text{total}}(t) = 2r_j \left|\sin\!\left(\frac{\omega t}{2f}\right)\right| \cdot S = 2r_j S \left|\sin\!\left(\frac{\pi t}{T_{\text{rot}} f}\right)\right|$$

Maximum occurs at half-rotation ($t = T_{\text{rot}} f / 2$): $d_j^{\max} = 2r_j S$.

---

## (a) Fraction of Breakdancing Frames Exceeding Search Radius

### Search Radius Recap (from Eq. 1.3, 7.2)

With correlation radius $R = 4$ and backbone stride $s = 4$:

| Scale Level | Feature Radius | Effective Pixel Radius $R_{\ell}$ |
|-------------|---------------|----------------------------------|
| L0 (single-scale) | $R = 4$ | $R \cdot s = 16$ px |
| L1 (2× pooled) | $R = 4$ | $2R \cdot s = 32$ px |
| L2 (4× pooled) | $R = 4$ | $4R \cdot s = 64$ px |

### Per-Frame Displacement Table

Using $d_j^{\text{frame}} = 2\pi r_j S / (T_{\text{rot}} f)$:

| Body Part | $r_j$ | 384p/30 | 384p/60 | 720p/30 | 720p/60 | 1080p/30 | 1080p/60 |
|-----------|-------|---------|---------|---------|---------|----------|----------|
| Shoulders | 0.15  | **4.5** | **2.2** | **8.4** | **4.2** | **12.6** | **6.3** |
| Head      | 0.35  | **10.5**| **5.2** | 19.6    | **9.8** | 29.4     | **14.7** |
| Hips      | 0.40  | **12.0**| **6.0** | 22.4    | **11.2**| 33.5     | 16.8     |
| Elbows    | 0.45  | **13.5**| **6.7** | 25.2    | **12.6**| 37.8     | 18.9     |
| Wrists    | 0.65  | 19.4   | **9.7** | 36.3    | 18.2    | 54.5     | 27.3     |
| Knees     | 0.75  | 22.5   | **11.2**| 41.9    | 20.9    | 62.8     | 31.4     |
| Ankles    | 0.95  | 28.5   | **14.2**| 53.1    | 26.5    | 79.7     | 39.8     |

Bold = within single-scale radius ($\leq 16$ px). Values in plain text exceed it.

### Failure Fraction by Configuration

Define $\Phi(R_\ell)$ = fraction of COCO keypoints where $d_j^{\text{frame}} > R_\ell$:

| Configuration | $\Phi(16)$ single-scale | $\Phi(32)$ L1 | $\Phi(64)$ L2 |
|---------------|------------------------|---------------|---------------|
| **384p / 30fps** | 6/17 = **35%** | 0% | 0% |
| **384p / 60fps** | 0% | 0% | 0% |
| **720p / 30fps** | 13/17 = **76%** | 4/17 = 24% | 0% |
| **720p / 60fps** | 6/17 = **35%** | 0% | 0% |
| **1080p / 30fps** | 15/17 = **88%** | 10/17 = 59% | 4/17 = **24%** |
| **1080p / 60fps** | 10/17 = **59%** | 4/17 = 24% | 0% |

**Key finding**: Multi-scale correlation (L2, 64px) eliminates the per-frame displacement problem at all resolutions ≤ 720p at 30fps, and at all resolutions at 60fps. Only 1080p/30fps retains a 24% failure rate even with multi-scale.

### But Per-Frame Displacement Is Not the Real Problem

CoTracker3's batch mode initializes **all frames at the query position**. The relevant metric is total displacement from query, not per-frame displacement. For a $T = 24$ frame sequence at 30fps (0.8s ≈ one windmill rotation):

$$d_j^{\text{total}}(t) = 2r_j S \left|\sin\!\left(\frac{\pi t}{30}\right)\right|$$

At frame $t = 15$ (half-rotation, maximum displacement):

| Body Part | 384p (px) | 720p (px) | 1080p (px) |
|-----------|-----------|-----------|------------|
| Shoulders | 43        | 80        | 120        |
| Head      | 100       | 187       | 281        |
| Hips      | 114       | 214       | 321        |
| Elbows    | 129       | 240       | 361        |
| Wrists    | 186       | 347       | 522        |
| Knees     | 215       | 401       | 602        |
| Ankles    | 272       | 507       | 762        |

**Multi-scale L2 search radius: 64 px. Every body part at every resolution exceeds this at the mid-sequence point.** Even shoulders at the lowest resolution (43 px) are within L2 range only marginally, and only for the first ~8 frames.

### Frame-Level Failure Curve

For each body part, the frame at which total displacement first exceeds $R_\ell = 64$ px:

$$t_{\text{fail}}(j) = \frac{f \cdot T_{\text{rot}}}{\pi} \arcsin\!\left(\frac{R_\ell}{2 r_j S}\right)$$

At 384p/30fps:

| Body Part | $t_{\text{fail}}$ (frames) | % of 24-frame sequence trackable |
|-----------|--------------------------|----------------------------------|
| Shoulders | 14.3 → never (max $d$ = 43 < 64) | **100%** |
| Head      | 5.9 | **25%** |
| Hips      | 5.1 | **21%** |
| Elbows    | 4.5 | **19%** |
| Wrists    | 3.1 | **13%** |
| Knees     | 2.7 | **11%** |
| Ankles    | 2.1 | **9%** |

**Weighted by COCO keypoint count**: at 384p/30fps in batch mode, only **~22%** of keypoint-frame pairs have total displacement within the multi-scale search radius. The remaining **78%** rely entirely on the iterative refinement propagation.

---

## (b) Multi-Scale Correlation: Recovery vs. Coarse Re-localization

The question: when the true position is within the L2 search window but outside L0, does multi-scale correlation provide enough signal for precise tracking, or only a coarse directional hint?

### Information Content Analysis

At each pyramid level $\ell$, the correlation (Eq. 7.2) is:

$$\mathbf{C}_{n,t}^{(\ell)} \in \mathbb{R}^{(2R+1)^2}$$

The **spatial precision** at level $\ell$ is determined by the pooling factor $2^\ell$:

$$\sigma_{\text{precision}}^{(\ell)} = 2^\ell \cdot s \text{ pixels}$$

| Level | Pooling | Precision (px) | Search Range (px) | Bits of position info |
|-------|---------|----------------|-------------------|-----------------------|
| L0    | 1×      | 4              | ±16               | $\log_2(33/4) \approx 3.0$ |
| L1    | 2×      | 8              | ±32               | $\log_2(65/8) \approx 3.0$ |
| L2    | 4×      | 16             | ±64               | $\log_2(129/16) \approx 3.0$ |

Each level provides ~3 bits of position information per axis. The key insight: **L2 has the same discriminative resolution as L0 within its own coordinate system** — it just operates at 4× coarser spatial resolution.

### Precision Bound for Multi-Scale Only

When the true position is at displacement $d$ from the estimated position:

**Case 1: $d \leq 16$ px (within L0)** — All three levels provide signal. The combined precision is dominated by L0:

$$\sigma_{\text{combined}} \approx s = 4 \text{ px (sub-pixel with bilinear interpolation: } \sigma \approx 1\text{ px)}$$

**Case 2: $16 < d \leq 64$ px (within L2 only)** — Only L2 provides a matching signal. L0 and L1 correlations are noise (no peak within their search windows).

The position update from the readout MLP (Eq. 2.5) is:

$$\Delta\mathbf{p}_{n,t} = \text{MLP}_{\text{out}}(\mathbf{z}_{n,t}^{(K)}) \in \mathbb{R}^2$$

The MLP receives concatenated correlation features $[\mathbf{C}^{(0)}; \mathbf{C}^{(1)}; \mathbf{C}^{(2)}]$. When only $\mathbf{C}^{(2)}$ has a peak:
- The MLP can extract direction and approximate magnitude from L2
- Precision of the correction: $\sigma \approx 2^2 \cdot s = 16$ px
- After correction, position error drops from $d$ to ~16 px

**This is coarse re-localization, not precise tracking.** The error drops to ~16 px, which is now within L0 range. But this requires a SECOND iteration with recomputed correlations.

### Multi-Scale Recovery Budget

For iterative refinement with $K = 4$ iterations:

| Iteration | Effective precision | Position error after |
|-----------|-------------------|-----------------------|
| 0         | L2: 16 px (if $d \leq 64$) | $\max(d - 48, 16)$ px |
| 1         | L0+L1+L2: 4 px (now within range) | ~4 px |
| 2         | L0 refinement: ~1 px | ~1 px |
| 3         | Sub-pixel: ~0.5 px | ~0.5 px |

**Conclusion**: Multi-scale correlation can recover tracks with displacements up to 64px in **2 iterations** — one for coarse re-localization (L2), one for fine refinement (L0). This consumes half the iteration budget ($K = 4$), leaving only 2 iterations for further refinement.

For displacements beyond 64px, **multi-scale provides no signal at any level** — recovery depends entirely on temporal attention propagation, which is analyzed in section (c).

### The Coarse-to-Fine Bandwidth Bottleneck

Define the **information bandwidth** per iteration as the maximum position correction the MLP can reliably produce:

$$c_{\max}^{(\ell)} \approx R \cdot 2^\ell \cdot s$$

| Level providing signal | $c_{\max}$ |
|-----------------------|------------|
| L0 only               | 16 px      |
| L1 only               | 32 px      |
| L2 only               | 64 px      |
| No correlation signal  | Depends on temporal attention (~$c_{\text{attn}}$) |

The maximum displacement recoverable in $K$ iterations (assuming each iteration gets the best available correlation signal):

$$d_{\max}^{\text{recover}} = c_{\max}^{(2)} + (K-1) \cdot c_{\max}^{(0)} = 64 + 3 \times 16 = 112 \text{ px}$$

Comparing to windmill ankle displacements at 384p: $d_{\max} = 272$ px. The multi-scale correlation alone can only recover **41%** of the maximum displacement. The remaining 160 px must come from temporal attention.

---

## (c) Cascading Failure Analysis: Does One Bad Frame Poison the Refinement Loop?

This is the most subtle question. I'll analyze three propagation mechanisms: (c.1) error persistence through iterations, (c.2) temporal attention propagation dynamics, and (c.3) cross-track contamination through spatial attention.

### (c.1) Error Persistence Through Iterations

Define the error at frame $t$ after iteration $k$:

$$e_t^{(k)} = \left\|\hat{\mathbf{p}}_{n,t}^{(k)} - \mathbf{p}_{n,t}^{\text{gt}}\right\|_2$$

At iteration 0: $e_t^{(0)} = d_j^{\text{total}}(t)$ (displacement from query position).

The update rule (from Eq. 2.5):

$$e_t^{(k+1)} = e_t^{(k)} - \underbrace{\Delta p_{\text{corr},t}^{(k)}}_{\text{from correlation}} - \underbrace{\Delta p_{\text{attn},t}^{(k)}}_{\text{from temporal attention}}$$

**When correlation has signal** ($e_t^{(k)} \leq R_\ell$):

$$\Delta p_{\text{corr},t}^{(k)} \approx \beta_\ell \cdot e_t^{(k)}, \quad \beta_\ell \in (0.5, 0.9)$$

The correlation provides a strong correction proportional to the error. Convergence is geometric: $e_t^{(k)} \to 0$ exponentially.

**When correlation has NO signal** ($e_t^{(k)} > R_{\text{max}}$):

$$\Delta p_{\text{corr},t}^{(k)} \approx \xi_t^{(k)}, \quad \xi_t^{(k)} \sim \mathcal{N}(0, \sigma_\xi^2)$$

The correlation volume is essentially noise — it may contain spurious peaks from similar-looking features, producing a random (or worse, misleading) correction $\xi$.

The **net correction** when correlation fails:

$$\Delta p_{\text{net},t}^{(k)} = \Delta p_{\text{attn},t}^{(k)} + \xi_t^{(k)}$$

The noise term $\xi$ can actively work against the attention-based correction. If $\sigma_\xi$ is comparable to $|\Delta p_{\text{attn}}|$, the refinement makes no progress on average.

### (c.2) Temporal Attention Propagation Dynamics

The temporal attention (Eq. 2.2) for point $n$ at frame $t$:

$$\tilde{\mathbf{z}}_{n,t} = \sum_{t'=0}^{T-1} \alpha_{t,t'} \mathbf{z}_{n,t'}$$

where $\alpha_{t,t'} = \frac{\exp(\mathbf{q}_t^T \mathbf{k}_{t'} / \sqrt{d_k})}{\sum_{t''} \exp(\mathbf{q}_t^T \mathbf{k}_{t''} / \sqrt{d_k})}$.

**Key question**: how are the attention weights $\alpha_{t,t'}$ distributed?

At iteration 0, all tokens are constructed from:
- Position: identical (all at query point)
- Displacement: zero for all frames
- Visibility: initialized uniformly
- Correlation: informative for frames near query, noise elsewhere

The tokens for "good" frames (near query, with peaked correlations) differ from "bad" frames (distant, with noisy correlations) primarily in their correlation features $\mathbf{C}_{n,t}$.

The attention weights depend on key-query similarity. If the model learns that "peaked correlation" tokens are informative:
- Good frames produce distinctive keys
- All frames' queries attend strongly to good frames

This creates an **asymmetric propagation pattern**: information flows from good→bad frames, but bad frames' noise also leaks into good frames' representations.

**Effective propagation speed**: Define the "correction reach" at iteration $k$ as the maximum frame distance from the query at which the position error is brought within $R_{\text{max}} = 64$ px.

Model the attention-based correction as a function of the number of informative frames within effective attention range. For a frame $t$ with no direct correlation signal:

$$\Delta p_{\text{attn},t}^{(k)} \approx \sum_{t': e_{t'}^{(k)} \leq R_{\max}} \alpha_{t,t'} \cdot g(\mathbf{z}_{n,t'})$$

where $g$ extracts motion information from the well-tracked frames.

The magnitude of this correction depends on:

1. **Total attention weight on informative frames**: $A_t^{(k)} = \sum_{t': e_{t'}^{(k)} \leq R_{\max}} \alpha_{t,t'}$
2. **Quality of motion model from those frames**: the MLP must extrapolate from nearby corrected frames to predict motion at frame $t$

For a windmill at 384p/30fps, the informative frames at iteration 0 are frames 0–2 (displacement ≤ 64px). Out of $T = 24$ frames, this is 12.5%. If attention weights are roughly uniform:

$$A_t^{(0)} \approx 3/24 = 0.125$$

Even if the motion model from those 3 frames is perfect, the correction is attenuated by factor 0.125. This means:

$$|\Delta p_{\text{attn},t}^{(0)}| \leq 0.125 \times d_t \approx 0.125 \times 272 = 34 \text{ px (best case for frame 15)}$$

After iteration 0: $e_{15}^{(1)} \approx 272 - 34 = 238$ px. Still way outside search radius.

But frames closer to the query get larger corrections (higher attention to nearby informative frames). Frame 5 might get: $e_5^{(1)} \approx 136 - 0.25 \times 136 = 102$ px. Still outside.

**Iteration-by-iteration propagation** (best-case model, windmill ankles, 384p/30fps):

| Iteration $k$ | Informative frames ($e \leq 64$) | Frame 5 error | Frame 10 error | Frame 15 error |
|---------------|----------------------------------|---------------|----------------|----------------|
| 0             | 0, 1, 2                         | 136           | 235            | 272            |
| 1             | 0, 1, 2, 3 (maybe)              | ~100          | ~200           | ~240           |
| 2             | 0–4 (maybe)                      | ~70           | ~170           | ~210           |
| 3             | 0–5 (maybe)                      | ~45           | ~140           | ~185           |

After $K = 4$ iterations: **frame 15 still has ~185 px error**. Tracking has failed catastrophically.

Even with optimistic assumptions about attention-based correction, the wavefront of "corrected" frames advances only ~1–2 frames per iteration. With 4 iterations, correction reaches frames 0–6 at best. Frames 7–18 remain lost.

### Formal Propagation Bound

**Theorem (Search Radius Propagation Limit):**

Given:
- Sequence length $T$, iteration count $K$
- Maximum displacement $d_{\max}$ in the sequence
- Multi-scale search radius $R_{\max}$
- Maximum attention-based correction per iteration: $c_{\text{attn}} \leq R_{\max}$ (the MLP cannot reliably correct by more than one search radius per step)

The maximum recoverable displacement in $K$ iterations is:

$$d_{\text{recover}} = R_{\max} + K \cdot c_{\text{attn}} \leq (K + 1) \cdot R_{\max}$$

For $K = 4$, $R_{\max} = 64$: $d_{\text{recover}} \leq 320$ px.

This bound is optimistic (assumes perfect propagation). Windmill ankle displacement at 384p: $d_{\max} = 272 \leq 320$, suggesting recovery is **theoretically possible** at the lowest resolution. But at 720p ($d_{\max} = 507$) and 1080p ($d_{\max} = 762$), recovery is **provably impossible** within 4 iterations.

However, $c_{\text{attn}} = R_{\max} = 64$ px per iteration is extremely optimistic. A more realistic estimate is $c_{\text{attn}} \approx 15\text{–}30$ px based on the training distribution (pseudo-labels biased toward slow motion, Eq. 4.1):

$$d_{\text{recover}}^{\text{realistic}} = 64 + 4 \times 25 = 164 \text{ px}$$

This fails even at 384p for frames with displacement > 164 px (frames 8–16 of 24).

### (c.3) Cross-Track Contamination via Spatial Attention

The most insidious cascading effect: lost tracks corrupt OTHER tracks through spatial attention (Eq. 2.2).

At each frame $t$, spatial attention computes:

$$\tilde{\mathbf{z}}_{:,t}^{\text{space}} = \text{softmax}\!\left(\frac{\mathbf{Q}_t \mathbf{K}_t^T}{\sqrt{d_k}}\right) \mathbf{V}_t$$

where $\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t \in \mathbb{R}^{N \times d_k}$.

If point $a$ is "lost" (position estimate far from truth), its token $\mathbf{z}_{a,t}$ contains:
1. **Wrong correlation** — matched against wrong image location → encodes wrong texture
2. **Wrong displacement** — $\Delta\hat{\mathbf{p}}_{a,t} \approx 0$ when true displacement is large
3. **Wrong positional encoding** — $\gamma(\hat{\mathbf{p}}_{a,t}) \approx \gamma(\mathbf{p}_{a,0})$ encodes query position, not current

When point $b$ (correctly tracked) attends to point $a$, it receives corrupted information. The corruption magnitude scales with the attention weight $\alpha_{b,a}^{\text{space}}$.

**Corruption fraction during power moves:**

During a windmill at 384p/30fps (frame 15, mid-sequence):
- Lost keypoints: 13/17 = 76% (all except shoulders)
- If tracking 2,500 dense points (hierarchical strategy from prior analysis), the fraction of lost points depends on their distribution

For dense points on the dancer's body, with the same radial velocity distribution:
- Points near core/rotation center ($r < 0.2$m): ~15% of body surface → likely tracked
- Points on extremities ($r > 0.5$m): ~60% of body surface → likely lost

**Conservative estimate**: 50–60% of tracked body points are lost at the mid-sequence frame during a power move.

The spatial attention sees 50–60% corrupted tokens. The impact on correctly-tracked points:

$$\tilde{\mathbf{z}}_{b,t} = \underbrace{\sum_{b': \text{good}} \alpha_{b,b'} \mathbf{v}_{b'}}_{\text{clean signal}} + \underbrace{\sum_{a: \text{lost}} \alpha_{b,a} \mathbf{v}_a}_{\text{corruption}}$$

If attention weights are roughly uniform: corruption fraction ≈ 60%. Even if the model learns to down-weight corrupted tokens (plausible — their correlations look different), the remaining corruption degrades position updates for good tracks.

**False match identity swaps**: The worst specific failure mode. When a lost ankle track's correlation is computed at the wrong location, it may find a strong match to the OTHER ankle (body parts are bilaterally symmetric). This produces a high-confidence but WRONG correlation peak, causing the MLP to produce a large correction in the wrong direction.

The probability of an identity swap is:

$$P(\text{swap}) \approx P(\text{other limb within search radius at wrong position}) \times P(\text{feature similarity} > \tau)$$

During a windmill, both ankles sweep through similar spatial regions on opposite phases. At the wrong position estimate (query position), the OTHER ankle's trajectory may pass within the search radius:

$$P(\text{spatial overlap}) = \frac{2R_{\max} \cdot T_{\text{overlap}}}{d_{\text{separation}} \cdot T}$$

where $T_{\text{overlap}}$ is the number of frames where the other ankle passes within $R_{\max}$ of the wrong position. For a windmill with both legs sweeping: $T_{\text{overlap}} \approx T/4$ (the other leg passes through the query region ~once per rotation). With $R_{\max} = 64$, $d_{\text{separation}} \approx 50\text{–}100$ px between limb paths:

$$P(\text{spatial overlap}) \approx 0.3\text{–}0.6$$

Combined with high feature similarity between symmetric body parts ($P(\text{similarity}) \approx 0.5\text{–}0.8$):

$$P(\text{swap}) \approx 0.15\text{–}0.5 \text{ per lost track}$$

**This is not a rare event.** Identity swaps during power moves are expected for 15–50% of lost extremity tracks.

### Summary: The Poisoning Effect

The cascading failure has three compounding mechanisms:

1. **Self-poisoning** (mild): A lost track's correlation noise adds variance to its own position updates, slowing convergence. Effect: $\sim 2\times$ slower convergence.

2. **Temporal stagnation** (severe): With only $K = 4$ iterations and global temporal attention, correction propagation reaches $\sim$6 frames from the query. Frames beyond this are irrecoverable in batch mode. Effect: **50–75% of sequence frames lost** for fast power moves.

3. **Spatial contamination** (catastrophic when it occurs): Lost tracks corrupt correctly-tracked points through spatial attention. Identity swaps create large, confident errors that propagate through subsequent iterations. Effect: **15–50% of lost tracks swap identity**, and the swap propagates to corrupt neighboring tracks.

### Critical Parameter: Frame Rate vs. Processing Mode

The analysis reveals an unexpected design implication:

**Online (causal) mode at 60fps** may outperform **batch mode at 30fps** for breakdancing:

| Mode | Search Radius Challenge | Per-frame Displacement (ankles, 384p) | Recoverable? |
|------|------------------------|---------------------------------------|-------------|
| Batch, 30fps | Total displacement: 272 px | N/A (all frames initialized at query) | No ($\gg R_{\max}$) |
| Batch, 60fps | Total displacement: 272 px | N/A | No ($\gg R_{\max}$) |
| Online, 30fps | Per-frame only: 28.5 px | One frame at a time | Marginal (L1 covers it) |
| Online, 60fps | Per-frame only: 14.2 px | One frame at a time | **Yes (within L0)** |

In online mode, each frame's initial position is the previous frame's tracked position, so the displacement challenge is per-frame only. At 60fps, the per-frame displacement for even the fastest body parts (14.2 px for ankles) is within the single-scale search radius (16 px).

The tradeoff: online mode suffers from **drift** (accumulated error with no future-frame correction). But for power moves specifically, drift over 1 second of circular motion may be less severe than complete tracking failure in batch mode.

$$\text{Drift}_{\text{online}} = \sum_{t=1}^{T} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$$

$$\mathbb{E}\left[\text{Drift}_{\text{online}}^2\right] = T \sigma_\epsilon^2$$

For $T = 60$ frames (1s at 60fps) and per-frame noise $\sigma_\epsilon \approx 0.5$ px: $\text{Drift}_{\text{RMS}} = \sqrt{60} \times 0.5 \approx 3.9$ px. Acceptable.

### Mitigation Strategies

The analysis suggests three concrete interventions for the breakdancing pipeline:

**M1: Use online mode at 60fps for power move segments**. Pre-classify segments as power-move vs. non-power-move, and switch tracking mode accordingly. Batch mode for toprock/freezes (where future context helps), online mode for windmills/headspins (where search radius is critical).

**M2: Increase $K$ to 8–12 iterations** during power moves (at the cost of $2\text{–}3\times$ compute). This extends the propagation reach to $\sim$12–18 frames and provides more coarse-to-fine refinement steps.

**M3: Warm-start initialization**. Instead of initializing all frames at the query position, use a simple motion model (e.g., constant velocity extrapolation from the first few frames, or optical flow initialization) to provide better starting positions:

$$\hat{\mathbf{p}}_{n,t}^{(0)} = \mathbf{p}_{n,0} + t \cdot \hat{\mathbf{v}}_n$$

where $\hat{\mathbf{v}}_n$ is estimated from short-range optical flow near the query frame. For circular motion, linear extrapolation is wrong, but it puts the initial estimate closer to the true position than the query point, potentially bringing more frames within search radius.
