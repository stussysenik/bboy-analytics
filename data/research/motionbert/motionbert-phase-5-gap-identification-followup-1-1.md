# motionbert — Follow-up 1.1: The analysis produces detailed error estimates (78mm fused MPJPE, velocity SNR ~1:1) but never asks:

_Generated: 2026-03-23T02:51:38.179Z_

---

I don't have filesystem access in this environment — I'll work directly from the comprehensive prior research context provided. Let me address the critical gap: **Validation Protocol Design for Breaking Pose Estimation Without MoCap Ground Truth**.

---

# Validation Protocol for Breaking Pose Estimation: Closing the Unfalsifiability Gap

## The Core Problem

Every quantitative claim in the research — 78mm fused MPJPE, velocity SNR ≈ 1:1, quadratic rotation degradation, 62.2mm composite MPJPE — is **unfalsifiable** in its current form. The pipeline targets breakdancing, but:

1. **No MoCap ground truth exists for breaking** — no lab has captured windmills, headspins, or airflares with marker-based systems
2. **BRACE dataset capabilities are unverified** (Gap 5D) — it may lack 3D annotations entirely
3. **AIST++ contains no breaking** — it covers 10 street dance genres but explicitly excludes breaking
4. **H36M and AMASS** contain no inverted poses, no floor contact, no power moves

Without a measurement protocol, the system cannot be iteratively improved. You can't gradient-descend on a metric you can't compute.

## Proposed Evaluation Framework: Five Complementary Proxy Metrics

No single proxy replaces MoCap ground truth. The strategy is to define **multiple independent proxy metrics** that, taken together, constrain the error space from different directions. If a system scores well on all five, the probability that it's secretly terrible is low.

### Proxy 1: Synthetic Rotation Benchmark from AMASS

**Principle**: If we can't get MoCap data of breaking, we can **synthetically rotate** existing MoCap data to simulate the distribution shift.

**Construction**:

Take AMASS sequences with known 3D ground truth. Apply synthetic rotations to create "pseudo-breaking" poses:

$$\mathbf{p}_{rotated} = R(\theta, \phi, \psi) \cdot \mathbf{p}_{original}$$

where $R(\theta, \phi, \psi)$ is a rotation matrix parameterized by Euler angles:
- $\theta \in [0°, 180°]$: sagittal plane rotation (simulates inversions)
- $\phi \in [0°, 360°]$: transverse plane rotation (simulates spins)
- $\psi \in [-45°, 45°]$: coronal plane rotation (simulates tilts)

**Rendering pipeline**:
1. Take AMASS sequence $\mathbf{P} \in \mathbb{R}^{T \times J \times 3}$ (T frames, J=24 SMPL joints)
2. Apply global rotation: $\mathbf{P}' = R \cdot \mathbf{P}$
3. Project to 2D via random camera: $\mathbf{p}_{2D} = \Pi(K, [R_{cam} | t_{cam}], \mathbf{P}')$
4. Render synthetic image using SMPL mesh + texture (SURREAL-style rendering)
5. Run full pipeline on rendered video
6. Compare output to known $\mathbf{P}'$

**Metric**: Standard MPJPE, but stratified by rotation angle:

$$\text{MPJPE}(\theta) = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \| \hat{\mathbf{p}}_{t,j} - \mathbf{p}'_{t,j} \|_2$$

**What this validates**: The rotation degradation model. The research claims $\Delta\text{MPJPE} \propto \theta^2$ — this benchmark directly tests it. Sample $\theta$ at 15° intervals from 0° to 180° and fit:

$$\text{MPJPE}(\theta) = a + b\theta + c\theta^2$$

If $c \gg b$, the quadratic model holds. If $b \gg c$, degradation is linear. If neither fits, the model is more complex.

**Limitations**: Rotated walking ≠ actual breaking. The body articulation (limb positions relative to torso) is still "normal motion" — only the global orientation changes. This isolates the rotation factor but doesn't capture the combined rotation + unusual articulation challenge.

**Tensor shapes for implementation**:
- AMASS input: $(B, T, 24, 3)$ — batch of sequences
- Rotation matrices: $(B, 3, 3)$ — one per sequence (or per-frame for spin simulation)
- Rendered images: $(B, T, H, W, 3)$ — synthetic video frames
- Pipeline output: $(B, T, J, 3)$ — estimated 3D joints
- Ground truth: $(B, T, J, 3)$ — rotated AMASS joints (after SMPL→H36M mapping)

**Expected cost**: ~2 days to build renderer + evaluation script. AMASS is freely available. No new data collection needed.

---

### Proxy 2: Cross-View Geometric Consistency (Multi-Camera Breaking Footage)

**Principle**: If two cameras observe the same dancer from different angles, their 3D reconstructions must agree (up to a rigid transform). Disagreement quantifies error without needing ground truth.

**Setup**: Obtain or record breaking footage from 2+ synchronized cameras. Competition footage sometimes has multiple broadcast angles with timecode. Alternatively, record a controlled session with 2 consumer cameras.

**Formulation**: Let $\hat{\mathbf{P}}_A \in \mathbb{R}^{T \times J \times 3}$ and $\hat{\mathbf{P}}_B \in \mathbb{R}^{T \times J \times 3}$ be the 3D pose estimates from cameras A and B. After Procrustes alignment:

$$R^*, t^*, s^* = \arg\min_{R, t, s} \sum_{t,j} \| s R \hat{\mathbf{p}}_{A,t,j} + t - \hat{\mathbf{p}}_{B,t,j} \|^2$$

The **cross-view consistency error** (CVCE) is:

$$\text{CVCE} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \| s^* R^* \hat{\mathbf{p}}_{A,t,j} + t^* - \hat{\mathbf{p}}_{B,t,j} \|_2$$

**Key property**: CVCE provides a **lower bound** on actual MPJPE. If the true pose is $\mathbf{P}^*$:

$$\text{CVCE} \leq \text{MPJPE}_A + \text{MPJPE}_B$$

by the triangle inequality. So $\text{CVCE}/2$ is a lower bound on average MPJPE. More precisely, if errors from the two views are independent with equal variance:

$$\mathbb{E}[\text{CVCE}^2] = \mathbb{E}[\| \epsilon_A - \epsilon_B \|^2] = 2\sigma^2$$

$$\sigma \approx \frac{\text{CVCE}}{\sqrt{2}}$$

This gives a **statistical estimate** of per-view MPJPE from consistency alone.

**Stratified analysis**: Compute CVCE separately for:
- Toprock frames (upright) — baseline, should match standard benchmarks
- Footwork frames (low to ground) — moderate challenge
- Power move frames (inverted/rotating) — maximum challenge
- Freeze frames (static) — tests stability

This stratification is the most valuable output — it reveals **which move categories** degrade most.

**Caveat**: Procrustes alignment removes global scale, rotation, and translation. This means CVCE doesn't capture systematic errors shared between views (e.g., if both views produce the same depth error). It also means PA-MPJPE is the closest standard metric, not MPJPE. To get MPJPE-equivalent, calibrate the cameras and use the known extrinsics instead of Procrustes.

With calibrated cameras, the alignment is known, and:

$$\text{CVCE}_{calibrated} = \frac{1}{TJ} \sum_{t,j} \| T_{A \to B} \hat{\mathbf{p}}_{A,t,j} - \hat{\mathbf{p}}_{B,t,j} \|_2$$

where $T_{A \to B}$ is the known rigid transform between camera coordinate systems.

**Expected cost**: Medium — requires either multi-angle competition footage (obtainable from Red Bull BC One broadcasts, which use 6+ camera angles) or a controlled recording session (~1 day setup, 1 day processing).

---

### Proxy 3: Physical Plausibility Constraints (Self-Supervised)

**Principle**: Even without ground truth, we know physics. Estimated poses that violate physical laws are wrong. The **rate** of physical violations is a proxy for error magnitude.

**Constraints to check**:

**3a. Bone length consistency**:

$$\mathcal{L}_{bone} = \frac{1}{T \cdot |\mathcal{B}|} \sum_{t=1}^{T} \sum_{(i,j) \in \mathcal{B}} \left| \| \hat{\mathbf{p}}_{t,i} - \hat{\mathbf{p}}_{t,j} \| - l_{ij} \right|$$

where $\mathcal{B}$ is the set of bone pairs, $l_{ij}$ is the bone length (estimated from the median over the sequence). For a perfect reconstruction, $\mathcal{L}_{bone} = 0$. Under noise with MPJPE $= \sigma$:

$$\mathbb{E}[\mathcal{L}_{bone}] \approx \sigma \sqrt{\frac{2}{\pi}} \left(1 - \frac{1}{2} \frac{l_{ij}^2}{\sigma^2 + l_{ij}^2}\right)$$

For $\sigma = 70\text{mm}$ and typical bone lengths $l \approx 250-450\text{mm}$: $\mathcal{L}_{bone} \approx 30-40\text{mm}$.

**Measurement**: Track $\mathcal{L}_{bone}$ per-frame. Spikes indicate catastrophic failures (joint swaps, identity switches). The **coefficient of variation** of bone lengths is perhaps more informative than the mean:

$$\text{CV}_{bone}(i,j) = \frac{\text{std}_t(\| \hat{\mathbf{p}}_{t,i} - \hat{\mathbf{p}}_{t,j} \|)}{\text{mean}_t(\| \hat{\mathbf{p}}_{t,i} - \hat{\mathbf{p}}_{t,j} \|)}$$

For clean MoCap: $\text{CV} < 0.01$. For good estimation: $\text{CV} < 0.05$. For degraded estimation: $\text{CV} > 0.10$.

**3b. Ground penetration**:

$$\mathcal{V}_{ground}(t) = \sum_{j=1}^{J} \max(0, -\hat{z}_{t,j})$$

where $\hat{z}_{t,j}$ is the height of joint $j$ relative to the estimated floor plane. Any negative value means ground penetration. The floor plane can be estimated from the lowest joint positions across the sequence (robust fit via RANSAC on the 5th percentile of joint heights).

**Why this is particularly informative for breaking**: Floor contact is constant during power moves. Ground penetration rate directly measures how well the system handles the floor constraint:

$$\text{Penetration rate} = \frac{|\{(t,j) : \hat{z}_{t,j} < -\epsilon\}|}{T \times J}$$

For $\epsilon = 10\text{mm}$ (tolerance), a good system should have penetration rate $< 5\%$. Higher rates indicate systematic failure in depth estimation near the floor.

**3c. Joint angle limits**:

Human joints have biomechanical limits. The knee can't hyperextend beyond ~10°. The elbow can't bend backward. Define angle limits $[\alpha_{min}, \alpha_{max}]$ for each joint and count violations:

$$\mathcal{V}_{angle}(t) = \sum_{j \in \mathcal{J}} \mathbb{1}[\hat{\alpha}_{t,j} \notin [\alpha_{min}^j, \alpha_{max}^j]]$$

**Caveat for breaking**: Breakers have extreme flexibility. Standard biomechanical limits may need to be relaxed by 20-30%. A b-boy in a hollowback has thoracic extension that would be flagged as "impossible" for a normal person. Use relaxed limits (e.g., extend all ranges by 1.5×) to avoid false positives.

**3d. Acceleration smoothness (jerk)**:

$$\text{Jerk}_{t,j} = \| \dddot{\hat{\mathbf{p}}}_{t,j} \|$$

High jerk indicates temporal incoherence — the estimate is jumping between frames. For smooth human motion, jerk should be below a threshold. However, breaking involves genuinely high-jerk moments (impact transitions, stops). So this metric is useful for **non-transition frames** but not universally.

**Composite physical plausibility score**:

$$\text{PhyScore} = w_1 \cdot (1 - \text{CV}_{bone}) + w_2 \cdot (1 - \text{PenRate}) + w_3 \cdot (1 - \text{ViolRate}_{angle}) + w_4 \cdot (1 - \text{NormJerk})$$

where each component is normalized to $[0, 1]$. This gives a unitless quality score that can track improvement across pipeline iterations.

**Expected cost**: Low — all computed from estimated joints, no additional data needed. Implementation: ~1 day.

---

### Proxy 4: Expert Annotation on a Small Validation Set (Semi-Ground-Truth)

**Principle**: A small set of manually annotated frames provides sparse but trustworthy ground truth.

**Protocol**:

1. **Select 200 frames** from breaking footage, stratified:
   - 40 toprock (upright, baseline)
   - 40 footwork (crouching, moderate)
   - 40 power moves — aerial (windmills, flares)
   - 40 power moves — inverted (headspins, freezes)
   - 40 transitions (between categories)

2. **2D annotation**: Have 3 annotators mark all visible joint positions on each frame. Take the median as pseudo-ground-truth. Compute inter-annotator agreement as a noise floor:

$$\sigma_{annotator} = \frac{1}{3 \cdot J} \sum_{a=1}^{3} \sum_{j=1}^{J} \| \mathbf{p}_{a,j} - \bar{\mathbf{p}}_j \|$$

Expected $\sigma_{annotator} \approx 3-8$ pixels at 1080p for visible joints, higher for occluded joints.

3. **2D evaluation**: Compare pipeline's 2D detections to manual annotations:

$$\text{PCK}@k = \frac{|\{j : \| \hat{\mathbf{p}}_{2D,j} - \mathbf{p}^*_{2D,j} \| < k \cdot d_{torso}\}|}{J_{visible}}$$

where $d_{torso}$ is the torso diameter (standard normalization). Report PCK@0.1 and PCK@0.2.

4. **3D ordinal annotation**: For each frame, annotators provide **ordinal depth rankings** for joints rather than metric depth (which is impossible from a single image). E.g., "left hand is closer to camera than right hand", "head is further than pelvis."

$$\text{OrdinalAcc} = \frac{|\{(i,j) : \text{sign}(\hat{z}_i - \hat{z}_j) = \text{sign}(z^*_i - z^*_j)\}|}{|\mathcal{P}|}$$

where $\mathcal{P}$ is the set of annotated joint pairs with clear depth ordering.

This is the **only practical way** to evaluate 3D accuracy from monocular footage without MoCap. Ordinal depth is reliably annotatable by humans (inter-annotator agreement ~90-95% for clearly separated joints), and it directly tests the depth estimation component that is most likely to fail on inverted poses.

**What ordinal accuracy tells us**: For normal poses, lifting networks achieve ~95% ordinal accuracy. If breaking poses drop to ~70-75%, it quantifies the depth confusion. Below ~60% means the model is essentially guessing depth ordering — worse than random for some joint pairs (systematic left-right or front-back confusion).

**Expected cost**: 200 frames × 3 annotators × ~2 min/frame = ~20 person-hours. Plus ordinal depth annotation: ~10 person-hours. Feasible for a small validation effort.

---

### Proxy 5: Action Recognition as a Downstream Task Metric

**Principle**: If the 3D pose estimates are accurate enough for the pipeline's actual purpose (analyzing breaking), then a **downstream task** should work. Action recognition serves as an end-to-end proxy.

**Protocol**:

1. Collect 100+ breaking clips with labeled moves (windmill, flare, headspin, freeze type, etc.)
2. Extract 3D pose sequences using the pipeline
3. Train a simple classifier (linear probe or 1-layer LSTM) on the 3D pose features
4. Measure classification accuracy

**The key insight**: Classification accuracy from 3D pose features vs. classification accuracy from raw video features (InternVideo 2.5) reveals how much information the 3D estimation preserves.

$$\Delta_{info} = \text{Acc}_{video} - \text{Acc}_{3D pose}$$

- $\Delta_{info} \approx 0$: 3D pose captures all relevant information → estimation quality is sufficient
- $\Delta_{info} \approx 10-20\%$: Moderate information loss → estimation degrades some moves
- $\Delta_{info} > 30\%$: Severe information loss → 3D estimation is failing on the moves that distinguish categories

**Stratified by difficulty**: Report per-class accuracy. If headspins (inverted) have 40% accuracy while toprock has 90%, the gap directly quantifies where the pose estimation fails.

**Expected cost**: Requires labeled breaking clips (potentially derivable from YouTube compilations with manual labeling, or from BRACE if it has move labels). Classifier training is trivial. Total: ~3-5 days.

---

## Unified Evaluation Protocol

### Hierarchy of Evidence

The five proxies form a pyramid of decreasing rigor but increasing domain relevance:

```
        /\
       /  \         Proxy 1: Synthetic AMASS rotation
      /    \        (full 3D GT, but wrong motion domain)
     /------\
    /        \      Proxy 2: Cross-view consistency
   /          \     (no GT, but real breaking, geometric bound)
  /------------\
 /              \   Proxy 3: Physical plausibility
/                \  (no GT, no multi-view, but physics never lies)
------------------
Proxy 4: Expert annotation (sparse GT, real breaking, 2D + ordinal 3D)
Proxy 5: Downstream task accuracy (end-to-end, task-relevant, no explicit GT)
```

### Combined Reporting

For each pipeline configuration (MotionBERT alone, SAM-Body4D alone, fused), report:

| Metric | Toprock | Footwork | Power (aerial) | Power (inverted) | Freezes | Transitions |
|--------|---------|----------|----------------|-------------------|---------|-------------|
| MPJPE (Proxy 1, at θ=0°/90°/180°) | — | — | — | — | — | — |
| CVCE (Proxy 2) | — | — | — | — | — | — |
| Bone CV (Proxy 3a) | — | — | — | — | — | — |
| Penetration rate (Proxy 3b) | — | — | — | — | — | — |
| 2D PCK@0.2 (Proxy 4) | — | — | — | — | — | — |
| Ordinal depth acc (Proxy 4) | — | — | — | — | — | — |
| Move classification (Proxy 5) | — | — | — | — | — | — |

This table is the **evaluation dashboard**. Every pipeline change should move numbers in this table. If a change improves one metric but degrades another, the stratification reveals whether the tradeoff is acceptable (e.g., improving inverted poses at the cost of slightly degrading upright poses is a good trade for breaking).

### Minimum Viability Thresholds

Based on the pipeline's purpose (judging analysis, not surgical planning), define thresholds:

| Metric | Minimum Viable | Target | Source of Threshold |
|--------|---------------|--------|---------------------|
| Bone CV | < 0.08 | < 0.03 | 0.08 = visually noticeable jitter |
| Ground penetration rate | < 10% | < 2% | 10% = one joint underground every 10 frames |
| 2D PCK@0.2 (power moves) | > 70% | > 85% | 70% = most joints roughly correct |
| Ordinal depth accuracy (inverted) | > 75% | > 90% | 75% = clearly better than chance (50%) |
| Move classification accuracy | > 60% | > 80% | 60% = distinguishes major categories |
| CVCE (power moves) | < 120mm | < 60mm | 120mm ≈ half a forearm length |

**Critical red line**: If ordinal depth accuracy on inverted poses drops below 60%, the 3D estimation is not meaningfully capturing depth for these poses. The pipeline should fallback to 2D-only analysis for inverted frames rather than producing misleading 3D estimates.

---

## Addressing the Specific Numerical Claims

### Validating the 78mm Fused MPJPE Estimate

The 78mm claim (corrected from 72mm in Gap 2A) can be **bounded** using Proxy 2:

- If calibrated CVCE on power moves is ~110mm, then per-view MPJPE ≈ $110/\sqrt{2}$ ≈ 78mm — consistent
- If calibrated CVCE is ~60mm, then per-view MPJPE ≈ 42mm — much better than estimated
- If calibrated CVCE is ~200mm, then per-view MPJPE ≈ 141mm — much worse than estimated

The cross-view consistency check is the **fastest path** to validating or refuting the 78mm estimate.

### Validating the Velocity SNR ≈ 1:1 Estimate

Velocity SNR can be validated using Proxy 3 (jerk analysis):

$$\text{SNR}_v = \frac{\text{std}(\dot{\mathbf{p}}_{smooth})}{\text{std}(\dot{\mathbf{p}} - \dot{\mathbf{p}}_{smooth})}$$

where $\dot{\mathbf{p}}_{smooth}$ is obtained by Savitzky-Golay filtering with a physically motivated window (e.g., 100ms). If the residual after smoothing has comparable magnitude to the smooth signal, SNR ≈ 1:1 is confirmed. This can be computed from any pipeline output — no ground truth needed.

Alternatively, during **freeze frames** (where true velocity is ~0):

$$\hat{\sigma}_v = \text{std}(\dot{\hat{\mathbf{p}}}_{t,j}) \quad \text{during freeze}$$

This gives a direct estimate of velocity noise, since the true velocity signal is zero. Compare to velocity magnitude during power moves to get SNR.

### Validating the Quadratic Rotation Degradation

Proxy 1 directly tests this. Run the synthetic rotation benchmark at $\theta = 0°, 15°, 30°, \ldots, 180°$ and fit:

$$\text{MPJPE}(\theta) = \sum_{k=0}^{K} c_k \theta^k$$

Use AIC/BIC to select the polynomial order. If quadratic ($K=2$) is selected, the claim holds. Report the coefficients:

$$\text{MPJPE}(\theta) \approx 39.2 + c_1 \theta + c_2 \theta^2 \text{ mm}$$

where $c_1$ captures linear sensitivity and $c_2$ captures the accelerating degradation. The research predicts $c_1 \approx 0$ and $c_2 \approx 0.003$ (to reach ~90mm at 135°). Measuring these coefficients directly replaces speculation with data.

---

## Implementation Priority

1. **Proxy 3 (Physical plausibility)** — implement first. Zero additional data needed. Immediate signal. ~1 day.
2. **Proxy 1 (Synthetic rotation)** — implement second. Uses existing AMASS data. Directly validates the rotation model. ~2 days.
3. **Proxy 4 (Expert annotation)** — start annotation in parallel with 1 and 2. ~1 week for annotation, 1 day for evaluation code.
4. **Proxy 5 (Downstream task)** — implement after the pipeline produces 3D outputs. ~3-5 days.
5. **Proxy 2 (Cross-view)** — requires multi-camera footage. Start acquisition/search immediately but expect this to take longest. ~1-2 weeks.

The first two proxies can be running within 3 days, providing quantitative feedback for every pipeline decision thereafter. That's the minimum needed to make the research **falsifiable**.

---

## Key Conclusion

The gap isn't just "we need a benchmark" — it's that the research has been optimizing a system design based on **estimated error magnitudes that have never been measured**. The fusion weights, the progressive unfreezing schedule, the confidence thresholds — all are designed around the 78mm / 1:1 SNR estimates. If actual MPJPE on breaking is 40mm (better) or 150mm (worse), the entire design shifts.

The protocol above costs roughly 2-3 weeks of effort and produces a reusable evaluation harness. Every subsequent architectural decision (should we add floor contact constraints? should we use JOSH? should we retrain with rotation augmentation?) can be measured against the dashboard rather than argued from first principles.

The two most impactful immediate actions:

1. **Run bone length CV and ground penetration on any available breaking clip** — takes hours, gives immediate signal on whether the system is catastrophically failing or roughly working
2. **Build the synthetic rotation benchmark from AMASS** — takes days, resolves the quadratic degradation debate with actual numbers
