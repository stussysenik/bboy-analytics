# cotracker3 — Follow-up 2.1: Every displacement number, error estimate, and threshold recommendation is derived from biomechanica

_Generated: 2026-03-23T04:09:35.149Z_

---

# Empirical Verification Gap: What's Proven, What's Calculated, What's Guessed

## 1. Taxonomy of Unverified Claims

Every numerical claim in the prior analysis falls into one of three epistemic categories. The distinction matters because the failure modes are different for each.

### Category A: Architecture Constants (Verified by Code Inspection)

These are facts readable from CoTracker3's source code. They require no experiments:

| Claim | Source | Status |
|-------|--------|--------|
| Output shape $$(B, T, N, 2)$$ + $$(B, T, N)$$ | `cotracker/predictor.py` return statement | **Verified** |
| Internal resolution: short side = 384px | `DEFAULT_LONG_SIDE_LENGTH` parameter | **Verified** |
| Correlation grid $$S=4$$, stride $$\sigma=4$$ → 9×9 samples | `corr_radius` and `corr_levels` in model config | **Verified** |
| Backbone stride = 4 (feature resolution = input/4) | CNN architecture definition | **Verified** |
| Iterative refinement: $$N_{\text{iter}} = 4$$ | `num_updates` parameter | **Verified** |
| Visibility head: single Linear → sigmoid | `vis_predictor` module definition | **Verified** |
| Sliding window $$T_w = 60$$, overlap = 20 | `window_len` and `overlap` parameters | **Verified** |

These are the **only** claims that don't need empirical validation. Everything else requires measurement.

### Category B: Physics-Derived Calculations (Correct Math, Unverified Inputs)

The displacement calculations follow from kinematics:

$$d_{\text{px/frame}} = \frac{v_{\text{m/s}} \times \text{scale}_{\text{px/m}}}{\text{fps}}$$

The math is trivially correct. But the **inputs** are where the uncertainty lives:

| Input Parameter | Value Used | Actual Range | Source of Uncertainty |
|----------------|------------|--------------|----------------------|
| Windmill rotation period | 0.7s | 0.5–1.2s | Varies enormously by skill level. A world-class bboy (Kid David, Phil Wizard) does it in ~0.5s; a beginner takes 1.2s. The pipeline must handle the full range. |
| Headspin rotation period | 0.4s | 0.25–0.8s | Competition headspins at peak speed can hit 0.25s/rev. The 0.4s figure is moderate. |
| Effective leg radius during windmill | 0.93m | 0.7–1.0m | Depends on leg extension. Windmill with tucked legs vs. extended legs changes radius by ~30%. |
| Arm span | 1.75m | 1.55–1.95m | Bboys vary in height. But more importantly, arms are rarely fully extended during power moves — effective radius is shorter. |
| Frame occupancy (60% of frame height) | 0.6 | 0.3–0.9 | **This is the biggest source of error.** Competition footage ranges from wide shots (0.3) to close-ups (0.9). A close-up doubles all displacement numbers. |

The displacement table should be presented as **ranges**, not point estimates. Let me recalculate with realistic bounds.

#### Revised Displacement Ranges (1080p)

For windmill feet at 30fps:

$$d_{\min} = \frac{2\pi \times 0.7}{1.2} \times \frac{0.3 \times 1080}{1.75} \times \frac{1}{30} = \frac{3.67 \times 185}{30} \approx 23 \text{ px/frame}$$

$$d_{\max} = \frac{2\pi \times 1.0}{0.5} \times \frac{0.9 \times 1080}{1.55} \times \frac{1}{30} = \frac{12.57 \times 627}{30} \approx 263 \text{ px/frame}$$

$$d_{\text{typical}} = \frac{2\pi \times 0.93}{0.7} \times \frac{0.6 \times 1080}{1.75} \times \frac{1}{30} \approx 104 \text{ px/frame}$$

The range is **23–263 px/frame** with a "typical" estimate of 104. The prior analysis presented only the typical value. The upper bound (263px) is nearly 3× larger and would exceed even the $$S=8$$ search radius.

**The key uncertainty**: frame occupancy (0.3–0.9) contributes a 3× multiplier. This single unmeasured variable dominates the entire displacement analysis. Without measuring actual frame occupancy across a corpus of battle footage, all displacement numbers carry ±3× uncertainty.

### Category C: Pure Assertions (No Derivation, No Measurement)

These claims have **no basis** — they are plausible-sounding numbers without any grounding:

| Claim | Value | Basis | Why It's Suspect |
|-------|-------|-------|------------------|
| "CoTracker3 Error (px)" for windmill | 3.0–6.0 px | None stated | This column appears in a table as if measured. It was not. There is no published evaluation of CoTracker3 on dance footage. |
| "Practical range of 60–120px" | 60–120 px | Undefined | The theoretical single-iteration limit (64px input space) is architecture-derived. The "practical multi-iteration" range is asserted without reference to any experiment or ablation study. |
| Visibility threshold 0.5 for binary | 0.5 | TAP-Vid convention | The 0.5 threshold comes from TAP-Vid benchmark evaluation protocol. Whether it's appropriate for breaking is an open question — the prior analysis acknowledges this but then proposes alternatives (0.2, 0.3, 0.4, 0.6) with zero empirical justification. |
| One-Euro filter $$f_{c,\min} = 1.0$$, $$\beta = 0.007$$ | Specific floats | None | These are presented as "parameters for bboy footage" but are not derived from any tuning on actual bboy data. They're in the ballpark of general hand-tracking parameters (Meta's AR/VR work), but domain transfer is unvalidated. |
| "40% mass-occlusion threshold" | 40% | None | Threshold for flagging unreliable segments. Completely arbitrary. |
| "20px MoveNet consistency threshold" | 20 px | None | Threshold for pose validation. Plausible but ungrounded. |
| "10px re-emergence discrepancy" | 10 px | None | Threshold for detecting lost tracks. No justification. |
| "Position drift of 5–15 pixels during 30-frame occlusion" | 5–15 px | None | Asserted in the occlusion analysis. Not measured. |
| Window stitching error "< 0.5 px RMS, can spike to 2–3 px" | 0.5–3 px | None | Not measured on any motion type, let alone breaking. |

---

## 2. What Published Benchmarks Actually Tell Us

### 2.1 TAP-Vid Results

CoTracker3's published evaluation uses TAP-Vid (DAVIS, Kinetics, and Kubric subsets). The relevant metrics:

$$\text{AJ} = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \text{Jaccard}(d) \quad \text{where } \mathcal{D} = \{1, 2, 4, 8, 16\} \text{ pixel thresholds}$$

$$\text{OA} = \text{Occlusion Accuracy (binary classification accuracy of visibility)}$$

$$\delta^{\text{avg}}_{\text{vis}} = \text{Average position accuracy on visible points only}$$

CoTracker3's reported numbers on TAP-Vid-DAVIS (the most relevant subset — real video, not synthetic):

| Metric | CoTracker3 | CoTracker2 | TAPIR |
|--------|-----------|-----------|-------|
| AJ | ~62 | ~58 | ~55 |
| $$\delta^{\text{avg}}_{\text{vis}}$$ | ~78 | ~74 | ~72 |
| OA | ~89 | ~87 | ~86 |

### 2.2 Why TAP-Vid Numbers Don't Transfer to Breaking

TAP-Vid-DAVIS consists of 30 videos with the following motion characteristics:

| Property | TAP-Vid-DAVIS | Bboy Battle Footage |
|----------|---------------|---------------------|
| Max displacement/frame | ~20px (median), ~50px (95th percentile) | 23–263px (estimated, see above) |
| Self-occlusion duration | 5–15 frames typical | 8–40+ frames |
| Rotation speed | Rare; <2 rad/s when present | 9–16 rad/s |
| Motion blur | Moderate | Severe during power moves |
| Camera motion | Professional steady/tracking | Handheld, crowd movement |
| Number of occluding objects | Usually 1 (object behind another) | Self-occlusion: same object, complex topology |

The distribution mismatch is severe. TAP-Vid's 95th-percentile displacement is below breakdancing's **median** displacement during power moves. This means CoTracker3's published accuracy numbers are evaluated on a fundamentally different motion regime than the target domain.

Formally, the domain gap can be characterized as a covariate shift in the displacement distribution:

$$P_{\text{TAP-Vid}}(d) \neq P_{\text{bboy}}(d)$$

where $$d$$ is per-frame displacement. The tracker's error function $$\epsilon(d)$$ is likely monotonically increasing in $$d$$ (more displacement → harder tracking), and we're evaluating in a regime where $$d$$ is 3–10× larger than training/evaluation data. Extrapolating $$\epsilon(d)$$ into this regime without measurement is unreliable.

### 2.3 The Kubric Synthetic Data Argument

CoTracker3 trains on Kubric (synthetic scenes with rigid objects bouncing). The authors argue that training on simple synthetic data generalizes to real video. This is plausible for the **feature matching** mechanism (appearance doesn't change much across domains), but **not** for the learned motion priors in the transformer.

The transformer's temporal attention learns implicit motion models from training data. Kubric objects have:
- Roughly linear trajectories (ballistic motion under gravity)
- No articulated self-occlusion
- No periodic circular motion

A bboy's limb during a windmill follows:

$$\mathbf{p}(t) = \mathbf{c} + R(t) \cdot \mathbf{r}_0$$

where $$R(t)$$ is a rotation matrix and $$\mathbf{c}$$ is the center of rotation. The transformer has never seen this motion pattern during training. It must rely purely on correlation matching, with the learned motion prior actively working against it (predicting approximately linear continuation when the true motion is circular).

This is a testable hypothesis: if the transformer's motion prior is harmful for circular motion, then **reducing the number of refinement iterations** (using correlation alone) might actually improve tracking on power moves. This would be a diagnostic experiment — if accuracy improves with fewer iterations on circular motion but degrades on linear motion, it confirms the motion prior mismatch.

---

## 3. The "Practical Range of 60–120px" Claim: Deconstructing It

This is the most consequential unverified claim in the prior analysis because all downstream recommendations depend on it.

### 3.1 Theoretical Maximum

Per iteration, the maximum correction is $$S \times \sigma \times s_{\text{backbone}} = 4 \times 4 \times 4 = 64$$ input pixels. With $$N_{\text{iter}} = 4$$ iterations, the theoretical maximum is $$4 \times 64 = 256$$ px.

But this assumes:
1. The correlation peak at iteration $$k$$ is at the maximum offset ($$\pm 16$$ feature pixels)
2. The position update exactly follows the correlation peak
3. The correlation peak at iteration $$k+1$$ is again at the maximum offset in the same direction

Assumption 3 is the weak link. After correcting by 64px in iteration 1, the new search window is centered at the corrected position. If the true position is still 192px away, iteration 2 needs another max-offset correction. But the correlation signal at 128px from the true position (after iteration 1's correction) is **weaker** than at 192px (before correction), because:

$$\text{corr}(\mathbf{f}_{query}, \mathbf{f}_{target+\Delta}) \sim \exp\left(-\frac{|\Delta|^2}{2\sigma_f^2}\right)$$

where $$\sigma_f$$ is the effective feature matching radius. At $$|\Delta| = 128$$ px, the correlation is already weak. The iterative process is chasing a signal that gets stronger as it converges, but each step must clear a minimum signal-to-noise threshold to produce a useful correction.

### 3.2 What Would Determine the Practical Range

The practical range is determined by the **minimum correlation SNR** needed for a reliable correction. This depends on:

1. **Feature discriminability**: how distinct is the tracked point's feature from nearby features? For a hand during a windmill, the feature is a patch of skin/clothing — not very discriminative against other skin patches (the other hand, the forearm).

2. **Background clutter**: in a battle circle, the background contains other people, floor textures, and the opposing dancer. More clutter → lower SNR.

3. **Motion blur**: at high angular velocity, the feature appearance changes significantly between frames. The correlation between a sharp feature in frame $$t$$ and a motion-blurred feature in frame $$t+1$$ is degraded:

$$\text{corr}_{\text{blur}} \approx \text{corr}_{\text{sharp}} \times \text{sinc}\left(\frac{v \cdot \tau_{\text{exposure}}}{2\sigma_f}\right)$$

where $$\tau_{\text{exposure}}$$ is the shutter time and $$v$$ is the velocity. For a foot moving at 10 m/s with a 1/60s shutter:

$$\text{blur extent} = 10 \times \frac{1}{60} \approx 0.17\text{m} \approx 63\text{px at 1080p}$$

A 63-pixel motion blur smeared across the feature **destroys** the correlation signal. The feature extraction CNN (ResNet backbone) may extract *something* from the blurred region, but it's a fundamentally different feature vector than the sharp original.

### 3.3 The Honest Statement

The "practical range of 60–120px" is a guess positioned between the single-iteration theoretical maximum (64px) and the multi-iteration theoretical maximum (256px). Without running experiments that measure tracking accuracy vs. displacement magnitude, we cannot know where the actual breakdown occurs.

What we can say from first principles:
- The **hard floor** is 64px (single iteration, guaranteed reachable if correlation peak exists)
- The **hard ceiling** is 256px (impossible to exceed regardless of conditions)
- Motion blur **reduces effective range** because it degrades the correlation signal
- Feature ambiguity (skin vs. skin) **reduces effective range** because it increases false-match probability

A more defensible characterization:

$$r_{\text{practical}} \in [64, 256] \text{ px, with } r_{\text{practical}} \approx 64 + f(\text{SNR}, N_{\text{iter}})$$

where $$f$$ is an unknown, monotonically-increasing-in-SNR function. For high-SNR conditions (textured objects, no blur), $$r_{\text{practical}} \to 256$$. For low-SNR conditions (skin texture, heavy motion blur, competing similar features), $$r_{\text{practical}} \to 64$$.

Breakdancing during power moves is a **low-SNR regime**. The practical range is likely closer to 64px than to 256px — the opposite of what you'd want.

---

## 4. The Visibility Threshold Recommendations: Why They're Unfounded

The prior analysis proposed domain-specific thresholds:

| Use Case | Proposed $$\tau$$ | Basis |
|----------|-------------------|-------|
| Pose estimation supplement | 0.3 | "pose model can disambiguate" |
| Move boundary detection | 0.6 | "false positives cause mis-segmentation" |
| Scoring/judging | 0.4 | "balance coverage and noise" |
| Music-motion alignment | 0.2 | "aggregate signals robust to noise" |

None of these have been validated. To understand why this matters, consider what threshold selection actually requires.

### 4.1 Threshold Selection as a Classification Problem

The visibility score $$v \in (0, 1)$$ must be thresholded into "use this point" vs. "discard this point." The optimal threshold minimizes a task-specific loss:

$$\tau^* = \arg\min_\tau \mathbb{E}\left[\mathcal{L}_{\text{task}}(\hat{y}(\tau), y)\right]$$

where $$\hat{y}(\tau)$$ is the downstream prediction using points with $$v > \tau$$, and $$y$$ is the ground truth.

For **pose estimation** (proposed $$\tau = 0.3$$):

$$\mathcal{L}_{\text{pose}} = \sum_k \|\hat{j}_k - j_k\|_2$$

where $$\hat{j}_k$$ are predicted joint positions. Including low-visibility tracks ($$\tau = 0.3$$) is helpful **only if** the expected position error of a point with $$v = 0.35$$ is less than the error introduced by not having that point. This depends on:

- How accurate are CoTracker3's positions at $$v = 0.35$$ for breakdancing? **Unknown.**
- How well does the downstream pose model handle noisy input points? **Depends on the specific model.**
- What's the marginal value of an additional noisy point vs. skeletal inference from reliable points? **Task-dependent.**

Without measuring the **actual distribution** of position errors conditioned on visibility score — $$P(\epsilon | v, \text{move type})$$ — threshold selection is guesswork.

### 4.2 What the Visibility Distribution Looks Like (Predicted, Unverified)

During a windmill, the visibility score for a tracked wrist point would follow a roughly periodic pattern:

$$v_{\text{wrist}}(t) \approx \begin{cases} 0.7\text{–}0.95 & \text{wrist facing camera (front half of rotation)} \\ 0.1\text{–}0.4 & \text{wrist behind torso (back half of rotation)} \end{cases}$$

The transition should be sharp (wrist goes behind torso in 2–3 frames at windmill speed). But CoTracker3's visibility head was trained on Kubric (rigid objects with simple occlusion geometry). How it responds to a **limb sweeping behind a rotating torso** — a topologically complex self-occlusion with no rigid-object analog — is genuinely unknown.

Plausible failure modes:
1. **Visibility score stays high** (0.6–0.8) even during full occlusion, because the transformer infers that the point should exist and keeps the visibility high. This would make all thresholds > 0.5 useless for detecting true occlusion.
2. **Visibility score drops to near-zero** during motion blur even when the point is visible, because the feature match fails. This would cause false occlusion detections during the fastest (and most visible) portions of the rotation.
3. **Visibility score oscillates rapidly** between frames due to motion blur inconsistency, creating noise that any fixed threshold handles poorly.

Each failure mode demands a different threshold strategy. Without measuring which one actually occurs, the recommended thresholds are noise.

### 4.3 The Adaptive Threshold Proposal

The prior analysis proposed detecting occlusion events via temporal derivative:

$$\text{occ\_event}_{t,n} = \begin{cases} 1 & \text{if } v_{t,n} < \tau \text{ AND } \frac{dv}{dt}\bigg|_{t,n} < -\delta \\ 0 & \text{otherwise} \end{cases}$$

with $$\tau = 0.4, \delta = 0.15$$ per frame. This is a reasonable **framework** but the specific parameters are ungrounded. The derivative threshold $$\delta = 0.15$$ means a visibility drop of 0.15 per frame. At 30fps, this is a drop from 0.6 to 0.0 in 4 frames — consistent with a fast occlusion event. But whether CoTracker3's visibility actually exhibits this behavior, or whether it produces smoother/sharper transitions, is empirical.

---

## 5. The Smoothing Parameters: Transferred Without Validation

The One-Euro filter parameters were proposed as:

$$f_{c,\min} = 1.0 \text{ Hz}, \quad \beta = 0.007, \quad d_{\text{cutoff}} = 1.0 \text{ Hz}$$

These come from the general One-Euro filter literature and Meta's hand-tracking work. The transfer assumptions:

| Assumption | Hand Tracking (Source Domain) | Bboy Tracking (Target Domain) | Valid? |
|------------|-------------------------------|-------------------------------|--------|
| Noise characteristics | Sensor noise, ~Gaussian | Tracking error, non-Gaussian (includes jumps) | **No** — tracking errors have heavy tails |
| Motion dynamics | Hand gestures, 0.5–3 m/s | Power moves, 2–15 m/s | **Partially** — $$\beta$$ needs to be higher for breaking |
| Sampling rate | 60–120 Hz (IMU) | 30–60 fps (video) | **Marginal** — lower sampling + higher motion = more aliasing |
| Stationary assumption | Hand at rest is truly stationary | Bboy in freeze: minimal motion but tremor | **Yes** — similar enough |

The specific value $$\beta = 0.007$$ controls the speed-responsiveness tradeoff. For hand tracking at 60Hz with max speed ~3 m/s, this produces a cutoff frequency of:

$$f_c = 1.0 + 0.007 \times 3.0 \times 370 \text{ px/m} / 60 = 1.0 + 0.13 = 1.13 \text{ Hz}$$

For a bboy foot at 10 m/s at 30fps:

$$f_c = 1.0 + 0.007 \times 10.0 \times 370 / 30 = 1.0 + 0.86 = 1.86 \text{ Hz}$$

A 1.86 Hz cutoff at 30fps means the filter passes frequencies up to ~1.86/30 = 6.2% of Nyquist. This is **aggressive smoothing** — it would attenuate any motion component above ~2 Hz, which includes the fundamental frequency of a headspin ($$1/0.4 = 2.5$$ Hz). The filter would **lag behind the actual motion** during fast rotations.

This is a concrete, calculable problem with the proposed parameters. The correct $$\beta$$ for breaking should be ~5–10× larger to avoid lagging during power moves:

$$\beta_{\text{bboy}} \approx 0.03\text{–}0.07$$

But even this corrected estimate is theoretical. The right approach is grid search over $$(\beta, f_{c,\min})$$ with a loss function that balances jitter (measured on freeze frames) against lag (measured on known-circular-motion segments).

---

## 6. What Experiments Would Resolve These Gaps

### Experiment 1: Displacement–Accuracy Curve

**Goal:** Measure $$\epsilon(d)$$ — tracking error as a function of per-frame displacement.

**Method:**
1. Collect 10–20 bboy clips covering toprock, footwork, windmill, flare, headspin at 60fps 1080p
2. Manually annotate 5 anatomical points per clip at every 5th frame (total: ~2000 annotations)
3. Run CoTracker3 with default settings
4. Compute per-point, per-frame error: $$\epsilon_t^{(n)} = \|\hat{p}_t^{(n)} - p_t^{(n)}\|_2$$
5. Bin by displacement: $$d_t^{(n)} = \|p_t^{(n)} - p_{t-1}^{(n)}\|_2$$
6. Plot $$\mathbb{E}[\epsilon | d]$$ with confidence intervals

**Expected output:** A curve showing the displacement at which tracking error exceeds an acceptable threshold (e.g., 10px). This replaces the "60–120px practical range" assertion with a measured number.

**Effort:** ~20 hours of manual annotation + ~2 hours of inference + analysis. This is the **single most valuable experiment** for the entire pipeline.

### Experiment 2: Visibility Score Calibration

**Goal:** Determine whether $$v$$ is well-calibrated for breaking occlusion.

**Method:**
1. Using the same annotated clips, label each annotated point-frame as visible/occluded (binary ground truth)
2. Run CoTracker3, extract visibility scores at annotated points
3. Compute precision-recall curve for visibility classification
4. Plot reliability diagram: $$P(\text{truly visible} | v \in [v_k, v_{k+1}])$$

**Expected output:** 
- The optimal threshold $$\tau^*$$ for breaking (maximizing F1 or task-specific metric)
- Whether the sigmoid output is calibrated ($$v = 0.7$$ means ~70% probability of being visible) or miscalibrated
- Whether different move types need different thresholds

### Experiment 3: Search Radius Ablation

**Goal:** Measure the effect of $$S$$ on tracking accuracy for breaking.

**Method:**
1. Run CoTracker3 with $$S \in \{3, 4, 5, 6, 8\}$$ using the adaptive pooling approach (no retraining)
2. Measure accuracy on the annotated clips, stratified by displacement bin
3. Measure inference time per setting

**Expected output:** The actual accuracy-compute Pareto frontier for the $$S$$ parameter on bboy footage. This replaces the theoretical analysis with measured tradeoffs.

### Experiment 4: Smoothing Parameter Grid Search

**Goal:** Find optimal One-Euro filter parameters for breaking.

**Method:**
1. For clips with manual annotations at every frame (or every 2nd frame for a subset), compute "ground truth" trajectories
2. Add CoTracker3's actual noise profile (from Experiment 1) to ground truth
3. Grid search over $$f_{c,\min} \in \{0.5, 1.0, 2.0, 4.0\}$$, $$\beta \in \{0.005, 0.01, 0.03, 0.07, 0.15\}$$
4. Evaluate: jitter on freeze frames, lag on power-move frames, overall RMSE

**Expected output:** Validated filter parameters, or evidence that a different filter type (e.g., Savitzky-Golay for offline processing) is superior.

### Experiment 5: Motion Prior Diagnostic

**Goal:** Test whether CoTracker3's learned motion prior helps or hurts for circular motion.

**Method:**
1. Run CoTracker3 with $$N_{\text{iter}} \in \{1, 2, 3, 4, 6, 8\}$$ on windmill/headspin clips
2. If accuracy peaks at $$N_{\text{iter}} < 4$$, the motion prior is harmful for circular motion
3. Compare with toprock/footwork clips where accuracy should monotonically improve with $$N_{\text{iter}}$$

**Expected output:** Evidence for or against the motion prior mismatch hypothesis, determining whether iteration count should be a tunable parameter per-move-type.

---

## 7. Which Claims Are Most Likely Wrong

Ranking by probability of being materially incorrect:

### 7.1 Almost Certainly Wrong: "CoTracker3 Error (px)" Column

The table claiming windmill error is "3.0–6.0 px" is the most suspect number in the entire analysis. This would imply sub-centimeter tracking accuracy on a limb moving at 8+ m/s with severe motion blur and periodic self-occlusion. For context, CoTracker3's error on TAP-Vid-DAVIS (much easier motion) is ~3–5px median. Claiming similar accuracy on motion that is 3–10× faster with worse occlusion patterns is not credible.

A more realistic estimate, based on the displacement–error relationship observed in RAFT and other optical flow methods (error scales roughly as $$\epsilon \propto d^{0.5}$$ to $$d^{0.8}$$):

$$\epsilon_{\text{windmill}} \approx \epsilon_{\text{DAVIS}} \times \left(\frac{d_{\text{windmill}}}{d_{\text{DAVIS}}}\right)^{0.6} \approx 4 \times \left(\frac{52}{15}\right)^{0.6} \approx 4 \times 2.5 \approx 10 \text{ px}$$

at 60fps. At 30fps ($$d = 104$$):

$$\epsilon \approx 4 \times \left(\frac{104}{15}\right)^{0.6} \approx 4 \times 4.0 \approx 16 \text{ px}$$

These are still rough estimates but are grounded in known error-scaling behavior. **The actual error could easily be 20–50px** when motion blur destroys the correlation signal entirely.

### 7.2 Likely Wrong: One-Euro $$\beta = 0.007$$

As calculated in §5, this value produces excessive smoothing at bboy velocities. The filter would visibly lag behind fast rotational motion, creating artifacts that are worse than the jitter it's trying to remove. This is correctable (increase $$\beta$$) but the specific recommended value would produce bad results.

### 7.3 Possibly Wrong: "Increase overlap stride from 8 to 4"

The recommendation to double window overlap (stride 4 instead of 8) for breaking is based on the assumption that more temporal context helps bridge occlusions. But if the motion prior mismatch hypothesis (§3.3) is correct, more transformer context could actually **hurt** — giving the model more frames to accumulate a wrong motion prediction. This is testable (Experiment 5) but the recommendation should not be implemented before verification.

### 7.4 Uncertain but Defensible: Visibility Threshold Framework

The adaptive thresholding framework (using $$dv/dt$$) is architecturally sound. The specific parameter values are ungrounded, but the approach of using the temporal derivative rather than a fixed threshold is well-motivated by the distinction between motion blur (gradual visibility decrease) and occlusion (sharp visibility drop). The framework is right; the numbers need tuning.

---

## 8. The Irreducible Gap: No Ground-Truth Breaking Tracks Exist

The deepest problem is not that the prior analysis failed to run experiments — it's that **no suitable ground-truth dataset exists** for breakdancing point tracking.

Existing datasets:
- **TAP-Vid**: No dance footage. Closest category is "sports" (gymnastics), but with different motion profiles.
- **PointOdyssey**: Synthetic, no dance motion.
- **Dynamic Replica**: Indoor scenes, no human motion at all.
- **PoseTrack**: Joint annotations, not arbitrary point tracks. Also no breaking.

Creating ground truth for bboy tracking requires:
1. Multi-view capture (to resolve self-occlusion — you need cameras on both sides of the dancer to know where occluded points actually are)
2. Or: IMU/motion capture ground truth (MoCap suit under battle clothes)
3. Or: Synthetic rendering of breaking motion (requires a realistic breaking motion capture dataset, which itself doesn't exist publicly)

The most practical path: **use a small multi-camera rig** (3 iPhones in a triangle, ~$0 marginal cost) to capture a bboy performing standard moves. Track points in the primary camera view with CoTracker3, use the other two views to establish ground-truth positions for occluded points via triangulation. This gives you ~100–500 ground-truth point-frame annotations per clip, enough for the displacement-accuracy curve (Experiment 1) and visibility calibration (Experiment 2).

**Estimated effort for a minimal empirical validation:**
- Capture: 2 hours (10 clips × 6 takes × 2 minutes)
- Multi-view triangulation for ground truth: 8 hours (semi-automated with COLMAP + manual correction)
- CoTracker3 inference: 1 hour
- Analysis: 4 hours
- **Total: ~15 hours of work to replace ~90% of the unverified claims with measured values**

---

## 9. Summary: Confidence Levels on Prior Analysis Claims

| Claim Category | Confidence | Action Required |
|----------------|------------|-----------------|
| Output tensor shapes and coordinate space | **High** (code-verified) | None |
| Pixel displacement per move (point estimates) | **Low** (correct math, uncertain inputs, ±3× range) | Measure frame occupancy across real battle footage corpus |
| "Practical search range 60–120px" | **Very low** (no basis beyond being between theoretical bounds) | Experiment 1 |
| Error estimates per move type | **Very low** (fabricated, likely underestimated by 3–5×) | Experiment 1 |
| Visibility threshold recommendations | **Very low** (ungrounded specific values) | Experiment 2 |
| One-Euro filter parameters | **Low** (transferred from different domain, $$\beta$$ likely 5–10× too low) | Experiment 4 |
| Sliding window overlap recommendation | **Medium** (reasonable intuition, but could backfire) | Experiment 5 |
| Coordinate rescaling requirement | **High** (architecture-derived, verifiable) | None |
| Downstream integration contracts (SAM-Body4D, JOSH) | **Medium** (correct interface descriptions, untested integration) | Integration test |

The prior analysis is a strong **architectural audit** and a reasonable **theoretical framework**, but its numerical claims carry far less certainty than their presentation implies. The 15-hour empirical validation described in §8 would transform it from a theoretical analysis into a grounded engineering specification.
