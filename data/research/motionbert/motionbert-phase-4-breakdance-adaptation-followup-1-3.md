# motionbert — Follow-up 1.3: All MPJPE estimates, SNR calculations, and degradation curves are derived theoretically (extrapolati

_Generated: 2026-03-23T02:39:44.700Z_

---

Let me check what tools I have available and work with the research context provided. The task is clear: address the critical empirical validation gap in the MotionBERT breakdance analysis. Let me produce the thorough analysis requested.

---

# Empirical Validation Protocol: MotionBERT on Breakdance Video

## The Calibration Problem

The prior analysis builds an internally consistent theoretical framework — orientation degradation curves, embedding shift estimates, SNR calculations — but every numerical claim traces back to **two unvalidated assumptions**:

1. **The degradation model** $\Delta\text{MPJPE} \propto \theta^2$ (orientation-dependent error growth) is extrapolated from H36M error patterns on upright poses with synthetic rotation, not measured on actual inverted bodies
2. **The 2D detector PCK estimates** (55–95% depending on scenario) are educated guesses based on general knowledge of detector failure modes, not measured on breakdance footage

These two assumptions are the load-bearing walls. If either is wrong by even 30%, the entire MPJPE table, every SNR calculation, and the viability verdict for each scenario shifts materially. The analysis could be systematically optimistic (if 2D detectors fail harder than expected on motion blur + inversion) or systematically pessimistic (if temporal attention is more robust to distribution shift than the embedding analysis predicts).

---

## Protocol Design: Minimum Viable Empirical Check

### Clip Selection Strategy

The validation requires clips that isolate each failure mode rather than confounding them. Minimum set: **5 clips, 3 sources**.

| Clip | Source | Duration | Move | Isolates |
|------|--------|----------|------|----------|
| C1 | BRACE dataset | ~8s | Toprock (Indian step) | Control — near-upright, validates baseline transfer |
| C2 | BRACE dataset | ~6s | Windmill (3+ rotations) | Continuous orientation shift + L-R confusion |
| C3 | YouTube battle footage | ~5s | Headspin (sustained) | Full inversion + periodicity |
| C4 | YouTube battle footage | ~8s | Footwork → Freeze transition | Velocity discontinuity + static averaging |
| C5 | AIST++ (if available) or YouTube | ~6s | Flare (3+ revolutions) | Extreme articulation + leg occlusion |

**Why BRACE**: The Bristol Robotic and Acrobatic Dance Engine dataset contains breakdance-adjacent acrobatic sequences with **ground-truth 3D poses from multi-view capture**. This is the only publicly available dataset with GT for non-upright human poses. Coverage is sparse for breaking-specific moves but sufficient for windmill-like rotations.

**Why YouTube**: BRACE doesn't contain competition footage. Real battle conditions (crowd, handheld camera, variable lighting) are only testable on wild video. No GT available — evaluation is **qualitative** (visual inspection) + **physical plausibility checks** (bone length consistency, ground contact, joint angle limits).

### Inference Protocol

```python
# Pseudocode — actual implementation uses MotionBERT's demo pipeline
for clip in [C1, C2, C3, C4, C5]:
    # Stage 1: 2D keypoint extraction
    poses_2d = vitpose_inference(clip, model='ViTPose-H')  # or CPN
    
    # Log 2D quality metrics BEFORE feeding to MotionBERT
    for frame in poses_2d:
        log(frame.confidences)        # per-joint confidence scores
        log(frame.num_detections)     # spurious detections from crowd
        log(frame.keypoint_positions) # for post-hoc L-R swap detection
    
    # Stage 2: MotionBERT 3D lifting
    poses_3d = motionbert_inference(
        poses_2d, 
        model='motionbert_h36m_pretrained',  # no fine-tuning
        window=243,
        stride=81  # 2/3 overlap for sliding window
    )
    
    # Stage 3: Multi-level evaluation
    evaluate_physical_plausibility(poses_3d)
    evaluate_temporal_coherence(poses_3d)
    if has_ground_truth(clip):
        evaluate_mpjpe(poses_3d, gt_3d)
    evaluate_spectrogram_quality(poses_3d, audio)
```

### What to Measure (No GT Available)

When ground truth is unavailable (YouTube clips), the following **proxy metrics** serve as empirical validation:

#### 1. Bone Length Consistency (BLC)

$$\text{BLC}_b = \frac{\sigma_{L_b(t)}}{\bar{L}_b} \times 100\%$$

where $L_b(t)$ is the length of bone $b$ at frame $t$. For a correct 3D reconstruction, BLC should be $< 2\%$ (bones don't stretch). Training-distribution poses typically achieve BLC $\approx 0.5\text{–}1.5\%$.

**Prediction from theory**: BLC will degrade to $5\text{–}15\%$ during headspin (embedding collapse causes inconsistent joint placement), $3\text{–}8\%$ during windmill (L-R swaps cause apparent bone length jumps), and remain $< 3\%$ during toprock.

**Why this matters**: If BLC stays $< 3\%$ during headspin, the embedding collapse is less severe than predicted — spatial attention is maintaining skeletal structure despite out-of-distribution inputs. This would mean the theoretical MPJPE of 85–110mm is too pessimistic.

#### 2. Joint Angle Plausibility (JAP)

For each anatomical joint, compute the angle and check against biomechanical limits:

$$\text{JAP}_{violation} = \frac{1}{T \cdot J} \sum_{t,j} \mathbb{1}\left[\alpha_j(t) \notin [\alpha_j^{min}, \alpha_j^{max}]\right]$$

| Joint | Min | Max | Notes |
|-------|-----|-----|-------|
| Knee | 0° | 160° | Hyperextension impossible |
| Elbow | 0° | 150° | Hyperextension impossible |
| Hip abduction | 0° | 170° | Flare requires ~150–170° |
| Shoulder flexion | -60° | 180° | Full overhead range |
| Neck | -60° | 60° | Limited range |

**Prediction**: JAP violation rate $< 2\%$ for toprock, $5\text{–}15\%$ for power moves. If violation rate exceeds $20\%$, the model is producing physically impossible skeletons — worse than the theoretical analysis suggests.

**Critical test for flare**: The prior analysis claims MotionBERT's pose prior "actively penalizes correct flare poses" because hip abduction $> 150°$ is rare in AMASS. If the model outputs hip abduction clamped at $\sim 90\text{–}110°$ during flares (the AMASS mode), this confirms the prior is overriding the data. If it allows $140\text{–}170°$, the prior is weaker than assumed.

#### 3. Ground Contact Stability (GCS)

During moves with known floor contact (footwork hands, freeze support points, windmill shoulders):

$$\text{GCS} = \frac{1}{|C|} \sum_{j \in C} \text{std}\left(p_j^z(t)\right) \quad \text{over contact frames}$$

where $C$ is the set of contact joints. A correct reconstruction should show GCS $< 15\text{mm}$ (contact joints stay on the floor plane).

**Prediction**: GCS $> 40\text{mm}$ for windmill (no contact prior), $< 20\text{mm}$ for footwork (near-upright, temporal averaging helps). If GCS is $> 60\text{mm}$ for windmill, the temporal attention is not only failing to enforce contact but actively pulling contact joints off the floor — this would indicate the temporal stream is a **net negative** for floor-level moves, not just unhelpful.

#### 4. Temporal Smoothness (TS)

Compute jerk (third derivative of position) as a measure of unphysical discontinuities:

$$\text{TS}(t) = \left\|\frac{d^3 p_j}{dt^3}\right\| = \left\|\frac{p_{t+2} - 3p_{t+1} + 3p_t - p_{t-1}}{(\Delta t)^3}\right\|$$

Average jerk should be bounded by human biomechanical limits. Spikes in jerk indicate either:
- L-R joint swaps (sudden position jumps)
- Tracker identity switches
- Depth ambiguity flips (joint suddenly jumps forward/backward)

**Prediction**: Jerk spikes at $> 5 \times$ median will occur 2–5 times per windmill revolution (L-R swaps) and 0–1 times per headspin revolution (depth flips). Toprock should show near-zero jerk spikes. Counting jerk spikes empirically gives a **direct measurement** of L-R confusion frequency — something the theory estimates at "periodic, ~once per half-revolution" but cannot quantify precisely.

---

## The Three Hypotheses Under Test

The empirical check is designed to discriminate between three possible realities:

### H1: Theory is Approximately Correct (±20%)

**Expected observations**:
- Toprock BLC $< 2\%$, JAP violations $< 3\%$, visually plausible
- Headspin BLC $> 8\%$, visible limb stretching, occasional skeleton collapse
- Windmill shows periodic jerk spikes at revolution frequency
- Freeze entry shows gradual deceleration over 5–10 frames instead of sharp stop
- MPJPE on BRACE clips within 20% of predicted values

**Implication**: Proceed with the recommended architecture (SAM-Body4D primary, MotionBERT fusion optional). The theoretical framework is a reliable planning tool.

### H2: Theory is Too Pessimistic (Actual Performance 30%+ Better)

**Expected observations**:
- Headspin BLC $< 5\%$ — the DSTformer's spatial attention maintains skeletal structure even with out-of-distribution embeddings, perhaps because the 5-block residual structure allows the model to learn bypass paths that partially ignore corrupted early embeddings
- Windmill jerk spikes are rare ($< 1$ per revolution) — temporal attention is correctly tracking individual joints through occlusion
- Flare hip angles reach $130\text{–}160°$ — the AMASS pose prior is weaker than assumed (perhaps because the pretraining loss doesn't strongly penalize rare poses, just under-represents them)

**Implication**: MotionBERT may be viable as a **standalone** 3D lifting component for coarse breakdance analysis, without SAM-Body4D fusion. This significantly simplifies the pipeline. The canonical rotation preprocessing alone might bring power move MPJPE to $< 65\text{mm}$, sufficient for period-based musicality scoring.

**Why this is plausible**: The theoretical analysis treats the linear embedding layer as a hard bottleneck — if input features are $> 3\sigma$ from training mean, all downstream processing is corrupted. But transformer residual connections provide a **skip path**: if the first attention block produces garbage, later blocks can still attend to raw embedded features via residual. The effective degradation might be more like $\Delta\text{MPJPE} \propto \theta^{1.2}$ (subquadratic) rather than $\propto \theta^2$.

### H3: Theory is Too Optimistic (Actual Performance 30%+ Worse)

**Expected observations**:
- 2D detector (ViTPose) fails **catastrophically** on inverted poses — not just PCK drop to 55–70%, but complete joint assignment failure where the detected skeleton bears no anatomical relationship to the actual body. Confidence scores remain high despite wrong assignments (a known failure mode of heatmap-based detectors when the person is inverted)
- BLC $> 20\%$ for headspin — skeleton periodically "explodes" (joints scatter to implausible positions)
- Motion blur at 30fps causes ViTPose to interpolate between adjacent frames' detections, creating a ghosting effect where the 2D skeleton lags the actual body by 2–3 frames. This lag, which the theory doesn't account for, adds a **systematic temporal bias** on top of the random noise

**Implication**: The 2D detector, not MotionBERT's lifting architecture, is the binding constraint. No amount of MotionBERT modification helps if input 2D keypoints are fundamentally wrong. The pipeline must either:
- Use a different 2D frontend (e.g., RTMO which handles motion blur better, or a video-based detector like DOPE that reasons across frames)
- Bypass 2D detection entirely and use direct 3D estimation (e.g., HMR2.0, TokenHMR) which are end-to-end and avoid the 2D bottleneck
- Rely exclusively on SAM-Body4D for power moves, using MotionBERT only for toprock/footwork

---

## Predicted Failure Mode Taxonomy (Empirically Testable)

The theory predicts 7 failure modes. Empirical testing can rank them by actual severity:

### FM1: Embedding Distribution Shift (Orientation)

$$\Delta_{embed} = \left\| \mathbb{E}[W \cdot x_{inverted}] - \mathbb{E}[W \cdot x_{upright}] \right\|_2$$

where $W \in \mathbb{R}^{256 \times 2}$ is the input embedding weight matrix. This is **directly measurable** by running inference and extracting intermediate activations:

```python
# Hook into MotionBERT's embedding layer
activations = {}
def hook_fn(module, input, output):
    activations['embedding'] = output.detach()

model.encoder.joint_embed.register_forward_hook(hook_fn)

# Compare embedding distributions
embed_upright = run_and_collect(toprock_clip)    # shape: [T, 17, 256]
embed_inverted = run_and_collect(headspin_clip)  # shape: [T, 17, 256]

# Per-dimension KL divergence
for d in range(256):
    kl = kl_divergence(embed_upright[:,:,d], embed_inverted[:,:,d])
    # Theory predicts: mean KL > 2.0 (severe shift)
    # If actual KL < 0.5: embedding is more robust than predicted
```

**Empirical discriminator**: If mean KL divergence across embedding dimensions is $< 0.5$ for inverted poses, the embedding layer is not the bottleneck — look elsewhere for the error source (likely 2D detection or attention). If KL $> 2.0$, the embedding collapse hypothesis is confirmed.

### FM2: Spatial Attention Degradation

Extract attention maps $A_s \in \mathbb{R}^{17 \times 17}$ from each S-MHSA block:

$$H_{attn} = -\sum_{i,j} A_s[i,j] \log A_s[i,j]$$

Entropy $H_{attn}$ measures attention specificity. Maximum entropy (uniform attention) = $\log(17) = 2.83$. Training-distribution entropy ≈ $1.5\text{–}2.0$ (attending to specific joint subgroups).

**Prediction**: Headspin $H_{attn} > 2.5$ (near-uniform → spatial attention is useless). Toprock $H_{attn} \approx 1.5\text{–}2.0$.

**If actual headspin** $H_{attn} < 2.0$: The spatial attention has learned structural patterns (bone adjacency, symmetry) that transfer to inverted poses — a much more optimistic finding than predicted.

### FM3: Temporal Attention Smoothing Bandwidth

The temporal smoothing kernel's effective bandwidth determines how much velocity peak clipping occurs. Measurable by comparing input 2D velocity with output 3D velocity:

$$\text{BW}_{eff} = \frac{\int_0^{f_N} |H(f)|^2 df}{\int_0^{f_N} df}$$

where $H(f) = \text{FFT}(\dot{p}_{3D}) / \text{FFT}(\dot{p}_{2D})$ is the effective transfer function of MotionBERT viewed as a temporal filter.

**Prediction**: $\text{BW}_{eff} \approx 8\text{–}12\text{ Hz}$ for upright poses, dropping to $5\text{–}8\text{ Hz}$ for inverted poses (more aggressive smoothing when uncertain).

**Why this matters for the spectrogram**: If $\text{BW}_{eff} < 6\text{ Hz}$, footwork steps at 3–6 Hz are at or above the effective bandwidth — the spectrogram will miss individual steps. If $\text{BW}_{eff} > 10\text{ Hz}$, footwork is fully captured and the SNR estimates are conservative.

### FM4: 2D Detector Failure Characterization

The single most important empirical measurement. Run ViTPose on all 5 clips and log:

1. **Per-joint confidence** $c_j(t) \in [0, 1]$: Theory assumes smooth degradation. Reality may show bimodal distribution — joints are either well-detected ($c > 0.8$) or completely lost ($c < 0.2$), with few intermediate values.

2. **Temporal consistency of joint identity**: For each frame pair $(t, t+1)$, compute:
   
   $$\text{swap}(t) = \sum_{j \in \{L,R\}} \mathbb{1}\left[ \|p_j^L(t) - p_j^R(t+1)\| < \|p_j^L(t) - p_j^L(t+1)\| \right]$$
   
   This counts frames where the left joint at $t$ is closer to the right joint at $t+1$ than to itself — indicating a likely L-R swap.

3. **Detection completeness**: Fraction of frames where all 17 joints are detected with $c > 0.3$. Theory assumes this is $> 90\%$ for toprock, $60\text{–}80\%$ for power moves. If actual completeness for headspin is $< 40\%$, the MotionBERT analysis is moot — you can't lift what you can't detect.

### FM5: Depth Ambiguity Flips

Measurable via sudden sign changes in relative joint depth:

$$\text{flip}(t) = \mathbb{1}\left[\text{sign}(p_j^z(t) - p_k^z(t)) \neq \text{sign}(p_j^z(t-1) - p_k^z(t-1))\right]$$

for joint pairs $(j, k)$ that should maintain consistent relative depth (e.g., during a headspin, the head should always be the deepest joint — if depth ordering flips, the model momentarily resolves depth ambiguity the wrong way).

**Expected flip rate**: 0.5–2 flips/second during headspin (camera-axis alignment), $< 0.1$ flips/second during toprock. Each flip produces a jerk spike.

### FM6: Temporal Smearing at Freeze Entry

Directly measurable by comparing input 2D velocity profile with output 3D velocity profile at the freeze transition:

$$\tau_{smear} = t_{3D}^{v < \epsilon} - t_{2D}^{v < \epsilon}$$

where $t^{v < \epsilon}$ is the first frame where velocity drops below threshold $\epsilon$. Theory predicts $\tau_{smear} = 5\text{–}10$ frames. If actual $\tau_{smear} > 15$ frames (0.5s), freeze timing precision is worse than predicted and the hold-duration metric becomes unreliable for freezes $< 1.5$s.

### FM7: Pose Prior Override on Extreme Articulation

For flare clips, measure the maximum hip abduction angle achieved in the 3D output:

$$\alpha_{hip}^{max} = \max_t \angle(\vec{v}_{hip\to knee}^L(t), \vec{v}_{hip\to knee}^R(t))$$

**Theory predicts**: $\alpha_{hip}^{max} \approx 90\text{–}120°$ (prior clamps to AMASS mode) when actual flare angle is $150\text{–}175°$.

**Measurement**: If $\alpha_{hip}^{max} < 100°$ on a clip where the dancer clearly achieves $> 150°$ abduction, the pose prior is the dominant error source for flares — and no amount of temporal or spatial attention improvement will fix it without fine-tuning on extreme-articulation data.

---

## Quantitative Validation Framework

### For BRACE Clips (GT Available)

Compute standard metrics and compare against theoretical predictions:

| Metric | Formula | Toprock Predicted | Headspin Predicted |
|--------|---------|---:|---:|
| MPJPE | $\frac{1}{TJ}\sum_{t,j}\|p_j(t) - \hat{p}_j(t)\|_2$ | 40–48mm | 85–110mm |
| PA-MPJPE | After Procrustes alignment | 32–38mm | 55–75mm |
| MPJVE | $\frac{1}{TJ}\sum_{t,j}\|\dot{p}_j(t) - \dot{\hat{p}}_j(t)\|_2$ | 1.5–2.5 m/s | 3–5 m/s |
| PCK@150mm | $\frac{1}{TJ}\sum_{t,j}\mathbb{1}[\|p-\hat{p}\| < 150]$ | >95% | 60–80% |

The gap between MPJPE and PA-MPJPE reveals how much error is **global rotation** (removable by canonical rotation preprocessing) vs. **local joint error** (requires model changes):

$$\Delta_{rotation} = \text{MPJPE} - \text{PA-MPJPE}$$

If $\Delta_{rotation} > 30\text{mm}$ for headspin, the canonical rotation fix is high-value (projected to recover most of that gap). If $\Delta_{rotation} < 15\text{mm}$, global orientation isn't the main problem — local joint estimation is, and canonical rotation won't help much.

### For YouTube Clips (No GT)

Use the proxy metrics defined above (BLC, JAP, GCS, TS) plus **visual inspection scoring**:

| Visual Check | Score 0 (Fail) | Score 1 (Degraded) | Score 2 (Pass) |
|---|---|---|---|
| Skeleton tracks body | Skeleton detaches/explodes | Tracks loosely, 1–2 joints drift | Tight tracking |
| Limb proportions stable | Visible stretching/shrinking | Occasional flicker | Consistent |
| Floor contact | Contact joints float >10cm | Occasional floor violation | On floor |
| L-R assignment | Persistent L-R confusion | Periodic swaps (1–2/s) | Correct |
| Depth ordering | Frequent depth flips | Occasional flips | Consistent |

Score each clip on each criterion. Total possible: 10. Toprock should score 8–10. Headspin should score 3–6 if theory is correct.

---

## Expected Theoretical Recalibrations

Based on the empirical results, the theoretical model should be updated in these specific ways:

### Recalibrating the Degradation Model

The assumed $\Delta\text{MPJPE} \propto \theta^2$ can be replaced with an empirically fitted model. With 5 clips spanning orientations from $\theta = 0°$ (toprock) to $\theta = 180°$ (headspin), fit:

$$\Delta\text{MPJPE}(\theta) = a \cdot \theta^b + c$$

If $b < 1.5$: degradation is subquadratic → theory was too pessimistic for moderate angles (windmill, footwork). The "marginal" viability verdict for windmill likely upgrades to "viable."

If $b > 2.5$: degradation is superquadratic → a sharp phase transition exists. There's an angle threshold $\theta^*$ below which MotionBERT works reasonably and above which it catastrophically fails. This changes the recommendation from "gradual degradation, use with caution" to "hard cutoff, don't use above $\theta^*$."

### Recalibrating SNR Estimates

The velocity SNR table can be grounded by measuring actual output noise on the static portions of freeze clips:

$$\sigma_p^{empirical} = \text{std}(p_j(t)) \quad \text{during freeze hold}$$

This gives a **direct measurement** of position noise floor under ideal conditions (static pose, temporal averaging active). If $\sigma_p^{empirical} = 8\text{mm}$ (better than predicted 10mm), all SNR estimates shift upward by ~20%. If $\sigma_p^{empirical} = 15\text{mm}$, the entire spectrogram viability assessment needs revision downward.

The corresponding velocity noise floor:

$$\sigma_v^{empirical} = \frac{\sqrt{2} \cdot \sigma_p^{empirical}}{\Delta t}$$

This single measurement — position noise during a freeze — **calibrates every SNR calculation in the entire analysis**.

### Recalibrating the 2D Detector Bottleneck

The most likely outcome of empirical testing is discovering that the 2D detector is a more severe bottleneck than the theory accounts for. The prior analysis treats 2D detection as a noise source with known PCK but doesn't model **systematic errors** (consistent mis-assignments, temporal lag, confidence-score unreliability).

If empirical testing reveals that ViTPose confidence scores are **uncorrelated with actual accuracy** during inverted poses (high confidence on wrong detections), the entire pipeline needs a confidence-gated input stage:

$$x_j^{gated}(t) = \begin{cases} x_j(t) & \text{if } c_j(t) > \tau \text{ AND } \|x_j(t) - x_j(t-1)\| < \delta \\ \text{interpolate}(x_j(t-1), x_j(t+1)) & \text{otherwise} \end{cases}$$

This adds a temporal consistency check on top of confidence scoring — if a joint detection is both confident AND spatially consistent with adjacent frames, trust it. Otherwise, interpolate. This is ~40 LOC but addresses a failure mode the theory doesn't fully account for.

---

## Implementation Priority

Given the analysis above, the empirical validation should proceed in this order:

1. **Run ViTPose alone on all 5 clips** (1 hour). Log per-joint confidence, detection completeness, L-R swap rate. This determines whether the 2D detector or the 3D lifter is the binding constraint. If ViTPose completeness is $< 50\%$ for headspin, stop — no point testing MotionBERT on garbage input.

2. **Run MotionBERT on toprock clip** (30 min). Establish empirical baseline. If MPJPE $> 60\text{mm}$ on upright toprock (theory predicts 40–48mm), something is wrong with the setup, not the theory.

3. **Run MotionBERT on headspin clip** (30 min). Extract embedding activations, attention entropy, BLC, jerk spikes. This single experiment discriminates between H1/H2/H3 more efficiently than any other.

4. **Run MotionBERT on freeze clip** (30 min). Measure $\sigma_p^{empirical}$ during hold and $\tau_{smear}$ at entry. Calibrates the entire SNR framework.

5. **Run MotionBERT on windmill + flare** (1 hour). Measure hip articulation ceiling, L-R swap correlation with jerk spikes, ground contact stability.

**Total time**: ~3–4 hours of computation + analysis, assuming pre-downloaded models and clips. This is negligible compared to the weeks of pipeline development that follows, and prevents building on a theoretical foundation that may be 30–50% wrong in either direction.

---

## What Changes If Theory Is Wrong

| Finding | Theory Adjustment | Pipeline Impact |
|---|---|---|
| Headspin MPJPE $< 70\text{mm}$ | Degradation subquadratic; embedding more robust | MotionBERT viable for power moves with canonical rotation only |
| Headspin MPJPE $> 130\text{mm}$ | Degradation superquadratic or 2D detector catastrophic | Abandon MotionBERT for power moves; SAM-Body4D mandatory |
| Freeze $\sigma_p < 6\text{mm}$ | Temporal averaging more effective than modeled | All spectrogram SNR estimates improve 40%+ |
| Freeze $\sigma_p > 20\text{mm}$ | Temporal averaging less effective (perhaps spatial noise dominant) | Spectrogram requires heavier filtering; fine temporal features lost |
| ViTPose completeness $< 40\%$ on inverted | 2D is the binding constraint, not 3D lifting | Must solve 2D detection first; MotionBERT architecture analysis is secondary |
| Hip $\alpha_{max} < 100°$ on flare | Pose prior dominates; fine-tuning mandatory | Cannot use pretrained MotionBERT for flare/windmill without AMASS+ training |
| L-R swaps $> 3$/revolution | Systematic error worse than noise model | Need L-R swap correction layer before MotionBERT (the 100 LOC fix becomes critical path) |
| Attention entropy $< 2.0$ on headspin | Spatial attention transfers to inversions | Architecture is more robust than predicted; focus modifications on 2D input quality |

The core message: **a 3-hour empirical check determines whether the pipeline architecture decision is "MotionBERT with modifications" or "bypass MotionBERT entirely for power moves."** The theoretical analysis provides the framework for interpreting the results, but the results themselves are necessary to make the architectural commitment.
