# motionbert — Gap Identification

_Generated: 2026-03-23T02:48:04.285Z_

---

I don't have filesystem access in this environment. I'll work directly from the comprehensive research summary provided. This is a thorough body of work — here's the gap analysis.

---

# MotionBERT Reimplementation: Gap Analysis

## 1. Architectural Gaps

### 1A. DSTformer Dual-Stream Topology Mischaracterized — CRITICAL

**What's missing**: The research describes the DSTformer data flow as **sequential** (S-MHSA → T-MHSA):

```
h_s^l[t] = S-MHSA^l(X^{l-1}[t])     // spatial first
h_t^l[:,j] = T-MHSA^l(h_s^l[:,j])    // temporal receives spatial output
```

This is wrong. DSTformer uses a **parallel dual-stream** design where spatial and temporal attention operate on the **same input** and are fused via learned gating:

```
h_s = S-MHSA(X)       // spatial stream
h_t = T-MHSA(X)       // temporal stream (SAME input X, not h_s)
X' = α · h_s + (1-α) · h_t   // attention-based adaptive fusion
```

**Why it matters**: The entire Section 4 transfer analysis is built on the cascade dependency assumption — that random embedding destroys spatial attention, which then feeds garbage to temporal attention ("Block 1: ~0% transfer"). If the streams are parallel, T-MHSA receives raw embedded features, not S-MHSA output. The "cascade failure" argument collapses. T-MHSA transfer should be re-analyzed independently of S-MHSA quality.

Additionally, the **adaptive fusion weights α** are themselves learned parameters that must adapt during modality transfer. The research ignores these entirely.

**Suggested resolution**: Re-read the DSTformer source code (`lib/model/DSTformer.py` in the official repo). Verify parallel vs. sequential. Re-derive the transfer analysis with correct topology. The fusion weights α likely need their own unfreezing schedule.

---

### 1B. Pretraining Objective Not Analyzed

**What's missing**: The research mentions "AMASS pretraining" but never details MotionBERT's actual pretraining strategy:

1. **Unified 2D representation**: 3D AMASS motions are projected to 2D via **random camera parameters** (azimuth, elevation, distance). This creates synthetic (noisy 2D, clean 3D) pairs. The camera augmentation distribution determines what viewpoints the model has "seen."
2. **Noise injection**: Gaussian noise is added to synthetic 2D to simulate detector errors. The noise model (standard deviation, per-joint vs. global, occlusion simulation) directly determines robustness.
3. **Loss function**: Reconstruction L1/L2 + velocity loss + potentially bone length regularization.

**Why it matters**: For breaking, the critical question is: **does the camera augmentation include extreme elevation angles?** If azimuth is sampled uniformly but elevation is sampled near-horizontal (as is common in pose datasets), the model has never seen 2D projections of inverted bodies from above/below — which is exactly the camera angle in competition footage (camera at standing height, dancer on the floor inverted).

The noise injection model also matters: if noise is isotropic Gaussian, it doesn't capture the **structured** failures of 2D detectors on inverted poses (entire limb swaps, not random jitter).

**Suggested resolution**: Inspect the pretraining config in the official codebase (`configs/pretrain.yaml` or equivalent). Document camera parameter ranges. If elevation is limited, this partially explains the rotation degradation and suggests that augmenting the camera distribution during fine-tuning could recover significant performance.

---

### 1C. CPN Detector Coupling Not Addressed

**What's missing**: MotionBERT's H36M numbers (39.2mm MPJPE) are with **CPN-detected 2D input**. The revised pipeline uses **ViTPose**. These detectors have different noise profiles:

- CPN (2018): systematic errors on occluded joints, tends to snap to mean pose
- ViTPose (2022): higher accuracy but different failure modes (hallucinated joints in occluded regions)

**Why it matters**: MotionBERT's pretrained weights learned to correct for **CPN-specific noise patterns**. Switching to ViTPose may paradoxically degrade performance because the noise characteristics don't match training. This is a known phenomenon in the lifting literature — "better 2D ≠ better 3D" when the lifter was trained on a specific detector.

**Suggested resolution**: Either (a) use CPN for consistency with pretrained weights, or (b) fine-tune the lifting network with ViTPose-detected 2D input. Option (b) is better but requires a fine-tuning dataset with both ViTPose 2D detections and 3D ground truth.

---

### 1D. Temporal Window vs. Frame Rate Interaction

**What's missing**: MotionBERT uses a **243-frame temporal window**. At H36M's 50fps, this covers 4.86 seconds. The research mentions AIST++ is 60fps and breaking videos may be 30fps, but never addresses:

- At 30fps: 243 frames = 8.1 seconds (different temporal context)
- At 60fps: 243 frames = 4.05 seconds
- The temporal attention patterns (periodicity detection, velocity smoothing) are learned at 50fps cadence
- Same physical motion at 30fps occupies fewer frames → temporal attention windows don't align

**Why it matters**: A windmill at 50fps might have 25 frames per rotation. At 30fps it's 15 frames. The temporal attention heads that learned periodicity at 25-frame cycles won't detect the same periodicity at 15-frame cycles without adaptation.

**Suggested resolution**: Frame rate normalization via interpolation/decimation to target 50fps before feeding to MotionBERT, or document the expected degradation from frame rate mismatch.

---

## 2. Math Errors

### 2A. Correlated Fusion Error Underestimated

**What's wrong**: The research claims:

> With ρ = 0.6, σ_A = 90mm, σ_B = 85mm: σ_fused ≈ 72mm

The optimal linear fusion with correlated errors:

$$w^* = \frac{\sigma_B^2 - \rho\sigma_A\sigma_B}{\sigma_A^2 + \sigma_B^2 - 2\rho\sigma_A\sigma_B}$$

$$w^* = \frac{7225 - 0.6 \times 90 \times 85}{8100 + 7225 - 2 \times 0.6 \times 90 \times 85} = \frac{2635}{6145} = 0.429$$

$$\sigma_{fused}^2 = w^{*2}\sigma_A^2 + (1-w^*)^2\sigma_B^2 + 2\rho\sigma_A\sigma_B w^*(1-w^*)$$
$$= 0.184 \times 8100 + 0.326 \times 7225 + 2 \times 0.6 \times 7650 \times 0.245 = 6095$$
$$\sigma_{fused} \approx \mathbf{78\text{mm}}$$

The actual answer is **~78mm, not 72mm** — an 8% underestimate. The 72mm value only holds for ρ ≈ 0.35-0.4, which contradicts the ρ = 0.6 assumption.

**Why it matters**: 78mm vs. 72mm affects the downstream velocity SNR calculation. More importantly, it means fusion provides even less benefit than claimed when errors are correlated. The "still far from 45mm" conclusion is even more true.

**Suggested resolution**: Correct the calculation. Consider presenting a ρ sensitivity table rather than a single point estimate.

---

### 2B. MPJPE ≠ Position Noise Standard Deviation

**What's wrong**: The velocity SNR calculation treats MPJPE as positional noise σ:

$$\sigma_v \approx \frac{\sqrt{2} \times 70\text{mm}}{1/30\text{s}} \approx 2970\text{mm/s}$$

MPJPE is the **mean** absolute error, not the standard deviation. It includes both systematic bias (consistently wrong depth) and random noise. Only the random component drives velocity noise. For a Gaussian error distribution, MPJPE ≈ σ × √(2/π) ≈ 0.798σ, so σ ≈ MPJPE / 0.798 ≈ 87.7mm. But then only the **random component** matters for velocity:

If bias fraction is ~40-60% (common for out-of-distribution poses): σ_random ≈ 0.5-0.7 × σ_total.

**Why it matters**: The SNR could be ~1.5-2× better than reported (still bad, but the difference between "completely unusable" and "marginally usable with heavy filtering"). Alternatively, if the bias is non-stationary (varies with pose), it contributes to velocity noise after all, making the estimate approximately correct but for the wrong reasons.

**Suggested resolution**: Decompose MPJPE into bias + variance components. Bias can be estimated from consistent directional errors in depth. Report velocity SNR for both components separately.

---

### 2C. Linear Additivity of Degradation Factors

**What's wrong**: The composite MPJPE estimates add degradation factors:

$$39.2 + 8 + 7 + 3 + 5 = 62.2\text{mm}$$

But these factors are **correlated and interact nonlinearly**:
- Motion blur (factor 4) is **caused by** fast motion (factor 1) — they're the same physical phenomenon, double-counted
- Pose distribution shift (factor 2) and 2D detector quality (factor 4) compound: unusual poses cause BOTH internal model confusion AND worse 2D detection
- Self-occlusion is more severe during fast motion

Linear addition likely **underestimates** the true error because the interaction terms are positive: E[f₁ × f₂] > E[f₁] × E[f₂] when positively correlated.

**Suggested resolution**: Present as a range with explicit note about interaction effects. Consider a multiplicative model: MPJPE_total = MPJPE_base × (1 + δ₁) × (1 + δ₂) × ... which naturally captures compounding.

---

### 2D. Quadratic Degradation Model Unjustified

**What's wrong**: The claim Δ MPJPE ∝ θ² is stated without derivation. The intuition (training distribution is Gaussian, so log-density drops quadratically) conflates **data density** with **model error**. A model might:
- Extrapolate linearly in low-data regions (if the learned representation is smooth)
- Degrade exponentially (if there's a phase transition in representation quality)
- Plateau (if the model defaults to a mean pose for all OOD inputs)

**Why it matters**: The specific degradation predictions at 90°, 135°, 180° depend on this model. If degradation is worse-than-quadratic (exponential), the 135° estimate of 65-90mm is optimistic.

**Suggested resolution**: Flag the quadratic model as a hypothesis, not a fact. Empirical validation on AMASS with synthetic rotation augmentation would resolve this quickly — rotate test poses and measure actual MPJPE vs. angle.

---

## 3. Implementation Risks

### 3A. COCO→H36M Spine Interpolation During Extreme Torso Articulation

**What's missing**: The research correctly identifies the spine mapping problem but underestimates its severity. COCO provides: nose, left/right eye, left/right ear, left/right shoulder, left/right hip. H36M needs: pelvis, spine, thorax, neck, head.

The interpolation for spine/thorax assumes the spine is approximately straight between pelvis and neck. During:
- **Chest freezes**: The spine is hyperextended (concave back)
- **Hollowbacks**: The spine is in extreme extension
- **Windmill transitions**: The spine twists axially

The interpolated spine joints will be systematically wrong — not just noisy, but biased in a consistent direction that temporal smoothing cannot correct.

**Why it matters**: Spine joints are the **root of the kinematic chain** in H36M format. Errors in spine position propagate to all limb positions through the hierarchical skeleton structure. A 20mm spine error can create 30-40mm limb errors.

**Suggested resolution**: Either (a) work directly in COCO format (requires retraining MotionBERT, significant effort) or (b) use the SMPL mesh from SAM-Body4D to extract anatomically correct spine joints rather than interpolating from 2D keypoints.

---

### 3B. LayerNorm Statistics During Modality Transfer

**What's missing**: DSTformer uses LayerNorm after attention and in FFN blocks. When switching from 2D input (normalized pixel coordinates, range [-1, 1], variance ~0.15-0.20) to 3D input (mm-scale, range [-500, 500], variance ~(150-350)²), the **first LayerNorm** after the embedding layer will see completely different statistics.

While LayerNorm normalizes per-sample (unlike BatchNorm), the learned affine parameters (γ, β) were trained assuming 2D-derived feature distributions. With 3D input and a new embedding layer, the feature distribution entering LayerNorm is random — the learned γ, β will scale/shift inappropriately.

**Why it matters**: Training instability in early epochs. The progressive unfreezing schedule proposed in the research would freeze these γ, β along with their blocks, potentially forcing the unfrozen embedding to learn distributions compatible with frozen normalization parameters — an unnecessary constraint that slows convergence.

**Suggested resolution**: Reset LayerNorm parameters (γ=1, β=0) in at least Block 1 when switching to 3D input. Or use the warm-started embedding projection (W_emb^3D = W_emb^2D · P) which would produce approximately correct statistics for the first forward pass.

---

### 3C. The 243-Frame Window and Power Move Duration

**What's missing**: A typical power move combination in competition:
- Windmill: 1-3 seconds per set
- Transition to flare: 0.5s
- Flare: 1-3 seconds
- Freeze: 1-3 seconds
- Total round: 30-60 seconds

At 30fps, 243 frames = 8.1 seconds. This means:
- A single power combination fits within one window ✓
- But the **padding strategy at sequence boundaries** matters — how are the first/last 121 frames of a round handled?
- During a freeze (static pose for 2s = 60 frames), the temporal attention has 60 nearly identical frames — does it handle this well, or does it try to interpolate motion that isn't there?

**Why it matters**: Freezes are scored on duration and stability. If MotionBERT's temporal stream over-smooths or introduces phantom motion during static holds, freeze analysis is compromised.

**Suggested resolution**: Test with synthetic static pose sequences to measure MotionBERT's behavior during zero-motion windows. If it introduces jitter, that's a known failure mode that needs post-processing.

---

### 3D. Numerical Stability in Confidence-Weighted Fusion

The proposed fusion formula:

$$w_{t,j} = \sigma\left(\frac{\log(c_{t,j}) - \lambda \cdot e_t^{S}}{\tau}\right)$$

Risks:
- **c_{t,j} = 0**: log(0) = -∞. Need ε-clamping.
- **e_t^S undefined for failed SAM-Body4D frames**: If mesh reconstruction fails entirely (returns NaN), the fusion weight is undefined.
- **τ → 0**: sigmoid saturates, fusion becomes hard switching with no gradient for learning τ.
- **Per-joint vs. per-body weights**: The formula is per-joint, but SAM-Body4D error e_t^S is per-frame. Mixing granularities may cause inconsistent skeleton geometry (some joints from SAM-Body4D, others from MotionBERT, violating bone length constraints).

**Suggested resolution**: Define explicit fallback behavior (e.g., if either source fails, use the other exclusively). Clamp c_{t,j} ∈ [ε, 1]. Consider per-body fusion with per-joint refinement rather than fully per-joint fusion.

---

## 4. Breakdance-Specific Blind Spots

### 4A. Floor Contact Mechanics Completely Absent

**What's missing**: Breaking involves extensive floor contact: hands, head, elbows, forearms, back, shoulders. Contact fundamentally changes the problem:

1. **Contact points are fixed**: During a freeze, the supporting hand is pinned to a world-space position. Neither MotionBERT nor SAM-Body4D explicitly model contact constraints.
2. **Ground penetration**: Without a floor plane constraint, estimated meshes/skeletons frequently penetrate the floor — physically impossible.
3. **Weight distribution**: A baby freeze has weight on head + one hand. The kinematic chain from these contact points determines the entire body pose. Models that don't understand contact will produce floating/sinking artifacts.

**Why it matters**: Floor contact is perhaps the single most informative constraint for breaking pose estimation. A freeze is fully determined by: which body parts touch the floor + the floor plane + gravity. Without contact modeling, you're throwing away the strongest geometric prior available.

**Suggested resolution**: The JOSH paper (cited in the tech stack re-evaluation) explicitly handles human-scene contact constraints. This should be integrated more centrally — not as an optional addition but as a core constraint for any frame where floor contact is detected.

---

### 4B. Rotation Counting and Angular Velocity Estimation

**What's missing**: For judging, **quantity matters**: 7 continuous windmills is scored higher than 3. The research discusses velocity SNR but never addresses rotation counting.

To count rotations, you need to track cumulative angle around the rotation axis. With 70-100mm MPJPE and ~1:1 velocity SNR, this requires:
- Identifying the rotation axis (changes during transitions)
- Unwrapping the angle (detecting 360° wraps vs. direction reversals)
- Distinguishing complete rotations from partial ones

At the estimated noise levels, angular tracking would accumulate ~30-60° of error per rotation, making it unreliable after 2-3 continuous rotations.

**Why it matters**: Rotation quantity is one of the most objective and important scoring criteria in breaking. If the pipeline can't count rotations, its value for judging is severely limited.

**Suggested resolution**: Use video-level classification (InternVideo 2.5 or similar) for rotation counting rather than joint trajectory analysis. Visual periodicity detection from raw video is likely more robust than counting from noisy joint angles.

---

### 4C. Camera Lens Distortion

**What's missing**: Competition footage often uses:
- Wide-angle lenses (barrel distortion at frame edges)
- GoPro-style action cameras (severe fish-eye)
- Phone cameras at close range (moderate distortion)

MotionBERT assumes a pinhole camera model. Radial distortion shifts 2D keypoint positions, particularly at frame edges. A dancer in the periphery of a wide-angle shot can have 10-30 pixel distortion at 1080p.

**Why it matters**: 10-30 pixel distortion at 1080p translates to ~2-6% coordinate shift, which propagates through the lifting network. For a dancer filling ~400 pixels of frame height, that's ~8-24 pixel error on keypoint positions — comparable to the 2D detector's own error.

**Suggested resolution**: Apply lens distortion correction (undistort) as a preprocessing step before 2D keypoint detection. Camera intrinsics can be estimated from frame metadata (EXIF) or via automated calibration.

---

### 4D. Sweat, Reflections, and Stage Lighting

**What's missing**: Competition environments have:
- Dramatic stage lighting with harsh shadows (creates false body boundaries)
- Spotlight falloff (dancer partially in darkness)
- Reflective/wet surfaces (mirror doubles confuse segmentation)
- Sweat on skin causing specular highlights (confuses mesh fitting)

**Why it matters**: SAM 3 segmentation and SAM-Body4D mesh fitting rely on consistent appearance. Stage lighting creates strong gradients across the body that can fragment segmentation or create appearance discontinuities.

**Suggested resolution**: Acknowledge as a deployment risk. SAM 3's robustness to lighting variation is likely good (foundation model trained on diverse data) but hasn't been validated on stage lighting specifically.

---

### 4E. Head Contact During Headspins

**What's missing**: During headspins:
- The head is compressed against the floor under full body weight
- A beanie/cap is typically worn (changes head shape)
- The face is occluded (facing the floor)
- Hair or headwear creates a "base" that differs from the head keypoint

Both 2D detectors and mesh fitting models expect to see a face or at least a head shape. During a headspin, the visible head is a compressed blob on the floor.

**Why it matters**: Head position is critical for headspin analysis, but it's the least observable keypoint during the move that most depends on it.

**Suggested resolution**: For headspins specifically, infer head position from the floor contact point (known from the previous frame's transition) rather than from visual detection.

---

## 5. Integration Gaps

### 5A. CoTracker3 → SAM-Body4D Connection Is Unclear

**What's missing**: The pipeline shows:
```
② Segment → SAM 3
③ Track   → CoTracker3
④ Mesh    → SAM-Body4D
```

But the data flow between these is never specified:
- CoTracker3 outputs **2D point tracks** (x, y trajectories for N points over T frames)
- SAM-Body4D takes **video frames** as input (not point tracks)
- How do CoTracker3 outputs feed into SAM-Body4D? Are they used as initialization for mesh fitting? As correspondence constraints? Or do they simply run in parallel?

**Why it matters**: If CoTracker3 and SAM-Body4D don't actually interact, step ③ is wasted compute. If they do interact, the integration mechanism needs to be specified.

**Suggested resolution**: Clarify the architectural role of CoTracker3. Likely roles: (a) providing dense correspondences to constrain SAM-Body4D's temporal consistency, (b) tracking through occlusions to provide re-initialization points, or (c) providing an independent motion signal for fusion. Choose one and document the interface.

---

### 5B. Audio-Video Temporal Alignment

**What's missing**: The core innovation ("Audio × Movement spectrogram cross-correlation") requires **precise** temporal alignment between audio and video. Issues:

1. **A/V sync offset**: Even professional recordings can have 30-100ms A/V desync. At 60fps, that's 2-6 frames. For musicality scoring (hitting the beat), 30ms offset could mean the difference between "on beat" and "off beat."
2. **Spectrogram resolution mismatch**: Audio STFT with 2048-sample window at 44.1kHz = 46ms time resolution. Video at 30fps = 33ms time resolution. These grids don't align — interpolation is needed.
3. **Variable frame rate**: Some video formats (VFR) have non-uniform frame timing. The movement spectrogram must account for this.

**Why it matters**: The entire scoring system depends on temporal correlation. A systematic 50ms offset would make every dancer appear slightly behind the beat — or the system would have a constant bias that's indistinguishable from musicality skill.

**Suggested resolution**: Implement explicit A/V sync detection (e.g., detect audio onset → detect motion onset → compute offset). Resample both spectrograms to a common time grid before correlation.

---

### 5C. SMPL-24 to H36M-17 Joint Mapping

**What's missing**: SAM-Body4D outputs SMPL meshes with **24 joints**. MotionBERT works with H36M's **17 joints**. The research discusses COCO→H36M mapping but never addresses SMPL→H36M, which is the relevant mapping in the revised pipeline.

SMPL joints are **regressed from mesh vertices** using a learned regression matrix. H36M joints are **annotated by humans** on images. There are **systematic offsets** between these two joint definitions:
- SMPL "hip" ≈ H36M "hip" + 10-20mm offset
- SMPL "knee" center vs. H36M annotated knee can differ by 15-25mm
- SMPL "head top" doesn't exist — must be synthesized from neck + head joints

**Why it matters**: If SAM-Body4D and MotionBERT produce 3D joints using different joint definitions, fusion is comparing apples to oranges. A 15mm systematic offset between joint definitions would be confused with actual pose error.

**Suggested resolution**: Use a single joint definition throughout. Either: (a) always extract joints from SMPL mesh using the H36M regression matrix (available in standard SMPL toolkits), or (b) define a pipeline-specific joint format and convert all sources to it.

---

### 5D. BRACE Dataset Limitations Not Assessed

**What's missing**: BRACE is cited as solving the data scarcity problem, but:
- What annotations does it actually contain? (2D keypoints? 3D joints? Move labels? Quality scores?)
- What's the video quality? (Resolution, frame rate, camera angles)
- How much breaking content specifically? (vs. other dance styles)
- Is it multi-view? Single camera? Calibrated?

If BRACE provides only 2D annotations, it can't validate 3D pose accuracy. If it's single-view uncalibrated, 3D ground truth doesn't exist.

**Why it matters**: The research pivots from "no breaking data" to "BRACE exists, problem solved" without verifying that BRACE actually provides what's needed for validation.

**Suggested resolution**: Download and inspect BRACE. Document: annotation types, number of breaking-specific clips, video specifications, and whether 3D ground truth exists.

---

## 6. Citation Verification

### 6A. SAM-Body4D (arxiv:2512.08406) — UNVERIFIABLE

The paper is cited as "Training-free 4D human mesh recovery from video, Dec 2025." I cannot verify this paper exists at this arXiv ID. The name follows a plausible pattern (building on SAM + body reconstruction) but could be:
- A real paper I haven't seen (published after my knowledge cutoff)
- A conflation of multiple papers (SAM 3D Body + temporal extension)
- A hallucinated reference from the autoresearch loop

**Impact**: HIGH — SAM-Body4D is the **central component** of the revised pipeline. If it doesn't exist as described, the entire architecture needs revision.

**Suggested resolution**: Verify the arXiv ID. Check the GitHub repo link. If it doesn't exist, identify the actual closest paper and reassess capabilities.

---

### 6B. SAM 3 (arxiv:2511.16719) — UNVERIFIABLE

Similar concern. SAM 2 is real (Meta, 2024). "SAM 3" as a named release in Nov 2025 is plausible but unverified.

**Impact**: MEDIUM — SAM 2 can substitute for most claimed capabilities, but the "2x accuracy over SAM 2" claim may be fabricated.

---

### 6C. JOSH (ICLR 2026) — UNVERIFIABLE

Cited as ICLR 2026. ICLR 2026 would have acceptance notifications around late January 2026. Plausible but unverified.

**Impact**: MEDIUM — JOSH is proposed for occlusion handling, which has alternatives.

---

### 6D. DanceFormer "18.4mm on AIST" — METRIC AMBIGUITY

**What's missing**: Is 18.4mm MPJPE or PA-MPJPE? On all AIST++ or a subset? With GT 2D or detected 2D? The research uses this number to claim "3.4× better" than MotionBERT's estimated 62.2mm, but if the metrics are different (PA-MPJPE vs. MPJPE), the comparison is invalid. PA-MPJPE removes global rotation/translation/scale, which is exactly what makes breaking hard.

**Impact**: HIGH — this comparison motivates the "MotionBERT is insufficient" conclusion.

---

### 6E. MotionBERT "GT 2D MPJPE: 26.9mm" — LIKELY PA-MPJPE

The research lists 26.9mm as "GT 2D MPJPE." The MotionBERT paper reports:
- Protocol 1 (MPJPE) with GT 2D: **~35.1mm**
- Protocol 2 (PA-MPJPE) with GT 2D: approximately this range

If 26.9mm is actually PA-MPJPE mislabeled as MPJPE, this propagates as a baseline error into all degradation estimates.

**Suggested resolution**: Re-verify against Table 1 of the MotionBERT paper. Specify protocol explicitly for every number.

---

## Summary: Priority-Ranked Gaps

| # | Gap | Severity | Effort to Fix |
|---|-----|----------|---------------|
| **1A** | DSTformer parallel vs. sequential | **CRITICAL** | Medium — re-read source, re-derive transfer analysis |
| **6A** | SAM-Body4D existence unverified | **CRITICAL** | Low — check arXiv |
| **1B** | Pretraining camera augmentation unknown | **HIGH** | Low — inspect config |
| **6D** | DanceFormer metric ambiguity | **HIGH** | Low — read paper |
| **4A** | Floor contact mechanics absent | **HIGH** | High — needs architectural addition |
| **5A** | CoTracker3→SAM-Body4D interface undefined | **HIGH** | Medium — design decision |
| **2A** | Fusion error math wrong (78mm, not 72mm) | **MEDIUM** | Low — recalculate |
| **5B** | Audio-video temporal alignment | **MEDIUM** | Medium — implement sync detection |
| **1C** | CPN vs. ViTPose detector coupling | **MEDIUM** | Medium — fine-tuning needed |
| **2C** | Linear additivity assumption | **MEDIUM** | Low — reframe as multiplicative |
| **5C** | SMPL-24 → H36M-17 joint mapping | **MEDIUM** | Low — use standard regression matrix |
| **3A** | Spine interpolation during articulation | **MEDIUM** | High — may need COCO-native retraining |
| **4B** | Rotation counting unaddressed | **MEDIUM** | Medium — use video classification |
| **1D** | Frame rate / temporal window mismatch | **MEDIUM** | Low — resample to 50fps |
| **5D** | BRACE dataset capabilities unverified | **MEDIUM** | Low — download and inspect |
| **3B** | LayerNorm statistics during transfer | **LOW** | Low — reset γ/β in Block 1 |
| **4C** | Lens distortion correction | **LOW** | Low — standard undistort preprocessing |

The two **critical** gaps (1A and 6A) should be resolved before any implementation begins — one invalidates the transfer analysis, the other potentially invalidates the entire pipeline architecture.
