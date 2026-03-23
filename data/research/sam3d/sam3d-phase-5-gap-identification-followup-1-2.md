# sam3d — Follow-up 1.2: The analysis makes dozens of specific quantitative claims (37% boundary error, mAP drops to 28/24/18

_Generated: 2026-03-23T06:19:32.573Z_

---

I'll proceed directly with the experimental validation protocol based on the gap analysis provided.

---

# SAM3D Experimental Validation Protocol

## Motivation

The gap analysis identifies 14 gaps with quantitative claims but zero validation experiments. This document specifies a systematic experimental plan: datasets, metrics, ablations, tensor shapes, and binary pass/fail criteria for each claim. The goal is to determine whether SAM3D reimplementation is worth pursuing for breakdancing analysis — or whether the predicted failures are real and the effort should be redirected.

---

## 0. Experimental Design Philosophy

The claims fall into three tiers:

| Tier | Claims | Validation Strategy |
|------|--------|-------------------|
| **Tier 1: Gate claims** | Point cloud construction fails for moving subjects (§1.5), AP drops catastrophically (§2.4) | Validate first — if confirmed, remaining experiments are moot |
| **Tier 2: Quantitative precision** | 37% boundary error (§2.2), mAP 28/24/18 (§2.3), radial depth error (§3.2) | Measure empirically on controlled data |
| **Tier 3: Domain-specific** | Lighting, clothing, VFR, $V_{eff}$ | Measure on BRACE dataset |

Execute in tier order. If Tier 1 confirms catastrophic failure, document results and stop — don't spend compute validating the precise magnitude of a dead pipeline.

---

## 1. Datasets

### 1.1 Primary: ScanNet v2 (Controlled Baseline)

**Purpose:** Reproduce SAM3D's claimed results on its native benchmark before testing out-of-domain.

- **Subset:** ScanNet validation split scenes 0-20 (20 scenes, ~50K frames total)
- **Ground truth:** Instance segmentation labels, camera intrinsics $K_v$, camera poses $T_v \in SE(3)$, fused point clouds
- **Why this subset:** Small enough for rapid iteration (~2 days on single A100), large enough for statistically meaningful mAP

**Pass/fail gate:** Reproduce SAM3D mAP@50 within ±5 points of the paper's reported value. If we can't reproduce on the native benchmark, the implementation is wrong — stop and debug before testing anything else.

### 1.2 Secondary: AIST++ (Dynamic Human, Controlled Setting)

**Purpose:** Introduce human motion while retaining calibrated multi-view cameras and ground truth 3D poses.

- **Source:** AIST++ Dance Database (Google, ~1000 sequences, 9 cameras)
- **Subset:** 50 sequences spanning slow (locking), medium (popping), and fast (breaking) styles
- **Ground truth:** SMPL parameters, 17-joint 3D keypoints, calibrated multi-camera intrinsics/extrinsics
- **Key property:** Multi-view with known geometry BUT dynamic subjects — isolates the "moving subject" problem from the "single view" problem

**Selection criteria for the 50 sequences:**

$$S_{select} = \{s_i \in \text{AIST++} \mid \text{genre}(s_i) \in \{\text{lock}, \text{pop}, \text{break}\}, \; v_{max}(s_i) \in [v_{25}, v_{75}, v_{95}]\}$$

Stratify by peak joint velocity percentile to ensure coverage of slow, medium, and extreme motion.

### 1.3 Tertiary: BRACE (Real Battle Footage)

**Purpose:** End-to-end domain validation on actual breakdancing competition video.

- **Source:** BRACE dataset (Diller et al., 2023) — 1,352 clips from Red Bull BC One
- **Subset:** 100 clips stratified by move category:
  - 25 toprock (moderate motion, upright)
  - 25 footwork (fast legs, hands on ground)
  - 25 power moves (rotational, extreme velocity)
  - 25 freezes (static poses, extreme configurations)
- **Ground truth:** Pseudo-GT from WHAM/HMR 2.0 (not perfect, but establishes a baseline)
- **Key property:** Real battle conditions — stage lighting, crowd, single handheld camera, baggy clothing

**No 3D ground truth exists for BRACE.** This means we cannot compute mAP directly. Instead, we measure proxy metrics (see §3).

### 1.4 Synthetic Validation: Controlled Depth Error Injection

**Purpose:** Validate the mathematical models (§2.1-2.4) by injecting known errors into perfect data.

- **Source:** Take ScanNet GT depth maps, inject controlled noise:

$$\hat{d}(u,v) = d_{GT}(u,v) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_d^2)$$

$$\sigma_d \in \{0.01, 0.02, 0.05, 0.10, 0.20\} \text{ meters}$$

- Sweep noise levels to empirically measure mAP degradation vs. the analytical prediction

---

## 2. Experiments

### Experiment 1: Tier 1 Gate — Static Scene Reproduction

**Claim under test:** SAM3D achieves ~28-33 mAP@50 class-agnostic on ScanNet (gap §6.2), and the implementation is correct.

**Protocol:**

1. Implement SAM3D following the identified paper (pin to arxiv:2306.03908, Yang et al.)
2. Run on ScanNet val scenes 0-20
3. Evaluate with ScanNet official evaluation script

**Metrics:**
- mAP@25, mAP@50 (class-agnostic and class-specific)
- Per-class AP breakdown

**Tensor shapes:**
- Input point cloud per scene: $P \in \mathbb{R}^{N \times 6}$ (xyz + rgb), $N \approx 100\text{K}$–$500\text{K}$
- Superpoints: $S \in \{0, ..., N_{sp}-1\}^N$, $N_{sp} \approx 5\text{K}$–$20\text{K}$
- Vote matrix: $V \in \mathbb{R}^{N_{sp} \times M}$, $M = \sum_v m_v$ (total masks across views)
- Output instance labels: $L \in \{0, ..., K\}^N$, $K$ = number of predicted instances

**Pass/fail:**

$$|\text{mAP}_{ours}@50 - \text{mAP}_{paper}@50| \leq 5.0$$

If fail: debug implementation, do not proceed to Experiment 2.

**Estimated compute:** ~8 GPU-hours on A100 (SAM inference dominates: ~100 views × 20 scenes × 0.15s/view = 300 SAM calls + point cloud processing)

---

### Experiment 2: Tier 1 Gate — Point Cloud Construction Under Motion

**Claim under test:** TSDF fusion produces motion-smeared geometry for moving subjects, making SAM3D's input structurally invalid (gap §1.5).

**Protocol:**

Using AIST++ multi-view sequences:

1. **Baseline (oracle):** Use single-frame per-camera depth (no temporal fusion). Build per-frame point clouds, transform to world coordinates using GT camera extrinsics. Compute instance segmentation metrics per frame.

2. **TSDF fusion (SAM3D's assumption):** Fuse depth maps across 30 frames (1 second at 30fps) using standard TSDF:

$$\text{TSDF}(\mathbf{x}) = \frac{\sum_{v} w_v \cdot \text{sdf}_v(\mathbf{x})}{\sum_{v} w_v}$$

where $\mathbf{x}$ is a voxel center, $\text{sdf}_v$ is the signed distance from the depth map of view $v$.

3. **Measure geometric quality:**

$$\text{Chamfer}(P_{fused}, P_{GT}) = \frac{1}{|P_{fused}|}\sum_{p \in P_{fused}} \min_{q \in P_{GT}} \|p - q\|_2 + \frac{1}{|P_{GT}|}\sum_{q \in P_{GT}} \min_{p \in P_{fused}} \|p - q\|_2$$

where $P_{GT}$ is the SMPL mesh vertices at the median frame.

**Key measurement — Motion Smear Index:**

$$\text{MSI} = \frac{\text{Chamfer}(P_{TSDF}, P_{GT})}{\text{Chamfer}(P_{single}, P_{GT})}$$

If $\text{MSI} \gg 1$, temporal fusion actively degrades geometry compared to single-frame.

**Ablation — temporal window:**

$$W \in \{1, 5, 10, 30, 60, 150\} \text{ frames}$$

Plot MSI vs. $W$ for each motion category (toprock, footwork, power moves). Expect:

$$\text{MSI}(W) \approx 1 + \alpha \cdot v_{rms} \cdot W \cdot \Delta t / R_{body}$$

where $v_{rms}$ is RMS joint velocity, $\Delta t = 1/30$s, and $R_{body} \approx 0.3$m is characteristic body dimension.

**Pass/fail:**

- If $\text{MSI}(W=30) > 3.0$ for power moves: **CONFIRMED** — TSDF fusion is structurally invalid for fast motion. The gap analysis is correct.
- If $\text{MSI}(W=30) < 1.5$ for all categories: **REFUTED** — fusion is surprisingly robust.
- Intermediate: characterize the motion velocity threshold where fusion breaks.

**Estimated compute:** ~4 GPU-hours (50 sequences × 9 views × depth processing)

---

### Experiment 3: Tier 1 Gate — Superpoint Purity Under Motion

**Claim under test:** Motion reduces superpoint purity to 0.15 for power moves, causing AP to drop ~99% (gap §2.4).

**Protocol:**

Using AIST++ sequences with GT SMPL meshes as ground truth instance labels:

1. Build point cloud from single-frame multi-view depth (bypassing the TSDF problem from Exp 2)
2. Generate superpoints via graph-based oversegmentation
3. Compute superpoint purity:

$$\text{Purity}(s) = \frac{\max_k |s \cap I_k|}{|s|}$$

where $I_k$ is the set of points belonging to GT instance $k$ (body parts defined by SMPL segmentation: 24 parts).

$$\text{Purity}_{avg} = \frac{1}{N_{sp}} \sum_{s} \text{Purity}(s)$$

4. Now introduce temporal smearing: for each superpoint, count points from multiple time steps that fall within the same spatial voxel:

$$\text{Purity}_{temporal}(s, W) = \frac{\max_k |s \cap I_k^{t_{mid}}|}{|s \cap \bigcup_{t \in W} I_k^{t}|}$$

This measures how much temporal mixing degrades part-level purity.

**Ablation — velocity stratification:**

Partition frames by instantaneous max joint velocity:

| Regime | $v_{max}$ range | Expected purity |
|--------|----------------|-----------------|
| Freeze | $< 0.1$ m/s | $> 0.9$ |
| Toprock | $0.5$–$2.0$ m/s | $0.6$–$0.8$ |
| Footwork | $1.0$–$4.0$ m/s | $0.3$–$0.5$ |
| Power move | $3.0$–$8.0$ m/s | $< 0.2$ |

**Validating the Purity→AP model:**

Run SAM3D with synthetically degraded purity and measure actual AP. Compare:

$$\text{AP}_{predicted} = \text{AP}_{static} \times \text{Purity}^{\beta}$$

Fit $\beta$ empirically by least squares:

$$\hat{\beta} = \arg\min_{\beta} \sum_{i} \left(\log \text{AP}_i - \log \text{AP}_{static} - \beta \log \text{Purity}_i\right)^2$$

**Pass/fail:**

- Purity < 0.2 for power moves AND $\hat{\beta} > 1.5$: **CONFIRMED** — catastrophic AP drop is real.
- Purity > 0.5 for power moves: **REFUTED** — motion impact is overstated.
- Check whether $\hat{\beta} \approx 2.0$ (gap §2.3) or $\hat{\beta} \approx 2.5$ (gap §2.4) — resolves the internal inconsistency.

**Estimated compute:** ~6 GPU-hours (50 sequences × superpoint computation + SAM3D inference)

---

### Experiment 4: Tier 2 — Boundary Error Characterization

**Claim under test:** 37% boundary point assignment error (gap §2.2), conflation of depth-axis vs. lateral separation.

**Protocol:**

Using ScanNet GT depth + GT instance masks:

1. Identify boundary points: points within $\epsilon = 2$cm of an instance boundary in 3D
2. Inject depth noise $\sigma_d \in \{0.01, 0.02, 0.05, 0.10\}$m
3. Re-run mask assignment (project to 2D, assign via SAM mask, back-project)
4. Measure boundary point reassignment rate:

$$e_{boundary}(\sigma_d) = \frac{|\{p \in \mathcal{B} : \text{label}(p, \hat{d}) \neq \text{label}(p, d_{GT})\}|}{|\mathcal{B}|}$$

5. **Decompose by separation angle** $\phi$ (angle between boundary separation vector and camera depth axis):

$$e_{boundary}(\sigma_d, \phi) = f(\sigma_d, \phi)$$

The gap analysis predicts $e \approx 1 - \Phi(d_{boundary}/\sigma_d)$. The corrected prediction is:

$$e_{corrected} \approx 1 - \Phi\left(\frac{d_{boundary} \cdot |\cos\phi|}{\sigma_d}\right)$$

**Ablation grid:**

$$\sigma_d \times \phi: \{0.01, 0.02, 0.05, 0.10\} \times \{0°, 15°, 30°, 45°, 60°, 75°, 90°\}$$

**Validation target:** For $d_{boundary} = 5$cm, $\sigma_d = 5$cm:

| $\phi$ | Original prediction | Corrected prediction |
|---------|-------------------|---------------------|
| 0° (along depth) | 37% | 37% |
| 45° | 37% | 26% |
| 90° (lateral) | 37% | ~0% |

**Pass/fail:**

- If empirical $e(\phi=90°) < 5\%$ and $e(\phi=0°) > 30\%$: corrected formula validated, original formula is wrong for non-depth-aligned boundaries.
- If empirical $e$ is roughly constant across $\phi$: something else is going on (perhaps the assignment mechanism itself introduces errors regardless of geometry).

**Estimated compute:** ~2 GPU-hours

---

### Experiment 5: Tier 2 — Depth Estimator Comparison and mAP Impact

**Claim under test:** Replacing GT depth with MDE drops mAP to 28/24/18 for DepthPro/DepthAnything v2/ZoeDepth (gap §2.3).

**Protocol:**

Using ScanNet val scenes 0-20:

1. Run SAM3D with GT depth → $\text{mAP}_{GT}$
2. Run SAM3D with each MDE replacing GT depth:
   - DepthPro (Apple, metric)
   - Depth Anything v2 (relative, scale-shift aligned per frame)
   - ZoeDepth (metric)
   - UniDepth (metric + intrinsics)

3. For each MDE, compute:

$$\Delta\text{mAP} = \text{mAP}_{GT} - \text{mAP}_{MDE}$$

4. **Depth error characterization** per MDE:

$$\text{AbsRel}(v) = \frac{1}{|P_v|} \sum_{p \in P_v} \frac{|d_{pred}(p) - d_{GT}(p)|}{d_{GT}(p)}$$

$$\sigma_d(v) = \text{std}(d_{pred} - d_{GT}) \text{ over view } v$$

5. **Radial error measurement** (gap §3.2):

Partition each image into radial bins from center. Compute per-bin depth error:

$$\text{AbsRel}(r) = \frac{1}{|P_r|} \sum_{p \in P_r} \frac{|d_{pred}(p) - d_{GT}(p)|}{d_{GT}(p)}$$

where $r = \sqrt{(u - c_x)^2 + (v - c_y)^2}$ in pixels.

The gap analysis claims 5-10% radial error at edges from z-depth vs Euclidean depth confusion. Measure whether each MDE outputs z-depth or Euclidean depth by fitting:

$$d_{pred} \approx d_{GT} \cdot \sqrt{1 + \left(\frac{u - c_x}{f_x}\right)^2 + \left(\frac{v - c_y}{f_y}\right)^2}$$

If the residual after this correction is significantly lower than before, the MDE outputs Euclidean depth.

**Pass/fail matrix:**

| MDE | Predicted mAP@50 | Pass if within |
|-----|------------------|----------------|
| DepthPro | 28 | ±8 |
| Depth Anything v2 | 24 | ±8 |
| ZoeDepth | 18 | ±8 |

Wide tolerance (±8) because the prediction model is acknowledged as an order-of-magnitude estimate. The ranking should be preserved:

$$\text{mAP}_{DepthPro} > \text{mAP}_{DAv2} > \text{mAP}_{ZoeDepth}$$

If the ranking is violated, the underlying error model is wrong.

**Estimated compute:** ~24 GPU-hours (4 MDEs × 20 scenes × SAM3D pipeline)

---

### Experiment 6: Tier 2 — FLOPs Audit

**Claim under test:** FLOPs calculation is off by ~2× due to MAC/FLOP convention mixing (gap §2.1).

**Protocol:** This is a code-level audit, not a GPU experiment.

1. Instrument SAM ViT-H with `torch.profiler` or `fvcore.nn.FlopCountAnalysis`
2. Measure actual FLOPs for a single 1024×1024 forward pass
3. Compare against:
   - Gap analysis original estimate: 2.8 TFLOPs
   - Gap analysis corrected estimate: 5.4 TFLOPs
   - PyTorch profiler measurement: $F_{measured}$

```python
from fvcore.nn import FlopCountAnalysis
from segment_anything import sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
img = torch.randn(1, 3, 1024, 1024)
flops = FlopCountAnalysis(sam.image_encoder, img)
print(f"Image encoder FLOPs: {flops.total() / 1e12:.2f} TFLOPs")
```

**Pass/fail:**

- If $F_{measured} \in [4.5, 6.5]$ TFLOPs: gap analysis correction is right (2×MACs convention)
- If $F_{measured} \in [2.2, 3.5]$ TFLOPs: original estimate was correct (MAC convention)

**Estimated compute:** <0.1 GPU-hours

---

### Experiment 7: Tier 2 — Resolution Misalignment Quantification

**Claim under test:** Cross-tool resolution mismatches create centimeter-scale 3D boundary errors (gap §5.2).

**Protocol:**

Using ScanNet GT with known geometry:

1. Generate SAM masks at 1024×1024
2. Generate depth maps at 518×518 (Depth Anything v2 native)
3. Back-project SAM boundary points using depth from:
   - (a) Depth map resized to 1024×1024 (bilinear)
   - (b) SAM mask boundaries resized to 518×518 (nearest-neighbor)
   - (c) Both at native video resolution 1920×1080 with proper affine transforms

4. Measure 3D boundary displacement between methods:

$$\delta_{3D}(a, c) = \frac{1}{|\mathcal{B}|}\sum_{p \in \mathcal{B}} \|p^{(a)}_{3D} - p^{(c)}_{3D}\|_2$$

**Pass/fail:**

- $\delta_{3D} > 2$cm: resolution mismatch is a real problem requiring the canonical pipeline (gap §5.2 confirmed)
- $\delta_{3D} < 0.5$cm: negligible, not worth engineering effort

**Estimated compute:** ~1 GPU-hour

---

### Experiment 8: Tier 3 — Effective View Count for Handheld Video

**Claim under test:** Handheld camera provides $V_{eff} \approx 2$–$3$, not 1 (gap §4.1).

**Protocol:**

Using BRACE clips where camera motion is visible:

1. Run COLMAP SfM on each clip to estimate camera poses $\{T_t\}_{t=1}^{T}$
2. Compute inter-frame baseline:

$$b(t_1, t_2) = \|c_{t_1} - c_{t_2}\|_2$$

where $c_t$ is the camera center extracted from $T_t$.

3. Define effective view count as the number of frames whose baselines exceed a minimum useful baseline $b_{min}$:

$$V_{eff}(b_{min}) = \left|\left\{t : \max_{t' < t} b(t, t') > b_{min}\right\}\right|$$

For scene depth $d_{scene} \approx 3$m, useful stereo baseline requires $b_{min} \approx d_{scene}/20 = 0.15$m.

4. Stratify by clip type:
   - Tripod/static: expect $V_{eff} \approx 1$
   - Handheld stable: expect $V_{eff} \approx 2$–$5$
   - Handheld dynamic (following dancer): expect $V_{eff} \approx 5$–$15$

**Also measure angular diversity:**

$$\theta_{eff} = \max_{t_1, t_2} \arccos\left(\frac{(c_{t_1} - \bar{p}) \cdot (c_{t_2} - \bar{p})}{\|c_{t_1} - \bar{p}\| \|c_{t_2} - \bar{p}\|}\right)$$

where $\bar{p}$ is the estimated scene centroid. SAM3D on ScanNet typically has $\theta_{eff} > 180°$. Handheld battle footage will have $\theta_{eff} \approx 5°$–$20°$.

**Pass/fail:**

- If $V_{eff} < 3$ for 90% of clips: gap analysis is essentially correct ($V_{eff} \approx 1$–$2$)
- If $V_{eff} > 10$ for handheld clips: significant geometric diversity exists, worth exploiting

**Estimated compute:** ~8 GPU-hours (COLMAP on 100 clips)

---

### Experiment 9: Tier 3 — Battle Lighting Domain Gap

**Claim under test:** Stage lighting degrades SAM/MDE/CoTracker unpredictably (gap §4.4).

**Protocol:**

Using BRACE clips with visible lighting variation:

1. Classify 100 BRACE clips by lighting condition:
   - Even/white (rehearsal footage, if available)
   - Single spotlight
   - Colored gels
   - Strobe/dynamic

2. Run SAM automatic mask generation on each clip's frames. Measure:

$$\text{MaskStability}(clip) = \frac{1}{T-1}\sum_{t=1}^{T-1} \text{IoU}(\text{mask}_{t}, \text{warp}(\text{mask}_{t-1}, \text{flow}_{t \to t-1}))$$

where flow comes from RAFT or CoTracker3. High stability = consistent segmentation across frames.

3. Run DepthPro on each frame. Measure temporal consistency:

$$\text{DepthFlicker}(clip) = \frac{1}{T-1}\sum_{t=1}^{T-1} \frac{|\hat{d}_t(u,v) - \hat{d}_{t-1}(\text{warp}(u,v))|}{|\hat{d}_t(u,v)|}$$

4. Compare metrics across lighting conditions.

**Pass/fail:**

- If MaskStability drops > 20% under colored lighting vs. white: domain gap is real, preprocessing needed
- If DepthFlicker increases > 50% under stage lighting: MDE is unreliable for battle conditions

**Estimated compute:** ~12 GPU-hours

---

### Experiment 10: Tier 3 — Motion Velocity Ground Truth

**Claim under test:** Hand velocities reach 2-5 m/s (toprock) and 3-8 m/s (power moves) (gap §6.5).

**Protocol:**

Using AIST++ GT 3D keypoints (calibrated mocap):

1. Compute per-joint velocity:

$$v_j(t) = \frac{\|p_j(t+1) - p_j(t-1)\|_2}{2\Delta t}$$

using central differences for noise reduction.

2. Compute statistics stratified by dance style and joint:

$$v_{max,j} = \max_t v_j(t), \quad v_{rms,j} = \sqrt{\frac{1}{T}\sum_t v_j^2(t)}$$

3. Also compute from BRACE using WHAM pseudo-GT (noisier but real breaking data).

**Pass/fail:** This is a measurement experiment, not a hypothesis test. Report the actual distributions:

$$p(v_{max} | \text{joint}, \text{style})$$

If AIST++ breaking clips show $v_{max,hand} > 5$ m/s for even 10% of frames, the gap analysis velocity claims are grounded. If $v_{max,hand} < 3$ m/s universally, the claims are overstated.

**Estimated compute:** ~0.5 GPU-hours (pure numpy on pre-computed keypoints)

---

## 3. Metrics for BRACE (No 3D Ground Truth)

Since BRACE has no 3D instance segmentation ground truth, we need proxy metrics:

### 3.1 Temporal Mask Consistency (TMC)

$$\text{TMC} = \frac{1}{T-1}\sum_{t=1}^{T-1} \text{IoU}\left(\text{mask}_t^{(i)}, \text{warp}(\text{mask}_{t-1}^{(i)}, F_{t-1 \to t})\right)$$

Measures whether the same instance maintains consistent segmentation across frames. High TMC = stable segmentation. Use CoTracker3 or RAFT optical flow for warping.

### 3.2 Geometric Plausibility Score (GPS)

$$\text{GPS} = 1 - \frac{|\{p \in P_{dancer} : p_z < z_{floor} - \epsilon\}|}{|P_{dancer}|}$$

Fraction of dancer points that don't penetrate the detected floor plane. Measures physical plausibility of the 3D reconstruction.

### 3.3 Silhouette Reprojection Error (SRE)

$$\text{SRE} = 1 - \text{IoU}\left(\pi(P_{3D}), M_{2D}\right)$$

Project the 3D point cloud back to 2D and compare against the original SAM mask. If the 3D reconstruction is good, the reprojection should match the original mask. Degradation indicates 3D errors.

### 3.4 SMPL Alignment Error (SAE)

$$\text{SAE} = \frac{1}{|P_{dancer}|}\sum_{p \in P_{dancer}} \min_{v \in \mathcal{V}_{SMPL}} \|p - v\|_2$$

Compare SAM3D's 3D point cloud against the SMPL mesh recovered by HMR 2.0/WHAM. This is the most direct measure of whether SAM3D adds value over mesh recovery alone.

**Critical insight:** If $\text{SAE} > 5$cm on average, SAM3D's 3D output is less accurate than HMR's parametric mesh — meaning SAM3D provides no added value for body understanding and the gap analysis recommendation (§4.3: "use HMR directly, skip SAM3D for body analysis") is empirically validated.

---

## 4. Ablation Schedule

### 4.1 Superpoint Parameter Sweep (Experiment 3)

| Parameter | Values | Expected impact |
|-----------|--------|----------------|
| Voxel size | 1, 2, 3, 5 cm | Coarser = fewer superpoints, lower purity |
| k-NN neighbors | 10, 20, 30, 50 | More = smoother boundaries, larger segments |
| Min segment size | 10, 50, 100, 500 points | Larger = fewer but purer superpoints |
| Color weight $\lambda_c$ | 0, 0.3, 0.5, 0.7, 1.0 | Higher = more appearance-driven splits |

Total: $4 \times 4 \times 4 \times 5 = 320$ configurations. Use grid search on 5 ScanNet scenes, evaluate mAP, select top-3 configs for full evaluation.

### 4.2 SAM Mask Filtering Sweep (§1.4)

| Parameter | Values | Default |
|-----------|--------|---------|
| `pred_iou_thresh` | 0.7, 0.8, 0.88, 0.95 | 0.88 |
| `stability_score_thresh` | 0.8, 0.9, 0.95 | 0.95 |
| `box_nms_thresh` | 0.5, 0.7, 0.9, 1.0 | 0.7 |

Measure: masks per image, overlap rate, final mAP. The interaction between NMS threshold and 3D vote quality is the key unknown.

### 4.3 Temporal Window Sweep (Experiment 2)

$$W \in \{1, 2, 5, 10, 15, 30, 60, 150\} \text{ frames}$$

For each $W$, measure: MSI, superpoint purity, mAP (on AIST++). Plot all three vs. $W$ per motion category. The critical output is the **optimal temporal window** per motion regime:

$$W^* = \arg\max_W \text{mAP}(W)$$

Hypothesis: $W^*$ decreases with motion velocity. For freezes, $W^* \to \infty$. For power moves, $W^* = 1$ (single frame only).

---

## 5. Compute Budget

| Experiment | GPU-hours | Priority |
|-----------|-----------|----------|
| Exp 1: ScanNet reproduction | 8 | **GATE** |
| Exp 2: TSDF under motion | 4 | **GATE** |
| Exp 3: Superpoint purity | 6 | **GATE** |
| Exp 4: Boundary error | 2 | Tier 2 |
| Exp 5: MDE comparison | 24 | Tier 2 |
| Exp 6: FLOPs audit | 0.1 | Tier 2 |
| Exp 7: Resolution mismatch | 1 | Tier 2 |
| Exp 8: $V_{eff}$ measurement | 8 | Tier 3 |
| Exp 9: Lighting domain gap | 12 | Tier 3 |
| Exp 10: Velocity ground truth | 0.5 | Tier 3 |
| **Total** | **~66** | |

On a single A100: ~3 days. On 4×A100: ~1 day. The gate experiments (18 GPU-hours) should run first and complete in under a day.

---

## 6. Decision Framework

After running the gate experiments (1-3), apply this decision tree:

```
Exp 1: Can we reproduce SAM3D on ScanNet?
├── NO → Implementation bug. Fix and retry. Do not proceed.
└── YES → 
    Exp 2: Does TSDF fusion work for moving subjects?
    ├── MSI > 3.0 for power moves →
    │   Can we use single-frame multi-view instead?
    │   ├── YES (AIST++ multi-cam) → Proceed to Exp 3
    │   └── NO (BRACE single-cam) → SAM3D is structurally invalid
    │       for breakdancing. STOP. Redirect to HMR+WHAM pipeline.
    └── MSI < 1.5 →
        Exp 3: What is superpoint purity under motion?
        ├── Purity < 0.2 for power moves →
        │   AP drop is catastrophic. SAM3D fails for fast motion.
        │   Decision: Use SAM3D ONLY for freeze detection.
        │   For power moves/footwork: HMR+WHAM only.
        └── Purity > 0.5 →
            SAM3D may be viable. Run Tier 2+3 experiments
            to characterize operational envelope.
```

**The most likely outcome** (based on the gap analysis reasoning, which is qualitatively sound):

$$P(\text{MSI} > 3.0 \text{ for power moves}) \approx 0.9$$
$$P(\text{Purity} < 0.2 \text{ for power moves} \mid \text{MSI} > 3.0) \approx 0.95$$

Expected decision: **SAM3D is validated as structurally incompatible with dynamic breakdancing video.** The 66 GPU-hour budget is mostly insurance — the gate experiments (~18 hours) will likely produce the definitive answer.

---

## 7. What Constitutes "Done"

The experimental campaign is complete when:

1. **Gate decision is made:** SAM3D is either viable or not for breakdancing (Experiments 1-3)
2. **If viable:** Operational envelope is characterized (which motion regimes, which depth estimators, what temporal windows)
3. **If not viable:** The specific failure mode is documented with empirical evidence (not just analytical prediction)
4. **All quantitative claims** in the gap analysis are tagged as CONFIRMED / REFUTED / REVISED with measured values
5. **The Purity→AP model** has an empirically fitted exponent $\hat{\beta}$ with confidence interval, resolving the §2.3 vs §2.4 inconsistency
6. **A recommendation** is issued: either proceed with SAM3D reimplementation (with specified constraints) or redirect effort to the HMR+WHAM+CoTracker3 pipeline

The final deliverable is a table:

| Gap # | Claim | Predicted Value | Measured Value | Status |
|-------|-------|----------------|----------------|--------|
| 2.2 | 37% boundary error | 37% | ___ % | CONFIRMED/REFUTED |
| 2.3 | mAP drops to 28 (DepthPro) | 28 | ___ | CONFIRMED/REFUTED |
| 2.4 | AP drops 99% (power moves) | 99% | ___ % | CONFIRMED/REFUTED |
| ... | ... | ... | ... | ... |

This table closes the loop between analytical prediction and empirical measurement — exactly what the gap analysis is missing.
