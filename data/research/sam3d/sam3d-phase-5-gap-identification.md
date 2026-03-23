# sam3d — Gap Identification

_Generated: 2026-03-23T06:11:59.353Z_

---

# SAM3D Reimplementation Gap Analysis

## 1. Architectural Gaps

### 1.1 Which SAM3D Paper?

**What's missing:** The research never cites the specific paper. There are at least three papers with "SAM3D" in the title published in 2023-2024:
- Yang et al. — point cloud + multi-view superpoint merging
- Cen et al. ("SA3D") — NeRF-based
- Zhang et al. — zero-shot 3D object segmentation

The described pipeline (superpoints, bidirectional merging) matches Yang et al., but without an explicit arxiv ID, the entire analysis could be grounded in the wrong paper.

**Why it matters:** Different papers have fundamentally different pipelines. Mixing details across papers invalidates the analysis.

**Resolution:** Pin the exact paper (likely arxiv:2306.03908 or similar). Verify every architectural claim against that specific paper.

---

### 1.2 Bidirectional Merging Process Collapsed to a Single Criterion

**What's missing:** The merge criterion shown is:

$$\text{merge}(G_a, G_b) = \text{IoU}_{3D} > \tau_{merge} \text{ AND } \Delta\theta_{normal} < \tau_{boundary}$$

The actual SAM3D contribution is a **bidirectional** process: (a) 2D→3D: project superpoints into each view, assign to SAM masks by majority vote to form initial 3D groups; (b) 3D→2D: for each 3D group, check which SAM masks it spans across views and merge masks that share the same 3D group. This ping-pong is the paper's core novelty. The research reduces it to a static threshold test.

**Why it matters:** The bidirectional process resolves ambiguities that a single-pass merge cannot. Implementing the simplified version would produce significantly worse results, invalidating the baseline mAP estimates.

**Resolution:** Re-read the paper's Section 3 (or equivalent) and document both merge directions with their distinct thresholds and stopping criteria.

---

### 1.3 Superpoint Generation Method Completely Unspecified

**What's missing:** The research states "superpoint voxel size = 2cm" but never describes how superpoints are constructed. SAM3D uses graph-based oversegmentation on the point cloud (color + geometric affinity, k-NN graph, Felzenszwalb-style clustering). Key parameters: minimum segment size, k-nearest neighbors, color weight vs. spatial weight, normal smoothing radius.

**Why it matters:** Superpoints are the fundamental unit of the voting scheme. Bad superpoints → bad votes → garbage output. The 2cm voxel size is a *downsampling* parameter, not the superpoint parameter. These are different things. Confusing them would produce superpoints that are either too coarse (merging body parts) or too fine (noisy votes).

**Resolution:** Document the full superpoint generation pipeline: voxel downsampling → normal estimation → k-NN graph → graph-based segmentation with specific parameters.

---

### 1.4 SAM Mask Overlap Resolution Missing

**What's missing:** SAM's automatic mask generator produces **hierarchical, overlapping** masks (whole object, parts, sub-parts). A single pixel might belong to 3+ masks. SAM3D must resolve these overlaps before 3D projection. The research doesn't discuss how.

**Why it matters:** If overlapping masks are projected to 3D without resolution, a single 3D point gets assigned to multiple instances, corrupting the vote matrix. The overlap resolution strategy (keep highest-confidence only? keep all? NMS in 3D?) dramatically affects results.

**Resolution:** Document SAM3D's mask filtering cascade: `pred_iou_thresh`, `stability_score_thresh`, `box_nms_thresh`, and any post-hoc overlap resolution in 3D.

---

### 1.5 Point Cloud Construction Pipeline Missing

**What's missing:** Before superpoints can be created, you need a fused point cloud from multi-view RGB-D. This requires: depth unprojection per frame → world-frame transformation → voxel deduplication (or TSDF fusion) → outlier removal → normal estimation. None of these steps are discussed.

**Why it matters:** For breakdancing, the dancer is non-static. Standard TSDF fusion assumes a static scene. Fusing a moving dancer across frames produces motion-smeared geometry. The research analyzes this for motion voting but not for the point cloud construction itself — you can't even BUILD the point cloud correctly for a moving subject.

**Resolution:** Explicitly address that SAM3D's point cloud construction assumes a static scene, making it fundamentally inapplicable to dynamic subjects (not just degraded — structurally invalid).

---

## 2. Math Errors

### 2.1 FLOPs Convention Inconsistency

**What's wrong:** The FLOPs calculation mixes conventions:
- Projection layers use MAC count: $4d^2$ per token (should be $8d^2$ if counting FLOPs = 2×MACs)
- Attention computation uses FLOP count: $2wd$ per token

Using consistent FLOPs = 2×MACs throughout:
- Per windowed block: $n(8d^2 + 2wd) + 16nd^2 = n(24d^2 + 2wd) \approx 163$ GFLOPs (not 82.6)
- Per global block: $n(24d^2 + 2nd) \approx 204$ GFLOPs (not 123.5)
- Total: $28 \times 163 + 4 \times 204 \approx 5.4$ TFLOPs (not 2.8)

**Why it matters:** The conclusion (survey underestimates) is correct regardless, but the claimed 7.5× underestimate may actually be ~14.6×. Presenting wrong numbers undermines credibility when correcting someone else's wrong numbers.

**Resolution:** Pick one convention (FLOPs = 2×MACs is standard in ML papers) and apply consistently. Recompute all values.

---

### 2.2 Mask Assignment Error: Dimension Conflation

**What's wrong:** 

$$e_{assign} \approx 1 - \Phi\left(\frac{d_{boundary}}{\sigma_d}\right)$$

$d_{boundary}$ is the 3D Euclidean distance between body parts (5cm for hand-near-hip). $\sigma_d$ is depth error along the **z-axis only**. These are different axes. If the hand is 5cm laterally separated from the hip (perpendicular to depth axis), depth error doesn't cause intermixing regardless of magnitude.

The formula implicitly assumes all separation is along the depth axis — a worst case, not typical. For a hand passing in front of the hip, the separation is largely lateral.

**Why it matters:** The "37% boundary point error" claim is inflated for many poses. The analysis overestimates depth-induced errors for configurations where separation is lateral, and underestimates for configurations where separation IS along depth (e.g., dancer facing camera with hands at different depths).

**Resolution:** Decompose boundary distance into depth-axis and lateral components. The formula should use the depth-axis projection of $d_{boundary}$:

$$e_{assign} \approx 1 - \Phi\left(\frac{d_{boundary} \cdot |\cos\phi|}{\sigma_d}\right)$$

where $\phi$ is the angle between the boundary separation vector and the camera's depth axis.

---

### 2.3 mAP Projection Model is Unjustified

**What's wrong:**

$$\text{mAP}_{est} \approx \text{mAP}_{GT} \times (1 - e_{assign})^2 \times \text{purity}$$

This is presented as fact but is actually an unvalidated heuristic. Why squared? Why multiply by purity linearly rather than as a power? The relationship between point-level errors and instance-level mAP is non-trivial — it depends on object sizes, IoU thresholds, and the distribution of errors across instances.

**Why it matters:** The projected mAP values (28, 24, 18 for different depth estimators) are presented with false precision. They could easily be off by ±50% relative.

**Resolution:** Label this as an order-of-magnitude estimate, not a prediction. Better: cite any empirical study that relates superpoint purity to mAP (e.g., from the Mask3D or SPFormer papers that also use superpoints).

---

### 2.4 Motion Purity to AP Conversion Exponent

**What's wrong:** Section 7 uses $\text{AP} \approx \text{AP}_{\text{static}} \times \text{Purity}^{2.5}$ — a different formula than Section 2's $(1-e)^2 \times \text{purity}$. Two different projection models for the same quantity.

**Why it matters:** The power move prediction (AP drops 99%) is sensitive to this exponent. With purity 0.15: $0.15^{2.5} = 0.0087$ vs $0.15^{2.0} = 0.0225$ — a 2.6× difference. The qualitative conclusion (it's catastrophically bad) holds, but the inconsistency is sloppy.

**Resolution:** Use one model throughout and justify the exponent empirically.

---

## 3. Implementation Risks

### 3.1 SAM "Everything" Mode Mask Explosion

**Risk:** Running SAM in automatic mask generation mode produces 100-400 masks per 1024×1024 image. For 100 views: 10,000-40,000 masks. The vote matrix $V \in \mathbb{R}^{N_{sp} \times M_{total}}$ with 100K superpoints and 20K masks = 2 billion entries. Even as a sparse matrix, this requires careful memory management.

**Why it matters:** Naive implementation will OOM on any consumer GPU. ScanNet papers typically process ~50-100 views of 640×480 images; breakdancing at 1080p@30fps for 10s = 300 frames at higher resolution.

**Resolution:** Implement progressive vote accumulation (process views sequentially, accumulate votes, discard raw masks). Use sparse storage from the start.

---

### 3.2 Depth Map to Z-Buffer Ambiguity

**Risk:** MDEs output either **z-depth** (distance along camera's optical axis) or **Euclidean/ray depth** (distance from camera center). The back-projection equation $p_{cam} = d \cdot K^{-1}[u,v,1]^T$ requires z-depth. Using Euclidean depth without correction introduces systematic radial error that grows toward image edges.

**Why it matters:** DepthPro outputs metric depth — but which metric depth? The correction factor is $d_z = d_{euclidean} / \sqrt{1 + ((u-c_x)/f_x)^2 + ((v-c_y)/f_y)^2}$. At 1080p edges, this can be a 5-10% error if ignored.

**Resolution:** Verify each depth estimator's output convention. Apply correction if needed before back-projection.

---

### 3.3 Camera Intrinsics for Monocular Video

**Risk:** SAM3D assumes known camera intrinsics $K_v$ per view. ScanNet provides calibrated intrinsics. For iPhone video, you need to extract intrinsics from EXIF metadata or the AVFoundation API. Different recording modes (wide vs ultrawide, video stabilization on/off) give different intrinsics. Video stabilization applies a per-frame crop/warp that changes the effective intrinsics dynamically.

**Why it matters:** A 5% error in focal length produces a 5% systematic depth scale error across the entire point cloud. Combined with MDE errors, this pushes the total error above any reasonable tolerance.

**Resolution:** Record with video stabilization OFF. Extract per-frame intrinsics from CMSampleBuffer metadata if using iPhone. Alternatively, run COLMAP SfM to jointly estimate intrinsics and camera poses.

---

### 3.4 SAM Mask Boundary Quantization

**Risk:** SAM outputs masks at 256×256 internally, upsampled to input resolution via bilinear interpolation. The boundary region (2-4px at native resolution) is soft but binarized at a threshold (default 0.0). This means boundary points are assigned to one side or the other with effectively random choice. When back-projected to 3D, these boundary points create a noisy halo around each instance.

**Why it matters:** Body part boundaries (arm-torso junction, leg-torso junction) are exactly where breakdancing analysis needs the most precision. The boundary halo is systematically worst at the most informative locations.

**Resolution:** Use SAM's multi-mask output and select by stability score. Consider using boundary erosion before back-projection and separately processing boundary regions.

---

## 4. Breakdance-Specific Blind Spots

### 4.1 Handheld Camera Motion ≠ Zero Geometric Diversity

**What's missed:** The research concludes "single camera → $V_{eff} = 1$" assuming all frames share the same camera position. But battle footage is typically filmed handheld, providing small but nonzero baseline between frames. Over 10 seconds, a handheld camera might move 0.5-1m, providing SOME geometric diversity — more than the analysis claims, but far less than SAM3D expects from a structured scan.

**Why it matters:** The analysis is overly pessimistic for handheld footage and overly optimistic for tripod footage. The actual $V_{eff}$ depends on camera motion, which varies per clip.

**Resolution:** Model $V_{eff}$ as a function of camera displacement between frames: $V_{eff}(t_1, t_2) \approx f(\|c_{t_1} - c_{t_2}\| / d_{scene})$. For handheld at 3m distance with 50cm movement: $V_{eff} \approx 2-3$, not 1.

---

### 4.2 Variable Frame Rate (VFR) from Phone Cameras

**What's missed:** iPhone and Android cameras often record in VFR mode by default. Frames are not uniformly spaced. The motion analysis assumes constant $\Delta t = 1/30$s, but actual inter-frame intervals can vary 20-50ms.

**Why it matters:** The critical velocity threshold $v_{crit} = R_{seed}/\Delta t$ becomes time-varying. Temporal window optimization $W^*$ is invalid if $\Delta t$ is non-constant. CoTracker3 also assumes uniform frame spacing.

**Resolution:** Force CFR recording (iPhone: set format explicitly). For existing VFR footage, resample to constant framerate with motion-compensated interpolation before processing.

---

### 4.3 Loose/Baggy Clothing

**What's missed:** Breakdancers often wear baggy pants, oversized jackets, and hats. SAM may segment clothing folds as separate objects (pants leg flaps create disconnected regions). The visual silhouette differs substantially from the body mesh — SAM segments what it sees (cloth surface), but HMR 2.0/WHAM recover the body underneath.

**Why it matters:** SAM3D mask → 3D point cloud captures the cloth surface. HMR mesh recovery captures the body surface. These are different surfaces with 5-15cm disagreement. Fusing them directly creates ghost geometry.

**Resolution:** Acknowledge this as a fundamental mismatch. For body analysis, use HMR directly (skip SAM3D). Reserve SAM3D for scene segmentation only (isolating the dancer from background, not segmenting body parts).

---

### 4.4 Battle Stage Lighting

**What's missed:** Breaking battles use dramatic stage lighting — moving spotlights, colored gels, strobes, haze machines. Effects on the pipeline:
- SAM: contrast changes affect mask boundaries
- MDE: trained on natural lighting; colored lighting shifts depth estimates unpredictably  
- CoTracker3: appearance changes between frames break feature matching
- Color-based superpoint affinity: colored lighting makes the same surface appear different colors from different frames

**Why it matters:** Most benchmarks use constant, white lighting. The domain gap to battle conditions is unquantified.

**Resolution:** Test the pipeline on the BRACE dataset (which uses real battle footage) to empirically measure the degradation. Consider preprocessing (white-balance normalization, histogram equalization) before feeding frames to SAM/MDE.

---

### 4.5 Floor Contact Geometry Beyond Reflections

**What's missed:** The research mentions reflective floors affecting depth, but misses a subtler issue: during freezes and power moves, the dancer's body IS the floor contact surface. Head, hands, shoulders, and elbows press against and conform to the floor. In 3D, these contact points should coincide with the floor plane, but MDE predicts them at slightly different depths, creating impossible intersections (body penetrating floor).

**Why it matters:** If using 3D reconstruction for biomechanical analysis (force vectors, balance assessment), floor-penetrating geometry produces physically impossible conclusions.

**Resolution:** Detect the floor plane (RANSAC on background points), then clamp dancer point cloud to not penetrate it. Use the floor plane as a hard constraint in any optimization.

---

## 5. Integration Gaps

### 5.1 SAM3D Output → MotionBERT Input: Incompatible Data Types

**What's missing:** SAM3D outputs: 3D instance segments (point clouds with instance labels). MotionBERT consumes: 2D keypoint sequences $X \in \mathbb{R}^{T \times 17 \times 2}$ (Human3.6M skeleton format). There is no defined bridge between "3D point cloud segments" and "2D keypoint detections."

**Why it matters:** Without this bridge, the two systems can't be connected. You'd need either: (a) a separate 2D keypoint detector (making SAM3D's output irrelevant to MotionBERT), or (b) projecting SAM3D's 3D body-part segments back to 2D and fitting keypoints (defeating the purpose of 3D lifting).

**Resolution:** Clarify the pipeline architecture. SAM3D and MotionBERT likely operate in parallel on different tasks, not in series. SAM3D → scene understanding; MotionBERT → pose classification. They share the input video but don't feed into each other.

---

### 5.2 Resolution Pipeline Mismatches

**What's missing:**

| Tool | Native resolution | Aspect ratio |
|------|------------------|-------------|
| SAM ViT-H | 1024×1024 (square) | 1:1 |
| DepthPro | 1536×1536 | 1:1 |
| Depth Anything v2 | 518×518 | 1:1 |
| CoTracker3 | 384×512 or native | Variable |
| iPhone video | 1920×1080 | 16:9 |

Every tool requires resizing/padding the input differently. Mask coordinates from SAM at 1024×1024 don't directly align with depth maps at 518×518 or video frames at 1920×1080. Each resize introduces boundary misalignment.

**Why it matters:** A 1-2 pixel misalignment at mask boundaries, when back-projected to 3D, becomes centimeter-scale errors — exactly at the body part boundaries where precision matters most.

**Resolution:** Define a canonical resolution pipeline. Process everything at native video resolution, resize per-tool with tracked affine transforms, and map all outputs back to native coordinates before fusion.

---

### 5.3 CoTracker3 Temporal Windows vs. SAM3D Multi-View

**What's missing:** CoTracker3 processes 8-frame sliding windows. SAM3D conceptually processes all views simultaneously (global vote matrix). There's no discussion of how to partition temporal processing between the two, or whether CoTracker3's window boundaries create discontinuities in the 3D segmentation.

**Why it matters:** At window boundaries, CoTracker3 may shift point identities slightly, causing vote inconsistencies in SAM3D's matrix. This is a subtle synchronization bug that would be hard to debug.

**Resolution:** Use CoTracker3's online/causal mode (processes frames sequentially with memory) to avoid window-boundary artifacts. Alternatively, use overlapping windows and stitch in the overlap region.

---

### 5.4 Coordinate System Conventions Undefined

**What's missing:** The pipeline involves:
- SAM: pixel coordinates (u,v), origin top-left
- Depth maps: pixel-aligned, values in meters (or inverse depth)
- Back-projection: camera coordinates (right-handed, z-forward for OpenCV; z-backward for OpenGL)
- SMPL meshes (from HMR): body-centered coordinates, y-up
- CoTracker3: pixel coordinates, sub-pixel precision

No coordinate system alignment diagram or convention document exists.

**Why it matters:** A sign flip in z or a y-up vs z-up confusion silently produces mirrored or rotated geometry that looks plausible but is wrong. This is the #1 integration bug in multi-tool 3D pipelines.

**Resolution:** Define canonical coordinates (e.g., OpenCV convention: x-right, y-down, z-forward, right-handed). Document every tool's native convention and the transform needed to convert.

---

## 6. Citation Verification

### 6.1 Internal Contradiction: Hallucinated Tools Used in Pipeline

**Critical issue:** Section 5 of the research correctly identifies SAM 3, SAM-Body4D, and SAM-3D-Body as **likely hallucinated** (confidence 0.05-0.15). But the Tech Stack Re-Evaluation document builds the entire "NEW (March 2026 Stack)" pipeline around these exact tools:

- Step ② → SAM 3 (flagged as hallucinated, confidence 0.05)
- Step ④ → SAM-Body4D (flagged as hallucinated, confidence 0.10)

The research simultaneously says "these don't exist" and "build the pipeline on them." This is the most critical internal inconsistency in the entire analysis.

**Resolution:** The corrected pipeline from Section 6 (SAM 2 + Grounding DINO + HMR 2.0 + WHAM + CoTracker3) should be the canonical recommendation. The Tech Stack Re-Evaluation should be marked as superseded or corrected.

---

### 6.2 SAM3D's Claimed vs. Actual mAP@50

The research argues SAM3D achieves ~28-33 mAP@50, not the survey's ~46. The argument (class-agnostic vs class-specific) is plausible but not verified against the actual paper's tables. The actual paper reports specific numbers — check Table 1/2 of the source paper.

**Resolution:** Read the actual paper's result tables. The gap between class-agnostic and class-specific evaluation is real but the specific range (28-33) needs grounding.

---

### 6.3 DanceFormer Reference

The citation points to `sciencedirect.com/science/article/pii/S1110016825001814` — an Alexandria Engineering Journal article. The "18.4mm on AIST" claim is extremely specific. AIST++ evaluates in PA-MPJPE; 18.4mm would be state-of-the-art by a large margin (current SOTA on AIST++ is ~40-50mm PA-MPJPE). This number is likely wrong — it may be a different metric, a different dataset, or hallucinated.

**Resolution:** Verify this specific paper and metric. If the 18.4mm claim is wrong, the Gap #1 "LOW severity" assessment may need revision.

---

### 6.4 JOSH at ICLR 2026

Referenced as arxiv:2501.02158. ICLR 2026 decisions would have been announced ~Jan 2026. The preprint date (Jan 2025) is consistent with an ICLR 2026 submission. Plausible but should be verified on OpenReview.

---

### 6.5 Motion Velocities "from mocap literature"

Section 7 provides specific velocity tables (hand: 2-5 m/s toprock, 3-8 m/s power moves) attributed to "mocap literature" with no citations. These numbers are critical to the motion invalidation argument.

**Resolution:** Cite specific mocap studies. The AIST++ dataset includes some motion statistics. The 8 m/s hand velocity during power moves is plausible but should reference a specific measurement.

---

## Summary: Priority-Ordered Gaps

| # | Gap | Severity | Section |
|---|-----|----------|---------|
| 1 | Hallucinated tools used in canonical pipeline | **CRITICAL** | 6.1 |
| 2 | Bidirectional merging process missing | **HIGH** | 1.2 |
| 3 | Which SAM3D paper? | **HIGH** | 1.1 |
| 4 | Point cloud can't be built for moving subjects | **HIGH** | 1.5 |
| 5 | SAM3D ↔ MotionBERT interface undefined | **HIGH** | 5.1 |
| 6 | Coordinate system conventions undefined | **HIGH** | 5.4 |
| 7 | FLOPs counting inconsistency | **MEDIUM** | 2.1 |
| 8 | Resolution mismatches between tools | **MEDIUM** | 5.2 |
| 9 | DanceFormer 18.4mm claim likely wrong | **MEDIUM** | 6.3 |
| 10 | Mask assignment error formula conflates axes | **MEDIUM** | 2.2 |
| 11 | Superpoint generation unspecified | **MEDIUM** | 1.3 |
| 12 | VFR from phone cameras | **LOW** | 4.2 |
| 13 | Baggy clothing surface mismatch | **LOW** | 4.3 |
| 14 | Battle stage lighting | **LOW** | 4.4 |

The research is strong on the qualitative conclusions (SAM3D is incompatible with dynamic dance video) but has quantitative precision issues and a critical internal contradiction where hallucinated tools are simultaneously debunked and relied upon.
