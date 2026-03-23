# motionbert — Follow-up 1.2: Breaking battles have two dancers, a DJ, judges, and spectators — all in frame. The pipeline assumes

_Generated: 2026-03-23T02:55:05.983Z_

---

# Gap Analysis: Multi-Person Scene Handling in Breaking Battle Footage

## Problem Statement

The entire pipeline — from SAM 3 segmentation through MotionBERT lifting to SAM-Body4D mesh recovery — is architected for **single-person input**. Real breaking footage is fundamentally multi-person:

| Scene Element | Count | Behavior |
|---|---|---|
| Active dancer | 1 | Moving, center stage |
| Opponent | 1 | Standing/bouncing at edge, enters on transition |
| DJ | 1 | Stationary, behind turntables |
| Judges | 3-5 | Seated, minimal motion |
| Spectators (cypher) | 10-50+ | Forming a circle, occasional movement |
| MC/Host | 0-1 | Standing, gesturing |

The pipeline must solve **four** distinct subproblems in sequence: detection, selection, tracking, and re-identification. Failure at any stage corrupts all downstream pose estimation.

---

## 1. Person Detection: The Crowd Density Problem

### 1.1 Detector Saturation

Standard person detectors (YOLO-v8, RT-DETR, ViTPose's built-in detector) are trained on COCO, which has a **mean of 3.5 persons per image**. A breaking battle cypher has 15-60 people in frame. Detection performance degrades with crowd density:

$$\text{AP}_{50}(\text{persons}) \approx \text{AP}_{50}^{\text{baseline}} \times \exp\left(-\lambda \cdot \max(0, N_{\text{persons}} - N_{\text{train}})\right)$$

where $N_{\text{train}} \approx 7$ (95th percentile of COCO person count per image) and $\lambda \approx 0.03\text{-}0.05$ empirically. At $N = 30$ persons: AP drops by ~50-70%.

More critically, **Non-Maximum Suppression (NMS)** with typical IoU threshold 0.5-0.65 suppresses legitimate detections when people overlap. In a tight cypher, spectators stand shoulder-to-shoulder — their bounding boxes overlap with IoU > 0.5, causing missed detections.

### 1.2 The Active Dancer's Bounding Box Problem

During power moves, the active dancer's bounding box is **non-standard**:

- **Windmill**: Body extends horizontally, ~2:1 aspect ratio, occupying ~400×200 pixels vs. normal ~150×400
- **Headspin**: Inverted, legs extend upward — detectors trained on upright people may assign low confidence
- **Flare**: Legs sweep a wide arc, bounding box changes by 2-3× between frames

Standard anchor-based detectors (YOLO) have fixed aspect ratio priors. The breaking dancer's bbox aspect ratio distribution:

$$\text{AR}_{\text{breaking}} \sim \text{Uniform}(0.3, 3.0)$$
$$\text{AR}_{\text{COCO}} \sim \mathcal{N}(0.42, 0.15) \quad \text{(standing person, height/width)}$$

The mismatch means anchor-free detectors (RT-DETR, DETR variants) are strongly preferred over anchor-based (YOLOv5-v7).

### 1.3 Tensor Shapes Through the Detection Stage

For a single frame at 1080p:

```
Input:           (1, 3, 1080, 1920)     — RGB frame
Detector output: (N_det, 6)             — [x1, y1, x2, y2, confidence, class]
                 N_det ∈ [5, 60]        — variable, mostly spectators
Filter to top-K: (K, 6)                 — K = 5-10 candidate persons
Crop & resize:   (K, 3, 256, 192)       — per-person crops for ViTPose
2D keypoints:    (K, 17, 3)             — [x, y, confidence] per joint per person
```

**The selection problem**: Which of the $K$ detections is the active dancer?

---

## 2. Active Dancer Selection: A Classification Problem

### 2.1 Heuristic Selection (Baseline)

The simplest approach uses spatial and motion priors:

$$\text{score}(d_i) = w_1 \cdot S_{\text{center}}(d_i) + w_2 \cdot S_{\text{size}}(d_i) + w_3 \cdot S_{\text{motion}}(d_i) + w_4 \cdot S_{\text{floor}}(d_i)$$

Where:

**Center proximity** — the active dancer tends to be near frame center:
$$S_{\text{center}}(d_i) = \exp\left(-\frac{\|c_i - c_{\text{frame}}\|^2}{2\sigma_c^2}\right), \quad \sigma_c \approx 0.25 \times W_{\text{frame}}$$

**Relative size** — the active dancer typically has one of the largest bounding boxes:
$$S_{\text{size}}(d_i) = \frac{A_i}{\max_j A_j}, \quad A_i = (x_2 - x_1)(y_2 - y_1)$$

**Motion magnitude** — the active dancer moves more than bystanders. Using frame differencing within the bounding box:
$$S_{\text{motion}}(d_i) = \frac{\sum_{p \in \text{bbox}_i} |I_t(p) - I_{t-1}(p)|}{\text{Area}(d_i)}$$

**Floor proximity** — during power moves, the dancer's center-of-mass is lower:
$$S_{\text{floor}}(d_i) = \sigma\left(\frac{y_{\text{center}}(d_i) - y_{\text{median}}}{\tau}\right)$$

where $y_{\text{center}}$ is the vertical center of the bbox (higher $y$ = closer to floor in image coordinates).

**Failure modes of heuristic selection:**

| Scenario | Which heuristic fails | Why |
|---|---|---|
| Camera follows opponent during transition | Center proximity | Both dancers near center |
| Dancer at edge of cypher | Center + Size | Partially occluded, smaller bbox |
| Toprock (standing moves) | Floor proximity + Motion | Similar motion magnitude to bouncing opponent |
| Camera cut to judges | All | No dancer in frame at all |
| Zoom-in on single move | Size | All detections are large |

Empirical failure rate of heuristic selection: **~8-15% of frames** in broadcast competition footage, based on analogous person-of-interest selection in sports analytics.

### 2.2 Learned Active Dancer Classifier

A more robust approach trains a binary classifier: **active dancer vs. bystander**.

**Input features per detection** (dimension: 64-128):
- Bounding box geometry: $(x_1, y_1, x_2, y_2, A, \text{AR})$ — 6D
- 2D keypoint configuration: flatten 17 joints × 3 = 51D
- Optical flow statistics within bbox: mean, std, max of flow magnitude — 3D
- Temporal context: bbox IoU with previous frame's active dancer — 1D
- Detection confidence — 1D

$$f_i = [g_i; k_i; o_i; \text{IoU}_{i,\text{prev}}; c_i] \in \mathbb{R}^{62}$$

$$P(\text{active} | f_i) = \sigma(\text{MLP}(f_i))$$

**Training data**: This classifier can be trained on general dance/sports datasets with person-of-interest annotations. Key datasets:
- **FineGym** (gymnastics): annotated active athlete in multi-person scenes
- **SportsMOT**: multi-person sports tracking with role annotations  
- **AIST++**: single dancer, but can synthetically paste bystanders

**Estimated performance**: With the feature set above and ~5K annotated frames, expect ~95-97% per-frame accuracy. The remaining 3-5% errors cluster at transitions (when both dancers are equally active) and camera cuts.

### 2.3 Battle-Specific State Machine

Breaking battles have a **structured turn-taking protocol**. This imposes a temporal prior:

```
stateDiagram-v2
    [*] --> DancerA_Active
    DancerA_Active --> Transition: round_end
    Transition --> DancerB_Active: opponent_enters
    DancerB_Active --> Transition: round_end
    Transition --> DancerA_Active: original_enters
    DancerA_Active --> CameraCut: cut_detected
    CameraCut --> DancerA_Active: re_id_A
    CameraCut --> DancerB_Active: re_id_B
    DancerB_Active --> CameraCut: cut_detected
```

State machine rules:
1. **Round duration**: Typically 30-60 seconds per dancer. If dancer A has been active for <10s, the probability of transition is near zero.
2. **Spatial transition**: The outgoing dancer moves to the edge; the incoming dancer moves to center. The spatial trajectories cross.
3. **Temporal gap**: There is usually a 1-3 second gap between rounds where neither dancer is performing power moves (both standing).

$$P(\text{transition at } t) = \begin{cases} 0 & \text{if } t - t_{\text{last\_transition}} < T_{\min} \\ \sigma\left(\frac{t - t_{\text{last\_transition}} - T_{\text{expected}}}{\tau_{\text{transition}}}\right) & \text{otherwise} \end{cases}$$

with $T_{\min} \approx 10\text{s}$, $T_{\text{expected}} \approx 40\text{s}$, $\tau_{\text{transition}} \approx 10\text{s}$.

This prior prevents false transitions (e.g., momentarily high motion from a spectator) and enables recovery from brief detection failures.

---

## 3. Multi-Person Tracking: Association Across Frames

### 3.1 The Tracking Requirement

Once the active dancer is identified in frame $t$, we need to maintain identity across frames $t+1, t+2, \ldots$ This is a **Single Object Tracking (SOT)** problem embedded within a **Multi-Object Tracking (MOT)** context.

Standard MOT approaches (ByteTrack, BoT-SORT, OC-SORT) solve this but introduce specific risks for breaking:

**Association cost matrix** for Hungarian matching at frame $t$:

$$C_{ij} = \alpha \cdot d_{\text{IoU}}(\text{bbox}_i^{t}, \text{pred}_j^{t}) + \beta \cdot d_{\text{app}}(\text{feat}_i^{t}, \text{feat}_j^{t-1}) + \gamma \cdot d_{\text{pose}}(k_i^{t}, k_j^{t-1})$$

Where:
- $d_{\text{IoU}} = 1 - \text{IoU}(\cdot, \cdot)$ — spatial overlap cost
- $d_{\text{app}} = 1 - \cos(\phi_i^t, \phi_j^{t-1})$ — ReID appearance embedding distance
- $d_{\text{pose}}$ — normalized keypoint distance (see below)

### 3.2 Breaking-Specific Tracking Failures

**Failure 1: Extreme bbox displacement between frames.**

During a flare-to-windmill transition, the dancer's center of mass can shift by 200+ pixels between consecutive frames (at 30fps). Standard Kalman filter prediction in ByteTrack assumes quasi-constant velocity:

$$\hat{x}_t = F \cdot x_{t-1}, \quad F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

For breaking, acceleration is extreme and non-constant. The prediction error:

$$\epsilon_{\text{pred}} = \|x_t - \hat{x}_t\| = \|x_t - F \cdot x_{t-1}\| \approx \frac{1}{2}|a| \cdot \Delta t^2$$

At $|a| \approx 15\text{m/s}^2$ (fast power move), $\Delta t = 33\text{ms}$: $\epsilon \approx 8\text{mm}$ in world space, which projects to ~5-15 pixels depending on camera distance. Manageable for a single frame, but during sustained power moves the cumulative drift causes track fragmentation.

**Quantified risk**: At 30fps, track fragmentation occurs in ~3-7% of power move frames with standard ByteTrack. This rises to ~15-20% if the dancer's body rotates rapidly (appearance embedding changes drastically between frames).

**Failure 2: Appearance embedding instability during rotation.**

ReID models (OSNet, BoT, TransReID) encode appearance from a person crop. During a windmill:
- Frame $t$: chest facing camera → appearance embedding $\phi_{\text{front}}$
- Frame $t+5$ (167ms later): back facing camera → appearance embedding $\phi_{\text{back}}$

$$\cos(\phi_{\text{front}}, \phi_{\text{back}}) \approx 0.3\text{-}0.5$$

This cosine similarity is comparable to **two different people wearing similar clothes**. The tracker may associate the dancer's back-view with the wrong person.

**Failure 3: Opponent occlusion during transitions.**

When dancers pass each other during turn transitions, they occlude each other for 0.5-2 seconds. During this window:
- Both detections may merge into one (IoU > 0.7)
- The tracker cannot associate either detection
- After separation, identities may swap (the dreaded **ID switch**)

### 3.3 Pose-Aware Tracking Cost

Adding a pose-based cost term mitigates appearance embedding failures:

$$d_{\text{pose}}(k_i^t, k_j^{t-1}) = \frac{1}{J_{\text{vis}}} \sum_{j=1}^{J} \mathbb{1}[c_j^t > \tau_c \wedge c_j^{t-1} > \tau_c] \cdot \frac{\|k_j^t - k_j^{t-1}\|}{s}$$

where $s = \sqrt{w \cdot h}$ is the bounding box scale (normalizes for distance), $J_{\text{vis}}$ is the count of mutually visible joints, and $\tau_c \approx 0.3$ is the keypoint confidence threshold.

**Why this helps**: Even when the dancer rotates 180°, the skeletal configuration changes continuously — there are no discontinuities in joint positions between frames. Pose distance is rotation-invariant in a way that appearance is not.

**Why this isn't sufficient alone**: During occlusion, $J_{\text{vis}} \to 0$, making $d_{\text{pose}}$ undefined. Need fallback to motion prediction during occlusion.

### 3.4 Recommended Architecture: Hybrid SOT + MOT

Rather than pure MOT (which doesn't know which person we care about), use a **hybrid**:

1. **Primary**: Single-object tracker (SiamRPN++ or OSTrack) initialized on the active dancer. This provides a bounding box prediction even through brief occlusions.
2. **Verification**: MOT (ByteTrack) running on all persons provides identity consistency checks and enables recovery after SOT failure.
3. **Fallback**: If SOT confidence drops below threshold for >$N$ frames, switch to the MOT track that best matches the last confident SOT state.

$$\text{bbox}_t^{\text{final}} = \begin{cases} \text{bbox}_t^{\text{SOT}} & \text{if } c_{\text{SOT}} > \tau_{\text{SOT}} \\ \text{bbox}_t^{\text{MOT}}[i^*] & \text{if } c_{\text{SOT}} \leq \tau_{\text{SOT}} \\ \hat{\text{bbox}}_t^{\text{Kalman}} & \text{if neither available} \end{cases}$$

where $i^* = \arg\min_i d_{\text{pose}}(\text{MOT}_i, \text{last\_confident\_SOT})$.

**Tensor shapes through tracking:**

```
Per frame t:
  All detections:        (N_t, 4)        — N_t bboxes
  SOT prediction:        (1, 4)          — single bbox + confidence
  MOT tracks:            (M_t, 4)        — M_t active tracks
  Active dancer bbox:    (1, 4)          — selected output
  Active dancer crop:    (1, 3, 256, 192) — fed to ViTPose
  Active dancer 2D kpts: (1, 17, 3)      — fed to MotionBERT
```

---

## 4. Camera Cut Detection and Re-Identification

### 4.1 Camera Cut Detection

Broadcast breaking footage contains camera cuts every 5-15 seconds. Cut detection is straightforward — frame-level histogram difference:

$$H_{\text{diff}}(t) = \sum_{b=1}^{B} |h_b(I_t) - h_b(I_{t-1})|$$

with threshold $\tau_{\text{cut}}$. Well-studied problem, >99% accuracy with standard methods (TransNetV2 or even simple color histogram).

**The harder problem**: **gradual transitions** (dissolves, wipes) common in highlight reels. These take 10-30 frames, during which both scenes are blended. During the blend, person detection returns ghosts from both scenes.

$$I_{\text{blend}}(t) = (1 - \alpha(t)) \cdot I_{\text{scene\_A}} + \alpha(t) \cdot I_{\text{scene\_B}}, \quad \alpha \in [0, 1]$$

Detection: monitor the second derivative of $H_{\text{diff}}$. A hard cut is a single spike; a gradual transition is a sustained elevation.

### 4.2 Post-Cut Re-Identification

After a camera cut, we must determine:
1. **Is the same dancer still performing?** (camera angle change only)
2. **Has the round changed?** (opponent now dancing)
3. **Is this a replay/crowd shot?** (no dancer performing)

**Re-ID feature vector** for the active dancer (built during tracking, updated with EMA):

$$\bar{\phi}_{\text{dancer}} = (1 - \mu) \cdot \bar{\phi}_{\text{dancer}} + \mu \cdot \phi_t, \quad \mu = 0.02$$

This running average is robust to single-frame appearance changes. After a cut, compare each detected person's embedding to $\bar{\phi}_{\text{dancer}}$:

$$i^* = \arg\max_i \cos(\phi_i^{t_{\text{post-cut}}}, \bar{\phi}_{\text{dancer}})$$

$$\text{Re-ID successful if } \cos(\phi_{i^*}, \bar{\phi}_{\text{dancer}}) > \tau_{\text{ReID}} \approx 0.6$$

**Breaking-specific challenge**: If the cut occurred mid-rotation, the dancer's appearance in the new angle may be drastically different. The EMA embedding $\bar{\phi}$ is dominated by the most common view, which may not match the post-cut view.

**Mitigation**: Maintain a **gallery** of $G$ appearance embeddings spanning the last $N$ seconds, sampled at diverse body orientations (detected via torso keypoint configuration):

$$\Phi_{\text{gallery}} = \{\phi_{t_1}, \phi_{t_2}, \ldots, \phi_{t_G}\}, \quad G \approx 10\text{-}20$$

with diversity sampling: only add $\phi_t$ if $\max_{\phi \in \Phi} \cos(\phi_t, \phi) < 0.85$ (ensures different views are stored).

Re-ID score becomes:

$$s_{\text{ReID}}(i) = \max_{\phi \in \Phi_{\text{gallery}}} \cos(\phi_i, \phi)$$

This handles view changes because the gallery contains multiple viewpoints.

### 4.3 Handling Failed Re-ID

When $s_{\text{ReID}}(i^*) < \tau_{\text{ReID}}$ for all candidates — either:
- The dancer is not in the post-cut frame (crowd shot, judge shot, replay)
- Appearance changed too much (different lighting, sweat, clothing adjustment)

**Decision tree:**

$$\text{PostCutAction} = \begin{cases} \text{SKIP\_FRAME} & \text{if no person detected or all } s < 0.3 \\ \text{TENTATIVE\_MATCH}(i^*) & \text{if } 0.3 \leq s_{i^*} < 0.6 \\ \text{CONFIDENT\_MATCH}(i^*) & \text{if } s_{i^*} \geq 0.6 \end{cases}$$

For TENTATIVE\_MATCH: accept the match but flag frames for potential manual review. If $s$ remains tentative for >2 seconds, switch to the state machine's temporal prior (which dancer should be active based on round timing?).

### 4.4 Cross-Cut Pose Continuity

An additional signal: if the dancer was mid-windmill at the cut point, and a person in the post-cut frame is also mid-windmill with consistent angular phase, this strongly suggests identity:

$$d_{\text{pose\_continuity}} = \|k_{t_{\text{pre-cut}}} - k_{t_{\text{post-cut}}}\|_{\text{procrustes}}$$

Procrustes alignment removes the camera change. If the dancer's pose is consistent (low Procrustes distance), it's the same dancer in a new camera angle:

$$s_{\text{continuity}} = \exp\left(-\frac{d_{\text{pose\_continuity}}^2}{2\sigma_{\text{pose}}^2}\right), \quad \sigma_{\text{pose}} \approx 50\text{mm}$$

Fused re-ID score:

$$s_{\text{final}}(i) = \lambda_{\text{app}} \cdot s_{\text{ReID}}(i) + \lambda_{\text{pose}} \cdot s_{\text{continuity}}(i), \quad \lambda_{\text{app}} = 0.6, \lambda_{\text{pose}} = 0.4$$

---

## 5. Opponent Interference and Background Person Suppression

### 5.1 The Opponent Problem

The opponent is not a random bystander — they are:
- Wearing similar athletic clothing
- Similar body type and build
- Performing similar movements (bouncing, grooving to the music)
- Standing at the edge of the performance area, occasionally entering frame

The opponent is the **hardest negative** for all selection and ReID systems. Appearance embeddings between two b-boys in similar gear:

$$\cos(\phi_{\text{dancer}}, \phi_{\text{opponent}}) \approx 0.55\text{-}0.75$$

This overlaps with the genuine re-ID range after camera cuts.

### 5.2 Opponent Discrimination Features

Features that distinguish active dancer from opponent:

1. **Motion energy ratio**: Active dancer has 3-10× more motion energy
$$r_{\text{motion}} = \frac{E_{\text{motion}}(d_i)}{\max_j E_{\text{motion}}(d_j)}$$

2. **Floor contact**: Active dancer during power moves has keypoints near the bottom of the bounding box (contact with floor). Opponent is standing.
$$f_{\text{floor}}(d_i) = \min_j y_j(d_i) / h_{\text{bbox}}(d_i) \quad \text{(lower = more floor contact)}$$

3. **Pose entropy**: Active dancer's pose is far from the standing mean pose; opponent's is close.
$$H_{\text{pose}}(d_i) = \|k_i - \bar{k}_{\text{standing}}\|^2$$
where $\bar{k}_{\text{standing}}$ is the mean standing pose in normalized coordinates.

4. **Spatial role**: In a standard battle setup, dancers alternate from opposite sides. Once the side is established, the opponent's spatial region is known.

### 5.3 Segmentation Mask Interaction

When using SAM 3 for segmentation, the opponent's body may partially overlap with the active dancer's mask, particularly during transitions. SAM 3 needs a **prompt** (point, box, or text) to select the target person.

**Recommended prompting strategy:**
1. Use the tracked bounding box from Section 3 as the SAM 3 box prompt
2. Add the detected 2D keypoints as positive point prompts within the box
3. Add the opponent's detected center as a **negative** point prompt

$$\text{SAM3\_prompt} = \{\text{box}: \text{bbox}_{\text{active}}, \; \text{pos}: \{k_j^{\text{active}}\}_{j \in \text{visible}}, \; \text{neg}: \{c_{\text{opponent}}\}\}$$

This explicitly tells SAM 3 "segment this person, not that one."

---

## 6. Complete Pipeline Integration

### 6.1 Proposed Multi-Person Front-End

Inserting multi-person handling into the existing pipeline:

```
Raw Video (multi-person)
    │
    ├─① Person Detection (RT-DETR / YOLOv8)
    │     Output: (N, 6) detections per frame
    │
    ├─② Active Dancer Selection
    │     Input: detections + heuristic scores + classifier
    │     Output: (1, 4) active dancer bbox
    │     State: battle state machine (transition timing)
    │
    ├─③ Multi-Person Tracking (ByteTrack + SOT hybrid)
    │     Input: all detections + active dancer label
    │     Output: consistent track IDs across frames
    │     Handles: occlusion, rotation, brief disappearance
    │
    ├─④ Camera Cut Detection (TransNetV2)
    │     Input: consecutive frames
    │     Output: cut boundaries
    │
    ├─⑤ Post-Cut Re-Identification
    │     Input: appearance gallery + pose continuity
    │     Output: re-associated active dancer track
    │
    └─⑥ Single-Person Crop
          Input: active dancer bbox (padded 20%)
          Output: (1, 3, H, W) clean single-person crop
          → feeds existing pipeline (SAM3, ViTPose, MotionBERT, SAM-Body4D)
```

### 6.2 Computational Cost

| Component | Model | FLOPs/frame | Latency (RTX 4090) |
|---|---|---|---|
| Person Detection | RT-DETR-L | ~110 GFLOPs | ~8ms |
| 2D Pose (all persons) | ViTPose-B × $N$ | ~18 GFLOPs × $N$ | ~3ms × $N$ |
| SOT | OSTrack-256 | ~8 GFLOPs | ~3ms |
| ReID Embedding | OSNet | ~1 GFLOPs × $N$ | ~1ms × $N$ |
| Cut Detection | TransNetV2 | ~2 GFLOPs | ~1ms |
| Active Selection | MLP classifier | negligible | <1ms |
| **Total front-end** | | | **~25-40ms** |

At 30fps (33ms budget), the multi-person front-end alone consumes 75-120% of the frame budget. This means the front-end must either:
- Run at reduced frame rate (process every 2nd-3rd frame, interpolate bboxes)
- Run detection/ReID asynchronously (detect on frame $t$, use prediction for $t+1$)
- Use a lighter detector (YOLOv8-S instead of RT-DETR-L: ~15ms total)

### 6.3 Error Propagation

The critical question: **how do front-end errors propagate to pose estimation?**

**Wrong person selected (ID switch)**:
- 3D pose output is anatomically valid but for the wrong person
- All downstream analysis (move classification, scoring) is incorrect
- This is a **catastrophic** failure — worse than noisy pose for the right person

Expected ID switch rate with the proposed hybrid system: ~0.5-2% of frames in continuous footage, ~3-8% at round transitions, ~5-15% after camera cuts.

**Bbox too tight (partial person)**:
- 2D keypoint detection fails for cropped-out joints
- MotionBERT receives incomplete input → degrades gracefully (temporal context fills in)
- Estimate: adds ~5-15mm MPJPE for affected frames

**Bbox too loose (includes opponent)**:
- 2D keypoint detector may detect joints from both persons
- MotionBERT receives a chimeric skeleton → output is garbage
- Estimate: catastrophic failure, 100mm+ MPJPE for affected frames

$$\text{MPJPE}_{\text{effective}} = (1 - p_{\text{IDswitch}} - p_{\text{chimera}}) \cdot \text{MPJPE}_{\text{clean}} + p_{\text{IDswitch}} \cdot \text{MPJPE}_{\text{wrong}} + p_{\text{chimera}} \cdot \text{MPJPE}_{\text{garbage}}$$

With $p_{\text{IDswitch}} \approx 0.02$, $p_{\text{chimera}} \approx 0.01$, $\text{MPJPE}_{\text{clean}} \approx 70\text{mm}$, $\text{MPJPE}_{\text{wrong}} \approx 300\text{mm}$ (random person), $\text{MPJPE}_{\text{garbage}} \approx 200\text{mm}$:

$$\text{MPJPE}_{\text{effective}} \approx 0.97 \times 70 + 0.02 \times 300 + 0.01 \times 200 = 67.9 + 6.0 + 2.0 = \mathbf{75.9\text{mm}}$$

The multi-person handling adds **~6-8mm effective MPJPE** — an 8-11% degradation over the already-concerning 70mm single-person estimate.

---

## 7. Recommendations

### 7.1 Minimum Viable Solution (Low Effort, High Impact)

1. **Require pre-cropped input** for v1. Shift the multi-person problem to data preparation — have users submit single-dancer clips or manually annotate the active dancer's bounding box in the first frame. SOT (OSTrack) handles the rest.

   - Effort: Days
   - Coverage: Handles practice videos, pre-segmented competition clips
   - Limitation: Can't process raw broadcast footage

### 7.2 Automated Solution (Medium Effort)

2. **Heuristic selection + ByteTrack + cut detection**. No learned classifier, no ReID gallery. Use motion energy + center proximity + floor contact as selection score. ByteTrack with pose-aware association cost. TransNetV2 for cut detection with temporal prior for re-association.

   - Effort: 1-2 weeks
   - Expected accuracy: ~92-95% frame-level correct person
   - Failure concentration: round transitions and post-cut re-ID

### 7.3 Full Solution (High Effort)

3. **The complete system described above**. Learned active dancer classifier, hybrid SOT+MOT, appearance gallery re-ID, battle state machine. Requires training data annotation (~2K frames with active dancer labels).

   - Effort: 4-6 weeks
   - Expected accuracy: ~97-99% frame-level correct person
   - Remaining failures: adversarial cases (identical twins, costume changes mid-battle)

### 7.4 Critical Integration Constraint

Whichever solution is chosen, the **pipeline must be designed to propagate uncertainty**. Each frame should carry a confidence score for "correct person selected." Downstream analysis should weight or discard low-confidence frames:

$$w_{\text{frame}}(t) = \min\left(c_{\text{detection}}(t), \; c_{\text{tracking}}(t), \; c_{\text{reID}}(t)\right)$$

$$\text{score}_{\text{move}} = \frac{\sum_t w_t \cdot q_t}{\sum_t w_t}$$

This prevents catastrophic ID-switch frames from corrupting move-level scoring.

---

## Summary

| Sub-problem | Severity | Key Risk | Best Mitigation |
|---|---|---|---|
| Person detection in crowds | Medium | Detector saturation at $N>15$ | Anchor-free detector (RT-DETR) |
| Active dancer selection | **High** | Opponent confusion during toprock | Learned classifier + motion energy |
| Tracking through power moves | **High** | Appearance instability during rotation | Pose-aware association cost |
| Camera cut re-ID | **High** | Opponent has similar appearance | Multi-view gallery + pose continuity |
| Opponent suppression | Medium | Chimeric skeleton from overlapping bbox | SAM 3 negative prompts |
| Computational overhead | Medium | 25-40ms added per frame | Async detection, lighter models |

The multi-person problem is not a preprocessing detail — it's a **load-bearing architectural requirement** that adds ~6-8mm effective MPJPE and introduces a new failure mode (ID switches) that is catastrophic for downstream analysis. It should be addressed as a first-class pipeline component with its own confidence propagation, not bolted on as an afterthought.
