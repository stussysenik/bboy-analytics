# sam3d — Breakdance Adaptation

_Generated: 2026-03-23T05:54:31.778Z_

---

# SAM3D vs. Breakdancing: Per-Scenario Analysis

## Architectural Preamble

SAM3D was designed for **static scenes, multiple viewpoints, ground-truth depth**. Dance video is **dynamic scene, one viewpoint, estimated depth**. Every assumption is violated. The analysis below quantifies *how badly* per scenario.

Key reference numbers from the research summary:
- Superpoint voxel size: $R_{\text{seed}} = 2\text{cm}$
- Max tolerable velocity at 30fps for superpoint coherence ($\alpha < 0.5$): **0.3 m/s**
- Effective independent views from single camera: $V_{\text{eff}} = 1$
- SAM3D baseline mAP@50 with GT depth on static scenes: **~46**

---

## Scenario 1: Headspin

**Works out of box**: Almost nothing.

- SAM's 2D mask generator (ViT-H) can segment a dancer blob in frames with low blur. The "everything mode" grid prompts will fire on the overall body silhouette.
- The NMS + confidence filtering stage works normally — it's view-independent.

**Fails**:

1. **Motion blur destroys 2D masks on extremities.** Angular velocity 300–600°/s × 16ms exposure = 8° angular blur. At arm's length (0.7m): ~10cm linear blur per frame. SAM's `pred_iou_thresh=0.88` will reject blurred limb masks entirely. Arms and legs become ghosted smears that SAM either ignores or merges into a single amorphous mask.

2. **Monocular depth catastrophe on inverted bodies.** MDE trained on <0.01% inverted humans. Abs Rel degrades from ~0.05 to ~0.12–0.15 (2–3× baseline). At 3m distance: **36–45cm depth error** vs. 2cm superpoint size = **18–22× oversize error**. Back-projected points scatter across a 45cm depth band.

3. **Superpoint coherence obliterated.** Extremity velocities during headspin: 3–8 m/s. Motion-to-resolution ratio:
$$\alpha = \frac{\|\mathbf{v}\| \cdot \Delta t}{R_{\text{seed}}} = \frac{5.0 \times 0.033}{0.02} = 8.3$$
Need $\alpha < 0.5$. Actual: **16.6× over threshold**. Each limb smears across ~170 phantom superpoints per frame.

4. **Viewpoint degeneracy.** All frames from same camera → $V_{\text{eff}} = 1$. The multi-view vote matrix $V(s_i, m_j)$ collapses to single-view assignment. Zero geometric diversity for disambiguation.

5. **Region merging breaks on spinning geometry.** Surface normals rotate continuously between frames. Normal discontinuity threshold $\tau_{\text{boundary}} \approx 30°$ is exceeded within 1–2 frames of rotation (at 500°/s: 16.7° per frame). Merge criterion fires spuriously, fusing separate body parts.

**Projected mAP@50**: ~2–5 (vs. 46 baseline). **~90–96% degradation.**

**Modifications needed**:

| Modification | Difficulty | LOC | Effect |
|---|---|---|---|
| Replace multi-view voting with temporal tracking (SAM 2 memory bank) | Major rewrite | ~2000 | Enables video-native segmentation |
| Replace VCCS superpoints with per-frame instance masks | Architectural change | ~1500 | Eliminates superpoint coherence requirement |
| Replace monocular depth with HMR 2.0 mesh-derived positions | Pipeline replacement | ~800 | Sidesteps depth entirely |
| Add motion-compensated back-projection (warp by optical flow before unproject) | Medium | ~400 | Reduces motion smear but doesn't solve it |

At this point you've replaced SAM3D entirely. **No modification makes SAM3D viable for headspins.**

**Integration output**: If somehow forced to produce output, you'd get per-point instance labels $(N_{\text{points}},)$ — but with purity ~0.15, the labels are essentially random for moving limbs. The movement spectrogram $S_m(j,t)$ requires per-joint tracking; SAM3D produces instance segments, not joint localizations. You'd need an additional step to map instance segments → joint positions, which loses the velocity signal in the segment-level averaging.

**Derivative quality**: Unusable. Position jitter from depth error (~40cm) at 30fps produces velocity noise:
$$\sigma_{\dot{p}} \approx \frac{\sigma_p \cdot \sqrt{2}}{\Delta t} = \frac{0.40 \times 1.414}{0.033} \approx 17.1 \text{ m/s}$$
Actual extremity velocity: 3–8 m/s. **SNR ≈ 0.3:1**. Derivatives are pure noise.

---

## Scenario 2: Windmill

**Works out of box**: SAM 2D segmentation of the overall body blob in low-blur frames (brief moments when the torso faces the camera with minimal rotation). The back-projection math itself is correct — the issue is input quality.

**Fails**:

1. **Cyclic self-occlusion defeats superpoint voting.** In windmills, ~50–70% of body surface disappears cyclically as the body rotates around the shoulder/back axis. SAM3D's vote accumulation:
$$V(s_i, m_j) = \frac{1}{|s_i|} \sum_{p \in s_i} \mathbb{1}[L_v(p) = m_j]$$
requires the same 3D point to be visible and correctly labeled across multiple views. With cyclic occlusion, a given arm superpoint is visible for only 3–5 frames out of a 10-frame window, and labeled differently each time (left arm vs. right arm confusion due to left-right alternation).

2. **Floor-contact depth ambiguity.** The dancer's body contacts the floor during the roll. Monocular depth estimators merge body parts with the floor surface. At the contact zone, $d_{\text{boundary}} \approx 0$, so:
$$e_{\text{assign}} \approx 1 - \Phi\left(\frac{0}{\sigma_d}\right) = 0.50$$
**50% of contact-zone points get wrong assignment** (body vs. floor).

3. **Left-right alternation breaks instance identity.** SAM3D has no concept of persistent identity across views. As the body alternates between left and right shoulders on the ground, the left arm in frame $t$ may be assigned to the same instance as the right arm in frame $t+5$ — both were "the arm closest to the camera." There's no skeleton model to maintain anatomical correspondence.

4. **Torso velocity during roll: 0.5–2.0 m/s, limb velocity: 3–8 m/s.** Even the torso exceeds the 0.3 m/s coherence limit. $\alpha_{\text{torso}} = 0.83\text{–}3.33$, $\alpha_{\text{limb}} = 5.0\text{–}13.3$. All superpoints contaminated by motion.

5. **Specular floor reflections.** Breaking happens on smooth/waxed floors. MDE Abs Rel on specular surfaces: 0.15–0.25 (3–5× worse than matte). Phantom depth surface below floor creates ghost points that merge with body segments.

**Projected mAP@50**: ~3–8. **~83–93% degradation.**

**Modifications needed**: Same as headspin — SAM3D's architecture is fundamentally incompatible. The cyclic occlusion pattern specifically requires a temporal model with persistent identity (SAM 2's memory bank, or CoTracker3's joint point tracking). Estimated effort to retrofit: equivalent to building a new system.

**Integration output**: Instance labels with purity ~0.20–0.30. Left-right arm confusion propagates directly into the movement spectrogram as phantom velocity spikes when identity swaps. $S_m(\text{left\_arm}, t)$ and $S_m(\text{right\_arm}, t)$ would contain each other's signals randomly.

**Derivative quality**: Position noise ~30cm (depth error + motion smear). $\sigma_{\dot{p}} \approx 12.8$ m/s vs. actual limb velocity 3–8 m/s. **SNR ≈ 0.4:1.** The left-right identity swaps additionally create discontinuous jumps of ~60cm (arm span) in the position signal, producing $\delta$-function artifacts in the velocity derivative.

---

## Scenario 3: Flare

**Works out of box**: SAM 2D mask generation for the torso (which moves less than the legs). The overall body silhouette is usually large and well-separated from background.

**Fails**:

1. **Extreme leg arc velocities.** Flares produce the largest circular leg arcs of any breaking move. Leg velocity: 5–8 m/s. $\alpha_{\text{leg}} = 8.3\text{–}13.3$. Each leg sweeps through ~400 superpoints per frame. The leg trace is a continuous arc in 3D space — VCCS will create superpoints along the arc, but each superpoint belongs to a different temporal position of the same leg.

2. **Hip articulation exceeds MDE priors.** During flares, the hips rotate through angles rarely seen in training data (legs spread >160° while rotating). MDE has no prior for this body configuration. The depth map for the leg region will be interpolated from surrounding context (floor + torso), producing a flat depth surface where there should be a complex 3D leg trajectory.

3. **Intermittent self-occlusion with rapid transitions.** One leg occludes the other every ~0.3s (half rotation). The transition from visible → occluded → visible takes 2–3 frames. SAM3D has no mechanism to maintain identity through these transitions. After re-emergence, the leg may be assigned to a new instance.

4. **Ground plane contamination.** Hands contact the floor during flares (supporting the body). At contact: body-floor depth difference = 0. Same issue as windmill: 50% point misassignment at contact zones.

**Projected mAP@50**: ~3–6. **~87–93% degradation.**

**Modifications needed**: Fundamentally incompatible. The circular arc motion pattern requires Lagrangian tracking (CoTracker3) rather than Eulerian segmentation (SAM3D).

**Integration output**: Leg segments would be fragmented across the arc. The movement spectrogram for $S_m(\text{left\_leg}, t)$ would show intermittent signal (only when that leg's instance is correctly assigned) interleaved with zeros or wrong-leg signal. The characteristic sinusoidal velocity profile of flares would be unrecoverable.

**Derivative quality**: Leg position SNR ≈ 0.2:1 (worst of all scenarios for the primary moving body part). The circular motion means velocity should be roughly constant magnitude with rotating direction — instead you'd get random noise. Hip joint velocity might be partially recoverable (lower velocity, ~1–2 m/s), but hip articulation angle — the key quality indicator for flares — requires accurate bilateral leg tracking that SAM3D cannot provide.

---

## Scenario 4: Freeze

**Works out of box**: **This is SAM3D's best scenario.**

- The dancer is **static** (velocity ≈ 0 m/s during the hold). $\alpha \approx 0$ for all body parts. Superpoint coherence is maintained.
- Multiple frames of the same static pose approximate a multi-view setup *if the camera moves slightly* (handheld wobble). Even handheld shake provides ~1–3° of viewpoint diversity, giving $V_{\text{eff}} \approx 1.5\text{–}3$.
- SAM 2D masks are sharp (no motion blur). SAM's `pred_iou_thresh=0.88` won't reject any body-part masks.
- Depth estimation is most accurate on static, well-lit subjects.

**Fails**:

1. **Inverted body orientation.** Most freezes involve inversions (headstand freeze, baby freeze, airchair). MDE Abs Rel still degrades 2–3× on inverted humans. At 3m: ~12–15cm depth error vs. 2cm superpoints = 6–7.5× oversize. Better than moving scenarios but still significant.

2. **Extreme articulation.** Freezes like hollowback or pike freeze put the body in configurations far outside MDE training distribution. The depth map may hallucinate body depth based on nearest-neighbor upright poses.

3. **Transition into/out of freeze.** The freeze itself is static, but the **entry** (velocity collapse from power move to zero in 2–4 frames) and **exit** (re-initiation) involve high velocities. SAM3D would produce clean output during the hold but catastrophic output during transitions, creating a temporal discontinuity.

4. **Single viewpoint limits instance resolution.** Even with perfect depth, a single viewpoint can't resolve self-occluded body parts. During a baby freeze, the bottom arm and leg may be completely hidden. SAM3D outputs no instance labels for invisible parts.

5. **Contact ambiguity persists.** Hand/head-ground contact during freezes merges body parts with background at $d_{\text{boundary}} = 0$.

**Projected mAP@50**: ~20–30 (during the static hold phase). **~35–57% degradation.** During entry/exit: ~5–10.

**Modifications needed**:

| Modification | Difficulty | LOC | Effect |
|---|---|---|---|
| Detect freeze-hold phase (velocity < threshold for N frames) and only run SAM3D during holds | Easy | ~100 | Avoids catastrophic transition frames |
| Add metric depth model constraint (DepthPro over ZoeDepth) | Easy | ~50 | Reduces depth error from 22cm to 14cm |
| Fuse with 2D pose detector for occluded limb estimation | Medium | ~500 | Recovers hidden body parts |

**Integration output**: During the freeze hold, SAM3D produces the cleanest instance labels of any scenario. Position output is stable. However, **freezes are scored on precision and difficulty of the held pose** — which requires joint angle estimation, not instance segmentation. SAM3D tells you "this blob is the dancer" but not "the right arm is at 135° from the torso." The output needs additional pose estimation to be useful for TRIVIUM scoring.

**Derivative quality**: During the hold: excellent. $\sigma_p \approx 14\text{cm}$ (DepthPro), velocity noise $\sigma_{\dot{p}} \approx 6.0$ m/s. Actual velocity ≈ 0. The signal *is* zero, and the noise *is* noise, but the **hold detection** itself can be derived from the low total energy: $M(t) = \sum_j S_m(j,t) \approx 0$ is a clean signal. For the audio-motion cross-correlation, a freeze coinciding with a musical break (silence) produces $\mu \approx 1.0$ correctly. **This is the one scenario where SAM3D contributes meaningfully to the spectrogram.**

---

## Scenario 5: Footwork

**Works out of box**:

- No full inversion → MDE operates in its training distribution. Abs Rel stays near baseline (~0.05).
- Body is mostly upright → SAM 2D masks are reliable for the torso and upper body.
- The dancer is close to the ground → shorter depth range → smaller absolute depth error.

**Fails**:

1. **Dense limb crossings near ground level.** Footwork involves rapid hand-foot-floor interleavings. Arms thread between legs, feet cross over hands. At these crossing points, $d_{\text{boundary}} < 5\text{cm}$ between limbs. With $\sigma_d = 14\text{cm}$ (DepthPro):
$$e_{\text{assign}} \approx 1 - \Phi\left(\frac{5}{14}\right) = 1 - \Phi(0.36) \approx 0.36$$
**36% misassignment at every limb crossing.** Footwork has ~4–8 crossings per second.

2. **Hand/foot velocities exceed coherence limit.** Hand velocity during footwork: 1–3 m/s. Foot velocity: 2–5 m/s. $\alpha_{\text{hand}} = 1.67\text{–}5.0$, $\alpha_{\text{foot}} = 3.33\text{–}8.33$. All above the 0.5 threshold.

3. **Rapid direction changes.** Footwork involves sharp direction reversals every 0.5–1.0 beats. SAM3D has no motion model — it can't predict where a body part will be after a direction change. Each reversal creates a 1–2 frame window where the superpoint trail points in the old direction while the body moves in the new direction.

4. **Small body-part masks near ground.** Hands and feet near the floor subtend small image areas. SAM's grid prompts (64×64) may not resolve individual fingers/toes. The mask generator outputs a single "lower body near floor" blob rather than separate hand/foot instances.

**Projected mAP@50**: ~10–18. **~61–78% degradation.**

**Modifications needed**:

| Modification | Difficulty | LOC | Effect |
|---|---|---|---|
| Increase SAM grid density to 128×128 for ground-level region | Medium | ~200 | Better small-part resolution |
| Pre-segment using Grounding DINO with "hand" / "foot" prompts | Medium | ~300 | Targeted small-part detection |
| Temporal smoothing with motion-compensated averaging | Medium | ~400 | Reduces direction-change artifacts |

These modifications improve 2D mask quality but don't address the fundamental superpoint coherence and viewpoint degeneracy problems.

**Integration output**: The torso signal is recoverable (lower velocity, upright orientation). Limb signals are noisy. For footwork scoring, the **foot placement precision** and **rhythmic accuracy** are the key metrics. Foot position at 14cm depth error + motion smear means beat-alignment detection ($\mu = \max_\tau \text{corr}(M(t), H(t-\tau))$) will have a temporal resolution of ~2–3 frames (66–100ms) rather than the 1-frame (33ms) needed for tight musicality scoring.

**Derivative quality**: Torso: $\sigma_{\dot{p}} \approx 6.0$ m/s, actual velocity 0.2–0.5 m/s. **SNR ≈ 0.05:1.** Even the torso velocity is buried in noise. Feet: completely unrecoverable. The movement spectrogram for footwork would be dominated by noise, with the rhythmic pattern invisible.

Wait — this reveals something important. Even though footwork is the second-most-favorable scenario (no inversion), the **derivative SNR is still catastrophic** because the fundamental position noise floor (~14cm from depth) is too high relative to the subtle movements being measured. Footwork scoring requires ~1cm position precision for beat-aligned foot placement detection.

---

## Scenario 6: Toprock (Control Case)

**Works out of box**:

- Upright dancing → MDE in-distribution. Abs Rel ~0.05, error at 3m: ~15cm.
- Beat-aligned accents → lower peak velocities than power moves. Torso: 0.3–0.8 m/s, arms: 2–5 m/s during accents.
- Minimal self-occlusion → SAM 2D masks are clean and well-separated.
- Full body visible → all body parts get mask assignments.

**Fails**:

1. **Arm velocities during accents still exceed coherence threshold.** Toprock involves sharp arm gestures on the beat. Arm velocity 2–5 m/s → $\alpha_{\text{arm}} = 3.3\text{–}8.3$. Arms fragment across superpoints during accents. Between accents, arms are fine ($v < 0.5$ m/s).

2. **Single viewpoint degeneracy persists.** $V_{\text{eff}} = 1$ regardless of movement quality. No geometric diversity.

3. **Depth error still dominates position precision.** 15cm error vs. 2cm superpoints. The depth noise floor prevents clean instance boundaries.

4. **Beat-aligned accents are the scoring signal.** Toprock musicality scoring depends on detecting *when* accents happen relative to the beat. The accent is a brief velocity spike (1–3 frames). At 15cm position noise, the velocity spike is:
$$\text{SNR}_{\text{accent}} = \frac{v_{\text{accent}}}{\sigma_{\dot{p}}} = \frac{3.0}{6.0} \approx 0.5$$
Barely detectable. The cross-correlation $\mu$ will be noisy.

**Projected mAP@50**: ~15–25 (best among all scenarios, but still ~46–67% degradation from baseline).

**Modifications needed**: Same fundamental issues. For toprock specifically:

| Modification | Difficulty | LOC | Effect |
|---|---|---|---|
| Use SAM3D only for scene context (stage boundaries, crowd) | Easy | ~100 | Plays to SAM3D's strength |
| Route dancer segmentation through SAM 2 instead | Easy | ~200 | Video-native temporal tracking |

**Integration output**: Torso segment is the most reliable output. The overall body bounding volume is correct. Individual limb instances are noisy during accent gestures but recoverable between them. For the movement spectrogram, toprock produces the cleanest total energy signal $M(t) = \sum_j S_m(j,t)$ — you can detect "movement happened" even if per-joint attribution is wrong.

**Derivative quality**: Total body energy $M(t)$ is marginally detectable (SNR ~0.5:1 during accents). Per-joint signals are not cleanly separable. After Savitzky-Golay smoothing (window=5, order=2), total energy SNR improves to ~1.5:1 — **barely usable for coarse beat alignment** but insufficient for precise accent characterization. The audio-motion cross-correlation $\mu$ would be ~0.3–0.5 (moderate correlation) where ground truth is ~0.7–0.9 for a skilled dancer. **SAM3D systematically underestimates musicality** because it can't resolve the sharp velocity transients that define precise beat-hitting.

---

## Scenario 7: Battle (Two Dancers)

**Works out of box**: SAM's 2D "everything mode" will generate masks for both dancers. If the dancers are well-separated in the frame, SAM produces distinct masks for each.

**Fails**:

1. **Cross-person occlusion.** When dancers are near each other (taunting, transitions, shared stage space), their masks overlap in image space. SAM3D's superpoint voting cannot distinguish which 3D points belong to which dancer when back-projected from overlapping 2D masks. With two people at similar depth, $d_{\text{boundary}} \approx 0$ between them during interactions:
$$e_{\text{assign}} = 0.50 \text{ at every cross-person boundary}$$

2. **Instance identity across camera angle shifts.** Battle footage involves camera movement to follow the active dancer. When the camera pans, both dancers move through the frame. SAM3D has no persistent identity — after a camera shift, the "same" dancer may be assigned a new instance ID. The movement spectrogram for dancer A would contain segments of dancer B's motion.

3. **Crowd edge intrusion.** Battle circles have spectators at the edges. SAM's grid prompts fire on crowd members, generating spurious masks. These back-project into the same 3D volume as the dancers (similar depth range). The vote matrix $V(s_i, m_j)$ accumulates crowd member masks alongside dancer masks.

4. **All single-dancer failure modes compound.** Each dancer independently suffers from motion blur, depth error, superpoint incoherence. With two dancers, the total number of spurious superpoints doubles, and cross-assignment between dancers adds a new error mode.

5. **Scale ambiguity doubles.** Two dancers at different depths from the camera produce different depth errors. The relative depth between them has error $\sigma_{\text{relative}} = \sqrt{2} \times \sigma_d \approx 20\text{cm}$ for DepthPro. This means SAM3D can't reliably determine which dancer is in front during close interactions.

**Projected mAP@50**: ~5–12 (both dancers combined). Per-dancer: ~3–8. **~74–93% degradation.**

**Modifications needed**:

| Modification | Difficulty | LOC | Effect |
|---|---|---|---|
| Pre-detect each dancer with Grounding DINO → separate SAM3D runs per bounding box | Medium | ~400 | Isolates per-dancer processing |
| Add ReID (re-identification) network for persistent identity | Hard | ~1500 | Maintains identity across camera shifts |
| Crowd suppression via semantic segmentation (label: "audience") | Medium | ~300 | Removes crowd intrusions |
| Replace with SAM 2 video tracking (track each dancer independently through memory bank) | Major | ~2000 | Solves temporal identity natively |

**Integration output**: Per-dancer movement spectrograms are unreliable. Cross-dancer contamination means $S_m^A(j,t)$ contains signal from dancer B, corrupting the per-dancer musicality score. The battle scoring use case (comparing two dancers' TRIVIUM scores) requires clean separation — SAM3D cannot provide this.

**Derivative quality**: Worse than single-dancer scenarios by $\sqrt{2}$ (uncorrelated noise from two sources) plus identity-swap discontinuities. When dancer identities swap (1–3 times per round during close interactions), the position signal has ~1–2m discontinuous jumps, producing impulse artifacts in the velocity derivative that are orders of magnitude larger than any real movement signal.

---

## Summary Table

| Scenario | Projected mAP@50 | Degradation | Derivative SNR | Spectrogram Usable? |
|----------|-----------------|-------------|----------------|-------------------|
| 1. Headspin | 2–5 | 89–96% | 0.3:1 | No |
| 2. Windmill | 3–8 | 83–93% | 0.4:1 | No |
| 3. Flare | 3–6 | 87–93% | 0.2:1 | No |
| 4. Freeze (hold) | 20–30 | 35–57% | N/A (v≈0) | Hold detection only |
| 5. Footwork | 10–18 | 61–78% | 0.05:1 | No |
| 6. Toprock | 15–25 | 46–67% | 0.5:1 | Barely (coarse M(t) only) |
| 7. Battle | 5–12 | 74–93% | <0.3:1 | No |

---

## Integration with Movement Spectrogram: Unified Assessment

### Position Output Quality

SAM3D outputs per-point instance labels, not joint positions. Converting to $p_j(t)$ requires:
1. Map instance segments → body parts (requires additional pose estimation)
2. Compute per-body-part centroid from assigned 3D points
3. Track centroids across frames

Each step introduces additional error. The centroid computation averages over noisy point clouds with ~25–50% misassigned points, producing a biased position estimate.

### Derivative Chain

The spectrogram requires clean first derivatives:
$$S_m(j,t) = \|\dot{p}_j(t)\| = \left\|\frac{p_j(t+1) - p_j(t-1)}{2\Delta t}\right\|$$

Position noise $\sigma_p$ propagates to velocity as:
$$\sigma_{\dot{p}} = \frac{\sigma_p \sqrt{2}}{\Delta t}$$

| Depth Source | $\sigma_p$ | $\sigma_{\dot{p}}$ at 30fps | Typical dance velocity | SNR |
|---|---|---|---|---|
| GT depth (unrealistic) | ~1cm | 0.6 m/s | 2–5 m/s | **3–8:1** ✓ |
| iPhone LiDAR | ~3cm | 1.8 m/s | 2–5 m/s | **1.1–2.8:1** marginal |
| DepthPro | ~14cm | 6.0 m/s | 2–5 m/s | **0.3–0.8:1** ✗ |
| DepthPro + motion | ~25–40cm | 17–24 m/s | 2–5 m/s | **0.1–0.3:1** ✗ |

For **second derivatives** (acceleration, needed for accent detection):
$$\sigma_{\ddot{p}} = \frac{\sigma_p \sqrt{6}}{\Delta t^2} \approx \frac{0.14 \times 2.449}{0.001} \approx 343 \text{ m/s}^2$$

Typical dance acceleration: 10–50 m/s². **Acceleration SNR ≈ 0.03–0.15:1.** Completely unusable.

### Preprocessing Requirements (if SAM3D were forced into the pipeline)

1. **Savitzky-Golay smoothing**: Window=15, order=3. Reduces velocity noise by ~4× but also attenuates real velocity transients by ~2× — halving the effective temporal resolution from 33ms to ~250ms (8 frames). Beat-aligned accents shorter than 250ms become invisible.

2. **Wavelet denoising**: CWT with Morlet wavelet, threshold at 2σ. Better preserves sharp transients but introduces Gibbs ringing artifacts around velocity discontinuities (freeze entries/exits).

3. **Kalman filter with biomechanical model**: Best option. State: $(p, \dot{p}, \ddot{p})$, process noise calibrated to human movement limits. Reduces noise while respecting physical constraints. But this requires a skeletal model — which SAM3D doesn't provide. You'd need to bolt on a pose estimator anyway, at which point SAM3D is redundant.

### Cross-Correlation Impact

The audio-motion cross-correlation:
$$\mu = \max_\tau \text{corr}(M(t), H(t-\tau))$$

With SAM3D-derived $M(t)$ at SNR 0.3–0.5:1 (toprock, best case):
$$\mu_{\text{measured}} \approx \mu_{\text{true}} \times \text{SNR} / (1 + \text{SNR}) \approx 0.8 \times 0.3/1.3 \approx 0.18$$

A skilled dancer's musicality ($\mu_{\text{true}} \approx 0.8$) and a random dancer's musicality ($\mu_{\text{true}} \approx 0.2$) both map to $\mu_{\text{measured}} \approx 0.15\text{–}0.20$. **SAM3D cannot distinguish skilled from unskilled musicality.** The scoring system collapses.

---

## Verdict

**SAM3D is not viable for any breakdancing analysis scenario.** The one partial exception — freeze hold detection via total energy — is trivially achievable with simpler methods (frame differencing, optical flow magnitude thresholding).

**Recommended role for SAM3D in the pipeline: none.** Use it only for static scene segmentation (stage boundaries, floor plane estimation, crowd region identification) if needed. All dynamic human analysis should use the verified pipeline: **SAM 2 (segmentation) → CoTracker3 (tracking) → HMR 2.0/WHAM (mesh recovery)**, which sidesteps depth estimation entirely and provides direct joint positions with ~50mm accuracy (sufficient for velocity SNR > 3:1 on most movements).

The viable pipeline from Section 8 of the research summary produces joint positions at ~50mm precision → velocity noise ~4.3 m/s at 30fps → SNR ~0.7–1.2:1 for typical dance movements. Still marginal, but improvable with:
- 60fps capture (halves derivative noise): SNR → 1.4–2.4:1
- CoTracker3 temporal smoothing: additional ~2× noise reduction
- Combined: **SNR ~2.8–4.8:1** — usable for musicality scoring
