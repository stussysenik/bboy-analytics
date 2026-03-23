# cotracker3 — Breakdance Adaptation

_Generated: 2026-03-23T04:23:01.104Z_

---

I'll produce the full scenario-by-scenario analysis of CoTracker3 for breakdancing, building on the research context provided.

---

# CoTracker3 Breakdancing Scenario Analysis

## Scenario 1: Headspin

**Works out of box:**
- **Temporal window coverage**: CoTracker3 processes 24-frame sliding windows (~0.8s at 30fps). A headspin lasting 3–5s spans 90–150 frames — multiple windows, each individually tractable. The online mode handles arbitrary-length sequences by chaining windows.
- **Group correlation mechanism**: CoTracker3's joint attention across all tracked points means torso points (which stay relatively stable during axial rotation) anchor the extremity tracks. During the stable rotation phase, the periodic motion is actually *easier* than random motion — the model can exploit the repeating displacement pattern.
- **Pseudo-label training coverage**: HACS includes gymnastics/acrobatics with some inverted poses. Not headspins specifically, but the feature correlation patterns for rotating bodies have partial coverage.

**Fails:**
- **Extremity motion blur**: At 30fps with 180° shutter, hands during headspin whip at ~2700 px/s → motion blur σ ≈ 6.5px, nearly 10× the feature localization floor (0.7px). Per the research: $\Delta p_{\text{hand/foot whip}} > 20$ px/frame — exceeds the correlation search radius ($S=4$, effective 16px). **Hands are untrackable at 30fps during fast headspins.**
- **Visibility oscillation**: Per the occlusion model, $\text{occlusion fraction} \approx 0.5 + 0.15\cos(2\omega t)$, meaning 35–65% of body points are occluded at any instant. The visibility predictor has **30–40% false negative rate on re-appearance** — so when a limb rotates back into view, there's a ~35% chance the model thinks it's still occluded, creating artificial gaps.
- **Exponential track death**: $P_{\text{survive}}(n_{\text{rot}}) \approx \exp(-0.4 \cdot n_{\text{rot}})$. After 5 rotations: ~13% of extremity points survive. After 10: ~2%. The headspin body is essentially re-initialized every 3–4 rotations.
- **Cycle-consistency filter bias**: The pseudo-label pipeline filters out fast motion more aggressively (round-trip error ∝ $\|\mathbf{v}\|^2$). Training data is therefore biased *against* exactly this scenario.

**Modifications needed:**
1. **Rotation-aware re-initialization** (~200 LOC, medium): Detect periodic visibility patterns (FFT on per-point visibility signals). When periodicity detected, predict re-emergence timing and pre-initialize tracks at expected re-appearance locations. Uses the rotation period $T_{\text{rot}}$ estimated from visible torso points.
2. **Hierarchical point density** (~100 LOC, easy): Allocate ~100 points on head/neck (rotation axis, high survival), ~50 on torso, ~200/limb. Total ~1200 points. Head points anchor the rotation model.
3. **60fps minimum capture** (~0 LOC, operational): At 60fps, $\Delta p_{\text{hand}} \approx 10$ px/frame — within search radius. Motion blur σ drops to 3.5px. This is the single highest-impact change.

**Integration output:**
Head/neck points provide clean velocity (near-zero, rotation axis) → movement spectrogram shows characteristic "donut" pattern: low velocity at center (head), radially increasing velocity. The velocity magnitude profile:

$$S_m(j, t) \propto r_j \cdot \omega$$

where $r_j$ is the distance of joint $j$ from the rotation axis and $\omega$ is angular velocity. Velocity derivatives are clean for torso (SNR ~5:1) but unreliable for extremities (SNR < 1:1 at 30fps). At 60fps, extremity SNR improves to ~2:1.

---

## Scenario 2: Windmill

**Works out of box:**
- **Temporal continuity**: Windmill rotation period 0.8–1.2s = 24–36 frames at 30fps. Each full rotation fits within ~1.5 CoTracker3 windows, so the model sees enough context to track through the transition.
- **Ground contact points**: The shoulder/back making floor contact creates high-contrast, low-motion anchor points that are excellent for correlation-based tracking. These points have $\Delta p < 3$ px/frame — well within search radius.
- **Bilateral symmetry exploitation**: CoTracker3's group attention across all points can implicitly learn that left-side and right-side points alternate in visibility. When left arm disappears, right arm appears — the group correlation captures this complementary pattern.

**Fails:**
- **Left-right alternation creates identity swaps**: This is the killer. During a windmill, the body flips between left-side-down and right-side-down every half-rotation. The left hip and right hip occupy *nearly identical* image positions at half-rotation intervals. CoTracker3 tracks based on local feature correlation — but a left hip and right hip in the same position, wearing the same fabric, have nearly identical feature descriptors. **Identity swap probability per half-rotation: ~15–25%.**
- **Visibility oscillation**: Per the research, $N_{\text{visible}}(t) \approx N \cdot (0.5 + 0.3\cos(2\pi t / T_{\text{rot}}))$. At minimum visibility, ~500 of 2500 points visible; ~2000 hallucinated. The hallucinated positions are plausible but may belong to the wrong anatomical side.
- **Ground-level camera typical in battles**: Camera at floor level means the dancer's body fully occludes the far side during each rotation. No partial visibility — a binary visible/invisible transition.
- **Extremity whip**: Legs during windmill sweep at ~2000 px/s. At 30fps: $\Delta p_{\text{leg}} \approx 11$ px/frame — borderline for search radius. At the acceleration phase of each rotation, this spikes to ~15 px/frame.

**Modifications needed:**
1. **Anatomical identity constraint** (~300 LOC, hard): Post-process tracks with a biomechanical model that enforces bone-length consistency. If a "left knee" track suddenly has the bone length of a right knee, flag as identity swap and correct. Requires skeleton topology as prior.
2. **SAM-Body4D mesh projection as oracle** (~150 LOC, medium): Project 3D mesh vertices from SAM-Body4D back to 2D. Use these projections to disambiguate left/right identity. This is the recommended approach from the research — it converts the 2D tracking problem into a 3D-informed correction.
3. **Confidence-weighted derivatives** (per research §2): $\frac{\partial p_n}{\partial t}\big|_{\text{weighted}} = \frac{\partial p_n}{\partial t} \cdot \sigma(v_{n,t}) \cdot c_{n,t}$. Down-weight velocity contributions from points with low visibility confidence during the high-occlusion phase (~50 LOC, easy).

**Integration output:**
Windmill produces a characteristic sinusoidal velocity envelope in the movement spectrogram — each joint traces a periodic path. The fundamental frequency $f_0 = 1/T_{\text{rot}} \approx 0.8\text{–}1.25$ Hz should appear as a clear spectral peak. **However**, identity swaps create discontinuities that manifest as spurious high-frequency energy in the spectrogram. A single left/right swap injects a velocity spike of magnitude $\approx 2 \cdot v_{\text{joint}}$ — which corrupts the acceleration and jerk channels.

Expected derivative quality:
| Derivative | Torso | Limbs (no swap) | Limbs (with swap) |
|-----------|-------|-----------------|-------------------|
| Velocity SNR | ~8:1 | ~3:1 | **< 1:1** (corrupted) |
| Acceleration SNR | ~4:1 | ~1.5:1 | Meaningless |

The SAM-Body4D correction path is essential for windmill analysis.

---

## Scenario 3: Flare

**Works out of box:**
- **Hip as stable anchor**: During flares, the hips stay roughly centered and in contact with (or near) the ground. Hip points have $\Delta p \approx 3\text{–}5$ px/frame — trackable. These serve as the rotation center for the leg arcs.
- **Leg arcs are large but smooth**: Legs sweep in wide circles, which means the per-frame displacement is distributed across many frames. At the apex of the arc, legs move slower (changing direction). CoTracker3's temporal attention captures this deceleration-acceleration pattern.
- **Partial training coverage**: Gymnastics floor exercise in HACS includes leg circles on pommel horse — mechanically similar to flares. This provides some transfer.

**Fails:**
- **Extreme hip articulation creates feature distortion**: During flares, the hip joint undergoes near-maximal range of motion (>170° abduction). The visual appearance of the hip region changes dramatically frame-to-frame — clothing stretches, wrinkles shift, skin folds appear/disappear. Feature descriptors at the hip become unreliable because the local appearance is non-stationary.
- **Leg crossing occlusion**: When legs cross (once per half-rotation), both legs overlap in the image for 2–4 frames. CoTracker3 cannot distinguish which leg is which during the crossing — same identity swap problem as windmill but concentrated at the crossing point.
- **Speed at leg extremities**: Feet during flares move at ~2200 px/s. At 30fps: $\Delta p_{\text{foot}} \approx 12$ px/frame — right at the edge of the search radius. Tracking will intermittently fail at the fastest phase of each arc, creating periodic dropouts.
- **Self-occlusion of arms**: Arms are planted on the ground supporting body weight, often hidden beneath the torso. Arm tracks will die within the first rotation and not recover.

**Modifications needed:**
1. **Leg-specific high-density tracking** (~80 LOC, easy): Allocate 300+ points per leg (especially ankle-to-foot), with only 30 on arms (they're occluded anyway). Increases the probability that some leg points survive the fast phase.
2. **Arc-aware motion model** (~250 LOC, hard): Fit a circular arc model to leg trajectories. When tracking fails during the fast phase, extrapolate along the fitted arc rather than relying on correlation. Reset when the leg decelerates and tracking recovers.
3. **60fps capture** (operational): Reduces $\Delta p_{\text{foot}}$ to ~6 px/frame — comfortably within search radius. Single most effective mitigation.

**Integration output:**
Flares produce a distinctive movement spectrogram signature: two counter-rotating sinusoids (one per leg) at the flare frequency (~1–1.5 Hz), with the hip as a near-DC baseline. The cross-correlation with audio is interesting because flares are often performed *against* the beat (continuous circular motion rather than hit-aligned accents) — the musicality score should show low beat-alignment but high energy consistency:

$$\mu_{\text{flare}} = \text{corr}(M(t), \bar{H}) \quad \text{(energy matches average audio level, not beat structure)}$$

Derivative quality: velocity is clean for hips (SNR ~6:1). Leg velocity has periodic dropouts at the fast phase — interpolatable if arc model is applied. Acceleration is usable only with the arc model correction.

---

## Scenario 4: Freeze

**Works out of box:**
- **Static pose = trivial tracking**: Once the freeze is held, all points have $\Delta p \approx 0$ px/frame. CoTracker3 excels here — the correlation between consecutive frames is near-perfect. Tracking confidence should be >95% for all visible points.
- **Long temporal window helps**: The 24-frame window at 30fps covers 0.8s. Most freezes are held 0.5–3s. The model sees a long stretch of near-zero motion, which is easy to fit.
- **Visibility is stable**: Once in a freeze, the occlusion pattern is constant — no new occlusions appear, no hidden points re-emerge.

**Fails:**
- **The ENTRY is the hard part**: The velocity collapse from power move to freeze happens in 80–150ms (2–5 frames at 30fps). During those frames, there's simultaneous: (a) rapid deceleration of all points, (b) large pose change as body reconfigures, (c) possible inversion transition (e.g., from windmill into baby freeze). The entry window has the *worst* tracking conditions — high velocity + high pose change + high occlusion change simultaneously.
- **Inverted freeze poses are out-of-distribution**: A baby freeze or airchair has the body in an orientation rarely seen in training data. While CoTracker3 doesn't explicitly model pose (it tracks appearance), the visual features of an inverted human are less frequently encountered in pseudo-labels, potentially reducing matching quality.
- **Micro-adjustments during hold**: A held freeze isn't truly static — the dancer makes constant micro-adjustments for balance. These are sub-pixel movements (~0.1–0.5 px/frame) that are below CoTracker3's noise floor ($\sigma_{\text{feat}} = 0.7$ px). The movement spectrogram during a freeze should be near-zero, but tracking noise will dominate the signal.

**Modifications needed:**
1. **Freeze detection + derivative clamping** (~100 LOC, easy): When global velocity drops below threshold for >10 frames, declare "freeze state" and clamp all derivatives to zero. This prevents tracking noise from corrupting the spectrogram during holds. The threshold: $\bar{v} < 2\sigma_{\text{feat}} / \Delta t \approx 42$ px/s.
2. **Entry keyframe re-initialization** (~80 LOC, easy): Detect the velocity collapse (acceleration magnitude spike followed by near-zero velocity). Re-initialize all tracks at the first stable frame of the freeze. This gives clean tracks for the hold phase even if the entry corrupted some tracks.
3. **No modification needed for the hold itself** — CoTracker3 handles static scenes natively.

**Integration output:**
The freeze signature in the movement spectrogram is a sharp velocity cliff:

$$S_m(j, t_{\text{entry}}) \gg 0, \quad S_m(j, t > t_{\text{freeze}}) \approx 0$$

The *sharpness* of this transition is a quality metric in breaking judging — a clean freeze "hits" the beat precisely. The derivative at the entry point:

$$\dot{S}_m(j, t_{\text{entry}}) = \text{acceleration at freeze onset}$$

This acceleration magnitude correlates with freeze "power" and should align with a musical accent for high musicality scores. At 30fps, the 80ms entry spans ~2.4 frames — barely resolvable. At 60fps (4.8 frames), the entry shape becomes visible. The CWT-based jerk detection (scale $a = 3$, center freq 2.5 Hz) should detect freeze entries with JSNR ~17.4.

---

## Scenario 5: Footwork

**Works out of box:**
- **Similar to training data**: Footwork involves a crouching person making rapid limb movements — visually similar to many actions in HACS and YouTube-VOS (crawling, floor exercises). Expected AJ: 65–70, near DAVIS baseline.
- **No full inversion**: The body stays in a recognizable orientation (crouched, not inverted). Feature descriptors remain in-distribution.
- **Ground anchoring**: Hands and/or feet are always in contact with the ground, providing stable reference points. At least 2–3 contact points are visible at all times with near-zero displacement.
- **Torso tracking is reliable**: The torso is generally visible and moves slowly during footwork (~200–400 px/s). Tracking confidence should be high for all torso points.

**Fails:**
- **Dense limb crossings**: During 6-step, CC, and other footwork patterns, legs constantly cross over each other. At ground level, this creates rapid occlusion changes — a leg goes from fully visible to fully occluded in 1–2 frames. The visibility predictor's 30% false negative rate on re-appearance means ~30% of limb tracks are lost at each crossing.
- **Rapid direction changes**: Footwork involves quick changes in leg direction (kick-outs, hooks, sweeps). The acceleration at direction changes peaks at ~5000 px/s² — producing jerk events in the 4–8 Hz range. At 30fps with SG window 5, the -3dB cutoff is 9.5 Hz — these are *just barely* capturable.
- **Clothing occlusion at ground level**: Baggy pants (common in breaking) pool on the ground during footwork, creating large low-texture regions that defeat feature correlation. The actual leg position may be 10–20 pixels away from the visible clothing boundary.
- **Small displacements between crossings**: Between explosive direction changes, footwork often has small precise placements. These 2–5 px/frame movements are near the noise floor for lower-body points ($\sigma_r \approx 3\text{–}5$ px for torso, worse for extremities).

**Modifications needed:**
1. **High-density foot tracking** (~60 LOC, easy): 150+ points per foot, especially on shoe edges and distinctive features (laces, toe cap). The redundancy ensures some points survive crossings.
2. **Crossing-aware stitching** (~150 LOC, medium): Detect crossing events from the visibility signal (rapid drop + recovery for paired limb points). After crossing, use feature-space nearest-neighbor to reassign tracks rather than trusting the visibility predictor.
3. **Ground-plane constraint** (~100 LOC, medium): All contact points must lie on the ground plane. Enforce this as a hard constraint in post-processing — any tracked point that violates the ground plane by > 5px during a known contact phase is corrected.

**Integration output:**
Footwork produces the richest movement spectrogram of all scenarios — rapid, varied velocity patterns with strong beat alignment. The spectrogram should show:

- **6-step rhythm**: Clear periodicity at the round frequency (~1–2 Hz fundamental)
- **Accents**: Velocity spikes at kick-outs and directional changes, aligned with musical accents
- **Ground pattern**: Spatial trajectory of foot contacts traces the characteristic 6-step circle

Velocity SNR is good for upper body (~5:1) and moderate for legs (~2:1 between crossings, dropping during crossings). The musicality cross-correlation should work well here — footwork is inherently rhythmic, and the velocity accents should correlate strongly with the audio beat:

$$\mu_{\text{footwork}} = \max_\tau \text{corr}(M(t), H(t-\tau)) \quad \text{expected } \mu > 0.6 \text{ for skilled dancers}$$

---

## Scenario 6: Toprock (Control Case)

**Works out of box:**
- **Upright pose — fully in-distribution**: Toprock is a standing person dancing. This is the most common scenario in CoTracker3's training data. Expected AJ: ~67–70, matching or exceeding DAVIS benchmarks.
- **Moderate velocities**: Arm swings ~800 px/s, leg steps ~500 px/s, torso ~200 px/s. All within $\Delta p < 10$ px/frame at 30fps — comfortably within search radius.
- **Minimal occlusion**: The upright body has natural visibility of all limbs. Self-occlusion occurs briefly during arm crossings but is transient (1–2 frames).
- **Beat-aligned motion**: Toprock is inherently rhythmic with clear velocity accents on musical beats — exactly the signal the musicality scoring needs.
- **All body regions trackable**: Per the survival table, head/neck at 90% after 1 rotation, upper torso 80% — but toprock has no full rotations, so survival is even higher. Expected 5s survival: >85% for all body regions.

**Fails:**
- **Minimal failures expected**. The only notable issue:
- **Arm swing blur at 30fps**: During aggressive cross-body arm swings, hand velocity can spike to ~1500 px/s → $\Delta p \approx 8$ px/frame. Not a failure, but tracking precision degrades for 2–3 frames per swing. Produces small velocity artifacts.
- **Shoulder isolation**: Some toprock styles include sharp shoulder pops. These are 30–50ms transients — below Nyquist at 30fps. The pop appears as a single high-velocity frame rather than a resolved transient.

**Modifications needed:**
- **None required for tracking**. Toprock works out of the box.
- For **musicality scoring**: shoulder pops need the CWT-based event detection (scale $a = 1.5$, center freq 5 Hz) rather than SG-differentiated jerk. Without CWT, pops are indistinguishable from tracking noise (JSNR ~1.2 via SG at 30fps).

**Integration output:**
Toprock is the **gold standard** for the movement spectrogram pipeline. Expected outputs:

| Derivative | SNR | Quality |
|-----------|-----|---------|
| Velocity | >10:1 | Excellent |
| Acceleration | ~6:1 | Good |
| Jerk (SG) | ~2:1 | Marginal |
| Jerk (CWT) | ~5:1 | Good |

The musicality cross-correlation should produce the highest scores here because toprock is explicitly designed to hit the beat:

$$\mu_{\text{toprock}} = \max_\tau \text{corr}(M(t), H(t-\tau)) \quad \text{expected } \mu > 0.7 \text{ for skilled dancers}$$

The time lag $\tau$ should be near-zero for skilled dancers (hitting beats on time) and measurably positive/negative for dancers who are behind/ahead of the beat.

---

## Scenario 7: Battle (Multi-Person)

**Works out of box:**
- **CoTracker3 tracks points, not people**: CoTracker3 doesn't have a concept of "person" — it tracks individual points. If points are initialized on Dancer A, they will follow Dancer A regardless of Dancer B's presence, *as long as there's no occlusion*. In the non-overlapping phases (one dancer performing, the other standing), tracking is equivalent to single-person performance.
- **SAM 3 segmentation upstream**: The pipeline runs SAM 3 segmentation ("breakdancer") before CoTracker3. This isolates the active dancer's pixels, so CoTracker3 only receives points on the performing dancer. This is the **primary defense** against cross-person confusion.
- **Online mode handles arbitrary duration**: Battles are 2–5 minutes. CoTracker3's online mode processes streaming frames — no temporal limit.

**Fails:**
- **Cross-person occlusion during transitions**: When dancers transition (toprock face-off, cipher entry/exit), bodies overlap for 1–3 seconds. SAM 3 must maintain identity through the overlap — if segmentation fails, CoTracker3 receives points from the wrong dancer.
- **Crowd edge intrusion**: In a battle circle, spectators lean in, wave arms, or move into frame. SAM 3 may segment spectator limbs as part of the "breakdancer" prompt response, injecting spurious points into CoTracker3.
- **Camera angle shifts**: Battle footage often involves handheld cameras with sudden pans, zooms, and angle changes. Camera motion creates apparent displacement of all points simultaneously — CoTracker3 handles this via global correlation, but rapid zoom changes the effective resolution, changing $\sigma_{\text{track}}$.
- **Active dancer identification**: The pipeline needs to know *which* dancer is currently performing to attribute the movement spectrogram correctly. CoTracker3 doesn't solve this — it's an upstream problem.
- **Re-initialization between rounds**: When dancers swap, all tracks must be re-initialized on the new active dancer. If the transition isn't cleanly detected, tracks from the previous dancer may persist.

**Modifications needed:**
1. **SAM 3 identity-aware segmentation** (~200 LOC, medium): Use SAM 3 with two prompts: "dancer in blue" and "dancer in red" (or other distinguishing features). Maintain separate segmentation masks throughout the battle. Fall back to spatial prior (center of ring = active dancer) when appearance is ambiguous.
2. **Active dancer detection** (~150 LOC, medium): Classify which dancer is performing using motion magnitude — the active dancer has significantly higher total velocity than the standing dancer. Threshold: $\sum_j \|v_j\|_{\text{active}} > 5 \times \sum_j \|v_j\|_{\text{inactive}}$.
3. **Camera motion compensation** (~100 LOC, easy): Track 10–20 background points (on the floor or static objects). Subtract their median displacement from all dancer points to isolate dancer-relative motion. Essential for handheld battle footage.
4. **Crowd filtering** (~80 LOC, easy): Any tracked point that moves inconsistently with the dancer's body (e.g., moves opposite to all other points, or has velocity uncorrelated with neighboring points) is flagged as crowd intrusion and removed.
5. **Round boundary detection** (~120 LOC, medium): Detect round transitions via global velocity profile — both dancers have near-zero velocity during transitions. Use this to trigger re-initialization.

**Integration output:**
The battle scenario requires the movement spectrogram to be **dancer-attributed** — each spectrogram segment tagged with the performing dancer's identity:

$$S_m^{(d)}(j, t) \quad \text{where } d \in \{A, B\}$$

The musicality score becomes per-dancer:

$$\mu^{(d)} = \max_\tau \text{corr}(M^{(d)}(t), H(t-\tau))$$

Comparative scoring (who hit the beat better in each round) requires clean round boundaries and correct dancer attribution. Camera motion compensation is critical — without it, a camera pan looks like dancer motion in the spectrogram, inflating the inactive dancer's apparent energy.

Expected tracking quality during active performance (after upstream segmentation): equivalent to the relevant single-person scenario (toprock, footwork, power moves). The battle layer adds **attribution risk** but doesn't degrade per-point tracking quality.

---

## Integration with Movement Spectrogram: Derivative Quality Analysis

### Position Output Characteristics

CoTracker3 outputs per-point trajectories $\mathbf{p}_n(t) \in \mathbb{R}^2$ with:
- Position noise: $\sigma_{\text{track}} \approx 2\text{px (slow motion)}, 7\text{px (fast motion at 30fps)}$
- Confidence score: $c_{n,t} \in [0,1]$
- Visibility flag: $v_{n,t} \in \{0,1\}$ (with 30–40% false negative on re-appearance)

These are **2D image-space** trajectories. For the 3D movement spectrogram $S_m(j,t) = \|\dot{p}_j(t)\|$, two paths exist:

**Path A (2D derivatives, then lift):** Compute velocity in image space, scale by depth estimate from SAM-Body4D. Faster but couples depth noise with velocity noise.

**Path B (Lift to 3D, then differentiate):** Project CoTracker3 points onto SAM-Body4D mesh, extract 3D joint positions, differentiate in 3D. Preferred — decouples tracking noise from depth estimation noise.

### Preprocessing Pipeline

```
Raw CoTracker3 tracks → Visibility filter → Confidence weighting
→ Anatomical assignment (nearest mesh vertex) → 3D lifting via SAM-Body4D
→ Per-joint aggregation (median of assigned points) → SG smoothing
→ Velocity / Acceleration → CWT jerk events → Movement spectrogram
```

### Expected Derivative SNR by Scenario

| Scenario | Velocity SNR | Accel SNR | Jerk (CWT) JSNR | Spectrogram Usability |
|----------|-------------|-----------|-----------------|----------------------|
| Toprock (30fps) | 10:1 | 6:1 | 5:1 | Excellent |
| Footwork (30fps) | 5:1 (torso), 2:1 (legs) | 2.5:1 | 2:1 | Good (legs marginal) |
| Freeze hold (30fps) | N/A (clamped to 0) | N/A | 17:1 at entry | Excellent (with clamping) |
| Flare (30fps) | 6:1 (hips), 1.5:1 (feet) | 1:1 (feet) | 1.5:1 | Poor for legs without 60fps |
| Windmill (30fps) | 3:1 (w/o swap), <1:1 (w/ swap) | <1:1 | <1:1 | Requires SAM-Body4D correction |
| Headspin (30fps) | 8:1 (head), <1:1 (hands) | <1:1 (extremities) | <1:1 | Head only; extremities need 60fps |
| Battle (30fps) | Per-scenario + attribution noise | — | — | Depends on segmentation quality |

### Required Smoothing Parameters

Per the corrected analysis (SG window 7 order 4 was **wrong** — destroys hits):

| Derivative | Method | Parameters (30fps) | Parameters (60fps) |
|-----------|--------|-------------------|-------------------|
| Velocity | SG | $M$=3, $p$=3, $d$=1, $f_{-3\text{dB}}$≈11Hz | $M$=5, $p$=3, $d$=1, $f_{-3\text{dB}}$≈14Hz |
| Acceleration | SG | $M$=2, $p$=3, $d$=2, $f_{-3\text{dB}}$≈9Hz | $M$=3, $p$=3, $d$=2, $f_{-3\text{dB}}$≈12Hz |
| Jerk events | CWT | Scales {1.5, 2, 3, 4, 6} | Scales {2, 3, 5, 7, 10} |

### Critical Insight: The Three-Tier Architecture is Essential

The movement spectrogram should NOT be a single uniform representation. It should be three layers:

1. **Velocity field** (continuous, SG-smoothed): Reliable across all scenarios. This is the primary input to musicality cross-correlation.
2. **Acceleration field** (continuous, SG-smoothed): Reliable for toprock/footwork/freeze-entry. Marginal for power moves at 30fps. Useful for technique assessment.
3. **Jerk events** (sparse, CWT-detected): Timestamps + magnitudes of transient events. This is how hits, pops, and freeze entries appear in the spectrogram. The musicality score should use:

$$S_{\text{musicality}} = \alpha \cdot \text{corr}(M_{\text{velocity}}(t), H(t)) + (1-\alpha) \cdot \frac{1}{|\mathcal{E}|}\sum_k m_k \cdot \exp\left(-\frac{(\Delta t_k)^2}{2\sigma_{\text{sync}}^2}\right)$$

where the first term captures continuous energy alignment (are you moving with the music?) and the second captures discrete event alignment (are your hits landing on the beat?).

### Bottom Line

CoTracker3 produces position output that is **clean enough for velocity-based analysis** across all scenarios, **marginal for acceleration** in power moves at 30fps, and **insufficient for continuous jerk** at any frame rate. The architecture must:

1. Use **confidence-weighted, visibility-filtered** tracks (not raw output)
2. Lift to **3D via SAM-Body4D** before differentiating
3. Apply the **three-tier derivative architecture** (velocity continuous, acceleration continuous, jerk sparse events)
4. Capture at **60fps minimum** for any scenario involving power moves
5. Use **SAM-Body4D mesh projection** to correct identity swaps in windmill/flare
