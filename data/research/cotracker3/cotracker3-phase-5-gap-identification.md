# cotracker3 — Gap Identification

_Generated: 2026-03-23T04:44:10.254Z_

---

# CoTracker3 Reimplementation Gap Analysis

## 1. Architectural Gaps

### 1.1 Pseudo-Label Training Pipeline Misunderstood

**What's missing**: The research frames Kubric MOVi-F as a confirmed primary training source ("Synthetic: Kubric MOVi-F (confirmed)"). This inverts CoTracker3's core contribution. The paper's key finding is that a student model trained **only on pseudo-labeled real videos** matches or exceeds models trained on synthetic data. Kubric is used **indirectly** — to train the teacher that generates pseudo-labels — not as direct training data for the final model.

**Why it matters**: This misunderstanding propagates into the domain adaptation analysis. The research suggests "adding ~500 breakdancing clips with pseudo-labels" for fine-tuning, but doesn't grapple with the bootstrapping problem: pseudo-labels are only as good as the teacher, and the teacher already struggles on breakdancing. Fine-tuning the student on low-quality pseudo-labels for power moves could degrade rather than improve performance.

**Suggested resolution**: The domain adaptation path should be: (1) manually annotate a small set (~50 clips) of breakdancing tracks as ground truth, (2) use these to evaluate and filter teacher pseudo-labels on a larger breakdancing corpus, (3) only include pseudo-labels that pass a breakdancing-specific quality threshold (lower cycle-consistency threshold for slow moves, higher for power moves — the opposite of what the current filter does).

---

### 1.2 Sliding Window Boundary Artifacts

**What's missing**: CoTracker3 processes videos in overlapping sliding windows (typically T=8 or similar). The research discusses memory budgets and frame counts but never addresses what happens at **window boundaries**. Trajectories are stitched across windows, and this stitching can introduce discontinuities — exactly where the pipeline computes derivatives.

**Why it matters**: A velocity spike at a window boundary is indistinguishable from a dancer's "hit." At 30fps with T=8, window boundaries occur every ~0.27s — right in the bandwidth of freeze entries (80–150ms) and power move initiations (100–200ms). The three-tier derivative architecture would detect these as jerk events.

**Suggested resolution**: (1) Use CoTracker3's offline mode for analysis (bidirectional attention eliminates some boundary issues), (2) compute window boundary timestamps and exclude ±2 frames from jerk event detection, (3) use overlapping windows with majority-vote on trajectories in overlap regions.

---

### 1.3 Online vs. Offline Mode Performance Delta Not Quantified

**What's missing**: The research mentions "online real-time mode" but doesn't distinguish performance between online (causal) and offline (bidirectional) modes. Online mode typically drops 3–5 AJ points because it can't look ahead. For breakdancing, where forward context helps predict occluded re-emergences, this gap may be larger.

**Why it matters**: The pipeline latency budget (~330ms/frame) assumes offline mode. If real-time coaching (v2) requires online mode, the tracking quality degradation compounds with the already-poor power move tracking (predicted AJ 35–45 could drop to 30–40).

**Suggested resolution**: Benchmark both modes on proxy data and establish the actual delta. Plan for offline-only in v0.1 and treat the online degradation as a known v2 risk.

---

### 1.4 Virtual Tracks Mechanism Absent

**What's missing**: CoTracker3 uses "support" or "virtual" tracks — additional points tracked internally to provide spatial context even if they're not in the query set. The research doesn't mention this mechanism. The number and placement of virtual tracks affects tracking quality, especially in sparse regions.

**Why it matters**: The "hierarchical tracking density" strategy (Section 8A) allocates 1200 strategic points. But CoTracker3's own virtual tracks may duplicate or conflict with this strategy. Understanding how virtual tracks interact with user-specified points is necessary to avoid wasting compute on redundant tracking.

**Suggested resolution**: Read the CoTracker3 codebase to understand virtual track initialization. Test whether the 1200-point hierarchical strategy provides additive benefit over CoTracker3's default virtual track mechanism, or whether a simpler approach (fewer user points + default virtual tracks) performs comparably.

---

### 1.5 Point Query Temporal Initialization

**What's missing**: CoTracker3 queries are (t, x, y) tuples — you specify **when** to start tracking each point. The research discusses spatial initialization (SAM 3 masks, body-part allocation) but never addresses **temporal** initialization. Starting all 1200 points at frame 0 means many will drift or die before the interesting motion begins.

**Why it matters**: For a 30-second battle round at 60fps (1800 frames), points initialized at frame 0 on the dancer's back will be lost long before the first windmill. The point survival analysis (Section 2) models decay from initialization — but doesn't propose staggered re-initialization as an architectural choice.

**Suggested resolution**: Use rolling initialization: re-sample points from SAM 3 masks every N frames (e.g., every 30 frames = 0.5s). Maintain a target density per body region. New points inherit the body-part label from the mask region. This converts the "track lifetime" problem from a single exponential decay to a renewal process.

---

## 2. Math Errors

### 2.1 Visibility Oscillation Assumes Fixed Camera-Axis Geometry

**What's missing**: The formula $N_{\text{visible}}(t) \approx N \cdot (0.5 + 0.3\cos(2\pi t / T_{\text{rot}}))$ uses a fixed amplitude of 0.3. This is not derived — it's asserted. The actual amplitude depends on the angle between the camera viewing direction and the rotation axis.

For a windmill filmed from the battle circle edge (camera at ~1.5m height, ~3m distance, dancer on floor):
- Camera elevation angle: ~27°
- Rotation axis: roughly vertical (dancer's spine)
- Visibility oscillation amplitude: $A \approx 0.5 \cdot |\sin(\theta_{\text{camera}})| \approx 0.5 \times 0.45 \approx 0.23$

For a headspin filmed from above (common drone/overhead angle):
- Camera nearly aligned with rotation axis
- Visibility oscillation amplitude drops to ~0.1 (most points always visible or always occluded)

**Why it matters**: The formula drives the "at maximum self-occlusion: ~500/2500 visible" claim, which drives the "~2000 must be hallucinated" claim, which drives the failure mode analysis. If the amplitude varies 3× based on camera angle, the entire failure severity assessment shifts.

**Suggested resolution**: Replace the single formula with a camera-angle-parameterized version:

$$N_{\text{visible}}(t, \theta) \approx N \cdot \left(0.5 + A(\theta) \cos\left(\frac{2\pi t}{T_{\text{rot}}}\right)\right), \quad A(\theta) = 0.5 |\sin(\theta)|$$

And evaluate at the three common angles: floor-level (θ≈10°, A≈0.09), standing-in-circle (θ≈27°, A≈0.23), overhead (θ≈70°, A≈0.47).

---

### 2.2 Tracking Error Model Missing Appearance Change Term

**What's missing**: The tracking error model $\sigma_{\text{track}}^2(\Delta t) = \sigma_{\text{feat}}^2 + (v \cdot \Delta t / 6.93)^2 + (\alpha \cdot v \cdot \Delta t)^2$ has three terms: feature localization, motion blur, and search error. But **appearance change due to viewpoint rotation** is absent. During a windmill, the same body point cycles through drastically different appearances (front of shirt → side → back → side) every rotation.

The correlation-based matching in CoTracker3 degrades as appearance changes. This should be modeled as:

$$\sigma_{\text{appear}}^2(\Delta\phi) \approx \beta^2 \cdot (1 - \cos(\Delta\phi))$$

where $\Delta\phi$ is the angular change in viewpoint. For $\omega = 6.28$ rad/s at 30fps, $\Delta\phi = 0.21$ rad/frame, giving $\sigma_{\text{appear}} \approx \beta \cdot 0.15$ per frame. This compounds across frames.

**Why it matters**: The search error coefficient $\alpha = 0.02$ is too low for rotational motion precisely because it doesn't account for appearance change. The actual effective $\alpha$ during power moves could be 0.05–0.10, which changes the 30fps tracking error from 6.80px to 10–15px — significantly worse than reported.

**Suggested resolution**: Add the appearance term to the error model and re-derive the fps recommendations. This may push the minimum recommended fps from 60 to 90+ for power moves.

---

### 2.3 Musicality Score Formula Has Unit/Normalization Issues

**What's missing**: The musicality formula:

$$S_{\text{musicality}}(n) = \frac{1}{|\mathcal{E}_n|} \sum_{(t_k, a_k, m_k) \in \mathcal{E}_n} m_k \cdot \exp\left(-\frac{\min_j |t_k - m_j|^2}{2\sigma_{\text{sync}}^2}\right)$$

has $m_k$ (jerk magnitude) in the sum. Jerk magnitudes vary by orders of magnitude across body parts (hand whip = 10× torso jerk). This means a single large hand movement dominates the score regardless of how many other body parts hit the beat.

**Why it matters**: A dancer who precisely hits every beat with subtle full-body accents would score lower than one who flails one hand near random beats — the accidental near-beat hand flail has higher $m_k$.

**Suggested resolution**: Normalize $m_k$ per body region: $\hat{m}_k = m_k / \sigma_{m, \text{region}(k)}$. Or use $\text{sign}(m_k)$ (binary: event happened or not) and weight by the number of body parts that simultaneously fire, which better captures the "full-body hit" aesthetic that judges reward.

---

### 2.4 The "16px Search Radius" Analysis Conflates Single-Iteration and Multi-Iteration Tracking

**What's missing**: The claim that "correlation radius S=3–4 at stride-4 gives effective search of 16px" and that "hand/foot tips during fast motion exceed this" treats the search radius as a hard boundary. But CoTracker3 uses **iterative refinement** (4–6 iterations). Each iteration re-centers the correlation search around the previous estimate. A point moving 20px/frame can be tracked if each iteration recovers ~5px of the displacement.

**Why it matters**: The per-move tracking quality table rates power moves at D/D+ partially based on the search radius argument. If iterative refinement effectively extends the capture range to ~3–4× the single-iteration radius, fast extremity motion may be trackable with AJ 5–10 points higher than predicted.

**Suggested resolution**: Run an empirical test with synthetic motion exceeding the single-iteration search radius to measure actual capture range across iterations. Update the per-move predictions accordingly.

---

## 3. Implementation Risks

### 3.1 Clothing Tracking ≠ Body Tracking

**What's missing**: CoTracker3 tracks **visual surfaces**, not body surfaces. The research consistently refers to "body surface points" and "body region" tracking. But a point initialized on a dancer's baggy pant leg tracks the fabric surface, which moves independently of the underlying knee. During windmills, clothing can lag, flap, and oscillate separately from the body.

**Why it matters**: This is architecturally fundamental. The movement spectrogram needs body-part velocities, not fabric velocities. A pant leg flapping during a freeze creates spurious high-frequency jerk events — exactly the signature the pipeline uses to detect hits/pops.

**Suggested resolution**: Two-layer tracking: (1) CoTracker3 on visual surface, (2) SAM-Body4D mesh vertices projected to 2D as "body truth." Use CoTracker3 points only where they agree with mesh projections within a tolerance. Where they diverge, trust the mesh. The research mentions this in Section 8D but frames it as optional — it should be **mandatory** for any clothed dancer.

---

### 3.2 SAM 3 Mask Flicker Propagates to Track Death

**What's missing**: The pipeline assumes SAM 3 reliably segments the dancer every frame. But segmentation models flicker — a frame where the mask drops the dancer's arm means CoTracker3 has no mask to constrain that region. If using masks to filter/weight tracks, a single-frame segmentation failure can kill tracks that were perfectly good.

**Why it matters**: SAM 3 is the first stage. Any failure propagates through the entire pipeline. During fast motion (exactly when tracking matters most), segmentation is least reliable.

**Suggested resolution**: Decouple segmentation from tracking temporally. Use SAM 3 every N frames (e.g., every 10th frame) to generate "anchor" masks. Between anchors, use CoTracker3 tracks to propagate the mask. This inverts the dependency: tracking maintains segmentation continuity instead of segmentation constraining tracking.

---

### 3.3 GPU Memory During Model Switching

**What's missing**: Running SAM 3 (~2GB), CoTracker3 (~4–6GB), and SAM-Body4D (~8–12GB estimated for diffusion-based) on an RTX 4090 (24GB) requires sequential loading/unloading. The research adds latency numbers but doesn't account for:
- Model loading time (~2–5s per model from disk, ~0.5s from CPU RAM)
- GPU memory fragmentation after load/unload cycles
- PyTorch CUDA memory allocator overhead

**Why it matters**: For a 30-second clip at 60fps (1800 frames), if models are loaded per-frame, loading overhead dominates computation. If loaded once, all three don't fit in 24GB simultaneously.

**Suggested resolution**: Process in stages, not per-frame: (1) Run SAM 3 on entire video → save masks, unload SAM 3; (2) Run CoTracker3 on entire video → save trajectories, unload; (3) Run SAM-Body4D on entire video → save meshes. This is the obvious approach but the research's latency budget implies per-frame interleaving.

---

### 3.4 Numerical Stability of CWT at Small Scales

**What's missing**: The CWT modulus maxima extraction at scale a=1.5 (50ms) uses a third-derivative Gaussian wavelet. At small scales and discrete sampling (30–60fps), the wavelet has very few non-negligible samples (3–5 points at 30fps), making the continuous wavelet transform poorly approximated by discrete summation. Edge effects at trajectory segment boundaries add spurious maxima.

**Why it matters**: The JSNR estimates at scale 1.5 (claimed 1.7) assume a well-resolved continuous wavelet. The discrete approximation error could reduce effective JSNR to <1.0 at 30fps, making the "marginal but detectable" assessment overoptimistic.

**Suggested resolution**: At 30fps, abandon CWT at scale 1.5 — the sampling is insufficient. Use matched filtering instead (correlate with a discrete template of the expected jerk pulse shape). At 60fps+, the CWT becomes viable. This reinforces the 60fps minimum recommendation but changes the detection methodology at the boundary.

---

## 4. Breakdance-Specific Blind Spots

### 4.1 Camera Shake Conflated with Dancer Motion

**What's missing**: Battle circle footage is handheld (phone or small camera). Camera shake at 2–8Hz with 5–15px amplitude is typical. CoTracker3 tracks in image coordinates — camera shake appears as coherent motion of all tracked points. The derivative pipeline would interpret camera shake as whole-body movement.

**Why it matters**: Camera shake frequency (2–8Hz) overlaps directly with the jerk event detection band (2–11Hz). A camera bump during a freeze would register as a hit. The "musicality score" would detect correlation between camera-person's movement and music rather than the dancer's.

**Suggested resolution**: Global motion compensation is mandatory. Options: (1) Track background points (floor, walls) and subtract global motion from dancer tracks, (2) Use gyroscope data from phone IMU (if available), (3) Compute homography from background points per-frame and warp dancer tracks to stabilized coordinates. This should be Stage 0 in the pipeline, before any derivative computation.

---

### 4.2 Floor Contact Creates Degenerate Tracking Regions

**What's missing**: During footwork, freezes, and power moves, large portions of the dancer are in contact with the floor. Contact patches have:
- Zero relative motion (stationary body + stationary floor)
- Ambiguous boundaries (skin/fabric blends into floor visually)
- Texture-poor regions (uniform floor surface)

CoTracker3's correlation matching fails in texture-poor regions. Points initialized on body-floor contact patches will drift to the nearest texture — often a floor marking or the edge of the contact patch.

**Why it matters**: Contact points are biomechanically important — they define the support polygon, which determines freeze stability and power move mechanics. Losing these points means losing the most physically meaningful data.

**Suggested resolution**: Use SAM-Body4D's mesh-floor intersection (or JOSH's human-scene contact constraints) as ground truth for contact regions. Don't rely on CoTracker3 for body parts within 5px of the floor surface. Instead, project the nearest mesh vertices.

---

### 4.3 Crew Battles and Simultaneous Dancers

**What's missing**: The research assumes "one dancer at a time." But:
- **Routines**: In crew battles, 2–6 dancers perform simultaneously
- **Transitions**: Outgoing and incoming dancers overlap for 1–3 seconds
- **Crowd intrusion**: Hype men, MCs, other crew members enter frame

CoTracker3 tracks points regardless of which person they belong to. The hierarchical point allocation strategy (Section 8A) assumes a single segmented dancer. Multi-person scenarios require per-identity point tracking.

**Why it matters**: Without dancer-specific tracking, the movement spectrogram blends multiple dancers' motions. The musicality score becomes meaningless when two dancers hit different beats.

**Suggested resolution**: Run SAM 3 with per-instance segmentation. Assign separate CoTracker3 point sets per dancer identity. Use the BoT-SORT + Re-ID pipeline from the prior analysis (ANALYSIS_v2.md) for identity maintenance across occlusions. The pipeline should support N simultaneous dancers with independent spectrograms.

---

### 4.4 Variable Frame Rate from Phone Cameras

**What's missing**: iPhone variable frame rate (VFR) is not discussed. When shooting at 30fps, iPhones occasionally drop to 24fps or spike to 30+ in variable lighting. The SG filter parameters and CWT scales assume **constant frame rate**. VFR produces non-uniform temporal sampling that:
- Breaks the SG polynomial fit assumptions
- Shifts CWT center frequencies per-segment
- Creates artificial acceleration artifacts from frame timing jitter

**Why it matters**: The entire derivative computation framework assumes uniform $\Delta t$. A single dropped frame at 30fps doubles the local $\Delta t$, halving the apparent velocity and creating a spurious negative-then-positive acceleration pulse.

**Suggested resolution**: (1) Transcode to constant frame rate before processing (ffmpeg `-vsync cfr`), accepting the interpolation artifacts, or (2) use non-uniform time stamps in derivative computation (irregular SG fitting, which requires different coefficient computation per window). Option 1 is simpler and probably sufficient for v0.1.

---

### 4.5 Extreme Lens Distortion at Close Range

**What's missing**: Battle footage is often shot with wide-angle phone lenses from close range (1–2m). Barrel distortion is significant at frame edges — a dancer's extended limb at the frame periphery appears curved. CoTracker3 operates in distorted pixel space. Velocity measurements at frame edges are systematically wrong because pixel distances don't correspond to physical distances.

**Why it matters**: A dancer moving from center to edge of frame appears to decelerate (barrel distortion compresses the periphery). The movement spectrogram would show false deceleration for linear motion.

**Suggested resolution**: Undistort frames before tracking using phone-specific lens calibration (available from EXIF metadata on modern iPhones). Alternatively, apply distortion correction to CoTracker3 output trajectories using a lens model.

---

## 5. Integration Gaps

### 5.1 The Movement Spectrogram Construction Is Unspecified

**What's missing**: This is the most critical gap. The research describes the "core innovation" as cross-correlating audio and movement spectrograms, but **never specifies how to construct the movement spectrogram**. Specifically:
- What is the frequency axis? (Joint velocity FFT? Wavelet scales? Something else?)
- What is the time resolution? (Must match audio STFT hop size)
- How are 17+ joints reduced to a single spectrogram? (Sum? Max? Per-joint channel?)
- What normalization makes cross-correlation meaningful between audio (energy in dB) and movement (velocity in px/s)?

**Why it matters**: This is literally "the 3%" — the core differentiator. Everything else in the pipeline exists to feed this computation, and it's the least specified component.

**Suggested resolution**: Define the movement spectrogram explicitly:
- Time axis: same hop size as audio STFT (e.g., 512 samples at 44.1kHz = 11.6ms)
- Frequency axis: CWT of per-joint velocity magnitude at scales matching audio STFT frequency bins
- Joint aggregation: weighted sum where weights are learned from beat-alignment data
- Normalization: z-score both spectrograms per-frequency-band before cross-correlation

---

### 5.2 CoTracker3 2D Points → SAM-Body4D 3D Mesh: No Camera Model

**What's missing**: Section 8D proposes "3D mesh → project vertices to 2D → compare with CoTracker3 → correct drifted tracks." This requires a camera projection model (intrinsic matrix K, extrinsic [R|t]). But:
- Single-camera battle footage has no calibration
- SAM-Body4D likely outputs meshes in a canonical camera frame
- Without known focal length, projecting 3D vertices to 2D is underconstrained

**Why it matters**: The mesh-guided correction is proposed as the key mitigation for CoTracker3's tracking failures during power moves. Without a camera model, this correction cannot be computed.

**Suggested resolution**: Estimate camera intrinsics from EXIF data (focal length, sensor size) or use a fixed "typical phone" model (f ≈ 26mm equivalent, 1080p). For extrinsics, SAM-Body4D should output the estimated camera-to-body transform — use that directly. Flag this as a validation requirement: verify that projected mesh vertices align with image within 5px tolerance before enabling mesh-guided correction.

---

### 5.3 MotionBERT's Role Is Contradictory

**What's missing**: The cross-paper summary says MotionBERT serves as "real-time triage" — classifying move phases to route segments to the appropriate processing path. But the revised pipeline (TECH_STACK_REEVALUATION.md) doesn't include MotionBERT at all. The pipeline goes SAM 3 → CoTracker3 → SAM-Body4D with no triage step.

**Why it matters**: If MotionBERT is needed for routing, it must process frames before the full pipeline runs — adding to latency. If it's not needed (SAM-Body4D handles everything), then the MotionBERT research was wasted. The architecture needs a clear decision.

**Suggested resolution**: Drop MotionBERT from v0.1. SAM-Body4D processes all frames uniformly. Add MotionBERT as a latency optimization in v2 only if profiling shows that SAM-Body4D on easy frames (toprock, standing) is a bottleneck worth bypassing.

---

### 5.4 Audio-Visual Temporal Alignment Precision

**What's missing**: The musicality scoring uses $\sigma_{\text{sync}} \approx 30–50$ ms, meaning the pipeline needs sub-frame temporal alignment between audio and video. But:
- Video frames have timestamps quantized to frame period (16.7ms at 60fps)
- Audio beat timestamps have ~10ms resolution from BeatNet+
- iPhone video/audio can have variable sync offset (0–50ms, device-dependent)
- No calibration procedure is specified

**Why it matters**: A systematic 30ms offset between audio and video would halve the musicality score for a perfectly on-beat dancer, because the timing error equals $\sigma_{\text{sync}}$.

**Suggested resolution**: Include an audio-visual sync calibration step: detect a sharp physical event visible in both domains (e.g., a foot stomp creates both an image displacement and an audio transient) and measure the offset. Apply this correction before musicality computation.

---

## 6. Citation Verification

### 6.1 Post-May-2025 Papers Are Unverifiable

The following citations are beyond my knowledge cutoff (May 2025) and I **cannot confirm they exist or verify their claimed capabilities**. The LLM research loop that produced these citations is known to occasionally hallucinate plausible-sounding papers:

| Citation | Claimed Date | Risk Level |
|---|---|---|
| SAM-Body4D (arxiv:2512.08406) | Dec 2025 | **HIGH** — central to pipeline |
| SAM 3 (arxiv:2511.16719) | Nov 2025 | **HIGH** — central to pipeline |
| JOSH (ICLR 2026) | Jan 2026 | **MEDIUM** — used for occlusion |
| DanceFormer (ScienceDirect) | 2025 | **MEDIUM** — claimed 18.4mm AIST accuracy |
| CoWTracker | 2026 | **LOW** — listed as alternative, not in pipeline |
| Fast SAM 3D Body | Nov 2025 | **LOW** — mentioned but not used directly |
| Carnegie Mellon Multi-View 4D | Jan 2026 | **LOW** — not in core pipeline |
| InternVideo 2.5 | Dec 2025 | **LOW** — mentioned as alternative |
| MSNet (Nature) | 2025 | **MEDIUM** — in audio layer |

**Why it matters**: If SAM-Body4D or SAM 3 don't exist (or don't work as described), the revised pipeline collapses. The TECH_STACK_REEVALUATION.md downgrades Gap #1 from CRITICAL to LOW entirely based on SAM-Body4D being "training-free."

**Suggested resolution**: Before any implementation, verify each paper exists by checking arxiv, GitHub repos, and model checkpoints. Prioritize verifying SAM-Body4D and SAM 3 — the pipeline cannot proceed without them. Maintain ViTPose + rotation augmentation as a fallback until verification is complete.

---

### 6.2 CoTracker3 Benchmark Numbers Need Source Check

**What's missing**: The claimed TAP-Vid-DAVIS AJ of 67.8 and the PointOdyssey estimates need verification against the actual paper tables. The CoTracker3 paper (arxiv:2410.11831) reports multiple configurations — the number may correspond to a specific variant (offline, specific resolution, specific training data mix) not the default configuration.

**Why it matters**: The entire "predicted AJ on breakdancing: 35–50" analysis chains from this number. A 3-point error in the baseline propagates through all per-move predictions.

**Suggested resolution**: Pull the exact numbers from Table 1 (or equivalent) of the CoTracker3 paper. Note which configuration (online/offline, resolution, number of points) each number corresponds to. Use the configuration that matches the planned deployment (likely offline, 384×512, 1200 points).

---

### 6.3 BRACE Dataset Scope May Be Overstated

**What's missing**: The research claims BRACE provides breakdancing annotations from Red Bull BC One footage. BRACE (ECCV 2022) does exist, but it focuses on **pose estimation in acrobatic scenarios** — it contains single-frame 2D pose annotations, not dense point tracking or 3D mesh ground truth. It may not have the temporal density needed to validate CoTracker3 trajectories.

**Why it matters**: The research says "BRACE exists for validation and fine-tuning" — but fine-tuning CoTracker3 requires dense point correspondences across frames, which BRACE likely doesn't provide. Validation of tracking requires ground truth trajectories, not just per-frame poses.

**Suggested resolution**: Verify exactly what BRACE provides: frame-level 2D keypoints? Temporal sequences? Video with timestamps? If it only provides sparse keypoints per frame, it's useful for validating SAM-Body4D/DanceFormer but not CoTracker3. For CoTracker3 validation, you'd need manual point annotation on breakdancing footage (Experiment #3 in Section 9).

---

## Summary: Top 5 Gaps by Severity

| # | Gap | Section | Impact |
|---|---|---|---|
| 1 | Movement spectrogram construction is unspecified | 5.1 | **Blocks the core innovation** |
| 2 | Camera shake not compensated — contaminates all derivatives | 4.1 | **Invalidates musicality scoring on real footage** |
| 3 | Clothing tracking ≠ body tracking | 3.1 | **Systematic bias in velocity/jerk measurements** |
| 4 | Post-2025 citations unverified — pipeline may depend on hallucinated models | 6.1 | **Architecture may need complete revision** |
| 5 | Pseudo-label quality on breakdancing — teacher struggles on the same domain | 1.1 | **Domain adaptation path may backfire** |
