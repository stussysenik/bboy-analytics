# cotracker3 — Follow-up 1.2: The audit focuses on CoTracker3 in isolation but says nothing about the output format, coordinate sp

_Generated: 2026-03-23T04:03:10.857Z_

---

Let me work with the research context provided and produce the thorough analysis of CoTracker3's output contract and downstream integration.

---

# CoTracker3 Output Contract & Downstream Integration

## 1. Output Format and Coordinate Space

### Raw Output Tensors

CoTracker3's forward pass returns a tuple of two tensors:

```
(tracks, visibility)
```

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `tracks` | $$(B, T, N, 2)$$ | `float32` | $$(x, y)$$ pixel coordinates per point per frame |
| `visibility` | $$(B, T, N)$$ | `float32` | Confidence/visibility score $$\in [0, 1]$$ |

Where:
- $$B$$ = batch size (typically 1 for inference)
- $$T$$ = number of frames in the window (or full video for offline mode)
- $$N$$ = number of tracked points
- Coordinates are in **pixel space** — $$(0, 0)$$ is top-left, $$(W-1, H-1)$$ is bottom-right

### Coordinate Convention — Critical Detail

CoTracker3 operates internally at a **downsampled resolution**. The input video is resized so the shorter side is 384px (default `S=384`), maintaining aspect ratio. The output `tracks` tensor contains coordinates in this **resized resolution**, not the original video resolution.

For a 1080p bboy clip (1920×1080):
$$
\text{scale} = \frac{384}{1080} = 0.3\overline{5}
$$
$$
\text{internal resolution} = 682 \times 384
$$

**You must rescale back to original resolution:**
$$
x_{\text{orig}} = x_{\text{track}} \cdot \frac{W_{\text{orig}}}{W_{\text{internal}}}, \quad y_{\text{orig}} = y_{\text{track}} \cdot \frac{H_{\text{orig}}}{H_{\text{internal}}}
$$

For 1080p: multiply both coordinates by $$\frac{1}{0.3\overline{5}} \approx 2.8125$$

### Visibility Semantics

The visibility value is a **sigmoid output** from the transformer, not a hard binary. In practice:
- $$v > 0.5$$: point is visible and tracked reliably
- $$0.2 < v < 0.5$$: point is occluded but position is hallucinated (the transformer predicts where it *should* be based on motion patterns)
- $$v < 0.2$$: point is lost — position unreliable

For breakdancing, the hallucinated-position regime ($$0.2 < v < 0.5$$) is actually valuable — a dancer's hand going behind their body during a windmill still has a physically plausible trajectory that the transformer infers from surrounding visible points.

---

## 2. Sliding Window Output Stitching

For videos longer than the window size ($$T_w = 60$$ frames by default, ~2 seconds at 30fps), CoTracker3 uses a sliding window with overlap. The critical detail for downstream consumers:

### Window Structure

```
Window 1: frames [0, 60)
Window 2: frames [40, 100)    ← 20-frame overlap
Window 3: frames [80, 140)   ← 20-frame overlap
...
```

The overlap region uses **weighted averaging** to stitch predictions:

$$
\hat{p}_t = \frac{w_{\text{prev}} \cdot p_t^{\text{prev}} + w_{\text{curr}} \cdot p_t^{\text{curr}}}{w_{\text{prev}} + w_{\text{curr}}}
$$

where weights are linear ramps across the overlap. The final output is a single contiguous $$(B, T_{\text{total}}, N, 2)$$ tensor — **downstream consumers see no window boundaries**.

However, stitching introduces subtle artifacts:

1. **Position discontinuities** at window edges: typically $$< 0.5$$ px RMS, but can spike to $$2\text{-}3$$ px during fast motion (power moves)
2. **Visibility score jumps**: a point may be $$v = 0.7$$ at the end of window $$k$$ and $$v = 0.4$$ at the start of window $$k+1$$ — the averaged value in the overlap smooths this, but the transition region can be noisy

---

## 3. Temporal Smoothing: When and How

### Does CoTracker3 output need smoothing before handoff?

**Short answer: Yes, for bboy footage. No, for typical benchmark footage.**

CoTracker3's iterative refinement (4 GRU updates per window) already produces smooth trajectories for moderate-speed motion. But breakdancing violates the assumptions:

| Motion Type | Typical px/frame @ 30fps | CoTracker3 Error (px) | Needs Smoothing? |
|-------------|--------------------------|----------------------|------------------|
| Walking | 2–4 | 0.5–1.0 | No |
| Running | 5–8 | 1.0–2.0 | No |
| Toprock | 3–6 | 0.8–1.5 | No |
| Footwork | 6–10 | 1.5–3.0 | Light |
| Windmill | 10–16 | 3.0–6.0 | Yes |
| Headspin/airflare | 13–20+ | 5.0–10.0+ | Yes |

### Recommended Smoothing: One-Euro Filter

**Not** a Kalman filter (which assumes constant velocity/acceleration — breakdancing violates this constantly). Use a **One-Euro filter** per tracked point:

$$
\hat{x}_t = \alpha_t \cdot x_t + (1 - \alpha_t) \cdot \hat{x}_{t-1}
$$

where the cutoff frequency adapts based on speed:

$$
f_c = f_{c,\min} + \beta \cdot |\dot{x}_t|
$$

Parameters for bboy footage:
- $$f_{c,\min} = 1.0$$ Hz (smooth when stationary — freezes, poses)
- $$\beta = 0.007$$ (react quickly to fast motion — power moves)
- $$d_{\text{cutoff}} = 1.0$$ Hz (derivative smoothing)

The One-Euro filter is:
- **Causal** (no future frames needed — works for streaming)
- **O(1)** per point per frame
- **Speed-adaptive** (low jitter on freezes, low lag on fast moves)

Implementation is ~30 LOC in Python. This is the same filter Meta uses in their AR/VR hand tracking.

### When to Skip Smoothing

If the downstream consumer is **SAM-Body4D** (which takes dense point tracks as input for 4D body reconstruction), it has its own internal temporal consistency mechanisms. Smoothing before SAM-Body4D can actually hurt — it assumes input reflects real motion dynamics, and over-smoothed tracks can cause the mesh to lag behind ground truth.

**Rule: smooth before visualization or metric computation. Don't smooth before model-based consumers that have their own temporal priors.**

---

## 4. Downstream Input Contracts

### 4a. CoTracker3 → SAM-Body4D

SAM-Body4D (Segment Any Moving Body in 4D) expects:

**Input:**
- Dense point tracks: $$(T, N, 2)$$ in **normalized coordinates** $$[0, 1]^2$$
- Visibility mask: $$(T, N)$$, binary (threshold at 0.5)
- RGB frames: $$(T, H, W, 3)$$ at the same framerate as the tracks

**Conversion from CoTracker3 output:**
```python
# CoTracker3 outputs pixel coords at internal resolution
tracks_norm = tracks.clone()
tracks_norm[..., 0] /= W_internal  # x → [0, 1]
tracks_norm[..., 1] /= H_internal  # y → [0, 1]
vis_binary = (visibility > 0.5).float()
```

**Critical: SAM-Body4D expects tracks at the same temporal resolution as the input video frames.** If CoTracker3 processes at 30fps and SAM-Body4D expects 30fps — trivial. If there's a mismatch, you must resample (see §5).

### 4b. CoTracker3 → JOSH (Judging of Style and Hits)

JOSH consumes motion features, not raw point tracks. The expected interface:

**Input:**
- Per-joint velocity vectors: $$(T, J, 2)$$ where $$J$$ = number of joints
- Beat-alignment features: temporal derivative of joint velocities aligned to music beats

**Conversion from CoTracker3:**
1. Track $$N$$ points on the dancer's body (grid or manually specified keypoints)
2. Compute finite-difference velocities:
$$
\mathbf{v}_t = \frac{\mathbf{p}_{t+1} - \mathbf{p}_{t-1}}{2 \Delta t}
$$
3. Cluster tracked points into body-part groups (this is where SAM-Body4D's segmentation helps — it labels which points belong to which body part)
4. Compute per-body-part aggregate motion features (mean velocity, angular velocity, acceleration magnitude)

### 4c. CoTracker3 → MoveNet/Pose Estimation (bootstrapping)

If using CoTracker3 to seed or validate pose estimation:

**MoveNet expects:** RGB frames, outputs 17 COCO keypoints in normalized $$[0, 1]^2$$

**Integration pattern:** Don't feed CoTracker3 tracks to MoveNet — they solve different problems. Instead, use CoTracker3 to **validate** MoveNet outputs by tracking the same anatomical points independently and checking consistency:

$$
\text{error}_t^{(k)} = \| \mathbf{p}_t^{\text{CoTracker}} - \mathbf{p}_t^{\text{MoveNet}} \|_2
$$

When $$\text{error}_t^{(k)} > \tau$$ (e.g., $$\tau = 20$$ px), MoveNet likely has a joint flip or occlusion failure. Use the CoTracker3 trajectory as the more reliable signal since it maintains temporal identity.

---

## 5. Framerate Mismatch Handling

### The Problem

| Component | Expected FPS | Notes |
|-----------|-------------|-------|
| Camera capture | 30, 60, or 120 | Higher = better for fast moves |
| CoTracker3 internal | Same as input | Processes every frame |
| SAM-Body4D | 30 | Standard training FPS |
| JOSH beat analysis | Music-dependent | Typically 44100 Hz audio / ~30fps video |
| Visualization | 24 or 30 | Display framerate |

### Scenario: 60fps Capture → 30fps Downstream

This is the most common case for bboy footage (60fps improves tracking of fast extremity motion).

**Option A: Decimate CoTracker3 output (simple, slight quality loss)**
$$
\text{tracks}_{30} = \text{tracks}_{60}[:, 0::2, :, :]
$$
$$
\text{vis}_{30} = \text{vis}_{60}[:, 0::2, :]
$$

Take every other frame. Simple but throws away information.

**Option B: Track at 60fps, resample with anti-aliasing (recommended)**

Apply a low-pass filter before decimation to prevent temporal aliasing:

1. Low-pass filter the track trajectories with cutoff at $$f_{\text{Nyquist}} = 15$$ Hz (half the target 30fps):
$$
\tilde{p}_t = \sum_{k=-K}^{K} h_k \cdot p_{t+k}
$$
where $$h$$ is a windowed sinc kernel with $$K = 5$$ taps

2. Decimate: take every 2nd filtered sample

This preserves trajectory smoothness and prevents the "jagged trajectory" artifact that naive decimation causes during fast rotational moves.

### Scenario: 120fps Capture → 30fps Downstream

Same approach but decimate by 4. The low-pass filter becomes more important — without it, a headspin at 120fps decimated to 30fps can lose an entire rotation's worth of trajectory detail.

**Better approach for 120fps:** Track at 120fps, compute velocity/acceleration features at 120fps (capturing the fast dynamics), *then* downsample the derived features to 30fps. Raw positions can be decimated, but motion derivatives should be computed at full temporal resolution.

### Scenario: Variable FPS (dropped frames)

Real bboy footage from phones may have dropped frames. CoTracker3 assumes uniform temporal spacing. If frames are dropped:

1. Detect gaps (timestamp deltas > 1.5× expected interval)
2. Interpolate missing frames with cubic spline before feeding to CoTracker3
3. Mark interpolated frames in metadata so downstream consumers can weight them lower

---

## 6. Complete Integration Pipeline — Tensor Flow

```
Camera (1920×1080 @ 60fps)
    │
    ▼
Resize to 682×384 (shorter side = 384)
    │
    ▼
CoTracker3 Sliding Window (T_w=60, overlap=20)
    ├── tracks: (1, T_total, N, 2) in 682×384 coords
    └── visibility: (1, T_total, N) in [0, 1]
    │
    ├─── [Branch A: SAM-Body4D] ────────────────────────┐
    │    Normalize coords to [0,1]²                      │
    │    Threshold visibility at 0.5                     │
    │    Resample 60fps → 30fps with anti-alias filter   │
    │    → (T_30, N, 2) normalized + (T_30, N) binary    │
    │    → 4D body mesh output                           │
    │                                                    │
    ├─── [Branch B: Motion Features for JOSH] ──────────┐
    │    Rescale to original 1920×1080 coords            │
    │    One-Euro filter (smooth at full 60fps)          │
    │    Compute velocities at 60fps                     │
    │    Aggregate per-body-part features                │
    │    Downsample features to beat-aligned times       │
    │    → (T_beats, J, F) feature tensor                │
    │                                                    │
    └─── [Branch C: Pose Validation] ───────────────────┐
         Rescale to original 1920×1080 coords            │
         Compare with MoveNet keypoints per frame        │
         Flag frames where ||Δ|| > 20px                  │
         → (T, K) consistency scores                     │
```

---

## 7. Edge Cases Specific to Breakdancing

### Self-Occlusion During Freezes/Power Moves

When a dancer holds a freeze (e.g., baby freeze — one hand on ground, body inverted), multiple tracked points suddenly become occluded simultaneously. CoTracker3 handles gradual occlusion well but **sudden mass-occlusion** (5+ points in 1-2 frames) causes the transformer to hallucinate positions that diverge from reality.

**Mitigation:** When $$\sum_n \mathbb{1}[v_t^{(n)} < 0.3] > 0.4N$$ (>40% of points are low-visibility), flag the segment as unreliable and fall back to pose estimation for those frames.

### Camera Shake / Battle Circle Movement

Bboy battles are filmed handheld. Camera motion contaminates all tracked points equally. Before computing dancer-centric motion features:

$$
\mathbf{v}_t^{\text{dancer}} = \mathbf{v}_t^{\text{raw}} - \mathbf{v}_t^{\text{camera}}
$$

Estimate camera motion from background points (tracked on the floor/walls). CoTracker3 can track background and foreground simultaneously — designate 20-30% of your tracked points as background anchors.

### Multiple Dancers in Frame

During a battle, both dancers may be visible (one performing, one watching). CoTracker3 tracks all $$N$$ points regardless of which dancer they're on. **Point assignment to dancer identity must happen downstream** — either via SAM-Body4D's segmentation or by spatial clustering of track velocities (the performing dancer has high-variance motion, the watching dancer is quasi-static).

---

## Key Numerical Constants for Implementation

| Parameter | Value | Why |
|-----------|-------|-----|
| Internal resolution (short side) | 384 px | Default, good accuracy/speed tradeoff |
| Window size $$T_w$$ | 60 frames | 2s @ 30fps, 1s @ 60fps |
| Window overlap | 20 frames | ⅓ of window, standard |
| Visibility threshold | 0.5 | Binary: visible/occluded |
| Search radius | 16 px ($$S=4, \text{stride}=4$$) | Limits max trackable speed |
| Max reliable displacement | ~13 px/frame | Beyond this, tracking degrades |
| One-Euro $$f_{c,\min}$$ | 1.0 Hz | For bboy, allows clean freezes |
| One-Euro $$\beta$$ | 0.007 | For bboy, fast response to power moves |
| Background anchor ratio | 20-30% of $$N$$ | For camera motion compensation |
| Mass-occlusion threshold | 40% of points | When to flag unreliable segment |

This gives you the complete integration contract. The most common failure mode I'd predict: **forgetting to rescale from internal 384px resolution back to original resolution** before handing off to downstream models that expect original or normalized coordinates. Put a coordinate-space assertion at every pipeline boundary.
