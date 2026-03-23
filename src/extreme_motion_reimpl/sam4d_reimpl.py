"""SAM4D reimplementation — temporal cross-modal segmentation with memory.

Paper-faithful numpy-only reimplementation of the SAM4D pipeline, which extends
SAM-style segmentation with two key innovations:

1. **UMPE (Unified Multi-modal Prompt Encoder)**: Fuses RGB, depth, spatial, and
   gradient features from within a mask region into a compact 64-dimensional
   embedding via a learned (here: fixed random) projection. This replaces the
   single-modality prompt encoders of SAM/SAM2 with a cross-modal representation
   that captures appearance, geometry, and spatial layout simultaneously.

2. **MCMA (Motion-aware Cross-Modal Attention)**: A temporal attention mechanism
   that queries a memory bank of past embeddings, applying motion-gated recency
   weighting. Fast-moving objects emphasise recent memories (high motion gate),
   while stationary objects weight older, more stable memories (low motion gate).
   The gating uses a sigmoid of velocity magnitude, inspired by optical-flow-
   conditioned attention in video transformers.

The per-frame pipeline simulates SAM-quality masks via noise injection on GT,
encodes each frame with UMPE, queries the temporal memory bank, refines the
embedding via MCMA attention, and decides whether to trust the current mask or
fall back to centroid-displaced previous masks based on embedding consistency.

Reference: SAM4D project — https://sam4d-project.github.io/
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import synth_data
from .audio_motion import audio_motion_alignment


# ---------------------------------------------------------------------------
# Constants for the reimplementation configuration
# ---------------------------------------------------------------------------
NOISE_LEVEL = 0.10       # Fraction of mask pixels to flip (simulated SAM noise)
MEMORY_CAPACITY = 32     # Max entries in the temporal memory bank
DECAY_RATE = 0.95        # Exponential recency decay per frame
MOTION_SCALE = 0.5       # Sigmoid scale for motion gating
PROJECTION_DIM = 64      # UMPE output embedding dimensionality
FEATURE_DIM = 16         # Raw feature vector size before projection
SEED = 42


# ---------------------------------------------------------------------------
# Simulated SAM masks
# ---------------------------------------------------------------------------

def _simulate_sam_mask(
    gt_mask: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a noisy SAM segmentation mask from ground truth.

    Applies two corruption strategies:
    - Random pixel flips: ``noise_level`` fraction of pixels are toggled.
    - Morphological erosion/dilation via box filter: a 3x3 majority vote that
      smooths jagged boundaries, mimicking SAM's tendency to produce clean but
      slightly shifted contours.

    Args:
        gt_mask: Binary ground-truth mask (H, W), values in {0, 1}.
        noise_level: Probability of flipping each pixel.
        rng: Numpy random generator for reproducibility.

    Returns:
        Noisy binary mask (H, W) with values in {0.0, 1.0}.
    """
    mask = gt_mask.copy()
    H, W = mask.shape

    # --- Pixel flips ---
    flip = rng.random((H, W)) < noise_level
    mask = np.where(flip, 1.0 - mask, mask)

    # --- Morphological smoothing via 3x3 box majority vote ---
    # This simulates the smooth-boundary bias of SAM's mask decoder.
    padded = np.pad(mask, 1, mode="edge")
    vote = np.zeros_like(mask)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            vote += padded[1 + dy : H + 1 + dy, 1 + dx : W + 1 + dx]
    mask = (vote >= 5.0).astype(np.float64)

    return mask


# ---------------------------------------------------------------------------
# UMPE — Unified Multi-modal Prompt Encoder
# ---------------------------------------------------------------------------

def _compute_gradient_magnitude(image_2d: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude of a 2D image via forward differences.

    Uses simple finite differences (Sobel-lite) to approximate the spatial
    gradient magnitude, which captures edge and texture information within
    the mask region — important for distinguishing dancer limbs from
    background in the prompt embedding.
    """
    gy = np.zeros_like(image_2d)
    gx = np.zeros_like(image_2d)
    gy[:-1, :] = image_2d[1:, :] - image_2d[:-1, :]
    gx[:, :-1] = image_2d[:, 1:] - image_2d[:, :-1]
    return np.sqrt(gx ** 2 + gy ** 2)


def _umpe_encode(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    """Encode an RGB-D frame within a mask region into a 64-dim embedding.

    The Unified Multi-modal Prompt Encoder extracts a fixed-size feature vector
    from four feature groups:

    1. **RGB statistics** (6 values): Per-channel mean and std within the mask.
       Captures dominant colour and texture variability — useful for re-identifying
       the same object across frames even under lighting changes.

    2. **Depth statistics** (4 values): Mean, std, min, max depth within the mask.
       Encodes the object's distance and depth extent — critical for separating
       overlapping objects at different depths.

    3. **Spatial statistics** (4 values): Centroid (x, y), area ratio, and aspect
       ratio of the mask's bounding box. These capture WHERE the object is and
       its rough shape, enabling spatial consistency checks across frames.

    4. **Gradient statistics** (2 values): Mean gradient magnitude of the
       greyscale RGB and depth within the mask. High gradients indicate textured
       or geometrically complex regions — useful for distinguishing articulated
       body parts from flat surfaces.

    The 16-dim feature vector is projected to 64 dimensions via a fixed random
    matrix, simulating a learned linear projection layer.

    Args:
        rgb: (H, W, 3) RGB image, float64 in [0, 1].
        depth: (H, W) depth map, float64 in metres.
        mask: (H, W) binary mask, float64 in {0, 1}.
        projection_matrix: (16, 64) fixed random projection.

    Returns:
        (64,) L2-normalised embedding vector.
    """
    mask_bool = mask > 0.5
    n_pixels = mask_bool.sum()
    H, W = mask.shape

    if n_pixels < 1:
        # Degenerate case: empty mask -> zero embedding
        return np.zeros(PROJECTION_DIM)

    # --- 1. RGB statistics (6 values) ---
    rgb_masked = rgb[mask_bool]  # (N, 3)
    rgb_mean = rgb_masked.mean(axis=0)  # (3,)
    rgb_std = rgb_masked.std(axis=0)    # (3,)

    # --- 2. Depth statistics (4 values) ---
    depth_masked = depth[mask_bool]
    depth_mean = depth_masked.mean()
    depth_std = depth_masked.std()
    depth_min = depth_masked.min()
    depth_max = depth_masked.max()

    # --- 3. Spatial statistics (4 values) ---
    ys, xs = np.where(mask_bool)
    centroid_x = xs.mean() / W
    centroid_y = ys.mean() / H
    area_ratio = float(n_pixels) / (H * W)
    # Bounding box aspect ratio
    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1
    aspect_ratio = bbox_w / max(bbox_h, 1)

    # --- 4. Gradient statistics (2 values) ---
    greyscale = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    grad_rgb = _compute_gradient_magnitude(greyscale)
    grad_depth = _compute_gradient_magnitude(depth)
    grad_rgb_mean = grad_rgb[mask_bool].mean()
    grad_depth_mean = grad_depth[mask_bool].mean()

    # --- Assemble 16-dim feature vector ---
    features = np.array([
        rgb_mean[0], rgb_mean[1], rgb_mean[2],
        rgb_std[0], rgb_std[1], rgb_std[2],
        depth_mean, depth_std, depth_min, depth_max,
        centroid_x, centroid_y, area_ratio, aspect_ratio,
        grad_rgb_mean, grad_depth_mean,
    ])  # (16,)

    # --- Project to 64 dimensions ---
    embedding = features @ projection_matrix  # (64,)

    # --- L2 normalise ---
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm

    return embedding


# ---------------------------------------------------------------------------
# Temporal Memory Bank
# ---------------------------------------------------------------------------

class TemporalMemoryBank:
    """Fixed-capacity memory bank storing per-frame embeddings with recency decay.

    The memory bank is the key temporal component of SAM4D. It stores past frame
    embeddings and allows the current frame to attend to them with exponentially
    decaying weights. When capacity is exceeded, the oldest entry is evicted
    (FIFO), simulating a sliding-window memory of bounded size.

    Attributes:
        capacity: Maximum number of stored entries.
        decay_rate: Per-frame exponential decay factor for recency weighting.
        entries: List of (frame_id, embedding, confidence) tuples.
    """

    def __init__(self, capacity: int, decay_rate: float) -> None:
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.entries: list[tuple[int, np.ndarray, float]] = []

    def add(self, frame_id: int, embedding: np.ndarray, confidence: float) -> None:
        """Store a new embedding. Evict oldest if at capacity."""
        if len(self.entries) >= self.capacity:
            self.entries.pop(0)
        self.entries.append((frame_id, embedding.copy(), confidence))

    def query(
        self,
        current_embedding: np.ndarray,
        current_frame: int,
        top_k: int = 8,
    ) -> list[tuple[int, np.ndarray, float, float]]:
        """Query memory bank by cosine similarity with recency decay.

        For each stored entry, the score is:
            score = cosine_sim(query, stored) * decay_rate^(current_frame - stored_frame)

        This biases retrieval toward recent, similar embeddings — a proxy for
        the paper's temporal attention weights.

        Args:
            current_embedding: (64,) query embedding.
            current_frame: Current frame index.
            top_k: Number of top results to return.

        Returns:
            List of (frame_id, embedding, confidence, score) tuples, sorted
            by score descending.
        """
        if not self.entries:
            return []

        results = []
        q_norm = np.linalg.norm(current_embedding)

        for fid, emb, conf in self.entries:
            e_norm = np.linalg.norm(emb)
            if q_norm < 1e-8 or e_norm < 1e-8:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(current_embedding, emb) / (q_norm * e_norm))
            recency = self.decay_rate ** (current_frame - fid)
            score = cos_sim * recency
            results.append((fid, emb, conf, score))

        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# MCMA — Motion-aware Cross-Modal Attention
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ez = np.exp(x)
    return ez / (1.0 + ez)


def _mcma_attend(
    query_embedding: np.ndarray,
    memory_results: list[tuple[int, np.ndarray, float, float]],
    current_frame: int,
    velocity_magnitude: float,
    motion_scale: float,
    decay_rate: float,
) -> np.ndarray:
    """Motion-aware Cross-Modal Attention over the temporal memory bank.

    MCMA modulates attention weights by a motion gate derived from the current
    frame's velocity. The intuition:
    - **High motion** (e.g., mid-windmill): Recent memories are most relevant
      because the pose is changing rapidly. The motion gate upweights recent
      entries and downweights older, potentially stale ones.
    - **Low motion** (e.g., freeze): Older memories are more reliable since the
      object is stable. The motion gate downweights recent jittery observations.

    The gate is computed as:
        motion_gate = sigmoid(||velocity|| / motion_scale)

    For each memory entry:
        - If recent (within last 4 frames): weight *= motion_gate
        - If older: weight *= (1 - motion_gate)

    Final embedding = softmax(adjusted_scores) @ memory_embeddings.

    Args:
        query_embedding: (64,) current frame embedding.
        memory_results: Output of TemporalMemoryBank.query().
        current_frame: Current frame index.
        velocity_magnitude: L2 norm of centroid displacement from previous frame.
        motion_scale: Denominator for sigmoid gating.
        decay_rate: For recency calculation.

    Returns:
        (64,) refined embedding via attention-weighted memory aggregation.
    """
    if not memory_results:
        return query_embedding.copy()

    # --- Motion gate ---
    motion_gate = _sigmoid(velocity_magnitude / motion_scale)

    # --- Compute attention scores ---
    scores = []
    embeddings = []
    q_norm = np.linalg.norm(query_embedding)

    for fid, emb, _conf, _raw_score in memory_results:
        e_norm = np.linalg.norm(emb)
        if q_norm < 1e-8 or e_norm < 1e-8:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(query_embedding, emb) / (q_norm * e_norm))

        recency = decay_rate ** (current_frame - fid)
        age = current_frame - fid

        # Motion-gated reweighting: recent memories (age <= 3) get motion_gate,
        # older memories get (1 - motion_gate)
        if age <= 3:
            gated_weight = cos_sim * recency * motion_gate
        else:
            gated_weight = cos_sim * recency * (1.0 - motion_gate)

        scores.append(gated_weight)
        embeddings.append(emb)

    scores = np.array(scores)
    embeddings = np.array(embeddings)  # (K, 64)

    # --- Softmax over scores ---
    # Numerical stability: subtract max before exp
    scores_shifted = scores - scores.max()
    exp_scores = np.exp(scores_shifted)
    weights = exp_scores / (exp_scores.sum() + 1e-8)  # (K,)

    # --- Weighted sum of memory embeddings ---
    refined = weights @ embeddings  # (64,)

    # L2 normalise
    norm = np.linalg.norm(refined)
    if norm > 1e-8:
        refined = refined / norm

    return refined


# ---------------------------------------------------------------------------
# Mask IoU utility
# ---------------------------------------------------------------------------

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks."""
    a_bool = a > 0.5
    b_bool = b > 0.5
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Full SAM4D per-frame pipeline
# ---------------------------------------------------------------------------

def run_sam4d_pipeline(
    rgb: np.ndarray,
    depth: np.ndarray,
    masks_gt: np.ndarray,
    noise_level: float = NOISE_LEVEL,
    memory_capacity: int = MEMORY_CAPACITY,
    decay_rate: float = DECAY_RATE,
    motion_scale: float = MOTION_SCALE,
    seed: int = SEED,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run the full SAM4D temporal segmentation pipeline.

    For each frame:
    1. Generate a noisy SAM mask from ground truth.
    2. Encode the frame's RGB+depth within the mask region (UMPE).
    3. Query the temporal memory bank for similar past embeddings.
    4. If memory is non-empty, run MCMA attention to get a refined embedding.
       Compare refined vs raw embedding (cosine similarity). If sim > 0.5,
       trust the current noisy mask; otherwise, fall back to shifting the
       previous best mask by centroid displacement.
    5. Store the embedding in the memory bank and record the final mask.

    Args:
        rgb: (T, H, W, 3) RGB video.
        depth: (T, H, W) depth maps.
        masks_gt: (T, H, W) ground-truth binary masks.
        noise_level: Noise for simulated SAM masks.
        memory_capacity: Temporal memory bank capacity.
        decay_rate: Recency decay factor.
        motion_scale: Motion gating scale for MCMA.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (final_masks (T, H, W), embeddings list).
    """
    rng = np.random.default_rng(seed)
    T, H, W = masks_gt.shape

    # Fixed random projection matrix for UMPE (seeded for reproducibility)
    proj_rng = np.random.default_rng(seed + 1000)
    projection_matrix = proj_rng.standard_normal((FEATURE_DIM, PROJECTION_DIM))
    projection_matrix /= np.linalg.norm(projection_matrix, axis=1, keepdims=True)

    memory = TemporalMemoryBank(capacity=memory_capacity, decay_rate=decay_rate)
    final_masks = np.zeros((T, H, W))
    embeddings = []
    prev_centroid = None
    prev_mask = None

    for t in range(T):
        # --- Step 1: Simulated SAM mask ---
        noisy_mask = _simulate_sam_mask(masks_gt[t], noise_level, rng)

        # --- Step 2: UMPE encode ---
        embedding = _umpe_encode(rgb[t], depth[t], noisy_mask, projection_matrix)

        # --- Step 3: Query memory ---
        mem_results = memory.query(embedding, t, top_k=8)

        # --- Compute current centroid for motion gating ---
        mask_bool = noisy_mask > 0.5
        if mask_bool.sum() > 0:
            ys, xs = np.where(mask_bool)
            current_centroid = np.array([xs.mean(), ys.mean()])
        else:
            current_centroid = np.array([W / 2.0, H / 2.0])

        # Velocity magnitude from centroid displacement
        if prev_centroid is not None:
            velocity_mag = float(np.linalg.norm(current_centroid - prev_centroid))
        else:
            velocity_mag = 0.0

        # --- Step 4: MCMA attention + mask decision ---
        if mem_results:
            refined_embedding = _mcma_attend(
                embedding, mem_results, t,
                velocity_mag, motion_scale, decay_rate,
            )

            # Cosine similarity between raw and refined embeddings
            raw_norm = np.linalg.norm(embedding)
            ref_norm = np.linalg.norm(refined_embedding)
            if raw_norm > 1e-8 and ref_norm > 1e-8:
                consistency = float(np.dot(embedding, refined_embedding) / (raw_norm * ref_norm))
            else:
                consistency = 0.0

            if consistency > 0.5:
                # Trust current noisy mask
                chosen_mask = noisy_mask
            else:
                # Fall back: shift previous best mask by centroid displacement
                if prev_mask is not None and prev_centroid is not None:
                    dx = int(round(current_centroid[0] - prev_centroid[0]))
                    dy = int(round(current_centroid[1] - prev_centroid[1]))
                    chosen_mask = np.zeros_like(noisy_mask)
                    # Shift prev_mask by (dx, dy)
                    src_y0 = max(-dy, 0)
                    src_y1 = min(H - dy, H)
                    src_x0 = max(-dx, 0)
                    src_x1 = min(W - dx, W)
                    dst_y0 = max(dy, 0)
                    dst_y1 = min(H + dy, H)
                    dst_x0 = max(dx, 0)
                    dst_x1 = min(W + dx, W)
                    h_copy = min(src_y1 - src_y0, dst_y1 - dst_y0)
                    w_copy = min(src_x1 - src_x0, dst_x1 - dst_x0)
                    if h_copy > 0 and w_copy > 0:
                        chosen_mask[dst_y0:dst_y0 + h_copy, dst_x0:dst_x0 + w_copy] = \
                            prev_mask[src_y0:src_y0 + h_copy, src_x0:src_x0 + w_copy]
                else:
                    chosen_mask = noisy_mask
        else:
            # First frame: no memory, use noisy mask directly
            chosen_mask = noisy_mask

        # --- Step 5: Store embedding + record mask ---
        confidence = _mask_iou(chosen_mask, masks_gt[t])
        memory.add(t, embedding, confidence)
        final_masks[t] = chosen_mask
        embeddings.append(embedding)

        prev_centroid = current_centroid
        prev_mask = chosen_mask

    return final_masks, embeddings


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    final_masks: np.ndarray,
    masks_gt: np.ndarray,
    inversion_mask: np.ndarray,
    joints_3d: np.ndarray,
    audio: np.ndarray,
    fps: float,
    sample_rate: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    seed: int = SEED,
) -> dict[str, float]:
    """Compute all canonical and applied SAM4D metrics.

    Metrics:
    - temporal_consistency: Average IoU between consecutive final masks.
      Captures how stable segmentation is over time — a hallmark of SAM4D's
      memory mechanism.
    - cross_modal_prompt_transfer: Ratio of RGBD pipeline performance to
      RGB-only (depth zeroed). Measures how much depth contributes to
      segmentation quality via the UMPE encoder.
    - inversion_tracking_stability: Average mask IoU during inversion frames
      only (head below hips). Stress-tests the pipeline under the most
      challenging breakdance poses.
    - audio_motion_stability: Cross-correlation between motion energy and
      audio energy, via the shared audio_motion module. Validates that the
      tracked dancer's motion remains physically correlated with music.
    """
    T = final_masks.shape[0]

    # --- Temporal consistency: avg IoU between consecutive masks ---
    consecutive_ious = []
    for t in range(1, T):
        iou = _mask_iou(final_masks[t], final_masks[t - 1])
        consecutive_ious.append(iou)
    temporal_consistency = float(np.mean(consecutive_ious)) if consecutive_ious else 0.0

    # --- Cross-modal prompt transfer ---
    # RGBD performance: avg IoU against GT
    rgbd_ious = [_mask_iou(final_masks[t], masks_gt[t]) for t in range(T)]
    avg_iou_rgbd = float(np.mean(rgbd_ious))

    # RGB-only performance: re-run pipeline with depth zeroed
    depth_zeroed = np.zeros_like(depth)
    rgb_only_masks, _ = run_sam4d_pipeline(
        rgb, depth_zeroed, masks_gt,
        noise_level=NOISE_LEVEL, memory_capacity=MEMORY_CAPACITY,
        decay_rate=DECAY_RATE, motion_scale=MOTION_SCALE, seed=seed,
    )
    rgb_only_ious = [_mask_iou(rgb_only_masks[t], masks_gt[t]) for t in range(T)]
    avg_iou_rgb_only = float(np.mean(rgb_only_ious))

    cross_modal = avg_iou_rgbd / max(avg_iou_rgb_only, 0.01)

    # --- Inversion tracking stability ---
    inv_frames = inversion_mask > 0.5
    if inv_frames.any():
        inv_ious = [
            _mask_iou(final_masks[t], masks_gt[t])
            for t in range(T)
            if inv_frames[t]
        ]
        inversion_tracking = float(np.mean(inv_ious)) if inv_ious else 0.0
    else:
        inversion_tracking = 1.0

    # --- Audio-motion stability ---
    am = audio_motion_alignment(
        joints_3d, audio, fps, sample_rate,
    )
    audio_stability = float(am.alignment_stability)

    return {
        "temporal_consistency": round(temporal_consistency, 4),
        "cross_modal_prompt_transfer": round(cross_modal, 4),
        "inversion_tracking_stability": round(inversion_tracking, 4),
        "audio_motion_stability": round(audio_stability, 4),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run SAM4D reimplementation and print results as JSON to stdout."""
    t0 = time.monotonic()

    # --- Generate synthetic RGBD data ---
    data = synth_data.generate_rgbd_sequence(n_frames=32, H=64, W=64, seed=SEED)
    rgb = data["rgb"]
    depth = data["depth"]
    masks_gt = data["masks_gt"]
    joints_3d = data["joints_3d"]
    audio = data["audio"]
    intrinsics = data["intrinsics"]
    fps = data["fps"]
    sample_rate = data["sample_rate"]
    inversion_mask = data["inversion_mask"]

    # --- Run pipeline ---
    final_masks, embeddings = run_sam4d_pipeline(
        rgb, depth, masks_gt,
        noise_level=NOISE_LEVEL,
        memory_capacity=MEMORY_CAPACITY,
        decay_rate=DECAY_RATE,
        motion_scale=MOTION_SCALE,
        seed=SEED,
    )

    # --- Compute metrics ---
    metrics = compute_metrics(
        final_masks, masks_gt, inversion_mask,
        joints_3d, audio, fps, sample_rate,
        rgb, depth, seed=SEED,
    )

    wall = time.monotonic() - t0
    loc = len(Path(__file__).read_text().splitlines())

    payload = {
        "paper_id": "sam4d",
        "mode": "reimplementation",
        "canonical_metrics": {
            "temporal_consistency": metrics["temporal_consistency"],
            "cross_modal_prompt_transfer": metrics["cross_modal_prompt_transfer"],
        },
        "applied_metrics": {
            "inversion_tracking_stability": metrics["inversion_tracking_stability"],
            "cross_modal_prompt_transfer": metrics["cross_modal_prompt_transfer"],
            "audio_motion_stability": metrics["audio_motion_stability"],
        },
        "runtime_cost": {
            "wall_clock_seconds": round(wall, 4),
            "gpu_hours": 0.0,
        },
        "code_stats": {
            "implementation_loc": loc,
            "files_touched": 2,
            "shared_module_ratio": round(
                1.0 - loc / (loc + len(Path(__file__).parent.joinpath("synth_data.py").read_text().splitlines())
                             + len(Path(__file__).parent.joinpath("audio_motion.py").read_text().splitlines())),
                4,
            ),
        },
        "artifacts": [],
        "open_questions": [
            "UMPE linear projection is fixed-random; a learned projection would likely improve cross-modal transfer.",
            "MCMA motion gate threshold (age <= 3) is hardcoded; adaptive windowing could help with variable-tempo moves.",
            "Memory bank FIFO eviction is naive; importance-weighted eviction may improve long-sequence performance.",
        ],
        "notes": (
            "Numpy-only SAM4D reimplementation with UMPE cross-modal encoder, "
            "temporal memory bank, and MCMA motion-gated attention. "
            f"noise_level={NOISE_LEVEL}, capacity={MEMORY_CAPACITY}, "
            f"decay={DECAY_RATE}, motion_scale={MOTION_SCALE}."
        ),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
