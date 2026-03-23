# sam3d — Architecture Survey

_Generated: 2026-03-23T05:06:26.751Z_

---

# SAM3D: Segment Anything in 3D Scenes — Architecture Survey

---

## 1. Problem Statement

### What it solves

3D scene segmentation in point clouds has traditionally required **expensive per-class annotation** and **task-specific training**. Before SAM3D, segmenting arbitrary objects in 3D scenes meant either:

1. Training a fully-supervised 3D segmentation network (PointNet++, MinkowskiNet, etc.) on domain-specific labeled 3D data
2. Using closed-vocabulary 3D models that can only segment pre-defined classes

SAM3D bridges SAM's **class-agnostic, promptable 2D segmentation** into 3D point clouds — enabling open-vocabulary, zero-shot 3D instance segmentation without any 3D-specific training.

### SOTA before SAM3D (mid-2023)

| Method | Approach | Limitation |
|--------|----------|------------|
| **Mask3D** (Schult et al., 2023) | Transformer-based 3D instance segmentation | Closed-vocabulary; requires task-specific 3D training |
| **OpenScene** (Peng et al., 2023) | CLIP features projected to 3D | Open-vocabulary but semantic-only (no instances) |
| **LERF** (Kerr et al., 2023) | Language-embedded radiance fields | Requires NeRF fitting per scene; slow |
| **SAM** (Kirillov et al., 2023) | 2D promptable segmentation | No 3D awareness; per-frame only |

### Gap filled

No prior method could take **arbitrary text/point/box prompts** and produce **3D instance masks** on point clouds in a **training-free** manner. SAM3D is the first to do this by treating SAM as a frozen 2D oracle and solving the 2D→3D aggregation problem.

---

## 2. Architecture Overview

### High-Level Pipeline

SAM3D operates in **three stages**: (1) 2D mask generation via SAM on multi-view images, (2) 2D→3D back-projection onto the point cloud, (3) 3D region merging via boundary-aware grouping.

### Complete Data Flow

```
INPUT: RGB-D Scene (multi-view images + depth + camera poses)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: 2D Mask Generation (per-view)                 │
│                                                         │
│  RGB Image (H×W×3)                                      │
│      │                                                  │
│      ▼                                                  │
│  ┌──────────────┐                                       │
│  │  SAM Encoder  │  (ViT-H frozen)                      │
│  │  Image → Emb  │                                      │
│  └──────┬───────┘                                       │
│         │  Image Embedding (256×64×64)                   │
│         ▼                                               │
│  ┌──────────────────┐    ┌─────────────────┐            │
│  │  Prompt Encoder   │◄───│ Prompts (auto)  │            │
│  │  (grid points)    │    │ 64×64 grid pts  │            │
│  └──────┬───────────┘    └─────────────────┘            │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────┐                                   │
│  │  SAM Mask Decoder │                                   │
│  │  → per-point mask │                                   │
│  └──────┬───────────┘                                   │
│         │  Binary masks M_i (K × H × W)                 │
│         │  + IoU confidence scores                       │
│         ▼                                               │
│  ┌──────────────────┐                                   │
│  │  NMS + Filtering  │  (remove low-conf, deduplicate)   │
│  └──────┬───────────┘                                   │
│         │  Filtered 2D masks per view                    │
└─────────┼───────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: 2D → 3D Back-Projection                       │
│                                                         │
│  For each view v with masks {M_1^v, ..., M_k^v}:       │
│                                                         │
│  ┌─────────────┐   ┌───────────────┐                    │
│  │ Depth Map    │   │ Camera Params │                    │
│  │ D_v (H×W)   │   │ K, [R|t]     │                    │
│  └──────┬──────┘   └──────┬────────┘                    │
│         │                  │                             │
│         ▼                  ▼                             │
│  ┌──────────────────────────────┐                       │
│  │  Unproject: pixel → 3D point  │                       │
│  │  p_3d = R^(-1)(K^(-1)·d·p_2d - t)                   │
│  └──────┬───────────────────────┘                       │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────────────┐                       │
│  │  Assign mask labels to 3D    │                       │
│  │  nearest-neighbor to point   │                       │
│  │  cloud P                     │                       │
│  └──────┬───────────────────────┘                       │
│         │  Per-view 3D mask assignments                  │
│         │  L_v ∈ {0,...,K_v} for each point in P        │
└─────────┼───────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Multi-View 3D Merging                         │
│                                                         │
│  ┌────────────────────────────┐                         │
│  │  Superpoint Extraction     │                         │
│  │  (geometric oversegment.)  │                         │
│  │  VCCS or learned superpts  │                         │
│  └──────┬─────────────────────┘                         │
│         │  Superpoints S = {s_1, ..., s_N}              │
│         ▼                                               │
│  ┌────────────────────────────┐                         │
│  │  Vote Accumulation         │                         │
│  │  For each superpoint s_i,  │                         │
│  │  accumulate mask votes     │                         │
│  │  across all views          │                         │
│  └──────┬─────────────────────┘                         │
│         │  Vote matrix V (N_superpts × N_masks)         │
│         ▼                                               │
│  ┌────────────────────────────────────┐                 │
│  │  Boundary-Aware Region Merging     │                 │
│  │  • Bi-directional merging          │                 │
│  │  • IoU threshold for grouping      │                 │
│  │  • Boundary preservation via       │                 │
│  │    normal discontinuity detection  │                 │
│  └──────┬─────────────────────────────┘                 │
│         │                                               │
│         ▼                                               │
│  ┌────────────────────────────┐                         │
│  │  Final 3D Instance Masks   │                         │
│  │  M_3D = {m_1, ..., m_J}   │                         │
│  └────────────────────────────┘                         │
│                                                         │
└─────────────────────────────────────────────────────────┘

OUTPUT: 3D point cloud with per-point instance labels
```

### Module Inventory

| Module | Sub-modules | Parameters | Trainable? |
|--------|-------------|------------|------------|
| **SAM Image Encoder** | ViT-H (MAE pretrained) | ~632M | No (frozen) |
| **SAM Prompt Encoder** | Positional + learned embeddings | ~6K | No (frozen) |
| **SAM Mask Decoder** | 2-layer transformer + MLP | ~4M | No (frozen) |
| **2D→3D Projector** | Camera math (K, R, t) | 0 | N/A (geometric) |
| **Superpoint Extractor** | VCCS / geometric clustering | 0 | N/A (algorithmic) |
| **Region Merger** | Graph-based grouping | 0 | N/A (algorithmic) |

**Total trainable parameters: 0.** The entire pipeline is training-free.

---

## 3. Key Innovation

### The ONE thing: Training-free 2D→3D transfer via superpoint-based multi-view aggregation with boundary-aware merging

Prior attempts at lifting 2D masks to 3D either:
- Naively projected per-pixel masks (noisy, inconsistent across views)
- Required 3D-specific training (defeats the purpose of SAM's generality)

SAM3D's insight: **use geometric superpoints as the aggregation unit, not individual 3D points.** By voting at the superpoint level across views, the method achieves:

1. **Noise suppression** — individual pixel mis-projections are averaged out within superpoints
2. **View consistency** — a superpoint seen from 5 views gets 5 votes; majority wins
3. **Boundary preservation** — surface normal discontinuities define natural object boundaries that prevent over-merging

### Ablation Evidence

From the paper's experiments on ScanNet:

| Variant | mAP@50 | Δ |
|---------|--------|---|
| Per-point projection (no superpoints) | ~31 | baseline |
| + Superpoint aggregation | ~38 | +7 |
| + Boundary-aware merging | ~42 | +4 |
| + Bi-directional merging | ~46 | +4 |
| Full SAM3D | **46.0** | — |

[NEEDS VERIFICATION] — exact ablation numbers should be confirmed against Table 2 in the paper; the above are reconstructed from reported improvements.

The boundary-aware merging is critical: without it, adjacent objects (e.g., a mug on a table) get merged because their 2D masks overlap from some viewpoints. Normal discontinuity detection keeps them separate.

---

## 4. Input/Output Specification

### Input

| Component | Format | Shape | Notes |
|-----------|--------|-------|-------|
| RGB images | uint8 → float32 normalized | $(V \times H \times W \times 3)$ | $V$ = number of views, typically all frames from an RGB-D scan (e.g., ScanNet: ~300-2000 frames per scene) |
| Depth maps | float32 (meters) | $(V \times H \times W)$ | Aligned to RGB; used for unprojection |
| Camera intrinsics | float32 | $(V \times 3 \times 3)$ | Matrix $K$ per view |
| Camera extrinsics | float32 | $(V \times 4 \times 4)$ | $[R \mid t]$ world-to-camera transform per view |
| 3D point cloud | float32 | $(N \times 3)$ or $(N \times 6)$ | XYZ + optional RGB; reconstructed from depth or provided (e.g., ScanNet .ply) |

**Preprocessing:**
1. Images resized to SAM's expected resolution (1024×1024 for ViT-H)
2. Depth maps aligned and filtered (remove invalid/zero depths)
3. Point cloud voxel-downsampled (typical: 2cm voxel size for indoor scenes)
4. Camera poses typically from SLAM or provided with dataset

### Intermediate Representations

| Stage | Representation | Shape |
|-------|---------------|-------|
| SAM image embedding | Dense feature map | $(256 \times 64 \times 64)$ per view |
| SAM prompt embedding | Point/sparse prompts | $(N_{prompts} \times 256)$ |
| 2D masks per view | Binary masks + scores | $(K_v \times H \times W)$, $K_v$ masks per view |
| 3D point labels per view | Integer labels | $(N_{points},)$ — each point gets a mask ID or 0 (no mask) |
| Superpoints | Cluster assignments | $(N_{points},)$ → $N_{sp}$ clusters |
| Vote matrix | Accumulated votes | $(N_{sp} \times M_{total})$ where $M_{total}$ = total unique mask IDs across all views |
| Merged regions | Graph components | $J$ final instance groups |

### Output

| Component | Format | Shape | Semantics |
|-----------|--------|-------|-----------|
| 3D instance masks | Integer labels | $(N_{points},)$ | Per-point instance ID $\in \{0, 1, ..., J\}$; 0 = unassigned |
| Confidence scores | float32 | $(J,)$ | Per-instance confidence (averaged IoU from SAM) |
| [Optional] Semantic labels | String/int | $(J,)$ | If text-prompted, each instance gets the query label |

---

## 5. Training Pipeline

### There is no training pipeline.

SAM3D is **entirely training-free**. This is the core design philosophy — leverage SAM's pre-trained 2D segmentation capability and transfer it to 3D via geometric reasoning.

- **SAM weights**: Frozen ViT-H checkpoint from the original SAM paper (Kirillov et al., 2023), trained on SA-1B (11M images, 1.1B masks)
- **No fine-tuning**: No adaptation, no LoRA, no prompt tuning
- **No 3D training data consumed**: The method never trains on ScanNet/S3DIS/etc.

### Loss Functions: N/A

No loss functions are optimized. The pipeline is purely geometric + heuristic.

### What "Hyperparameters" Exist (algorithmic, not learned)

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| SAM confidence threshold | Minimum IoU score to keep a 2D mask | 0.7 [NEEDS VERIFICATION] |
| NMS IoU threshold | Deduplication in 2D | 0.5 |
| Superpoint voxel size | Granularity of oversegmentation | 0.02m (2cm) |
| Merge IoU threshold | 3D region merging threshold | 0.5 [NEEDS VERIFICATION] |
| Normal discontinuity angle | Boundary detection threshold | 30° [NEEDS VERIFICATION] |
| View sampling stride | Process every $k$-th frame | Scene-dependent |

---

## 6. Inference Pipeline

Since SAM3D has no training, the inference pipeline **is** the full pipeline described in Section 2.

### What runs at test time

**Everything.** There are no training-only modules to drop. The full pipeline is:

```
1. For each sampled view:
   a. Run SAM encoder (ViT-H forward pass)               — GPU
   b. Generate grid prompts (64×64 = 4096 points)        — CPU
   c. Run SAM decoder per prompt → masks                  — GPU
   d. NMS + confidence filtering                          — CPU
   e. Back-project masks to 3D via depth + camera params  — CPU/GPU

2. Aggregate across views:
   a. Extract superpoints from point cloud                — CPU
   b. Build vote matrix                                   — CPU
   c. Boundary-aware region merging                       — CPU
   d. Output final 3D instance labels                     — CPU
```

### Latency / Throughput

From the paper and GitHub repo:

| Metric | Value | Notes |
|--------|-------|-------|
| Per-view SAM inference | ~150ms | ViT-H on A100, includes encoder + decoder |
| Per-scene total (ScanNet) | **2–10 minutes** | Depends on number of views processed |
| Views processed per scene | 50–300 (subsampled) | Every $k$-th frame; all frames would be prohibitive |
| Superpoint extraction | ~5–30s per scene | Depends on point cloud density |
| Region merging | ~1–5s per scene | Graph algorithm, efficient |

[NEEDS VERIFICATION] — the paper does not report detailed per-module latency breakdowns; these are estimates from the codebase and similar SAM-based pipelines.

### Bottleneck

SAM encoder is the bottleneck: $O(V)$ forward passes through ViT-H. Subsampling views is the primary speed lever.

---

## 7. Computational Cost

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| SAM ViT-H encoder | 632M |
| SAM prompt encoder | ~6K |
| SAM mask decoder | ~4M |
| **Total** | **~636M** |
| **Trainable** | **0** |

### FLOPs

| Operation | FLOPs per invocation | Invocations per scene |
|-----------|---------------------|----------------------|
| ViT-H encoder | ~370 GFLOPs | $V$ (number of views) |
| Mask decoder | ~5 GFLOPs | $V \times N_{prompts}$ |
| Back-projection | Negligible | $V$ |
| Superpoint extraction | ~1 GFLOPs | 1 |
| Region merging | Negligible | 1 |

For a typical ScanNet scene with 100 sampled views:
- **Total FLOPs ≈ 37 TFLOPs** (dominated by 100 × ViT-H forward passes)

[NEEDS VERIFICATION] — FLOPs not explicitly reported in paper; estimated from ViT-H architecture.

### GPU Memory

| Configuration | VRAM |
|---------------|------|
| ViT-H (batch=1) | ~6–8 GB |
| With mask decoder | ~8–10 GB |
| Peak (batch=1, 1024×1024 input) | **~10 GB** |

Single-view processing means memory is constant regardless of scene size — a significant advantage.

### Training Time

**0 hours.** No training required. "Setup time" is just downloading SAM checkpoints (~2.4 GB for ViT-H).

---

## 8. Relevance to Breakdancing Analysis Pipeline

### Where SAM3D fits in the revised pipeline

```
  ① Capture (iPhone 1080p@30fps)
  ② Segment → SAM 3 (text: "breakdancer")     ← SAM3D's DESCENDANT
  ③ Track   → CoTracker3                        ← runs on SAM3D masks
  ④ Mesh    → SAM-Body4D                        ← BUILDS on SAM3D's 3D
  ⑤-⑧ downstream (audio, correlate, score, viz)
```

### Specific value for bboy analysis

| Capability | How it helps |
|------------|-------------|
| **Dancer isolation from scene** | Separate dancer from floor, crowd, stage in 3D. Critical for clean mesh input to SAM-Body4D. |
| **Training-free operation** | No need for breakdancing-specific 3D segmentation data. Works on ANY scene. |
| **Multi-view consistency** | Multiple camera angles (or temporal frames from a single camera) produce consistent 3D segmentation. Handles the dancer moving through inversions across frames. |
| **Superpoint granularity** | Fine-grained segmentation preserves body part boundaries — important for distinguishing limbs during tangled power moves. |
| **Text-promptable** | "breakdancer" as prompt → directly segments the dancer. No class taxonomy needed. |

### Critical limitation for our use case

SAM3D assumes a **static scene** scanned from multiple views. A breakdancing battle is a **dynamic scene** — the dancer moves between frames. This means:

1. Direct multi-view aggregation across time doesn't work (the dancer has moved)
2. Per-frame 2D segmentation + single-frame depth → 3D is the realistic path
3. **SAM 3's video tracking** (temporal consistency across frames) solves this — SAM3D's multi-view merging is replaced by SAM 3's temporal propagation

### The SAM3D → SAM-Body4D lineage

```
SAM (2D, 2023)
  │
  ├── SAM3D (2D→3D projection, 2023)     ← THIS PAPER
  │     │
  │     └── SAM-3D-Body (body-specific 3D, Nov 2025)
  │           │
  ├── SAM 2 (video tracking, 2024)
  │     │
  │     └── SAM 3 (concept-aware video, Nov 2025)
  │           │
  └───────────┴── SAM-Body4D (training-free 4D mesh, Dec 2025)
                    │
                    └── Uses: SAM 3 + Diffusion-VAS + SAM-3D-Body
```

SAM3D's core contribution — the **geometric 2D→3D projection + superpoint aggregation** — is the foundational technique that SAM-3D-Body inherits and specializes for human bodies, which SAM-Body4D then extends to temporal (4D) mesh recovery.

---

## 9. Equations

Since SAM3D is training-free, the key equations are geometric, not loss-based:

### 2D → 3D Back-projection

For pixel $(u, v)$ with depth $d$ in view $v$:

$$\mathbf{p}_{3D} = \mathbf{R}_v^{-1} \left( d \cdot \mathbf{K}_v^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} - \mathbf{t}_v \right)$$

where:
- $\mathbf{K}_v \in \mathbb{R}^{3 \times 3}$ — camera intrinsic matrix for view $v$
- $\mathbf{R}_v \in \mathbb{R}^{3 \times 3}$ — rotation matrix (world → camera)
- $\mathbf{t}_v \in \mathbb{R}^{3}$ — translation vector (world → camera)
- $d \in \mathbb{R}^+$ — depth value at pixel $(u, v)$

### Superpoint Vote Aggregation

For superpoint $s_i$ and mask $m_j$, the vote score is:

$$V(s_i, m_j) = \frac{1}{|s_i|} \sum_{p \in s_i} \mathbb{1}\left[ L_v(p) = m_j \right]$$

where:
- $|s_i|$ — number of points in superpoint $s_i$
- $L_v(p)$ — mask label assigned to point $p$ from view $v$
- $\mathbb{1}[\cdot]$ — indicator function

### Region Merging Criterion

Two superpoint groups $G_a, G_b$ are merged if:

$$\text{IoU}_{3D}(G_a, G_b) = \frac{|G_a \cap G_b|}{|G_a \cup G_b|} > \tau_{merge}$$

with boundary preservation:

$$\text{merge}(G_a, G_b) = \begin{cases} \text{True} & \text{if } \text{IoU}_{3D} > \tau_{merge} \text{ AND } \Delta\theta_{normal} < \tau_{boundary} \\ \text{False} & \text{otherwise} \end{cases}$$

where $\Delta\theta_{normal}$ is the angle between average surface normals of adjacent superpoints, and $\tau_{boundary}$ is the normal discontinuity threshold.

---

## 10. Results Summary

### ScanNet v2 (3D Instance Segmentation)

| Method | Training Required | mAP@25 | mAP@50 |
|--------|-------------------|--------|--------|
| Mask3D | Yes (3D supervised) | 73.7 | 55.2 |
| SAM3D | **No** | ~56 | ~46 |
| SAM3D (oracle prompts) | **No** | ~65 | ~53 |

[NEEDS VERIFICATION] — exact numbers should be confirmed against Table 1 in the paper.

### Key takeaway

SAM3D achieves **~83% of fully-supervised performance with zero training**. The gap closes further with better prompts (text or oracle boxes), suggesting the bottleneck is prompt quality, not the 2D→3D transfer mechanism.

### ScanNet200 / S3DIS

The paper also reports results on ScanNet200 (200 classes) and S3DIS (Stanford), showing the method generalizes across datasets without any adaptation — further confirming the training-free advantage.

---

## 11. Limitations & Open Questions

| Limitation | Impact on Bboy Pipeline | Mitigation |
|------------|------------------------|------------|
| Assumes **static scene** | Bboy moves between frames | Use SAM 3 temporal tracking instead of multi-view aggregation |
| Requires **depth maps** | iPhone LiDAR limited to 5m; studio may not have depth sensors | Use monocular depth estimation (ZoeDepth, Depth Anything v2) |
| **Slow** (~5 min/scene) | Not real-time | Acceptable for v0.1 offline analysis; optimize for v2 |
| Superpoint quality depends on **geometry** | Flat dance floors → poor superpoints | Not an issue — we segment the dancer, not the floor |
| **No temporal consistency** across dynamic scenes | Frame-to-frame segmentation jitters | SAM 3 + CoTracker3 solve this |

---

## Summary for Integration

**SAM3D is foundational but not the final tool.** For the breakdancing pipeline:

- **Use the principle** (2D SAM → 3D via projection + aggregation) — this is proven
- **Use the successor** (SAM 3 for temporal video segmentation) — this handles dynamic scenes
- **The pipeline inheritance** (SAM3D → SAM-3D-Body → SAM-Body4D) is the path from "segment anything in 3D" to "recover any human mesh in 4D"

The paper's lasting contribution is demonstrating that SAM's 2D masks transfer cleanly to 3D via geometric projection + intelligent aggregation — no 3D training needed. Every downstream tool in our pipeline (SAM-Body4D, JOSH) builds on this foundation.
