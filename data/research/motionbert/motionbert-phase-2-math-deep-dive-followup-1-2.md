# motionbert — Follow-up 1.2: The document reports 6.3M parameters but provides zero analysis of FLOPs per forward pass, memory fo

_Generated: 2026-03-23T02:01:51.234Z_

---

# MotionBERT Computational Complexity Analysis

This fills the gap identified in the math deep dive: FLOPs, memory footprint, and wall-clock inference profiling.

---

## 9. FLOPs Analysis (Forward Pass)

### Constants

$$B = 1, \quad T = 243, \quad J = 17, \quad D = 256, \quad d_k = d_v = 32, \quad n_h = 8, \quad D_{ff} = 1024, \quad L = 5$$

Total token count per sample:

$$N_{tok} = T \times J = 243 \times 17 = 4{,}131$$

### Equation 9.1: Spatial MHSA FLOPs (per block)

S-MHSA operates **per-frame**: for each of $T$ frames, attention is computed over $J=17$ tokens.

**QKV projections** (3 linear maps $[J, D] \to [J, D]$, run $T$ times):

$$F_{QKV}^{(s)} = T \cdot 3 \cdot 2JD^2 = 243 \times 3 \times 2 \times 17 \times 256^2 = 6{,}684{,}672 \times 243 = 1{,}624{,}375{,}296$$

**Attention map computation** ($\mathbf{Q}\mathbf{K}^\top$: $[J, d_k] \times [d_k, J]$, per head per frame):

$$F_{attn}^{(s)} = T \cdot n_h \cdot 2J^2 d_k = 243 \times 8 \times 2 \times 17^2 \times 32 = 35{,}956{,}224$$

**Attention-value product** ($[J, J] \times [J, d_k]$, per head per frame):

$$F_{AV}^{(s)} = T \cdot n_h \cdot 2J^2 d_k = 35{,}956{,}224$$

**Output projection** ($[J, D] \to [J, D]$, run $T$ times):

$$F_{out}^{(s)} = T \cdot 2JD^2 = 243 \times 2 \times 17 \times 256^2 = 541{,}458{,}432$$

**Total S-MHSA per block:**

$$\boxed{F_{S\text{-}MHSA} = F_{QKV}^{(s)} + F_{attn}^{(s)} + F_{AV}^{(s)} + F_{out}^{(s)} = 2{,}237{,}746{,}176 \approx 2.24 \text{ GFLOPs}}$$

Note: The attention map terms ($F_{attn}^{(s)} + F_{AV}^{(s)} \approx 72\text{M}$) are **3.2%** of S-MHSA cost. The linear projections dominate because $J=17$ is small — the $O(J^2)$ attention maps are cheap.

### Equation 9.2: Temporal MHSA FLOPs (per block)

T-MHSA operates **per-joint**: for each of $J$ joints, attention is computed over $T=243$ tokens.

**QKV projections** ($[T, D] \to [T, D]$, run $J$ times):

$$F_{QKV}^{(t)} = J \cdot 3 \cdot 2TD^2 = 17 \times 3 \times 2 \times 243 \times 256^2 = 1{,}624{,}375{,}296$$

**Attention map computation** ($[T, d_k] \times [d_k, T]$, per head per joint):

$$F_{attn}^{(t)} = J \cdot n_h \cdot 2T^2 d_k = 17 \times 8 \times 2 \times 243^2 \times 32 = 513{,}962{,}496$$

**Attention-value product:**

$$F_{AV}^{(t)} = J \cdot n_h \cdot 2T^2 d_k = 513{,}962{,}496$$

**Output projection:**

$$F_{out}^{(t)} = J \cdot 2TD^2 = 541{,}458{,}432$$

**Total T-MHSA per block:**

$$\boxed{F_{T\text{-}MHSA} = 3{,}193{,}758{,}720 \approx 3.19 \text{ GFLOPs}}$$

Here the attention map terms ($F_{attn}^{(t)} + F_{AV}^{(t)} \approx 1.03\text{G}$) are **32.1%** of T-MHSA cost — significantly larger than in S-MHSA because $T^2 = 59{,}049 \gg J^2 = 289$.

### Equation 9.3: Temporal-to-Spatial Attention FLOPs Ratio

$$\frac{F_{attn}^{(t)}}{F_{attn}^{(s)}} = \frac{J \cdot T^2}{T \cdot J^2} = \frac{T}{J} = \frac{243}{17} \approx 14.3\times$$

The temporal stream's attention maps require **14.3× more FLOPs** than the spatial stream's. This ratio is independent of $D$, $n_h$, and $d_k$ — it depends purely on the sequence length asymmetry.

### Equation 9.4: FFN FLOPs (per block)

Applied to all $N_{tok} = T \times J = 4{,}131$ tokens:

$$F_{FFN} = 2 \times N_{tok} \times D \times D_{ff} \times 2 = 2 \times 2 \times 4{,}131 \times 256 \times 1{,}024$$

$$\boxed{F_{FFN} = 4{,}331{,}667{,}456 \approx 4.33 \text{ GFLOPs}}$$

(Two linear layers: expand $D \to D_{ff}$ and compress $D_{ff} \to D$.)

### Equation 9.5: Total FLOPs per DSTformer Block

$$F_{block} = F_{S\text{-}MHSA} + F_{T\text{-}MHSA} + F_{FFN} + F_{gate}$$

The fusion gate FLOPs are negligible (GAP + one linear → sigmoid):

$$F_{gate} = N_{tok} \times D + D \approx 1{,}058{,}000 \approx 0.001 \text{ GFLOPs}$$

$$\boxed{F_{block} = 2.24 + 3.19 + 4.33 + 0.001 \approx 9.76 \text{ GFLOPs}}$$

**Breakdown by component:**

| Component | GFLOPs | % of block |
|-----------|--------|------------|
| S-MHSA | 2.24 | 22.9% |
| T-MHSA | 3.19 | 32.7% |
| FFN | 4.33 | 44.4% |
| Gate | 0.001 | ~0% |

The FFN is the single largest compute consumer per block, followed by T-MHSA, then S-MHSA. This is counterintuitive — one might expect the $O(T^2)$ attention to dominate, but at $T=243$ the quadratic term hasn't yet overtaken the linear projections (which scale as $O(N_{tok} \cdot D^2)$).

### Equation 9.6: Total Forward Pass FLOPs

$$F_{total} = F_{embed} + L \cdot F_{block} + F_{head}$$

$$F_{embed} = 2 \times N_{tok} \times C_{in} \times D = 2 \times 4{,}131 \times 2 \times 256 = 4{,}229{,}632 \approx 0.004 \text{ GFLOPs}$$

$$F_{head} = 2 \times N_{tok} \times D \times 3 = 2 \times 4{,}131 \times 256 \times 3 = 6{,}337{,}536 \approx 0.006 \text{ GFLOPs}$$

$$\boxed{F_{total} = 0.004 + 5 \times 9.76 + 0.006 \approx 48.8 \text{ GFLOPs}}$$

**Context**: For comparison, a single ResNet-50 forward pass on 224×224 image is ~4.1 GFLOPs. MotionBERT is **~12× more expensive than ResNet-50**, but operates on 243 frames of temporal context rather than a single image. Per-frame amortized cost: $48.8 / 243 \approx 0.2$ GFLOPs/frame — cheaper than ResNet-50 per frame.

### Equation 9.7: FLOPs Scaling with Temporal Window

$$F_{total}(T) = L \left[ 2 \cdot 4JD^2 T + 2n_h J^2 d_k T + 2n_h T^2 d_k J + 4TJD \cdot D_{ff} + 4JD^2 T + 2n_h T^2 d_k J \right]$$

Simplifying by collecting terms linear and quadratic in $T$:

$$F_{total}(T) = L \left[ \underbrace{(8JD^2 + 4JD \cdot D_{ff} + 2n_h J^2 d_k)}_{\alpha} \cdot T + \underbrace{4n_h J d_k}_{\beta} \cdot T^2 \right]$$

With concrete values:

$$\alpha = 8 \times 17 \times 256^2 + 4 \times 17 \times 256 \times 1024 + 2 \times 8 \times 289 \times 32$$

$$\alpha = 8{,}912{,}896 + 17{,}825{,}792 + 147{,}968 = 26{,}886{,}656$$

$$\beta = 4 \times 8 \times 17 \times 32 = 17{,}408$$

The **crossover point** where the $O(T^2)$ term equals the $O(T)$ term:

$$T^* = \frac{\alpha}{\beta} = \frac{26{,}886{,}656}{17{,}408} = 1{,}544 \text{ frames}$$

At $T = 243$, the quadratic term is only $\beta T^2 / (\alpha T + \beta T^2) = 17{,}408 \times 59{,}049 / (26{,}886{,}656 \times 243 + 17{,}408 \times 59{,}049) = 1.028\text{G} / (6.533\text{G} + 1.028\text{G}) \approx 13.6\%$ of total. **The linear term dominates at $T = 243$.**

At $T = 1{,}544$ (crossover), total FLOPs would be $\approx 830$ GFLOPs — a hypothetical 6× increase. The quadratic scaling only becomes the dominant cost for temporal windows exceeding ~1,500 frames (~50 seconds at 30fps).

---

## 10. Memory Footprint Analysis

### Equation 10.1: Parameter Memory

$$M_{params} = |\theta| \times b = 6.3\text{M} \times b$$

| Precision | Bytes/param ($b$) | Total |
|-----------|-------------------|-------|
| FP32 | 4 | 25.2 MB |
| FP16 / BF16 | 2 | 12.6 MB |
| INT8 | 1 | 6.3 MB |

### Equation 10.2: Temporal Attention Map Memory (The O(T²) Problem)

This is the critical memory consumer identified in the question. Per layer, per MHSA type:

**Temporal attention maps:**

$$M_{attn}^{(t)} = J \times n_h \times T^2 \times b = 17 \times 8 \times 243^2 \times b$$

| Precision | Values | Memory |
|-----------|--------|--------|
| FP32 (4B) | 8,024,136 | **30.7 MB** |
| FP16 (2B) | 8,024,136 | **15.4 MB** |

**Spatial attention maps:**

$$M_{attn}^{(s)} = T \times n_h \times J^2 \times b = 243 \times 8 \times 17^2 \times b$$

| Precision | Values | Memory |
|-----------|--------|--------|
| FP32 (4B) | 561,816 | **2.1 MB** |
| FP16 (2B) | 561,816 | **1.1 MB** |

**Ratio:**

$$\frac{M_{attn}^{(t)}}{M_{attn}^{(s)}} = \frac{J \cdot T^2}{T \cdot J^2} = \frac{T}{J} = 14.3\times$$

Temporal attention maps consume **14.3× more memory** than spatial attention maps. Across all 5 layers:

$$M_{attn,total}^{(t)} = 5 \times 30.7 = 153.5 \text{ MB (FP32)}$$

This is **the single largest memory allocation** during forward pass — larger than all model parameters combined (25.2 MB).

### Equation 10.3: Attention Score Count (Verifying the Claim)

Total attention scores per sample:

**Temporal:**

$$N_{attn}^{(t)} = T^2 \times n_h \times L \times J = 243^2 \times 8 \times 5 \times 17 = 59{,}049 \times 680 = \mathbf{40{,}153{,}320}$$

**Spatial:**

$$N_{attn}^{(s)} = J^2 \times n_h \times L \times T = 17^2 \times 8 \times 5 \times 243 = 289 \times 9{,}720 = \mathbf{2{,}809{,}080}$$

**Total:**

$$\boxed{N_{attn} = 40{,}153{,}320 + 2{,}809{,}080 = 42{,}962{,}400 \approx 43\text{M attention scores per sample}}$$

The question's estimate of "~40M before spatial" is confirmed: the temporal stream alone produces **40.15M** attention scores, and the spatial stream adds another **2.81M**. The temporal stream accounts for **93.5%** of all attention computation.

### Equation 10.4: Complete Activation Memory (per block, per sample)

During **forward pass** (inference), peak memory per block:

| Tensor | Shape | FP32 Size |
|--------|-------|-----------|
| Input $\mathbf{H}^{(l-1)}$ | $[T, J, D]$ = $[243, 17, 256]$ | 4.04 MB |
| S-MHSA Q, K, V | $3 \times [T, J, D]$ | 12.12 MB |
| S-MHSA attention maps | $[T, n_h, J, J]$ | 2.15 MB |
| T-MHSA Q, K, V | $3 \times [T, J, D]$ | 12.12 MB |
| T-MHSA attention maps | $[J, n_h, T, T]$ | **30.71 MB** |
| FFN intermediate | $[T, J, D_{ff}]$ = $[243, 17, 1024]$ | 16.15 MB |
| Layer norms (2×) | $2 \times [T, J, D]$ | 8.08 MB |
| Fusion outputs $\mathbf{Z}_s, \mathbf{Z}_t, \hat{\mathbf{H}}$ | $3 \times [T, J, D]$ | 12.12 MB |
| **Block total** | | **~97 MB** |

During **training** (must retain activations for backpropagation):

$$M_{train}^{(act)} \approx L \times M_{block} = 5 \times 97 = 485 \text{ MB per sample (FP32)}$$

With gradient checkpointing (recompute instead of store), this drops to $O(1)$ blocks stored ≈ $97$ MB, at the cost of ~33% more compute.

### Equation 10.5: Training Memory Budget

$$M_{train} = M_{params} + M_{grads} + M_{optimizer} + B \times M_{act}$$

| Component | Formula | FP32 Size |
|-----------|---------|-----------|
| Parameters | $6.3\text{M} \times 4$ | 25.2 MB |
| Gradients | $6.3\text{M} \times 4$ | 25.2 MB |
| Adam optimizer states | $6.3\text{M} \times 8$ (m + v) | 50.4 MB |
| Activations (per sample) | from Eq. 10.4 | 485 MB |

$$M_{train}(B) \approx 101 \text{ MB} + B \times 485 \text{ MB}$$

| Batch size | Total memory | Fits on... |
|------------|-------------|------------|
| B = 1 | 586 MB | Any modern GPU |
| B = 8 | 3.98 GB | GTX 1080 Ti (11GB) ✓ |
| B = 16 | 7.86 GB | V100 16GB ✓ |
| B = 32 | 15.62 GB | V100 32GB ✓ |
| B = 64 | 31.14 GB | V100 32GB ✗ (needs checkpointing or A100) |
| B = 128 | 62.18 GB | A100 80GB ✓ |

Mixed precision (FP16 forward/backward, FP32 master weights):

$$M_{mixed}(B) \approx 101 \text{ MB} + B \times 243 \text{ MB}$$

This roughly doubles the feasible batch size at each GPU memory tier.

### Equation 10.6: The Temporal Attention Map as Memory Bottleneck

Fraction of activation memory consumed by temporal attention maps:

$$\rho_{t\text{-}attn} = \frac{L \times M_{attn}^{(t)}}{M_{train}^{(act)}} = \frac{5 \times 30.7}{485} \approx 31.6\%$$

Nearly a third of all activation memory is temporal attention maps. If the temporal window were extended:

$$M_{attn}^{(t)}(T) = J \times n_h \times T^2 \times b = 17 \times 8 \times T^2 \times 4$$

| $T$ | Per-layer temporal attention | 5-layer total | Context duration (30fps) |
|-----|----------------------------|---------------|--------------------------|
| 243 | 30.7 MB | 153.5 MB | 8.1 s |
| 486 | 122.8 MB | 614 MB | 16.2 s |
| 729 | 276.3 MB | 1.38 GB | 24.3 s |
| 1024 | 545 MB | 2.73 GB | 34.1 s |

The $O(T^2)$ scaling means doubling the temporal window quadruples the attention map memory. Going from 243 to 1024 frames increases attention memory by **17.8×**.

---

## 11. Wall-Clock Inference Time

### Equation 11.1: Compute-Bound Lower Bound

$$t_{compute} = \frac{F_{total}}{\text{Peak FLOPS}} = \frac{48.8 \times 10^9}{\text{Peak FLOPS}}$$

| GPU | Peak FP32 (TFLOPS) | Peak FP16 (TFLOPS) | $t_{compute}$ FP32 | $t_{compute}$ FP16 |
|-----|--------------------|--------------------|-------|-------|
| V100 | 14.0 | 112 (Tensor) | 3.49 ms | 0.44 ms |
| A100 | 19.5 | 312 (Tensor) | 2.50 ms | 0.16 ms |
| RTX 3090 | 35.6 | 71 (Tensor) | 1.37 ms | 0.69 ms |
| RTX 4090 | 82.6 | 165 (Tensor) | 0.59 ms | 0.30 ms |

These are **theoretical lower bounds** assuming 100% compute utilization — unachievable in practice.

### Equation 11.2: Memory-Bandwidth-Bound Estimate

For inference (batch size 1), linear layers become memory-bandwidth-bound because the weight matrices must be loaded from DRAM for each forward pass but are only used against a single batch element.

Total weight memory that must be streamed per forward pass:

$$M_{weights} = |\theta| \times b = 6.3\text{M} \times 4 = 25.2 \text{ MB (FP32)}$$

Total activation read/write traffic per block (rough estimate):

$$M_{act\text{-}traffic}^{(block)} \approx 2 \times (M_{QKV}^{(s)} + M_{attn}^{(s)} + M_{QKV}^{(t)} + M_{attn}^{(t)} + M_{FFN}) \approx 2 \times 73 \approx 146 \text{ MB}$$

(Factor of 2 for read + write cycles.)

Total memory traffic:

$$M_{traffic} \approx L \times (M_{act\text{-}traffic}^{(block)} + M_{weights}^{(block)}) \approx 5 \times (146 + 10) \approx 780 \text{ MB}$$

| GPU | Bandwidth (GB/s) | $t_{bandwidth}$ |
|-----|-------------------|-----------------|
| V100 | 900 | 0.87 ms |
| A100 | 2,039 | 0.38 ms |
| RTX 4090 | 1,008 | 0.77 ms |

### Equation 11.3: Roofline Analysis

The **operational intensity** (FLOPs per byte of memory traffic) determines whether a kernel is compute-bound or memory-bound:

$$I = \frac{F_{total}}{M_{traffic}} = \frac{48.8 \times 10^9}{780 \times 10^6} \approx 62.6 \text{ FLOPs/byte}$$

Ridge points:

| GPU | Ridge point (FP32) | Ridge point (FP16 Tensor) | MotionBERT @ 62.6 | Regime |
|-----|-------------------|--------------------------|-------------------|--------|
| V100 | 15.6 | 124.4 | 62.6 | Compute-bound (FP32), near-ridge (FP16) |
| A100 | 9.6 | 153.0 | 62.6 | Compute-bound (FP32), memory-bound (FP16) |
| RTX 4090 | 81.9 | 163.7 | 62.6 | Memory-bound (FP32 & FP16) |

At batch size 1, the operational intensity is misleading because the weight-loading cost amortizes poorly. Increasing batch size improves utilization:

$$I(B) = \frac{B \times 48.8 \times 10^9}{B \times M_{act\text{-}traffic} + M_{weights\text{-}traffic}} \approx \frac{48.8B}{0.73B + 0.05} \text{ GFLOPs/GB}$$

At $B = 1$: $I \approx 62.6$. At $B = 32$: $I \approx 66.9$ (modest improvement, since activations dominate over weights).

### Equation 11.4: Realistic Inference Latency (Empirical Model)

Real GPU utilization for transformer inference at this scale is typically 30–60% of peak (due to kernel launch overhead, memory access patterns, attention softmax being bandwidth-bound, and operator fusion limits). Using a utilization factor $\eta$:

$$t_{real} \approx \frac{F_{total}}{\eta \times \text{Peak FLOPS}}$$

| GPU | $\eta$ (typical) | $t_{real}$ FP32 | $t_{real}$ FP16 |
|-----|-------------------|-----------------|-----------------|
| V100 | 0.35–0.50 | 7–10 ms | 0.9–1.3 ms |
| A100 | 0.40–0.55 | 4.5–6.3 ms | 0.3–0.4 ms |
| RTX 4090 | 0.35–0.50 | 1.2–1.7 ms | 0.6–0.9 ms |

For the breaking pipeline's target hardware (RTX 4090):

$$\boxed{t_{inference}^{4090} \approx 1\text{–}2 \text{ ms (FP32)}, \quad 0.6\text{–}0.9 \text{ ms (FP16)}}$$

Per 243-frame window (8.1 seconds of video at 30fps), this is **< 0.025% of real-time** — MotionBERT is not the inference bottleneck in any pipeline.

### Equation 11.5: Throughput and Real-Time Analysis

**Sliding window throughput** (stride-1, processing every frame):

For real-time 30fps processing, each frame must complete in $< 33.3$ ms. With stride-$s$ sliding window:

$$t_{per\text{-}frame} = \frac{t_{inference}}{s}$$

At stride-1 on RTX 4090 (FP16): $t_{per\text{-}frame} = 0.7$ ms — supports **~1,400 fps equivalent throughput**, or **47× real-time**.

**Amortized cost per frame** (non-overlapping windows, stride = $T$):

$$t_{amortized} = \frac{t_{inference}}{T} = \frac{0.7}{243} \approx 0.003 \text{ ms/frame}$$

**Practical batch inference** (offline processing of a full breaking battle):

A 5-minute battle at 30fps = 9,000 frames = $\lceil 9000 / 243 \rceil = 38$ windows.

$$t_{battle} = 38 \times t_{inference} \approx 38 \times 0.7 \text{ ms} = 26.6 \text{ ms}$$

The entire 5-minute battle's 3D pose estimation completes in **~27 ms** on RTX 4090 (FP16, batch=1 sequential). With batch processing ($B = 38$ fits comfortably in memory at ~38 × 485 MB / 2 ≈ 9.2 GB FP16):

$$t_{battle}^{(batched)} \approx t_{inference}(B=38) \approx 5\text{–}10 \text{ ms}$$

---

## 12. Comparative Complexity Across the Pipeline

### Equation 12.1: MotionBERT vs. Upstream/Downstream Cost

For context within the bboy analysis pipeline:

| Component | Input | GFLOPs | Latency (4090 FP16) | Memory |
|-----------|-------|--------|---------------------|--------|
| YOLOv8-Pose (2D detection) | 640×640 image | ~7 per frame | ~2 ms/frame | ~50 MB |
| MotionBERT (2D→3D lifting) | 243×17×2 tensor | 48.8 per window | ~0.7 ms/window | ~500 MB |
| **MotionBERT amortized** | per frame | **0.2** | **0.003 ms** | — |
| Movement spectrogram (FFT) | 243×17×3 | ~0.001 | ~0.01 ms | ~1 MB |

**Key insight**: MotionBERT's 48.8 GFLOPs is concentrated in a single forward pass over 243 frames. The **per-frame amortized cost (0.2 GFLOPs)** is 35× cheaper than a single YOLOv8-Pose frame. The 2D pose detector, not the 3D lifter, is the computational bottleneck in the pipeline.

### Equation 12.2: Attention Memory as Function of Skeleton Size

For breaking-specific analysis with extended skeletons (e.g., 25-joint COCO-body25 or 133-joint whole-body):

$$M_{attn}^{(s)}(J) \propto J^2, \qquad M_{attn}^{(t)}(J) \propto J$$

| Skeleton | $J$ | $M_{attn}^{(s)}$ per layer | $M_{attn}^{(t)}$ per layer | Ratio change |
|----------|-----|---------------------------|---------------------------|-------------|
| H36M | 17 | 2.1 MB | 30.7 MB | baseline |
| COCO-25 | 25 | 4.6 MB | 45.2 MB | 1.47× total |
| COCO-133 (whole-body) | 133 | 130 MB | 240 MB | 7.0× total |

Whole-body 133-joint estimation with MotionBERT architecture would require **370 MB of attention maps per layer** (1.85 GB across 5 layers), pushing against V100 16GB limits at batch size > 4.

---

## 13. The Dual-Stream Overhead

### Equation 13.1: Dual-Stream vs. Sequential-Stream Cost

The DSTformer runs S-MHSA and T-MHSA in **parallel** (conceptually — both read from the same $\mathbf{H}^{(l-1)}$). A sequential design (S→T or T→S) would have the same total FLOPs but different memory and parallelism characteristics.

**FLOPs**: Identical between dual-stream and sequential. Both compute S-MHSA + T-MHSA; only the data flow differs.

**Peak memory**: Dual-stream requires both $\mathbf{Z}_s$ and $\mathbf{Z}_t$ to coexist simultaneously for the fusion gate:

$$M_{peak}^{(dual)} = M_{SMHSA} + M_{TMHSA} + M_{fusion}$$

Sequential only needs one stream's activations at a time:

$$M_{peak}^{(seq)} = \max(M_{SMHSA}, M_{TMHSA}) + M_{residual}$$

The overhead:

$$\Delta M = M_{peak}^{(dual)} - M_{peak}^{(seq)} \approx \min(M_{SMHSA\text{-}act}, M_{TMHSA\text{-}act})$$

$$\approx 12 + 2.1 = 14.1 \text{ MB per block (the S-MHSA intermediate activations)}$$

$$\Delta M_{total} = 5 \times 14.1 = 70.5 \text{ MB}$$

This is a **14.5% memory overhead** ($70.5 / 485$) for the dual-stream design — a modest cost for the architectural flexibility of learned fusion.

**GPU parallelism**: On GPUs with multiple SMs, the dual streams can theoretically execute concurrently (different CUDA streams). In practice, PyTorch's default execution mode is sequential within a forward pass, so this parallelism is only realized with explicit `torch.cuda.Stream` usage or graph-mode compilation. The paper's codebase does not exploit this.

---

## 14. Training FLOPs and Cost

### Equation 14.1: Training FLOPs Budget

Training involves forward + backward pass. The backward pass is approximately 2× the forward pass FLOPs (gradient computation through the same operations, plus gradient accumulation):

$$F_{train\text{-}step} = 3 \times F_{forward} \times B = 3 \times 48.8 \times B \text{ GFLOPs}$$

From the paper: training on Human3.6M for 120 epochs, ~1.5M training samples, batch size 32:

$$\text{Steps} = \frac{120 \times 1{,}500{,}000}{32} \approx 5{,}625{,}000$$

$$F_{train\text{-}total} = 5{,}625{,}000 \times 3 \times 48.8 \times 32 \approx 2.63 \times 10^{16} \text{ FLOPs} = 26.3 \text{ PFLOPs}$$

On 8× V100 (112 TFLOPS FP16 each, ~40% utilization):

$$t_{train} = \frac{26.3 \times 10^{15}}{8 \times 112 \times 10^{12} \times 0.4} \approx 73{,}300 \text{ s} \approx 20.4 \text{ hours}$$

Consistent with the paper's reported training time of ~24 hours on 8× V100 (the slight discrepancy accounts for data loading, validation, and I/O overhead).

Pretraining on AMASS (~30-40 hours reported) involves more data but similar architecture, aligning with roughly 1.5–2× the fine-tuning compute budget.

---

## Summary: Key Numerical Results

| Metric | Value | Note |
|--------|-------|------|
| **Total FLOPs (forward)** | **48.8 GFLOPs** | Per 243-frame window |
| Amortized FLOPs per frame | 0.2 GFLOPs | 35× cheaper than 2D detector |
| Attention scores per sample | **43M** (40.2M temporal + 2.8M spatial) | 93.5% from temporal stream |
| Temporal attention memory (per layer) | **30.7 MB** (FP32) | O(T²), 14.3× spatial |
| Total activation memory (training) | **485 MB/sample** (FP32) | 31.6% is temporal attention |
| Inference latency (RTX 4090 FP16) | **0.6–0.9 ms** | Per 243-frame window |
| Real-time throughput | **47× real-time** | At stride-1 on 4090 |
| O(T²) crossover | **T = 1,544 frames** | Below this, linear terms dominate |
| Dual-stream memory overhead | **14.5%** | vs. sequential S→T or T→S |
| Training cost | **~26 PFLOPs** | ~24h on 8×V100 |
| 5-min battle processing | **~27 ms** | Sequential; ~5-10 ms batched |

The most critical takeaway for the breaking pipeline: **MotionBERT is emphatically not a computational bottleneck.** At 0.003 ms per frame amortized on RTX 4090, the 3D lifting step is ~700× faster than the upstream 2D pose detection. The $O(T^2)$ temporal attention scaling only becomes problematic beyond ~1,500-frame windows (~50 seconds), well above the 243-frame design point. The primary concern is accuracy degradation on out-of-distribution poses (§8.2 in the math deep dive), not compute.
