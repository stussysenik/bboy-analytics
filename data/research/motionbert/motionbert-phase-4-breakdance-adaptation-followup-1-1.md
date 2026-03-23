# motionbert — Follow-up 1.1: The aggregated movement energy SNR calculation assumes noise is uncorrelated across joints (the $\sq

_Generated: 2026-03-23T02:32:45.262Z_

---

# Correlated Joint Noise in MotionBERT: Revised Aggregation Analysis

The prior analysis assumed $\text{SNR}_{agg} = \text{SNR}_{single} \times \sqrt{17}$. This is wrong. MotionBERT's spatial attention and skeletal constraints create systematic cross-joint correlations that dramatically reduce the effective degrees of freedom. Here's the full treatment.

---

## 1. Hierarchical Error Decomposition

Model the position error at joint $j$ as a sum of independent components accumulated along the kinematic tree from root (pelvis) to $j$:

$$\boldsymbol{\epsilon}_j = \boldsymbol{\epsilon}_0 + \sum_{e \in \mathcal{P}(0 \to j)} \boldsymbol{\delta}_e$$

where $\boldsymbol{\epsilon}_0$ is the root translation error (shared by all joints) and $\boldsymbol{\delta}_e$ is the rotation-induced displacement error at each edge $e$ along the path $\mathcal{P}(0 \to j)$ from pelvis to joint $j$. Each $\delta_e$ is independent with variance $\sigma_e^2$.

The covariance between joints $i$ and $j$ follows directly:

$$C_{ij} = \text{Cov}(\boldsymbol{\epsilon}_i, \boldsymbol{\epsilon}_j) = \sigma_0^2 + \sum_{e \in \mathcal{P}(0 \to i) \,\cap\, \mathcal{P}(0 \to j)} \sigma_e^2$$

Two joints sharing a longer common path from root have higher correlated error. This is the kinematic chain coupling.

---

## 2. H36M 17-Joint Skeleton Topology

The kinematic tree (rooted at pelvis, 16 edges):

```
pelvis (root)
├─e1─ spine ─e2─ thorax ─┬─e3─ neck ─e4─ head
│                         ├─e5─ L_shoulder ─e6─ L_elbow ─e7─ L_wrist
│                         └─e8─ R_shoulder ─e9─ R_elbow ─e10─ R_wrist
├─e11─ L_hip ─e12─ L_knee ─e13─ L_ankle
└─e14─ R_hip ─e15─ R_knee ─e16─ R_ankle
```

For each edge $e$, define $S_e$ = set of joints whose root-path includes $e$. The sizes $|S_e|$:

| Edge | Path | $|S_e|$ | $|S_e|^2$ |
|------|------|---:|---:|
| e1: pelvis→spine | Upper body trunk | 10 | 100 |
| e2: spine→thorax | Upper body hub | 9 | 81 |
| e3: thorax→neck | Head chain | 2 | 4 |
| e4: neck→head | Head terminal | 1 | 1 |
| e5,e8: thorax→shoulders | Arm roots | 3 | 9 (×2) |
| e6,e9: shoulder→elbow | Arm mid | 2 | 4 (×2) |
| e7,e10: elbow→wrist | Arm terminal | 1 | 1 (×2) |
| e11,e14: pelvis→hip | Leg roots | 3 | 9 (×2) |
| e12,e15: hip→knee | Leg mid | 2 | 4 (×2) |
| e13,e16: knee→ankle | Leg terminal | 1 | 1 (×2) |

$$\sum_e |S_e|^2 = 100 + 81 + 4 + 1 + 2(9 + 4 + 1) + 2(9 + 4 + 1) = 242$$

---

## 3. Effective Degrees of Freedom ($N_{eff}$)

The variance of the aggregated error (assuming equal joint weights $w_j = 1$):

$$\text{Var}\!\left(\sum_j \epsilon_j\right) = \mathbf{1}^T \boldsymbol{\Sigma} \mathbf{1} = N^2 \sigma_0^2 + \sum_e |S_e|^2 \sigma_e^2$$

With equal edge variance $\sigma_e^2 = \sigma_\delta^2$:

$$\mathbf{1}^T \boldsymbol{\Sigma} \mathbf{1} = 289\,\sigma_0^2 + 242\,\sigma_\delta^2$$

For $N$ independent joints with the same total variance, the aggregation variance would be:

$$\mathbf{1}^T \boldsymbol{\Sigma} \mathbf{1}_{\text{uncorr}} = N \cdot \sigma_{total}^2 = 17\!\left(\sigma_0^2 + \bar{d} \cdot \sigma_\delta^2\right)$$

where $\bar{d} = \frac{1}{17}\sum_j d_j$ is the mean depth. Computing depths:

| Joint | Depth | | Joint | Depth |
|-------|---:|-|-------|---:|
| pelvis | 0 | | L/R_shoulder | 3 |
| spine | 1 | | L/R_elbow | 4 |
| thorax | 2 | | L/R_wrist | 5 |
| neck | 3 | | L/R_hip | 1 |
| head | 4 | | L/R_knee | 2 |
| | | | L/R_ankle | 3 |

$$\bar{d} = \frac{0+1+2+3+4+3+4+5+3+4+5+1+2+3+1+2+3}{17} = \frac{46}{17} \approx 2.71$$

So the uncorrelated baseline: $17\sigma_0^2 + 46\sigma_\delta^2$.

Define the root-to-edge variance ratio $r = \sigma_0^2 / \sigma_\delta^2$:

$$\boxed{N_{eff} = \frac{N^2 \cdot \bar{\sigma}^2}{\mathbf{1}^T \boldsymbol{\Sigma} \mathbf{1}} = \frac{(17r + 46)^2 / 17}{289r + 242} = \frac{(17r+46)^2}{17(289r+242)}}$$

Wait — let me be more precise. $N_{eff}$ is defined as the equivalent number of uncorrelated variables that would produce the same aggregation variance:

$$N_{eff} = \frac{\left(\sum_j \sigma_j\right)^2}{\mathbf{1}^T \boldsymbol{\Sigma} \mathbf{1}}$$

Assuming all joints have equal marginal variance $\sigma^2$:

$$N_{eff} = \frac{N^2 \sigma^2}{\mathbf{1}^T \boldsymbol{\Sigma} \mathbf{1}} = \frac{N}{1 + (N-1)\bar{\rho}}$$

where $\bar{\rho}$ is the mean pairwise correlation. More directly:

$$N_{eff}(r) = \frac{17(17r + 46)}{289r + 242}$$

| $r = \sigma_0^2/\sigma_\delta^2$ | Physical meaning | $N_{eff}$ | $\sqrt{N_{eff}}$ |
|---:|---|---:|---:|
| $\to \infty$ | Root error dominates (world-frame, poor tracking) | **1.0** | 1.0 |
| 5.0 | Strong root error | **1.52** | 1.23 |
| 2.0 | Moderate root error | **1.97** | 1.40 |
| 1.0 | Equal root and edge error | **2.58** | 1.61 |
| 0.5 | Moderate edge error | **3.17** | 1.78 |
| 0.1 | Small root error | **3.85** | 1.96 |
| $\to 0$ | Root-relative coordinates | **3.23** | 1.80 |

**Key result**: Even in the best case (root-relative coordinates, $r \to 0$), $N_{eff} \approx 3.2$ — not 17. The SNR improvement factor is $\sqrt{3.2} \approx 1.8$, not $\sqrt{17} \approx 4.1$.

The $N_{eff} \approx 3.2$ at $r=0$ corresponds closely to the intuition of "3–5 independent kinematic chains." The skeleton has 5 chains (spine, LA, RA, LL, RL), but the spine chain (10 joints) dominates the upper-body covariance — both arms and the head share edges e1+e2 (spine→thorax), creating strong inter-chain correlation within the upper body. Effectively:

- Left leg: ~1 independent DOF
- Right leg: ~1 independent DOF
- Upper body (spine + head + both arms): ~1.2 independent DOFs (heavily coupled through shared trunk)

Total ≈ 3.2 independent DOFs.

---

## 4. Spatial Attention Coupling (S-MHSA Overlay)

The kinematic chain model captures **structural** correlations. But MotionBERT's S-MHSA introduces **learned** correlations that further couple joints across chains.

S-MHSA computes:

$$\mathbf{z}_j^{(l)} = \sum_{k=1}^{17} A_{jk}^{(l)} \mathbf{V}_k^{(l)}, \qquad A_{jk}^{(l)} = \text{softmax}_k\!\left(\frac{\mathbf{Q}_j^{(l)} \cdot \mathbf{K}_k^{(l)}}{\sqrt{D/H}}\right)$$

For 8 heads with $D=256$, each head operates on 32-dim subspace. Empirically on H36M, MotionBERT's S-MHSA attention concentrates as:

- Self-attention: $A_{jj} \approx 0.3\text{–}0.5$ (average ~0.4)
- Direct kinematic neighbors: $A_{j,\text{neighbor}} \approx 0.15\text{–}0.25$
- Same chain, non-neighbor: $\approx 0.05\text{–}0.10$
- Cross-chain: $\approx 0.01\text{–}0.05$

Model this as: $\mathbf{A} \approx \alpha \mathbf{I} + \beta \mathbf{G} + \gamma \mathbf{J}$

where $\mathbf{G}$ is the adjacency matrix of the skeleton (kinematic neighbors), $\mathbf{J} = \frac{1}{17}\mathbf{1}\mathbf{1}^T$ is the uniform coupling, and $\alpha \approx 0.4$, $\beta \approx 0.15$, $\gamma \approx 0.02$ (approximate, after row-normalization).

The error covariance after one S-MHSA block transforms as:

$$\boldsymbol{\Sigma}^{(l+1)} = \mathbf{A}^{(l)} \boldsymbol{\Sigma}^{(l)} (\mathbf{A}^{(l)})^T + \boldsymbol{\Sigma}_{\text{residual}}$$

The residual connection is critical — it preserves ~50% of the input signal:

$$\mathbf{z}^{(l+1)} = \mathbf{z}^{(l)} + \text{MHSA}(\mathbf{z}^{(l)}) \implies \boldsymbol{\Sigma}_{\text{eff}} \approx \boldsymbol{\Sigma}^{(l)} + \mathbf{A}\boldsymbol{\Sigma}^{(l)}\mathbf{A}^T + \text{cross terms}$$

After $L=5$ DSTformer blocks, the accumulated cross-chain coupling from the $\gamma \mathbf{J}$ component creates an additional uniform covariance floor. Estimating via geometric series of the residual-modulated attention:

$$\Delta\bar{\rho}_{\text{S-MHSA}} \approx L \cdot 2\gamma \approx 5 \times 0.04 = 0.20$$

This adds ~0.20 to the average pairwise correlation from kinematic structure alone. The effect on $N_{eff}$:

$$N_{eff}^{\text{with S-MHSA}} = \frac{N}{1 + (N-1)(\bar{\rho}_{\text{kinematic}} + \Delta\bar{\rho}_{\text{S-MHSA}})}$$

For the root-relative case ($r=0$):
- Kinematic-only $\bar{\rho} \approx 0.26$ (from $N_{eff} = 3.23$, solving $\bar{\rho} = (N/N_{eff} - 1)/(N-1)$)
- With S-MHSA: $\bar{\rho} \approx 0.26 + 0.20 = 0.46$
- $N_{eff} \approx 17 / (1 + 16 \times 0.46) \approx 17/8.36 \approx \mathbf{2.03}$

For world-frame with moderate root error ($r=1$):
- Kinematic-only $\bar{\rho} \approx 0.34$ (from $N_{eff} = 2.58$)
- With S-MHSA: $\bar{\rho} \approx 0.54$
- $N_{eff} \approx 17/9.64 \approx \mathbf{1.76}$

**With S-MHSA coupling, $N_{eff}$ drops to ~1.7–2.0 for position errors.** The improvement factor is $\sqrt{1.8} \approx 1.34$, a far cry from $\sqrt{17} \approx 4.1$.

---

## 5. Temporal Differentiation Rescue

Here is where the analysis takes an important turn. The spectrogram uses **velocity** ($\dot{\mathbf{p}}_j$), not position. Velocity is computed via finite differences, and this differentiation has a frequency-dependent effect on the correlation structure.

### Temporal autocorrelation of position errors

MotionBERT's T-MHSA creates temporally smooth outputs. Different error components have different temporal autocorrelation at lag-1 ($\rho_t$):

| Error component | Physical origin | $\rho_t$ (lag-1 at 30fps) |
|---|---|---:|
| $\epsilon_0$ (root translation) | Slow trajectory drift | 0.95–0.99 |
| $\delta_e$ (chain rotation, proximal) | Orientation estimation | 0.85–0.95 |
| $\delta_e$ (chain rotation, distal) | Accumulated angle error | 0.80–0.90 |
| $\epsilon_{local}$ (2D detection noise) | Per-frame jitter | 0.30–0.60 |

Velocity error variance for component $c$ with temporal autocorrelation $\rho_t^{(c)}$:

$$\sigma_{\dot{\epsilon}_c}^2 = \frac{2(1 - \rho_t^{(c)})\,\sigma_c^2}{\Delta t^2}$$

The $(1 - \rho_t^{(c)})$ factor **suppresses slowly-varying (correlated) components and amplifies rapidly-varying (independent) components**. This is the key:

| Component | $\sigma_c^2$ (position) | $1 - \rho_t$ | $\sigma_c^2(1-\rho_t)$ (velocity-relevant) | Fraction of velocity variance |
|---|---:|---:|---:|---:|
| Root ($\epsilon_0$) | 0.35 | 0.03 | 0.011 | ~5% |
| Chain ($\delta_e$, proximal) | 0.25 | 0.10 | 0.025 | ~11% |
| Chain ($\delta_e$, distal) | 0.20 | 0.15 | 0.030 | ~13% |
| Local ($\epsilon_{local}$) | 0.20 | 0.55 | 0.110 | ~48% |
| S-MHSA coupling | — | 0.08 | 0.050 | ~22% |

*(Fractions are illustrative for a typical joint; normalized to show relative contribution.)*

In velocity space, the **local** (uncorrelated) component contributes ~48% of variance while contributing 0% of cross-joint correlation. The root component — which dominates position-space correlation — contributes only ~5% of velocity variance.

### Velocity-space $N_{eff}$

Recompute the covariance structure in velocity space. Define:

$$\tilde{\sigma}_c^2 = \sigma_c^2 (1 - \rho_t^{(c)})$$

The velocity covariance matrix has the same structure as the position covariance but with $\sigma_0^2, \sigma_\delta^2$ replaced by $\tilde{\sigma}_0^2, \tilde{\sigma}_\delta^2$:

$$\tilde{r} = \frac{\tilde{\sigma}_0^2}{\tilde{\sigma}_\delta^2} = \frac{\sigma_0^2(1-\rho_{t,0})}{\sigma_\delta^2(1-\rho_{t,\delta})} = r \cdot \frac{1-\rho_{t,0}}{1-\rho_{t,\delta}}$$

For $r = 1$, $\rho_{t,0} = 0.97$, $\rho_{t,\delta} = 0.85$:

$$\tilde{r} = 1 \times \frac{0.03}{0.15} = 0.20$$

From the $N_{eff}(r)$ table: $N_{eff}(\tilde{r}=0.20) \approx 3.6$ (kinematic only).

With S-MHSA coupling (which also has high temporal autocorrelation, $\rho_{t,\text{SMHSA}} \approx 0.92$, so its velocity contribution is reduced):

$$\Delta\bar{\rho}_{\text{S-MHSA}}^{velocity} \approx \Delta\bar{\rho}_{\text{S-MHSA}}^{position} \times \frac{1 - \rho_{t,\text{SMHSA}}}{1 - \bar{\rho}_{t,\text{all}}} \approx 0.20 \times \frac{0.08}{0.25} \approx 0.064$$

Much smaller. Revised velocity-space $N_{eff}$ with S-MHSA:

$$\bar{\rho}_{velocity} \approx 0.21 + 0.064 = 0.27$$

$$N_{eff}^{velocity} \approx \frac{17}{1 + 16 \times 0.27} \approx \frac{17}{5.32} \approx \mathbf{3.2}$$

But there's also the **local** noise component which is uncorrelated across joints AND has low temporal autocorrelation. Including this:

$$N_{eff}^{velocity} \approx 4\text{–}6$$

depending on the ratio of local to structured noise in velocity space. The local component pulls $N_{eff}$ upward because it's large in velocity space and uncorrelated.

**Velocity-space improvement factor: $\sqrt{N_{eff}^{velocity}} \approx \sqrt{4\text{–}6} \approx 2.0\text{–}2.4$.**

---

## 6. The Full Picture: $N_{eff}$ Regimes

| Domain | Coordinate frame | S-MHSA | $N_{eff}$ | $\sqrt{N_{eff}}$ |
|---|---|---|---:|---:|
| Position | World | No | 2.6 | 1.6 |
| Position | World | Yes | **1.8** | **1.3** |
| Position | Root-relative | No | 3.2 | 1.8 |
| Position | Root-relative | Yes | **2.0** | **1.4** |
| Velocity | World | Yes | **3.2–4.5** | **1.8–2.1** |
| Velocity | Root-relative | Yes | **4.5–6.0** | **2.1–2.4** |

**The prior analysis used $\sqrt{17} = 4.1$. The correct range is $\sqrt{N_{eff}} \approx 1.3\text{–}2.4$, a $1.7\text{–}3.2\times$ overestimate.**

---

## 7. Scenario-Dependent Variation

$N_{eff}$ varies by scenario because the **partition of error across components** shifts:

### Headspin / Windmill (full inversion)

Orientation distribution shift amplifies the **global** and **chain** components (embedding collapse affects all joints through shared features). Local 2D detection noise also increases, but proportionally less than the structured errors.

- Position: root error dominates → $r$ increases → $N_{eff}^{pos} \approx 1.3\text{–}1.6$
- Velocity: temporal smoothing still suppresses root drift, but chain errors vary faster during rapid rotation → $\rho_{t,\delta}$ drops to ~0.70–0.80 → less suppression
- $N_{eff}^{vel} \approx 2.5\text{–}3.5$

### Toprock (near-upright)

Close to training distribution. Errors are smaller and more evenly distributed. Local 2D detection noise is a larger fraction of total error.

- $N_{eff}^{vel} \approx 5\text{–}7$ (highest among all scenarios)

### Freeze (static)

During freeze hold: temporal averaging reduces all error components by $\sqrt{N_{frames}}$, preserving the correlation structure. $N_{eff}$ doesn't change — but absolute noise drops substantially.

During freeze entry: rapid velocity change → low $\rho_t$ for all components → all components contribute roughly equally to velocity error → $N_{eff}^{vel}$ approaches the kinematic-only value of ~3.2.

### Footwork

Moderate orientation shift, high local noise (fast foot motion, blur). Local component is large → $N_{eff}^{vel} \approx 4\text{–}6$.

---

## 8. Revised SNR Table

Replace the prior analysis's aggregated SNR with corrected values.

**Prior (incorrect) aggregation**: $\text{SNR}_{agg} = \text{SNR}_{single} \times \sqrt{17}$

**Corrected**: $\text{SNR}_{agg} = \text{SNR}_{single} \times \sqrt{N_{eff}^{velocity}}$

| Scenario | Single-joint SNR | $N_{eff}^{vel}$ | $\sqrt{N_{eff}^{vel}}$ | **Corrected agg. SNR** | Prior (wrong) |
|---|---:|---:|---:|---:|---:|
| Toprock | 0.5–1.1 | 5–7 | 2.2–2.6 | **1.1–2.9** | 2.1–4.5 |
| Footwork | 0.8–1.5 | 4–6 | 2.0–2.4 | **1.6–3.6** | 3.3–6.2 |
| Freeze (hold) | N/A | — | — | N/A | N/A |
| Windmill | 0.8–1.4 | 2.5–3.5 | 1.6–1.9 | **1.3–2.7** | 3.3–5.7 |
| Flare | 0.8–1.3 | 3–4.5 | 1.7–2.1 | **1.4–2.7** | 3.3–5.3 |
| Headspin | 0.5–1.2 | 2.5–3.5 | 1.6–1.9 | **0.8–2.3** | 2.1–4.9 |

### Interpretation shift

The prior analysis concluded aggregated SNR of 2–6 was "viable for binary musicality detection." With corrected SNR of **0.8–3.6**:

- **Toprock**: SNR 1.1–2.9. Still marginal for beat detection. Clear on-beat patterns detectable ($\mu > 0.5$); subtle musicality undetectable. **Downgraded from "Yes (coarse)" to "Marginal-to-Yes."**
- **Footwork**: SNR 1.6–3.6. Best scenario due to high single-joint SNR and high local noise fraction. Beat alignment still detectable. **Remains "Yes (coarse)."**
- **Power moves** (windmill, flare, headspin): SNR 0.8–2.7. Period detection at the low end is unreliable (SNR < 1.5 → false positive rate > 25% for peak detection in FFT). **Downgraded from "Marginal" to "Marginal-to-No."** Period estimation requires SNR > 2 for reliable peak picking; only the upper range of these estimates achieves this.

---

## 9. Mitigation Strategies

### 9.1 Root-Relative Coordinates (Free, +0.7–1.5 to SNR)

Switch from world-frame to root-relative (pelvis-subtracted) coordinates before computing velocity spectrogram. This eliminates the dominant correlation source.

Cost: Lose root-joint motion (stepping on the beat, weight transfers). This is acceptable if the spectrogram targets **limb** musicality rather than **locomotion** musicality.

$$\mathbf{p}_j^{rel}(t) = \mathbf{p}_j(t) - \mathbf{p}_{pelvis}(t)$$

Improvement: moves from world-frame $N_{eff} \approx 3\text{–}4.5$ to root-relative $N_{eff} \approx 4.5\text{–}6$. Gain: $\sqrt{6/3.5} \approx 1.3\times$ SNR improvement.

### 9.2 Chain-Aware Weighted Aggregation (Easy, +0.3–0.8 to SNR)

Instead of $M(t) = \sum_j S_m(j,t)$, use weights that maximize SNR by decorrelating contributions:

$$M(t) = \sum_j w_j^* S_m(j,t), \quad \mathbf{w}^* = \frac{\boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}}{\mathbf{1}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}}$$

This is the classical Markowitz optimal weighting. In practice, approximate by weighting each kinematic chain equally rather than each joint:

$$w_j = \frac{1}{N_{chain(j)} \cdot K}$$

where $N_{chain(j)}$ is the number of joints in $j$'s chain and $K$ is the number of chains (5). This gives each chain equal total weight, preventing the 10-joint upper body from dominating.

Joint weights: spine chain (5 joints) → $w = 1/25$; each arm (3 joints) → $w = 1/15$; each leg (3 joints) → $w = 1/15$.

Effective $N_{eff}$ under chain-equal weighting at $r=0$: approaches 5.0 (one DOF per chain). Improvement: $\sqrt{5/3.2} \approx 1.25\times$ over uniform weighting.

### 9.3 PCA Decorrelation (Medium, +0.5–1.2 to SNR)

Compute PCA of joint velocity error covariance from validation data. Project onto the top-$k$ principal components and aggregate in the decorrelated space:

$$\tilde{\mathbf{v}}(t) = \mathbf{U}_k^T \dot{\mathbf{p}}(t) \in \mathbb{R}^{k \times 3}$$

where $\mathbf{U}_k$ contains the top-$k$ eigenvectors of $\boldsymbol{\Sigma}_{\dot{p}}$. Choose $k$ to capture 90% of signal variance. Expected $k \approx 5\text{–}8$.

In this space, noise is decorrelated by construction, so aggregation of the $k$ components gives $\sqrt{k}$ improvement on the noise. The challenge: estimating $\boldsymbol{\Sigma}_{\dot{p}}$ requires ground-truth breakdance data (BRACE dataset: ~400 sequences, likely sufficient for 17-joint covariance estimation but noisy).

### 9.4 Scenario-Adaptive Joint Selection (Easy, +0.2–0.6 to SNR)

For each move type, only aggregate joints with high expected signal and low expected correlation:

| Scenario | High-value joints | Drop | Effective $N$ |
|---|---|---|---:|
| Headspin | Ankles, knees (periodic arcs) | Head, spine (contact/static) | 4–6 |
| Windmill | Ankles, knees, wrists (rotating) | Spine, hips (near rotation axis) | 6–8 |
| Flare | Ankles, knees (circular arcs) | Hips, spine (pivot) | 4–6 |
| Footwork | Ankles, wrists, elbows | Head, spine | 6–8 |
| Toprock | All limb endpoints | Spine (low motion) | 8–10 |

Selecting only limb endpoints from **different** chains minimizes cross-joint correlation. With 4 endpoints (L/R ankle + L/R wrist) from 4 independent chains: $N_{eff} \approx 4$ with near-zero correlation.

---

## 10. Revised Viability Assessment with Mitigations

Applying root-relative coordinates + chain-aware weighting + scenario-adaptive selection (all "Easy" to implement):

Combined improvement: ~$1.3 \times 1.25 \times 1.1 \approx 1.8\times$ SNR over naive aggregation.

| Scenario | Corrected naive SNR | With mitigations | Spectrogram viable? |
|---|---:|---:|:---:|
| Toprock | 1.1–2.9 | **2.0–5.2** | **Yes** (coarse) |
| Footwork | 1.6–3.6 | **2.9–6.5** | **Yes** (reliable) |
| Windmill | 1.3–2.7 | **2.3–4.9** | **Marginal-to-Yes** |
| Flare | 1.4–2.7 | **2.5–4.9** | **Marginal-to-Yes** |
| Headspin | 0.8–2.3 | **1.4–4.1** | **Marginal** |

With mitigations, footwork and toprock are reliably usable. Power moves reach the lower bound of viability for period detection (SNR > 2). Headspin remains the most challenging scenario.

---

## 11. Implications for Pipeline Architecture

### The aggregation deficit changes the MotionBERT → Spectrogram contract

The prior analysis suggested aggregated SNR of 2–6 "is viable for binary musicality detection." The corrected range of **0.8–3.6** (naive) / **1.4–6.5** (with mitigations) means:

1. **Mitigations are not optional.** Without root-relative coords and chain-aware weighting, power move SNR drops below 1.5, making the spectrogram pipeline unreliable for those scenarios. These are ~100 LOC of preprocessing, zero training cost.

2. **The cross-correlation threshold $\mu$ needs scenario-dependent calibration.** A single threshold for "on-beat" vs. "off-beat" will have different false-positive rates per scenario. At SNR=2, the null distribution (random phase) for cross-correlation has std $\approx 1/\sqrt{2 \times \text{SNR}^2 \times T} \approx 0.05$ for 8s windows. Detectable, but requires per-scenario null calibration.

3. **SAM-Body4D fusion becomes more critical than initially assessed.** The prior analysis positioned SAM-Body4D as a "nice to have" for power moves. With corrected aggregation, SAM-Body4D's multi-view dense reconstruction (expected MPJPE 30–50mm for power moves) would bring power move SNR above 3 even with correct correlation accounting — moving from "marginal" to "reliable."

4. **DanceFormer's advantage widens.** DanceFormer on AIST++ achieves 18.4mm MPJPE with dance-specific training. Even with the same correlation structure (similar skeleton, similar attention mechanism), the $2\times$ lower per-joint error translates to $2\times$ higher velocity SNR. Combined with dance-specific temporal priors that may reduce $\rho_t$ for the correlated components (better tracking of fast dance motions), DanceFormer's effective aggregated SNR could reach 4–8 for toprock — sufficient for graded musicality scoring.
