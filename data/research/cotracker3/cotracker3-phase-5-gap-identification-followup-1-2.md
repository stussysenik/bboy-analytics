# cotracker3 — Follow-up 1.2: The analysis evaluates component-level metrics (AJ for tracking, AIST accuracy for pose) but never a

_Generated: 2026-03-23T04:51:14.836Z_

---

I don't have filesystem access in this environment, but I have all the context needed from the prior research. Let me provide the deep analysis on the critical validation gap.

---

# Gap 7: End-to-End Judging Validation Framework — The Missing Cornerstone

## The Problem Statement

The pipeline computes per-component metrics: AJ for CoTracker3, mm-error for SAM-Body4D, F1 for beat detection. But the final output — **a judging score** — has no validation path. This is the classic "last mile" problem in ML pipelines: every component can be individually correct while the composed system produces meaningless output.

The specific challenge is threefold:

1. **No ground truth dataset** maps `(video_segment) → (musicality_score, execution_score, creativity_score, ...)`
2. **Inter-judge disagreement** is inherent — Cohen's κ ≈ 0.4–0.6 means judges themselves disagree ~40% of the time on individual criteria
3. **The scoring function is unlearned** — it's a hand-crafted formula (§5.1's musicality score), not a learned mapping, so there's no training loop to optimize against ground truth even if it existed

## 1. What κ ≈ 0.4–0.6 Actually Means for System Validation

### 1.1 The Reliability Ceiling

Inter-rater reliability sets an **upper bound** on any automated system's achievable agreement with any single judge. If two expert judges agree at κ = 0.5, the best a system can do — even with perfect perception — is κ ≈ 0.5 with either judge individually.

The relevant statistic is the **Spearman-Brown prophecy formula** for aggregated judgments. If κ₁ is the reliability of a single judge, the reliability of the mean of k judges is:

$$\kappa_k = \frac{k \cdot \kappa_1}{1 + (k-1) \cdot \kappa_1}$$

For κ₁ = 0.5 and k = 5 judges:

$$\kappa_5 = \frac{5 \times 0.5}{1 + 4 \times 0.5} = \frac{2.5}{3.0} = 0.833$$

This means **the mean of 5 judges is a reliable target** (κ = 0.83), even though individual judges are noisy. The system should be validated against **consensus scores** (mean or median of multiple judges), not individual judge scores.

### 1.2 The Validation Metric

The correct metric is **agreement with consensus**, measured as:

$$\rho_{\text{system}} = \text{Spearman}(S_{\text{system}}, \bar{S}_{\text{judges}})$$

where $\bar{S}_{\text{judges}}$ is the mean of k ≥ 5 expert scores. The benchmark: a "replacement judge" — one judge compared against the consensus of the remaining judges — achieves:

$$\rho_{\text{human}} = \text{Spearman}(S_{\text{judge}_i}, \bar{S}_{\text{judges} \setminus i})$$

The system is "useful" if $\rho_{\text{system}} \geq \rho_{\text{human}} - \epsilon$ for a meaningful tolerance $\epsilon$. Based on figure skating and gymnastics automated judging literature (where similar κ values hold):

- $\rho_{\text{human}} \approx 0.70–0.85$ (single judge vs. panel consensus)
- $\epsilon = 0.15$ is a reasonable "useful assistant" threshold
- So the target is $\rho_{\text{system}} \geq 0.55–0.70$

### 1.3 Per-Criterion Decomposition

Breakdancing judging (under the 2024 Olympic framework) decomposes into criteria with different subjectivity levels:

| Criterion | Expected κ (judge-judge) | Automatable? | Why |
|-----------|-------------------------|--------------|-----|
| **Technique/Execution** | 0.55–0.65 | **Highest** | Clean landings, balance = measurable physics |
| **Musicality** | 0.45–0.55 | **Medium** | Beat-alignment is measurable; *interpretation* is not |
| **Vocabulary/Difficulty** | 0.50–0.60 | **Medium** | Move identification is objective; selection strategy is subjective |
| **Originality/Creativity** | 0.30–0.45 | **Lowest** | Requires cultural context, novelty detection against community norms |

The system should target **execution and musicality first** (higher κ = more trainable signal) and treat creativity as out-of-scope for v0.1.

---

## 2. Ground Truth Data Requirements

### 2.1 Minimum Dataset Size

To detect a correlation of $\rho = 0.60$ at $\alpha = 0.05$ with power $1 - \beta = 0.80$:

$$n \geq \left(\frac{z_{\alpha/2} + z_\beta}{\text{arctanh}(\rho)}\right)^2 + 3 = \left(\frac{1.96 + 0.84}{0.693}\right)^2 + 3 \approx 19.5$$

So **n ≥ 20 clips** suffices for a go/no-go validation. But to estimate the correlation with a 95% CI width of ±0.15, you need:

$$n \approx \frac{4}{w^2} + 3 \approx \frac{4}{0.15^2} + 3 \approx 181$$

A practical validation dataset: **~200 clips**, each ~30 seconds (one battle round), annotated by **5 judges** on 4 criteria, producing a $200 \times 5 \times 4$ tensor of scores.

### 2.2 Annotation Protocol

Each annotation unit is a **round** (one dancer's turn, typically 30–60 seconds). Each judge provides:

$$\mathbf{s}_{i,j} = (s_{\text{exec}}, s_{\text{music}}, s_{\text{vocab}}, s_{\text{orig}}) \in [1, 10]^4$$

for clip $i$ and judge $j$. Additionally, judges mark **temporal anchors**:

- Timestamp of each "moment of impact" (a hit, a freeze entry, a power move start)
- Binary: was this moment on-beat? (yes/no)
- Quality: 1–5 rating of execution quality for that specific moment

This temporal annotation is critical — it validates the intermediate representations (jerk events, beat-alignment), not just the final score.

### 2.3 Annotation Cost Estimate

- 200 clips × 30 seconds = 100 minutes of video
- Per-judge time: ~3× real-time for scoring + temporal anchors = ~300 minutes per judge
- 5 judges × 300 minutes = 1,500 judge-minutes = **25 judge-hours**
- At competitive bboy judge rates (~$50–100/hr): **$1,250–$2,500**
- Platform cost (Label Studio or similar): ~$0 (self-hosted)

This is remarkably cheap for a validation dataset. The bottleneck is finding 5 qualified judges, not cost.

---

## 3. Multi-Level Validation Architecture

The system needs validation at **four levels**, not just the final score:

### Level 0: Perception Validation (Does the system see what happened?)

**Test**: Given a clip, does the system correctly identify:
- Which body parts moved (binary per-joint activity detection)
- Approximate motion direction and magnitude
- Contact events (foot/hand/head touching floor)

**Ground truth**: Frame-level 2D pose annotations (BRACE dataset, if verified, or COCO-format keypoint annotation on breakdancing clips)

**Metric**: PCK@0.2 (Percentage of Correct Keypoints within 20% of head-body distance) on derived keypoint positions from CoTracker3 tracks

**Mathematical formulation**: For derived keypoint position $\hat{\mathbf{p}}_k(t)$ and ground truth $\mathbf{p}_k(t)$:

$$\text{PCK}@\tau = \frac{1}{K \cdot T} \sum_{k=1}^{K} \sum_{t=1}^{T} \mathbb{1}\left[\|\hat{\mathbf{p}}_k(t) - \mathbf{p}_k(t)\|_2 < \tau \cdot d_{\text{head-body}}(t)\right]$$

**Target**: PCK@0.2 ≥ 0.70 for toprock/footwork, ≥ 0.50 for power moves

### Level 1: Event Detection Validation (Does the system detect the right moments?)

**Test**: Does the jerk-event detection (CWT modulus maxima) fire at the same moments judges marked as "impacts"?

**Ground truth**: Temporal anchors from the annotation protocol (§2.2)

**Metric**: Event-level F1 with tolerance window $\delta$:

$$\text{Precision}(\delta) = \frac{|\{e_{\text{sys}} : \exists e_{\text{gt}}, |e_{\text{sys}} - e_{\text{gt}}| < \delta\}|}{|\{e_{\text{sys}}\}|}$$

$$\text{Recall}(\delta) = \frac{|\{e_{\text{gt}} : \exists e_{\text{sys}}, |e_{\text{sys}} - e_{\text{gt}}| < \delta\}|}{|\{e_{\text{gt}}\}|}$$

$$\text{F1}(\delta) = \frac{2 \cdot \text{Precision}(\delta) \cdot \text{Recall}(\delta)}{\text{Precision}(\delta) + \text{Recall}(\delta)}$$

At $\delta = 100$ms (the perceptual threshold for "on beat"):

**Target**: F1(100ms) ≥ 0.60 for toprock, ≥ 0.40 for power moves

### Level 2: Beat-Alignment Validation (Are detected events musically timed?)

**Test**: Among correctly detected events, does the system agree with judges on which are "on beat"?

**Ground truth**: Binary beat-alignment labels from annotation protocol

**Metric**: Binary classification accuracy, but weighted by event confidence:

$$\text{BA-Acc} = \frac{\sum_k w_k \cdot \mathbb{1}[\hat{b}_k = b_k]}{\sum_k w_k}, \quad w_k = \exp\left(-\frac{|t_k^{\text{sys}} - t_k^{\text{gt}}|^2}{2\delta^2}\right)$$

where $\hat{b}_k$ is the system's beat-alignment prediction and $b_k$ is the judge consensus. The weight $w_k$ downweights events where the temporal localization was poor (so timing errors don't double-penalize).

**Target**: BA-Acc ≥ 0.70

### Level 3: Score Validation (Does the final score match judge consensus?)

**Test**: The end-to-end metric — system score vs. judge panel consensus.

**Metric**: Spearman correlation per criterion:

$$\rho_c = \text{Spearman}\left(\{S_{\text{sys}, c}^{(i)}\}_{i=1}^N, \{\bar{S}_{\text{judges}, c}^{(i)}\}_{i=1}^N\right)$$

for criterion $c \in \{\text{exec, music, vocab, orig}\}$.

**Significance test**: Fisher z-transformation to test $H_0: \rho_c = 0$:

$$z = \frac{\text{arctanh}(\rho_c)}{\sqrt{1/(N-3)}}$$

**Target**: $\rho_{\text{exec}} \geq 0.55$, $\rho_{\text{music}} \geq 0.50$, $\rho_{\text{vocab}} \geq 0.45$, $\rho_{\text{orig}}$ = not targeted in v0.1

---

## 4. The Calibration Problem

### 4.1 Score Anchoring

The hand-crafted musicality formula (§5.1 in the gap analysis) outputs a dimensionless number. Judges score on a 1–10 scale. These are not comparable without calibration.

The calibration mapping from raw system scores $S_{\text{raw}}$ to calibrated scores $S_{\text{cal}}$ should be:

$$S_{\text{cal}} = \Phi^{-1}\left(F_{\text{raw}}(S_{\text{raw}})\right) \cdot \sigma_{\text{judge}} + \mu_{\text{judge}}$$

where $F_{\text{raw}}$ is the empirical CDF of raw system scores, $\Phi^{-1}$ is the inverse normal CDF, and $\mu_{\text{judge}}, \sigma_{\text{judge}}$ are the mean and standard deviation of judge consensus scores. This **quantile-matches** system output to the judge score distribution, preserving rank order while correcting scale.

### 4.2 Per-Criterion Calibration Sets

The calibration requires a held-out set (separate from the validation set):

- **Calibration set**: ~50 clips, scored by judges, used to fit the mapping
- **Validation set**: ~150 clips, scored by judges, used to measure final $\rho_c$

Never calibrate and validate on the same data — this inflates correlation estimates.

### 4.3 Judge Bias Correction

Individual judges have systematic biases (some are "harsh graders"). The consensus should use a **mixed-effects model**:

$$s_{i,j,c} = \mu_c + \alpha_{i,c} + \beta_{j,c} + \epsilon_{i,j,c}$$

where $\alpha_{i,c}$ is the clip effect (what we want), $\beta_{j,c}$ is the judge bias, and $\epsilon$ is noise. The BLUP (Best Linear Unbiased Predictor) for $\alpha_{i,c}$ corrects for judge bias and is a better consensus target than the raw mean.

---

## 5. The Bootstrapping Strategy

You can't collect 200 annotated clips before building anything. The practical path:

### Phase 0: Smoke Test (Before any judge involvement)
- **N = 5 clips** you personally curate (clear examples of good/bad musicality, clean/sloppy execution)
- **Validation**: Does the system rank them in the obviously correct order?
- **Cost**: $0, ~1 hour of your time
- **Pass criterion**: Spearman ρ > 0.80 on these obvious cases (anything less means the formula is broken at a basic level)

### Phase 1: Pilot (1 judge, 20 clips)
- Recruit one experienced judge for ~2 hours
- Score 20 clips on execution and musicality only
- Compute rank correlation
- **Pass criterion**: ρ ≥ 0.40 (above chance, directionally correct)
- **Purpose**: Identify systematic failures (e.g., "the system always overscores footwork")

### Phase 2: Full Validation (5 judges, 200 clips)
- Only proceed if Phase 1 passes
- Full annotation protocol (§2.2)
- Multi-level validation (§3)
- Calibration + validation split
- **Pass criterion**: Per-criterion targets from §3 Level 3

### Phase 3: Continuous Validation (Ongoing)
- As the system is used in practice, collect judge feedback on system outputs
- Use this as a growing validation set
- Monitor for distribution shift (new move styles the system hasn't seen)

---

## 6. The Creativity Problem (Why It's Out of Scope)

Creativity/originality scoring requires:

$$S_{\text{orig}}(x) \propto -\log p(x | \mathcal{D}_{\text{community}})$$

— the less likely a move sequence is under the distribution of "everything dancers have done before," the more original it is. This requires:

1. A generative model of breakdancing sequences (does not exist)
2. A representative corpus of breakdancing performances ($\mathcal{D}_{\text{community}}$ — partially exists in competition archives but is not annotated at the move level)
3. A move-level tokenization that makes $p(x)$ computable

Each of these is a research problem unto itself. Originality scoring is a **v3+ feature** that likely requires a learned model trained on thousands of annotated performances, not a hand-crafted formula.

---

## 7. Concrete Recommendations

| Priority | Action | Blocks | Cost |
|----------|--------|--------|------|
| **P0** | Define the smoke test (5 clips, personal ranking) | Everything | ~1 hour |
| **P0** | Specify the musicality formula completely (§5.1 in original gaps) | Smoke test | ~1 day of design |
| **P1** | Build the Level 1 event detection evaluator (F1@δ) | Phase 1 pilot | ~2 days |
| **P1** | Recruit 1 pilot judge | Phase 1 pilot | ~$100–200 |
| **P2** | Build Label Studio annotation interface with temporal anchoring | Phase 2 | ~3 days |
| **P2** | Recruit 5 judges, collect 200-clip annotations | Phase 2 | ~$2,500 |
| **P3** | Implement mixed-effects calibration model | Phase 2 analysis | ~1 day |

The key insight: **validation is not a post-hoc step — it shapes the architecture**. The decision to use a hand-crafted musicality formula vs. a learned scoring model depends entirely on whether you can collect enough ground truth data to train the latter. If 200 annotated clips are achievable (they are, for ~$2,500), a learned model will almost certainly outperform any hand-crafted formula. If you commit to the hand-crafted approach, validation is still mandatory but the formula's inductive biases (§2.3 in the original gap analysis — the jerk magnitude normalization issue) become load-bearing design decisions that need ablation testing against judge scores.

---

## Summary

The judging validation gap is **architecturally load-bearing** — it determines whether to hand-craft or learn the scoring function, sets the minimum dataset size, and defines what "success" means. Without it, you're building a pipeline that produces numbers with no way to know if those numbers mean anything. The good news: breakdancing validation data is cheap (~$2,500 for a statistically sufficient dataset), the statistical framework is well-established (borrowed from psychometrics and figure skating), and the bootstrapping path (5 → 20 → 200 clips) lets you fail fast before investing in full annotation.
