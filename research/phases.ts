/**
 * Research phase definitions: 6 phases × 3 papers = 18 total.
 *
 * Each phase has a prompt template with placeholders:
 *   {paper.name}, {paper.full_title}, {paper.authors}, {paper.venue},
 *   {paper.year}, {paper.why_bboy}, {paper.key_concepts}, {paper.official_sources},
 *   {context}, {cross_paper_context}, {prior_research}
 *
 * Phases are sequential within a paper. Papers run sequentially.
 */

import type { PaperConfig, ResearchPhase } from './types.js'

// ─── Gap Analysis Prompt (reused from bboy-battle-analysis) ──────────────

export const GAP_ANALYSIS_PROMPT = `You are a critical research reviewer. Given the following research output, identify the most important gaps:

## Research Output

{research}

## Task

Identify 1-3 gaps in the research. For each gap:
1. **What's missing**: Specific question or analysis that's absent
2. **Why it matters**: How this gap affects the overall system design
3. **Priority**: critical / important / nice-to-have

Only list gaps rated "critical" or "important". If the research is thorough and no meaningful gaps exist, respond with exactly: "NO_GAPS"

Format each gap as:
### Gap N: [title]
**Missing**: [specific question]
**Why**: [impact]
**Priority**: [critical/important]`

// ─── Phase Templates ─────────────────────────────────────────────────────

const PHASE_1_ARCHITECTURE = `You are a deep learning researcher performing a comprehensive paper architecture review.

## Paper
**{paper.full_title}**
Authors: {paper.authors} | Venue: {paper.venue} ({paper.year})

## Key Concepts
{paper.key_concepts}

## Official Sources
{paper.official_sources}

{prior_research}

## Task

Produce a comprehensive architecture survey of this paper. Cover EVERY section:

### 1. Problem Statement
What exactly does this paper solve? What was the SOTA before it? What gap does it fill?

### 2. Architecture Overview
Complete data flow from input to output. Name EVERY module, sub-module, and connection.
Draw an ASCII diagram of the full architecture showing tensor flow.

### 3. Key Innovation
What is the ONE thing this paper does differently? Why does it work better than prior work?
Be specific — cite ablation results from the paper.

### 4. Input/Output Specification
- Input: exact tensor shapes, data formats, preprocessing steps
- Output: exact tensor shapes, what each dimension represents
- Intermediate representations: shapes at each stage

### 5. Training Pipeline
- Loss functions: write the FULL math in LaTeX ($$...$$), define every variable
- Optimizer: which one, learning rate schedule, warmup
- Data augmentation: every transform applied
- Training data: datasets used, sizes, preprocessing

### 6. Inference Pipeline
What runs at test time? What modules/losses can be dropped?
Latency/throughput numbers from the paper.

### 7. Computational Cost
FLOPs, parameter count, GPU memory, training time as reported in the paper.

## Requirements
- Write actual equations in LaTeX notation ($$equation$$)
- Reference specific sections, figures, and tables from the paper
- Distinguish between what the paper CLAIMS and what is VERIFIED by ablations
- Mark anything uncertain with [NEEDS VERIFICATION]

## Context: Why We Care
This is for a breakdancing analysis system. {paper.why_bboy}`

const PHASE_2_MATH = `You are a mathematical ML researcher extracting and verifying equations from a deep learning paper.

## Paper
**{paper.full_title}**
Authors: {paper.authors} | {paper.venue} ({paper.year})

## Architecture Context (from Phase 1)
{context}

{prior_research}

## Task

Extract, explain, and verify EVERY key equation from this paper. For EACH equation:

$$equation$$

- **Name**: What this equation computes
- **Variables**: Define every symbol with type, shape, and units/dimensions
- **Intuition**: One paragraph plain-English explanation of what it does and WHY
- **Dimensions**: Verify input/output tensor shapes are compatible
- **Origin**: Is this equation novel to this paper, or standard (cite origin)?
- **Connection**: How does this equation's output feed into the next step?

### Required Equations (cover ALL of these):

1. **Loss Functions**: Every loss term individually, then the full composite loss with weights.
   Show the gradient direction — does minimizing this loss push the model in the right direction?

2. **Attention Mechanisms**: If transformer-based, write the FULL attention computation:
   - Standard self-attention: $$\\text{Attn}(Q,K,V) = \\text{softmax}(QK^T / \\sqrt{d_k})V$$
   - Any modifications (spatial attention, temporal attention, cross-attention, relative position encoding)
   - How attention heads are combined

3. **Core Forward Pass**: Write the complete mathematical function from raw input to final output.
   $$f(x) = ...$$ — write f explicitly, step by step.

4. **Pretraining Objective**: If self-supervised or masked prediction, write the objective function.
   What is masked? How is the reconstruction target defined?

5. **Evaluation Metrics**:
   - MPJPE: $$\\text{MPJPE} = \\frac{1}{T \\cdot J} \\sum_{t=1}^T \\sum_{j=1}^J \\| \\hat{p}_{t,j} - p_{t,j} \\|_2$$
   - P-MPJPE (procrustes-aligned), PA-MPJPE, or other metrics used
   - Write the exact formula, not just the name

### Verification Checklist
After listing all equations, verify:
- [ ] All dimensions are compatible in matrix operations
- [ ] Loss gradients push in the correct direction
- [ ] Attention scores sum to 1 (softmax applied correctly)
- [ ] Any implicit assumptions are stated (normalized inputs, centered coordinates, etc.)
- [ ] No circular dependencies between equations`

const PHASE_3_MINIMAL_IMPL = `You are a systems programmer planning a minimal reimplementation from first principles.

## Paper
**{paper.full_title}**

## Architecture + Math Context
{context}

{prior_research}

## Task

Design the "least keystrokes" reimplementation. The goal is: what is the MINIMUM code needed
to reproduce the core capability of this paper?

### ESSENTIAL (must implement — on the critical path)
For each component:
- **Module name**: e.g., "DSTformer", "CorrelationVolume"
- **What it does**: One sentence
- **Key functions**: List the 2-3 most important functions with signatures
- **Data structures**: Core tensors/dataclasses needed
- **Estimated LOC**: Be realistic
- **Pseudocode**: Write it out in Python-like syntax with tensor shape comments

\`\`\`python
# Example format:
# x: (B, T, J, 2) — batch, time_frames, joints, xy_coordinates
def forward(self, x):
    # spatial_attention: (B, T, J, J) — joint-to-joint attention per frame
    spatial_out = self.spatial_stream(x)
    ...
\`\`\`

### NICE-TO-HAVE (improves quality but not required for core)
For each:
- What you lose by skipping it
- Estimated quality impact (e.g., "+2mm MPJPE improvement")
- Estimated LOC

### SKIP (paper-specific overhead we don't need)
- Benchmark-specific code, visualization, distributed training, paper-specific ablation code
- For each: why it's safe to skip

### Dependency Audit
- **PyTorch version**: Minimum required
- **Required libraries**: einops, timm, etc. — version requirements
- **Pretrained weights**: What can be downloaded vs. must be retrained?
  - List exact checkpoint URLs if available
  - What dataset was it pretrained on? Does this matter for our use case?

### Total Estimate
- **Essential LOC**: N lines
- **With nice-to-have**: M lines
- **Time to implement**: rough estimate assuming familiarity with PyTorch`

const PHASE_4_BREAKDANCE = `You are adapting a computer vision model for breakdancing motion analysis.

## Paper
**{paper.full_title}**

## Architecture + Math + Implementation Context
{context}

## Cross-Paper Context (other models in the pipeline)
{cross_paper_context}

{prior_research}

## Breakdancing Challenge Scenarios
Analyze this model against EACH of these specific motion patterns:

1. **Headspin**: Continuous axial rotation, body fully inverted, heavy motion blur at extremities
2. **Windmill**: Floor-contact power move, left-right body alternation, continuous self-occlusion
3. **Flare**: Large circular leg arcs, extreme hip articulation, intermittent self-occlusion
4. **Freeze**: Sudden velocity collapse to zero, static pose held, must remain stable and confident
5. **Footwork**: Dense limb crossings near ground level, rapid direction changes, no full inversion
6. **Toprock**: Upright dancing with beat-aligned accents (control case — should work well)
7. **Battle**: Two dancers, cross-person occlusion, crowd edge intrusion, camera angle shifts

## Task

For EACH scenario (1-7), provide:

### Scenario N: {name}
**Works out of box**: Which components handle this natively? Why?
**Fails**: Specific failure modes with technical explanation.
  (e.g., "temporal window of 243 frames = 8.1s at 30fps, but headspins last 3-5s — FITS")
  (e.g., "trained on Human3.6M upright poses — never seen inverted body, attention map collapses")
**Modifications needed**: Concrete changes with estimated LOC and difficulty.
**Integration output**: How does this model's output feed downstream?

## Integration with Movement Spectrogram
The downstream scoring system needs clean derivatives from joint positions:

$$S_m(j, t) = \\|\\dot{p}_j(t)\\| = \\left\\| \\frac{d}{dt} p_j(t) \\right\\|$$

where $p_j(t)$ is the 3D position of joint $j$ at time $t$.

The audio-motion cross-correlation is:
$$\\mu = \\max_\\tau \\text{corr}(M(t), H(t - \\tau))$$

where $M(t) = \\sum_j S_m(j, t)$ is total movement energy and $H(t)$ is audio hotness
from the 8D psychoacoustic signature.

**Question**: How clean is this model's position output for computing first (velocity)
and second (acceleration) derivatives? What preprocessing is needed (smoothing, interpolation)?
What is the expected derivative SNR?`

const PHASE_5_GAPS = `You are a critical reviewer performing a gap analysis on a paper reimplementation plan.

## Paper
**{paper.full_title}**

## Complete Research So Far
{context}

## Cross-Paper Context
{cross_paper_context}

{prior_research}

## Task

Identify everything we MISSED, got WRONG, or left AMBIGUOUS:

### 1. Architectural Gaps
Components we didn't fully understand or glossed over.

### 2. Math Errors
Equations that might be wrong, incomplete, or have dimension mismatches.

### 3. Implementation Risks
Things that SEEM simple in the paper but are actually hard to implement correctly.
Common pitfalls, numerical stability issues, gradient problems.

### 4. Breakdance-Specific Blind Spots
Failure modes specific to breaking that we haven't considered.
What happens with extreme camera angles? Low resolution? Variable frame rates?

### 5. Integration Gaps
How this model connects to the other 2 papers in the pipeline.
Data format mismatches, resolution mismatches, timing/synchronization issues.

### 6. Citation Verification
Any claims that need source verification. Flag hallucinated or uncertain references.

For EACH gap:
- **What's missing**: Specific detail
- **Why it matters**: Impact on reimplementation success
- **Suggested resolution**: How to fill this gap

If no meaningful gaps exist, respond with: NO_GAPS`

const PHASE_6_SYNTHESIS = `You are writing a comprehensive first-principles reimplementation guide.

## Paper
**{paper.full_title}**
Authors: {paper.authors} | {paper.venue} ({paper.year})
arXiv: {paper.arxiv} | GitHub: {paper.github}

## All Research Artifacts (summarized)
{context}

## Cross-Paper Context
{cross_paper_context}

{prior_research}

## Output Structure (STRICT — include EVERY section below)

Write the complete guide following this EXACT structure:

# {paper.name} Reimplementation Guide

## Paper Metadata

| Field | Value |
|-------|-------|
| **Title** | {paper.full_title} |
| **Authors** | {paper.authors} |
| **Venue** | {paper.venue} ({paper.year}) |
| **arXiv** | [{paper.arxiv}](https://arxiv.org/abs/{paper.arxiv}) |
| **Code** | [{paper.github}](https://github.com/{paper.github}) |

## Why This Paper (for Breakdancing Analysis)

One paragraph connecting this paper to the bboy analysis pipeline. Reference the critical gap
it addresses and how its output feeds the movement spectrogram.

## Architecture

Include a mermaid diagram showing the complete data flow:
\`\`\`mermaid
flowchart TD
    A[Input] --> B[Module 1]
    B --> C[Module 2]
    ...
\`\`\`

Then prose explanation of each component (500-1000 words).

## Core Mathematics

For EACH key equation:
### Equation N: [Name]
$$equation$$
- **Variables**: Every symbol defined with shape and type
- **Intuition**: Plain English explanation
- **Connection**: How it feeds into the next equation

Include at minimum: the main loss function, the core forward pass, attention/correlation
mechanism, and the evaluation metric.

## "Least Keystrokes" Implementation Roadmap

### ESSENTIAL (~N LOC)
Numbered list of modules to implement, with LOC estimates.

### NICE-TO-HAVE (~M LOC)
What to add for production quality.

### SKIP
What to ignore and why.

## Pseudocode

\`\`\`python
# Complete critical-path pseudocode with tensor shape comments
# This should be detailed enough that someone can translate it to working PyTorch
\`\`\`

## Breakdance-Specific Modifications

### Headspin / Windmill / Flare / Freeze / Footwork / Toprock / Battle
Per-scenario: what works, what fails, what to modify.

## Known Limitations and Failure Modes

Honest numbered list of where this model breaks down.

## Integration Points

### With [Other Paper 1]
Data flow, format conversion, timing.

### With [Other Paper 2]
Data flow, format conversion, timing.

### With Movement Spectrogram Pipeline
How the output feeds S_m(j,t) and the audio-motion cross-correlation.

## References

All cited papers and resources. Use [UNVERIFIED] tag for anything not confirmed.

## Requirements
- The guide must be SELF-CONTAINED — a reader should understand the paper just from this guide
- Every equation must use LaTeX ($$...$$)
- Every claim must reference its source
- Mark uncertain information with [UNVERIFIED] or [NEEDS VERIFICATION]
- The mermaid diagram is REQUIRED
- Pseudocode must include tensor shape comments`

// ─── Phase Builder ───────────────────────────────────────────────────────

/**
 * Build the 6 research phases for a given paper.
 */
export function buildPhasesForPaper(paper: PaperConfig): ResearchPhase[] {
  const id = (n: number) => `${paper.id}-phase-${n}`
  const conceptsList = paper.key_concepts.map((c, i) => `${i + 1}. ${c}`).join('\n')
  const sourcesList = paper.official_sources.map(s => `- [${s.label}](${s.url})`).join('\n')

  function fillTemplate(template: string): string {
    return template
      .replace(/\{paper\.name\}/g, paper.name)
      .replace(/\{paper\.full_title\}/g, paper.full_title)
      .replace(/\{paper\.authors\}/g, paper.authors)
      .replace(/\{paper\.venue\}/g, paper.venue)
      .replace(/\{paper\.year\}/g, String(paper.year))
      .replace(/\{paper\.arxiv\}/g, paper.arxiv)
      .replace(/\{paper\.github\}/g, paper.github)
      .replace(/\{paper\.why_bboy\}/g, paper.why_bboy)
      .replace(/\{paper\.key_concepts\}/g, conceptsList)
      .replace(/\{paper\.official_sources\}/g, sourcesList)
  }

  return [
    {
      id: id(1),
      paperId: paper.id,
      phaseNumber: 1,
      name: 'Architecture Survey',
      description: `Map complete architecture of ${paper.name}`,
      seedQuestions: [
        `What is the complete data flow of ${paper.name} from raw input to final output? Name every module.`,
        `What are the exact input/output tensor shapes at each stage of the pipeline?`,
        `What is the key innovation — the ONE thing that makes ${paper.name} better than prior work?`,
      ],
      dependencies: [],
      maxFollowups: 2,
      timeBudgetMinutes: 15,
      promptTemplate: fillTemplate(PHASE_1_ARCHITECTURE),
    },
    {
      id: id(2),
      paperId: paper.id,
      phaseNumber: 2,
      name: 'Math Deep Dive',
      description: `Extract and verify all equations from ${paper.name}`,
      seedQuestions: [
        `What are ALL the loss functions in ${paper.name}? Write the complete composite loss.`,
        `How does the core attention/correlation mechanism work mathematically?`,
        `What is the complete forward pass as a mathematical function f(x) = ...?`,
      ],
      dependencies: [id(1)],
      maxFollowups: 2,
      timeBudgetMinutes: 15,
      promptTemplate: fillTemplate(PHASE_2_MATH),
    },
    {
      id: id(3),
      paperId: paper.id,
      phaseNumber: 3,
      name: 'Minimal Implementation Audit',
      description: `Design least-code reimplementation of ${paper.name}`,
      seedQuestions: [
        `What is the absolute minimum code needed to reproduce ${paper.name}'s core capability?`,
        `What pretrained weights are available and what would need to be retrained?`,
        `What are the critical implementation pitfalls that aren't obvious from the paper?`,
      ],
      dependencies: [id(1), id(2)],
      maxFollowups: 2,
      timeBudgetMinutes: 12,
      promptTemplate: fillTemplate(PHASE_3_MINIMAL_IMPL),
    },
    {
      id: id(4),
      paperId: paper.id,
      phaseNumber: 4,
      name: 'Breakdance Adaptation',
      description: `Analyze ${paper.name} for breakdancing motion patterns`,
      seedQuestions: [
        `How does ${paper.name} handle inverted human poses (headspins, windmills)?`,
        `What is the derivative SNR of ${paper.name}'s output for movement spectrogram computation?`,
        `What specific modifications are needed for breakdancing analysis?`,
      ],
      dependencies: [id(1), id(2), id(3)],
      maxFollowups: 3,
      timeBudgetMinutes: 12,
      promptTemplate: fillTemplate(PHASE_4_BREAKDANCE),
    },
    {
      id: id(5),
      paperId: paper.id,
      phaseNumber: 5,
      name: 'Gap Identification',
      description: `Find blind spots in ${paper.name} research`,
      seedQuestions: [
        `What did we miss about ${paper.name}'s architecture or math?`,
        `What integration issues exist between ${paper.name} and the other pipeline models?`,
      ],
      dependencies: [id(1), id(2), id(3), id(4)],
      maxFollowups: 1,
      timeBudgetMinutes: 10,
      promptTemplate: fillTemplate(PHASE_5_GAPS),
    },
    {
      id: id(6),
      paperId: paper.id,
      phaseNumber: 6,
      name: 'Synthesis',
      description: `Compile final reimplementation guide for ${paper.name}`,
      seedQuestions: [],
      dependencies: [id(1), id(2), id(3), id(4), id(5)],
      maxFollowups: 0,
      timeBudgetMinutes: 15,
      promptTemplate: fillTemplate(PHASE_6_SYNTHESIS),
    },
  ]
}
