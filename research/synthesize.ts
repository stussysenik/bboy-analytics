/**
 * Per-paper guide synthesis.
 *
 * Reads all phase artifacts for a paper, summarizes to ~40KB,
 * sends synthesis prompt, checks quality, retries once if below 60.
 */

import fs from 'fs/promises'
import path from 'path'
import { research } from './llm.js'
import { scoreGuide, formatQualityReport } from './quality-gates.js'
import type { PaperConfig, ExperimentState, QualityReport } from './types.js'
import { buildPhasesForPaper } from './phases.js'
import { saveState, ensurePaper } from './checkpoint.js'

const DATA_DIR = path.join(import.meta.dir, '..', 'data', 'research')
const GUIDES_DIR = path.join(import.meta.dir, '..', 'guides')

function timestamp(): string {
  return new Date().toISOString().replace('T', ' ').slice(0, 19)
}

/**
 * Gather and summarize all phase artifacts for a paper.
 */
async function gatherPaperContext(paperId: string): Promise<string> {
  const paperDir = path.join(DATA_DIR, paperId)
  let files: string[]
  try {
    files = (await fs.readdir(paperDir))
      .filter(f => f.endsWith('.md'))
      .sort()
  } catch {
    return '(No research artifacts found)'
  }

  const parts: string[] = []
  for (const file of files) {
    const content = await fs.readFile(path.join(paperDir, file), 'utf-8')
    parts.push(`--- ${file} ---\n${content}`)
  }

  const raw = parts.join('\n\n')

  // If under 40KB, use directly
  if (raw.length < 40_000) return raw

  // Summarize to stay within token limits
  console.log(`  [${timestamp()}] Context for ${paperId} is ${(raw.length / 1024).toFixed(1)}KB — summarizing...`)
  try {
    const summary = await research(
      `Summarize the following research findings into a comprehensive summary (max 10000 words).

PRESERVE exactly:
- ALL mathematical equations (keep LaTeX notation $$...$$)
- Tensor shapes and dimensions
- Architecture component names and data flow
- Specific numerical values (parameters, FLOPs, metrics, LOC estimates)
- Breakdancing-specific analysis and per-scenario results
- Pseudocode blocks
- Integration points between models

DROP:
- Verbose re-explanations of the same concept
- Generic ML background the reader would already know

${raw.slice(0, 120_000)}`,
      300_000, // 5 min for summarization
    )
    console.log(`  [${timestamp()}] Summarized to ${(summary.length / 1024).toFixed(1)}KB`)
    return summary
  } catch {
    console.log(`  [${timestamp()}] Summarization failed — truncating to last 40KB`)
    return raw.slice(-40_000)
  }
}

/**
 * Synthesize a guide for one paper and save to guides/.
 * Returns the quality report.
 */
export async function synthesizeGuide(
  state: ExperimentState,
  paper: PaperConfig,
  crossPaperContext: string,
): Promise<QualityReport> {
  console.log(`\n  [${timestamp()}] Synthesizing guide for ${paper.name}...`)

  const phases = buildPhasesForPaper(paper)
  const synthesisPhase = phases[5] // Phase 6 = index 5

  // Gather all phase artifacts
  const context = await gatherPaperContext(paper.id)

  // Build the synthesis prompt
  const prompt = synthesisPhase.promptTemplate
    .replace(/\{context\}/g, context)
    .replace(/\{cross_paper_context\}/g, crossPaperContext || '(This is the first paper — no cross-paper context yet.)')

  // Run synthesis
  const guide = await research(prompt, 900_000) // 15 min
  const header = `<!-- Generated: ${new Date().toISOString()} | Paper: ${paper.id} | Overnight Research Loop -->\n\n`
  const fullGuide = header + guide

  // Score quality
  let report = scoreGuide(paper.id, fullGuide)
  console.log(`  [${timestamp()}] ${formatQualityReport(report)}`)

  // Retry once if below threshold
  if (report.score < 60) {
    console.log(`  [${timestamp()}] Below threshold (60). Retrying with quality feedback...`)

    const missing: string[] = []
    if (!report.hasMermaidDiagram) missing.push('- Include a ```mermaid``` architecture diagram')
    if (!report.hasLatexMath) missing.push('- Include at least 3 LaTeX equations ($$...$$)')
    if (!report.hasPseudocode) missing.push('- Include ```python``` pseudocode with tensor shapes')
    if (!report.hasBreakdanceSection) missing.push('- Include a breakdance-specific modifications section')
    if (!report.hasIntegrationSection) missing.push('- Include an ## Integration Points section')
    if (!report.hasLimitationsSection) missing.push('- Include a ## Known Limitations section')
    if (report.totalSections < 9) missing.push(`- Need at least 9 H2 sections (currently ${report.totalSections})`)

    const retryPrompt = `${prompt}

## CRITICAL: Your previous attempt scored ${report.score}/100. It was MISSING:
${missing.join('\n')}

Fix ALL of these. The guide must be comprehensive and self-contained.`

    const retryGuide = await research(retryPrompt, 900_000)
    const retryFull = header + retryGuide

    const retryReport = scoreGuide(paper.id, retryFull)
    console.log(`  [${timestamp()}] Retry: ${formatQualityReport(retryReport)}`)

    if (retryReport.score > report.score) {
      report = retryReport
      await saveGuide(paper.id, paper.name, retryFull)
    } else {
      await saveGuide(paper.id, paper.name, fullGuide)
    }
  } else {
    await saveGuide(paper.id, paper.name, fullGuide)
  }

  // Update state
  const paperState = ensurePaper(state, paper.id)
  paperState.guideGenerated = true
  paperState.guideQualityScore = report.score
  await saveState(state)

  return report
}

async function saveGuide(paperId: string, paperName: string, content: string): Promise<void> {
  await fs.mkdir(GUIDES_DIR, { recursive: true })
  const filename = `${paperId.toUpperCase()}_REIMPL_GUIDE.md`
  const filepath = path.join(GUIDES_DIR, filename)
  await fs.writeFile(filepath, content)
  console.log(`  [${timestamp()}] Saved ${filename} (${(Buffer.byteLength(content) / 1024).toFixed(1)} KB)`)
}

/**
 * Generate a short cross-paper context summary from a completed guide.
 */
export async function summarizeForCrossRef(paperId: string): Promise<string> {
  const filename = `${paperId.toUpperCase()}_REIMPL_GUIDE.md`
  const filepath = path.join(GUIDES_DIR, filename)

  try {
    const content = await fs.readFile(filepath, 'utf-8')
    // Extract key sections: architecture, math, integration
    const sections = content.split(/^## /gm)
    const relevant = sections
      .filter(s => /architecture|math|integration|why this paper/i.test(s.slice(0, 50)))
      .map(s => s.slice(0, 2000))
      .join('\n\n---\n\n')

    return `### ${paperId} Summary\n${relevant.slice(0, 5000)}\n`
  } catch {
    return `### ${paperId}\n(Guide not yet generated)\n`
  }
}
