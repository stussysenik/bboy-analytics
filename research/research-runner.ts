/**
 * Overnight Research Runner — Main Orchestrator
 *
 * Runs 6 research phases per paper × 3 papers sequentially.
 * Each phase: send prompt → self-identification loop → checkpoint.
 * After all phases: synthesize → quality gate → save guide.
 *
 * Usage:
 *   bun run research-runner.ts              # Full run
 *   bun run research-runner.ts --status     # Progress summary
 *   bun run research-runner.ts --resume     # Resume interrupted run
 *   bun run research-runner.ts --paper X    # Run single paper
 */

import fs from 'fs/promises'
import path from 'path'
import yaml from 'js-yaml'
import { research } from './llm.js'
import { loadState, saveCheckpoint, isPhaseComplete, ensurePaper, saveState } from './checkpoint.js'
import { isStorageSafe } from './storage.js'
import { buildPhasesForPaper, GAP_ANALYSIS_PROMPT } from './phases.js'
import { synthesizeGuide, summarizeForCrossRef } from './synthesize.js'
import { getPriorContext, preloadPriorContext } from './prior-context.js'
import type { PaperConfig, ResearchPhase, ExperimentState } from './types.js'

const DATA_DIR = path.join(import.meta.dir, '..', 'data', 'research')
const MAX_HOURS = 8
const MAX_CONSECUTIVE_FAILURES = 3

function timestamp(): string {
  return new Date().toISOString().replace('T', ' ').slice(0, 19)
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function slugify(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '')
}

// ─── Config Loading ──────────────────────────────────────────────────────

async function loadPapers(): Promise<PaperConfig[]> {
  const configPath = path.join(import.meta.dir, 'papers.research.yaml')
  const raw = await fs.readFile(configPath, 'utf-8')
  const config = yaml.load(raw) as { papers: PaperConfig[] }
  return config.papers
}

// ─── Context Management ──────────────────────────────────────────────────

/**
 * Gather context from completed dependency phases.
 * Summarizes if over 30KB to prevent timeout issues.
 */
async function gatherContext(paperId: string, dependencies: string[]): Promise<string> {
  const paperDir = path.join(DATA_DIR, paperId)
  const parts: string[] = []

  for (const depId of dependencies) {
    try {
      const files = await fs.readdir(paperDir)
      const depFiles = files
        .filter(f => f.startsWith(depId) && f.endsWith('.md'))
        .sort()

      for (const file of depFiles) {
        const content = await fs.readFile(path.join(paperDir, file), 'utf-8')
        parts.push(`--- ${file} ---\n${content}`)
      }
    } catch {
      // Dependency may not exist yet
    }
  }

  if (parts.length === 0) {
    return '(No prior research context — this is the first phase.)'
  }

  const raw = parts.join('\n\n')

  if (raw.length < 30_000) return raw

  // Summarize via claude -p
  console.log(`    Context too large (${(raw.length / 1024).toFixed(1)}KB). Summarizing...`)
  try {
    const summary = await research(
      `Summarize the following research findings into a concise summary (max 8000 words). Preserve: key conclusions, specific numbers/metrics, mathematical formulas (keep LaTeX), model architecture details, tensor shapes, breakdancing-specific analysis. Drop: verbose explanations, repeated content.\n\n${raw.slice(0, 100_000)}`,
      300_000,
    )
    console.log(`    Summarized to ${(summary.length / 1024).toFixed(1)}KB`)
    return summary
  } catch {
    console.log(`    Summarization failed. Truncating to last 30KB.`)
    return raw.slice(-30_000)
  }
}

// ─── Gap Analysis ────────────────────────────────────────────────────────

function parseGaps(response: string): string[] {
  if (response.trim() === 'NO_GAPS' || response.includes('NO_GAPS')) {
    return []
  }

  const gaps: string[] = []
  const gapRegex = /\*\*Missing\*\*:\s*(.+?)(?:\n|$)/g
  let match
  while ((match = gapRegex.exec(response)) !== null) {
    gaps.push(match[1].trim())
  }

  if (gaps.length === 0) {
    const lines = response.split('\n').filter(l => /^###?\s*Gap/.test(l))
    for (const line of lines) {
      gaps.push(line.replace(/^###?\s*Gap\s*\d+:\s*/, '').trim())
    }
  }

  return gaps
}

// ─── Phase Execution ─────────────────────────────────────────────────────

async function executePhase(
  state: ExperimentState,
  phase: ResearchPhase,
  crossPaperContext: string,
  startFollowup = 0,
): Promise<void> {
  const { paperId, id: phaseId } = phase
  const paperDir = path.join(DATA_DIR, paperId)

  console.log(`\n  ╔══════════════════════════════════════════════════╗`)
  console.log(`  ║  ${phaseId}: ${phase.name.padEnd(38)}║`)
  console.log(`  ╚══════════════════════════════════════════════════╝`)
  console.log(`    Dependencies: ${phase.dependencies.join(', ') || 'none'}`)
  console.log(`    Budget: ${phase.timeBudgetMinutes}min | Max followups: ${phase.maxFollowups}`)

  const phaseStart = Date.now()
  const paperState = ensurePaper(state, paperId)
  let artifacts: string[] = paperState.phases[phaseId]?.artifacts || []
  let bytesWritten = paperState.phases[phaseId]?.bytesWritten || 0
  let followupsCompleted = startFollowup

  await saveCheckpoint(state, paperId, phaseId, {
    phaseId,
    paperId,
    status: 'in_progress',
    startedAt: paperState.phases[phaseId]?.startedAt || new Date().toISOString(),
    completedAt: null,
    followupsCompleted,
    artifacts,
    bytesWritten,
  })

  // Gather context from dependency phases
  console.log(`    [${timestamp()}] Gathering context...`)
  const context = await gatherContext(paperId, phase.dependencies)
  console.log(`    Context: ${(Buffer.byteLength(context) / 1024).toFixed(1)}KB`)

  // Load prior research context from bboy-battle-analysis
  const priorResearch = await getPriorContext(paperId)
  console.log(`    Prior research: ${(Buffer.byteLength(priorResearch) / 1024).toFixed(1)}KB`)

  // Initial research
  const initialPath = path.join(paperDir, `${phaseId}-${slugify(phase.name)}.md`)

  if (startFollowup === 0) {
    console.log(`    [${timestamp()}] Sending research prompt...`)

    const questionsBlock = phase.seedQuestions.map((q, i) => `${i + 1}. ${q}`).join('\n')
    const prompt = phase.promptTemplate
      .replace(/\{context\}/g, context)
      .replace(/\{questions\}/g, questionsBlock)
      .replace(/\{cross_paper_context\}/g, crossPaperContext || '(No cross-paper context yet.)')
      .replace(/\{prior_research\}/g, priorResearch)

    const result = await research(prompt)

    // Check for stale output
    if (result.length < 500) {
      console.log(`    [${timestamp()}] WARNING: Output suspiciously short (${result.length} chars). Retrying...`)
      const retry = await research(prompt)
      if (retry.length > result.length) {
        await saveArtifact(initialPath, phase, retry)
      } else {
        await saveArtifact(initialPath, phase, result)
      }
    } else {
      await saveArtifact(initialPath, phase, result)
    }

    const bytes = (await fs.stat(initialPath)).size
    bytesWritten += bytes
    artifacts.push(initialPath)
    console.log(`    [${timestamp()}] Initial research saved (${(bytes / 1024).toFixed(1)}KB)`)
  } else {
    console.log(`    [${timestamp()}] Resuming from followup ${startFollowup}...`)
  }

  // Self-identification loop
  for (let i = followupsCompleted; i < phase.maxFollowups; i++) {
    const elapsedMin = (Date.now() - phaseStart) / 60_000
    if (elapsedMin > phase.timeBudgetMinutes) {
      console.log(`    [${timestamp()}] Time budget exhausted (${elapsedMin.toFixed(1)}min).`)
      break
    }

    if (!isStorageSafe()) {
      console.log(`    [${timestamp()}] Storage low. Stopping followups.`)
      break
    }

    // Read all current research for gap analysis
    let currentResearch = ''
    for (const art of artifacts) {
      try {
        currentResearch += '\n\n' + await fs.readFile(art, 'utf-8')
      } catch { /* skip missing */ }
    }
    if (currentResearch.length > 80_000) {
      currentResearch = currentResearch.slice(-80_000)
    }

    console.log(`    [${timestamp()}] Self-identification ${i + 1}/${phase.maxFollowups}...`)
    const gapPrompt = GAP_ANALYSIS_PROMPT.replace('{research}', currentResearch)
    const gapResponse = await research(gapPrompt, 120_000)
    const gaps = parseGaps(gapResponse)

    if (gaps.length === 0) {
      console.log(`    [${timestamp()}] No gaps found. Research is thorough.`)
      break
    }

    console.log(`    [${timestamp()}] Found ${gaps.length} gap(s). Researching...`)

    for (let g = 0; g < gaps.length; g++) {
      const gap = gaps[g]
      console.log(`      Gap ${g + 1}: ${gap.slice(0, 80)}${gap.length > 80 ? '...' : ''}`)

      const followupPrompt = `You are continuing research on ${phase.name} for paper "${phase.paperId}" in a breakdancing analysis pipeline.

## Prior Research Context

${currentResearch.slice(-40_000)}

## Specific Question to Address

${gap}

## Requirements

- Provide mathematical formulations in LaTeX ($$...$$) where applicable
- Include concrete numerical estimates and tensor shapes
- Reference specific models, algorithms, or papers
- Address this gap thoroughly — it was identified as critical

Depth over breadth.`

      const followupResult = await research(followupPrompt)
      const followupPath = path.join(paperDir, `${phaseId}-${slugify(phase.name)}-followup-${i + 1}-${g + 1}.md`)
      await saveArtifact(followupPath, phase, followupResult, `Follow-up ${i + 1}.${g + 1}: ${gap.slice(0, 100)}`)
      const bytes = (await fs.stat(followupPath)).size
      bytesWritten += bytes
      artifacts.push(followupPath)
      console.log(`      Saved (${(bytes / 1024).toFixed(1)}KB)`)
    }

    followupsCompleted = i + 1

    await saveCheckpoint(state, paperId, phaseId, {
      phaseId,
      paperId,
      status: 'in_progress',
      startedAt: paperState.phases[phaseId]?.startedAt || new Date().toISOString(),
      completedAt: null,
      followupsCompleted,
      artifacts,
      bytesWritten,
    })
  }

  // Mark complete
  const totalMin = ((Date.now() - phaseStart) / 60_000).toFixed(1)
  await saveCheckpoint(state, paperId, phaseId, {
    phaseId,
    paperId,
    status: 'complete',
    startedAt: paperState.phases[phaseId]?.startedAt || new Date().toISOString(),
    completedAt: new Date().toISOString(),
    followupsCompleted,
    artifacts,
    bytesWritten,
  })

  console.log(`    [${timestamp()}] Phase complete: ${totalMin}min, ${artifacts.length} artifacts, ${(bytesWritten / 1024).toFixed(1)}KB`)
}

async function saveArtifact(
  filepath: string,
  phase: ResearchPhase,
  content: string,
  subtitle?: string,
): Promise<void> {
  const title = subtitle || phase.name
  const header = `# ${phase.paperId} — ${title}\n\n_Generated: ${new Date().toISOString()}_\n\n---\n\n`
  await fs.mkdir(path.dirname(filepath), { recursive: true })
  await fs.writeFile(filepath, header + content)
}

// ─── Safe Phase Execution with Retries ───────────────────────────────────

async function safeExecutePhase(
  state: ExperimentState,
  phase: ResearchPhase,
  crossPaperContext: string,
): Promise<boolean> {
  const maxRetries = 2

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const currentState = await loadState()
      const startFollowup = currentState.papers[phase.paperId]?.phases[phase.id]?.followupsCompleted || 0
      await executePhase(currentState, phase, crossPaperContext, startFollowup)
      return true
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error)
      console.error(`    [${timestamp()}] Attempt ${attempt + 1} failed: ${msg}`)

      if (attempt < maxRetries) {
        const waitMs = 30_000 * Math.pow(3, attempt)
        console.log(`    Retrying in ${(waitMs / 1000).toFixed(0)}s...`)
        await sleep(waitMs)
      } else {
        const currentState = await loadState()
        await saveCheckpoint(currentState, phase.paperId, phase.id, {
          phaseId: phase.id,
          paperId: phase.paperId,
          status: 'failed',
          startedAt: currentState.papers[phase.paperId]?.phases[phase.id]?.startedAt || null,
          completedAt: null,
          followupsCompleted: currentState.papers[phase.paperId]?.phases[phase.id]?.followupsCompleted || 0,
          artifacts: currentState.papers[phase.paperId]?.phases[phase.id]?.artifacts || [],
          bytesWritten: currentState.papers[phase.paperId]?.phases[phase.id]?.bytesWritten || 0,
          error: msg,
        })
        console.error(`    Phase ${phase.id} failed after ${maxRetries + 1} attempts.`)
        return false
      }
    }
  }
  return false
}

// ─── Status Command ──────────────────────────────────────────────────────

async function showStatus() {
  // Delegate to checkpoint.ts --status
  const { execSync } = await import('child_process')
  execSync('bun run checkpoint.ts --status', { cwd: import.meta.dir, stdio: 'inherit' })
}

// ─── Main ────────────────────────────────────────────────────────────────

async function main() {
  // Handle CLI flags
  if (process.argv.includes('--status')) {
    await showStatus()
    return
  }

  const papers = await loadPapers()

  // Optional single-paper mode
  const paperFlag = process.argv.indexOf('--paper')
  const targetPaper = paperFlag >= 0 ? process.argv[paperFlag + 1] : null
  const papersToRun = targetPaper
    ? papers.filter(p => p.id === targetPaper)
    : papers

  if (targetPaper && papersToRun.length === 0) {
    console.error(`Paper "${targetPaper}" not found. Available: ${papers.map(p => p.id).join(', ')}`)
    process.exit(1)
  }

  // Pre-load prior research context from bboy-battle-analysis
  await preloadPriorContext()

  console.log('\n════════════════════════════════════════════════════════')
  console.log('  Overnight Research Loop: Reimplementation Guides')
  console.log('  (with prior research from ANALYSIS_v2.md + TECH_STACK)')
  console.log('════════════════════════════════════════════════════════')
  console.log(`  Papers: ${papersToRun.map(p => p.name).join(', ')}`)
  console.log(`  Phases: 6 per paper × ${papersToRun.length} papers = ${papersToRun.length * 6} total`)
  console.log(`  Circuit breaker: ${MAX_HOURS}h max wall time`)
  console.log('════════════════════════════════════════════════════════\n')

  const deadline = Date.now() + MAX_HOURS * 60 * 60 * 1000
  let state = await loadState()

  // Heartbeat every 5 minutes
  const heartbeat = setInterval(() => {
    console.log(`  [${timestamp()}] ♥ Heartbeat — research still running`)
  }, 5 * 60_000)

  let crossPaperContext = ''
  let consecutiveFailures = 0

  try {
    for (const paper of papersToRun) {
      if (Date.now() > deadline) {
        console.log(`\n  Circuit breaker: ${MAX_HOURS}h exceeded. Stopping.`)
        break
      }

      console.log(`\n╔══════════════════════════════════════════════════════════╗`)
      console.log(`║  📄 ${paper.name.padEnd(50)}║`)
      console.log(`║  ${paper.full_title.slice(0, 54).padEnd(54)}║`)
      console.log(`╚══════════════════════════════════════════════════════════╝`)

      const phases = buildPhasesForPaper(paper)
      ensurePaper(state, paper.id)

      // Execute phases 1-5 (research)
      for (const phase of phases.slice(0, 5)) {
        if (Date.now() > deadline) break

        state = await loadState()
        if (isPhaseComplete(state, paper.id, phase.id)) {
          console.log(`\n    ${phase.id} already complete — skipping.`)
          continue
        }

        if (!isStorageSafe()) {
          console.error(`\n    Storage critically low. Stopping.`)
          break
        }

        const success = await safeExecutePhase(state, phase, crossPaperContext)
        if (success) {
          consecutiveFailures = 0
        } else {
          consecutiveFailures++
          if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            console.error(`\n    ${MAX_CONSECUTIVE_FAILURES} consecutive failures. Skipping ${paper.name}.`)
            break
          }
        }
      }

      // Phase 6: Synthesis → Guide
      state = await loadState()
      const paperState = state.papers[paper.id]
      const completedPhases = paperState
        ? Object.values(paperState.phases).filter(p => p.status === 'complete').length
        : 0

      if (completedPhases >= 3) {
        // At least 3/5 research phases done — synthesize what we have
        try {
          await synthesizeGuide(state, paper, crossPaperContext)
          // Accumulate cross-paper context
          crossPaperContext += await summarizeForCrossRef(paper.id)
          consecutiveFailures = 0
        } catch (error) {
          const msg = error instanceof Error ? error.message : String(error)
          console.error(`  [${timestamp()}] Synthesis failed for ${paper.name}: ${msg}`)
          consecutiveFailures++
        }
      } else {
        console.log(`\n  Only ${completedPhases}/5 phases complete for ${paper.name}. Skipping synthesis.`)
      }
    }

    // Mark experiment complete
    state = await loadState()
    state.completed = true
    await saveState(state)

    // Final summary
    console.log('\n════════════════════════════════════════════════════════')
    console.log('  RESEARCH LOOP COMPLETE')
    console.log('════════════════════════════════════════════════════════')

    for (const paper of papersToRun) {
      const ps = state.papers[paper.id]
      if (ps) {
        const phases = Object.values(ps.phases)
        const done = phases.filter(p => p.status === 'complete').length
        const failed = phases.filter(p => p.status === 'failed').length
        const guideIcon = ps.guideGenerated ? '✓' : '✗'
        const scoreStr = ps.guideQualityScore !== null ? ` (${ps.guideQualityScore}/100)` : ''
        console.log(`  ${paper.name}: ${done} phases ✓, ${failed} failed | Guide: ${guideIcon}${scoreStr}`)
      }
    }

    console.log(`\n  Total: ${(state.totalBytesWritten / 1024).toFixed(1)}KB written`)
    console.log(`  Duration: ${((Date.now() - new Date(state.startedAt).getTime()) / 3_600_000).toFixed(1)}h`)
    console.log('════════════════════════════════════════════════════════\n')

  } finally {
    clearInterval(heartbeat)
  }
}

main().catch(err => {
  console.error(`\nFatal error: ${err.message || err}`)
  process.exit(1)
})
