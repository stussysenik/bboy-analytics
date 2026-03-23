/**
 * Checkpoint/resume engine for multi-paper research.
 *
 * Persists experiment state to JSON so the pipeline can survive interruptions
 * and resume from where it left off. Adapted from bboy-battle-analysis for
 * nested papers → phases structure.
 *
 * Run with --status flag for a progress summary:
 *   bun run checkpoint.ts --status
 */

import fs from 'fs/promises'
import path from 'path'
import type { Checkpoint, ExperimentState, PaperState, PhaseStatus } from './types.js'

const DATA_DIR = path.join(import.meta.dir, '..', 'data', 'research')
const MANIFEST_PATH = path.join(DATA_DIR, 'manifest.json')

function checkpointsDir(paperId: string): string {
  return path.join(DATA_DIR, paperId, 'checkpoints')
}

/**
 * Load experiment state from manifest.json.
 * If manifest is missing/corrupt, creates a fresh state.
 */
export async function loadState(): Promise<ExperimentState> {
  try {
    const raw = await fs.readFile(MANIFEST_PATH, 'utf-8')
    return JSON.parse(raw) as ExperimentState
  } catch {
    return {
      experimentId: `reimpl-research-${Date.now()}`,
      startedAt: new Date().toISOString(),
      lastUpdated: new Date().toISOString(),
      papers: {},
      totalBytesWritten: 0,
      completed: false,
    }
  }
}

/**
 * Ensure a paper entry exists in state.
 */
export function ensurePaper(state: ExperimentState, paperId: string): PaperState {
  if (!state.papers[paperId]) {
    state.papers[paperId] = {
      paperId,
      phases: {},
      guideGenerated: false,
      guideQualityScore: null,
    }
  }
  return state.papers[paperId]
}

/**
 * Save experiment state to manifest.json.
 */
export async function saveState(state: ExperimentState): Promise<void> {
  state.lastUpdated = new Date().toISOString()
  await fs.mkdir(path.dirname(MANIFEST_PATH), { recursive: true })
  await fs.writeFile(MANIFEST_PATH, JSON.stringify(state, null, 2))
}

/**
 * Update a single phase's checkpoint and persist.
 */
export async function saveCheckpoint(
  state: ExperimentState,
  paperId: string,
  phaseId: string,
  update: Partial<Checkpoint>,
): Promise<void> {
  const paper = ensurePaper(state, paperId)
  const existing = paper.phases[phaseId] || createEmptyCheckpoint(paperId, phaseId)
  paper.phases[phaseId] = { ...existing, ...update }

  // Save individual checkpoint file
  const cpDir = checkpointsDir(paperId)
  await fs.mkdir(cpDir, { recursive: true })
  await fs.writeFile(
    path.join(cpDir, `${phaseId}.checkpoint.json`),
    JSON.stringify(paper.phases[phaseId], null, 2),
  )

  // Recalculate total bytes
  state.totalBytesWritten = Object.values(state.papers).reduce((sum, p) =>
    sum + Object.values(p.phases).reduce((s, cp) => s + cp.bytesWritten, 0), 0)

  await saveState(state)
}

/**
 * Check if a phase is already complete.
 */
export function isPhaseComplete(state: ExperimentState, paperId: string, phaseId: string): boolean {
  return state.papers[paperId]?.phases[phaseId]?.status === 'complete'
}

/**
 * Create an empty checkpoint for a phase.
 */
export function createEmptyCheckpoint(paperId: string, phaseId: string): Checkpoint {
  return {
    phaseId,
    paperId,
    status: 'pending',
    startedAt: null,
    completedAt: null,
    followupsCompleted: 0,
    artifacts: [],
    bytesWritten: 0,
  }
}

// CLI mode: --status flag (only when run directly, not imported)
if (import.meta.main && process.argv.includes('--status')) {
  const state = await loadState()
  console.log(`\n=== Research Experiment Status ===`)
  console.log(`ID: ${state.experimentId}`)
  console.log(`Started: ${state.startedAt}`)
  console.log(`Last Updated: ${state.lastUpdated}`)
  console.log(`Completed: ${state.completed}`)
  console.log(`Total Bytes Written: ${(state.totalBytesWritten / 1024).toFixed(1)} KB`)

  const papers = Object.values(state.papers)
  if (papers.length === 0) {
    console.log('\n  No papers started yet.')
  } else {
    for (const paper of papers) {
      const phases = Object.values(paper.phases)
      const done = phases.filter(p => p.status === 'complete').length
      const failed = phases.filter(p => p.status === 'failed').length
      const total = phases.length
      const guideIcon = paper.guideGenerated ? '✓' : '○'
      const scoreStr = paper.guideQualityScore !== null ? ` (score: ${paper.guideQualityScore})` : ''

      console.log(`\n  📄 ${paper.paperId}: ${done}/${total} phases complete${failed > 0 ? `, ${failed} failed` : ''}`)
      console.log(`     Guide: ${guideIcon}${scoreStr}`)

      for (const p of phases) {
        const icon = p.status === 'complete' ? '✓' : p.status === 'in_progress' ? '…' : p.status === 'failed' ? '✗' : '○'
        console.log(`     ${icon} ${p.phaseId}: ${p.status} (${p.artifacts.length} artifacts, ${p.followupsCompleted} followups)`)
        if (p.error) console.log(`       Error: ${p.error}`)
      }
    }
  }
  console.log()
}
