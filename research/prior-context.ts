/**
 * Prior research context extraction.
 *
 * Reads ANALYSIS_v2.md and TECH_STACK_REEVALUATION.md from the bboy-battle-analysis
 * experiment and extracts per-paper relevant sections to inject into research prompts.
 *
 * This grounds each phase in existing knowledge, preventing re-discovery of known
 * facts and ensuring the new guides build on top of prior research.
 */

import fs from 'fs/promises'
import path from 'path'

const BBOY_DIR = path.join(import.meta.dir, '..', '..', 'bboy-battle-analysis')
const ANALYSIS_PATH = path.join(BBOY_DIR, 'ANALYSIS_v2.md')
const TECH_STACK_PATH = path.join(BBOY_DIR, 'TECH_STACK_REEVALUATION.md')

let _techStack: string | null = null
let _analysisLines: string[] | null = null

async function loadTechStack(): Promise<string> {
  if (!_techStack) {
    try {
      _techStack = await fs.readFile(TECH_STACK_PATH, 'utf-8')
    } catch {
      _techStack = '(TECH_STACK_REEVALUATION.md not found)'
    }
  }
  return _techStack
}

async function loadAnalysisLines(): Promise<string[]> {
  if (!_analysisLines) {
    try {
      const content = await fs.readFile(ANALYSIS_PATH, 'utf-8')
      _analysisLines = content.split('\n')
    } catch {
      _analysisLines = []
    }
  }
  return _analysisLines
}

function extractLines(lines: string[], start: number, end: number): string {
  return lines.slice(start - 1, end).join('\n')
}

/**
 * Get prior research context for a specific paper.
 * Returns a markdown section (~4-8KB) with relevant excerpts.
 */
export async function getPriorContext(paperId: string): Promise<string> {
  const techStack = await loadTechStack()
  const lines = await loadAnalysisLines()

  const sections: string[] = [
    '## Prior Research from Bboy Battle Analysis',
    '',
    'The following excerpts are from a prior 370KB research analysis (ANALYSIS_v2.md)',
    'and its March 2026 tech stack re-evaluation. Use these as grounding — build on',
    'this knowledge rather than re-discovering it. Correct any errors you find.',
    '',
  ]

  switch (paperId) {
    case 'motionbert':
      sections.push(
        '### From ANALYSIS_v2.md: The Inverted Pose Problem + 3D Pose Lifting',
        '',
        extractLines(lines, 750, 850),
        '',
        '### From ANALYSIS_v2.md: Move Taxonomy (computational signatures)',
        '',
        extractLines(lines, 812, 847),
        '',
        '### From TECH_STACK_REEVALUATION.md: Gap #1 Resolution + Revised Pipeline',
        '',
        '> Note: MotionBERT is the 3D pose lifting foundation. SAM-Body4D (below) replaces',
        '> the ViTPose + rotation hack that fed into it. MotionBERT remains relevant as the',
        '> temporal transformer that provides clean 3D trajectories for the movement spectrogram.',
        '',
        techStack,
      )
      break

    case 'cotracker3':
      sections.push(
        '### From ANALYSIS_v2.md: Multi-Person Tracking Challenges',
        '',
        extractLines(lines, 785, 810),
        '',
        '### From ANALYSIS_v2.md: The Inverted Pose Problem (context for why tracking matters)',
        '',
        extractLines(lines, 750, 770),
        '',
        '### From TECH_STACK_REEVALUATION.md: Full Document',
        '',
        '> CoTracker3 appears in Gap #3 resolution, Vision Layer, and Revised Pipeline.',
        '> It is step ③ in the new pipeline: dense point tracking on segmented dancer.',
        '',
        techStack,
      )
      break

    case 'sam3d':
      sections.push(
        '### From ANALYSIS_v2.md: The Inverted Pose Problem (the gap SAM-Body4D solves)',
        '',
        extractLines(lines, 750, 770),
        '',
        '### From ANALYSIS_v2.md: 3D Pose Lifting Context',
        '',
        extractLines(lines, 771, 783),
        '',
        '### From TECH_STACK_REEVALUATION.md: Full Document',
        '',
        '> SAM3D is the 2D→3D segmentation foundation. SAM-Body4D builds on SAM 3 +',
        '> Diffusion-VAS + SAM-3D-Body for training-free 4D mesh recovery. This is',
        '> step ② (segment) and ④ (mesh) in the revised pipeline.',
        '',
        techStack,
      )
      break

    default:
      sections.push('(No prior research context available for this paper.)')
  }

  return sections.join('\n')
}

/**
 * Pre-load all context at startup to avoid repeated disk reads.
 */
export async function preloadPriorContext(): Promise<void> {
  await loadTechStack()
  await loadAnalysisLines()
  console.log(`  Prior context loaded: TECH_STACK_REEVALUATION.md + ANALYSIS_v2.md`)
}
