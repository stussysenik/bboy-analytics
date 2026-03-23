/**
 * Types for the overnight research loop.
 *
 * PaperConfig drives the YAML manifest.
 * ResearchPhase defines per-paper research steps.
 * Checkpoint + ExperimentState handle persistence.
 * QualityReport scores the final guide output.
 */

export interface PaperConfig {
  id: string
  name: string
  full_title: string
  authors: string
  venue: string
  year: number
  arxiv: string
  github: string
  project_page: string
  why_bboy: string
  critical_gap: string
  key_concepts: string[]
  official_sources: { label: string; url: string }[]
}

export interface ResearchPhase {
  id: string
  paperId: string
  phaseNumber: number
  name: string
  description: string
  seedQuestions: string[]
  dependencies: string[]
  maxFollowups: number
  timeBudgetMinutes: number
  promptTemplate: string
}

export type PhaseStatus = 'pending' | 'in_progress' | 'complete' | 'failed'

export interface Checkpoint {
  phaseId: string
  paperId: string
  status: PhaseStatus
  startedAt: string | null
  completedAt: string | null
  followupsCompleted: number
  artifacts: string[]
  error?: string
  bytesWritten: number
}

export interface PaperState {
  paperId: string
  phases: Record<string, Checkpoint>
  guideGenerated: boolean
  guideQualityScore: number | null
}

export interface ExperimentState {
  experimentId: string
  startedAt: string
  lastUpdated: string
  papers: Record<string, PaperState>
  totalBytesWritten: number
  completed: boolean
}

export interface QualityReport {
  paperId: string
  totalSections: number
  hasMermaidDiagram: boolean
  hasLatexMath: boolean
  hasPseudocode: boolean
  hasBreakdanceSection: boolean
  hasIntegrationSection: boolean
  hasLimitationsSection: boolean
  estimatedWordCount: number
  referencesFound: number
  score: number
}
