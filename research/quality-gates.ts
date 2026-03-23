/**
 * Quality gates for generated guide .md files.
 *
 * Deterministic string checks — no LLM call.
 * Pass threshold: 60/100.
 */

import type { QualityReport } from './types.js'

export function scoreGuide(paperId: string, content: string): QualityReport {
  const h2Count = (content.match(/^## /gm) || []).length
  const latexBlocks = (content.match(/\$\$.+?\$\$/gs) || []).length
  const hasMermaid = /```mermaid/.test(content)
  const hasPython = /```python/.test(content)
  const hasBreakdance = /breakdanc/i.test(content)
  const hasIntegration = /## Integration/i.test(content)
  const hasLimitations = /## Known Limitation/i.test(content)
  const wordCount = content.split(/\s+/).length
  const refs = (content.match(/arxiv|github\.com|doi\.org/gi) || []).length

  let score = 0
  if (hasMermaid) score += 10
  if (latexBlocks >= 3) score += 15
  if (hasPython) score += 10
  if (hasBreakdance) score += 10
  if (hasIntegration) score += 10
  if (hasLimitations) score += 5
  if (wordCount >= 5000) score += 15
  if (refs >= 5) score += 10
  if (h2Count >= 9) score += 15

  return {
    paperId,
    totalSections: h2Count,
    hasMermaidDiagram: hasMermaid,
    hasLatexMath: latexBlocks >= 3,
    hasPseudocode: hasPython,
    hasBreakdanceSection: hasBreakdance,
    hasIntegrationSection: hasIntegration,
    hasLimitationsSection: hasLimitations,
    estimatedWordCount: wordCount,
    referencesFound: refs,
    score,
  }
}

export function formatQualityReport(report: QualityReport): string {
  const pass = report.score >= 60 ? 'PASS' : 'FAIL'
  return [
    `Quality Report: ${report.paperId} — ${report.score}/100 (${pass})`,
    `  Sections: ${report.totalSections} | Words: ${report.estimatedWordCount} | Refs: ${report.referencesFound}`,
    `  Mermaid: ${report.hasMermaidDiagram ? '✓' : '✗'} | LaTeX: ${report.hasLatexMath ? '✓' : '✗'} | Pseudocode: ${report.hasPseudocode ? '✓' : '✗'}`,
    `  Breakdance: ${report.hasBreakdanceSection ? '✓' : '✗'} | Integration: ${report.hasIntegrationSection ? '✓' : '✗'} | Limitations: ${report.hasLimitationsSection ? '✓' : '✗'}`,
  ].join('\n')
}
