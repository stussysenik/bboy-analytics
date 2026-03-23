/**
 * LLM wrapper — calls `claude -p` headless for research.
 *
 * Adapted from bboy-battle-analysis/src/llm.ts.
 * Default timeout raised to 720s (12 min) for research phases.
 * Zero API key needed — uses the authenticated Claude CLI.
 */

import { spawn } from 'child_process'

/**
 * Send a research prompt to Claude and return the raw text response.
 *
 * @param prompt - The full research prompt
 * @param timeoutMs - Max wait time (default 12 minutes)
 * @returns Raw text response from Claude
 */
export async function research(prompt: string, timeoutMs = 720_000): Promise<string> {
  return callClaude(prompt, timeoutMs)
}

/**
 * Use `claude -p` (Claude Code headless mode) to perform research.
 * --tools "" disables all tools so Claude doesn't read files or add commentary
 * --output-format text gives raw response without JSON wrapping
 */
async function callClaude(prompt: string, timeoutMs: number): Promise<string> {
  const output = await spawnCapture('claude', [
    '-p',
    '--output-format', 'text',
    '--tools', '',
    '--model', 'sonnet',
  ], {
    stdin: prompt,
    timeout: timeoutMs,
  })
  return output
}

/**
 * Spawn a child process and capture its stdout.
 */
function spawnCapture(
  cmd: string,
  args: string[],
  opts: { stdin?: string; cwd?: string; timeout?: number } = {},
): Promise<string> {
  return new Promise((resolve, reject) => {
    let stdout = ''
    let stderr = ''

    const proc = spawn(cmd, args, {
      cwd: opts.cwd || process.cwd(),
      env: { ...process.env },
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: opts.timeout || 720_000,
    })

    proc.stdout.on('data', (chunk: Buffer) => { stdout += chunk.toString() })
    proc.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString() })

    if (opts.stdin) {
      proc.stdin.write(opts.stdin)
      proc.stdin.end()
    }

    proc.on('close', (code) => {
      if (code === 0 || stdout.trim()) {
        resolve(stdout)
      } else {
        reject(new Error(`${cmd} exited with code ${code}. stderr: ${stderr.slice(0, 500)}`))
      }
    })

    proc.on('error', reject)
  })
}
