from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .models import CommandOutput, CommandSpec, PaperResult, PaperSpec, RunSummary, Scenario
from .reporting import write_outputs
from .scoring import score_paper


def _expand_env(value: str) -> str:
    return os.path.expandvars(value)


def _command_output(payload: dict) -> CommandOutput:
    return CommandOutput(
        paper_id=payload["paper_id"],
        mode=payload["mode"],
        canonical_metrics={key: float(value) for key, value in payload["canonical_metrics"].items()},
        applied_metrics={key: float(value) for key, value in payload["applied_metrics"].items()},
        runtime_cost={key: float(value) for key, value in payload.get("runtime_cost", {}).items()},
        code_stats={key: float(value) for key, value in payload.get("code_stats", {}).items()},
        artifacts=list(payload.get("artifacts", [])),
        open_questions=list(payload.get("open_questions", [])),
        notes=payload.get("notes", ""),
    )


def execute_command(spec: CommandSpec, base_dir: Path) -> CommandOutput:
    env = os.environ.copy()
    env.update({key: _expand_env(value) for key, value in spec.env.items()})

    if spec.executor == "local":
        workdir = Path(_expand_env(spec.workdir))
        if not workdir.is_absolute():
            workdir = base_dir / workdir
        completed = subprocess.run(
            ["bash", "-lc", _expand_env(spec.cmd)],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    elif spec.executor == "ssh":
        if not spec.target:
            raise ValueError("SSH executor requires a target host")
        workdir = _expand_env(spec.workdir)
        remote_cmd = _expand_env(spec.cmd)
        if spec.env:
            exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items() if key in spec.env)
            remote_cmd = f"{exports} {remote_cmd}"
        remote_script = f"cd {shlex.quote(workdir)} && {remote_cmd}"
        completed = subprocess.run(
            ["ssh", _expand_env(spec.target), remote_script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        raise ValueError(f"Unsupported executor: {spec.executor}")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {spec.cmd}\nSTDERR:\n{completed.stderr}"
        )

    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"Command produced no stdout JSON: {spec.cmd}")

    try:
        return _command_output(json.loads(stdout))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON output from command: {spec.cmd}\nSTDOUT:\n{stdout}") from exc


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def execute_ladder(
    papers: list[PaperSpec],
    scenarios: list[Scenario],
    base_dir: Path,
    output_root: Path,
    continue_on_fail: bool = False,
) -> RunSummary:
    run_id = _run_id()
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[PaperResult] = []
    stopped_early = False

    for paper in papers:
        oracle = execute_command(paper.oracle_cmd, base_dir=base_dir)
        reimpl = execute_command(paper.reimpl_cmd, base_dir=base_dir)
        decision = score_paper(paper, oracle, reimpl)
        result = PaperResult(paper=paper, oracle=oracle, reimplementation=reimpl, decision=decision)
        results.append(result)

        if not decision.promoted and not continue_on_fail:
            stopped_early = True
            break

    summary = RunSummary(
        run_id=run_id,
        output_dir=output_dir,
        scenarios=scenarios,
        results=results,
        stopped_early=stopped_early,
    )
    write_outputs(summary)
    (output_root / "latest.txt").write_text(str(output_dir))
    return summary
