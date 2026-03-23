from __future__ import annotations

import json
from pathlib import Path

from .models import PaperResult, RunSummary


def _format_metric_table(paper_result: PaperResult) -> str:
    keys = sorted(
        set(paper_result.oracle.canonical_metrics)
        | set(paper_result.reimplementation.canonical_metrics)
        | set(paper_result.oracle.applied_metrics)
        | set(paper_result.reimplementation.applied_metrics)
    )
    lines = ["| Metric | Oracle | Reimpl |", "|---|---:|---:|"]
    for key in keys:
        oracle = paper_result.oracle.canonical_metrics.get(key, paper_result.oracle.applied_metrics.get(key))
        reimpl = paper_result.reimplementation.canonical_metrics.get(
            key,
            paper_result.reimplementation.applied_metrics.get(key),
        )
        oracle_value = "-" if oracle is None else f"{oracle:.4f}"
        reimpl_value = "-" if reimpl is None else f"{reimpl:.4f}"
        lines.append(f"| `{key}` | {oracle_value} | {reimpl_value} |")
    return "\n".join(lines)


def _score_summary(result: PaperResult) -> str:
    return (
        f"- Utility: `{result.decision.utility_score:.3f}`\n"
        f"- Parity: `{result.decision.parity_score:.3f}`\n"
        f"- Code economy: `{result.decision.code_economy_score:.3f}`\n"
        f"- Surrogate score: `{result.decision.surrogate_score:.3f}`\n"
        f"- Promoted: `{'yes' if result.decision.promoted else 'no'}`"
    )


def render_author_packet(result: PaperResult) -> str:
    sources = "\n".join(f"- [{source.label}]({source.url})" for source in result.paper.official_sources)
    open_questions = sorted(set(result.paper.author_packet_targets + result.reimplementation.open_questions))
    questions_md = "\n".join(f"- {question}" for question in open_questions) if open_questions else "- None"
    artifacts = "\n".join(f"- `{artifact}`" for artifact in result.reimplementation.artifacts) or "- None"

    return f"""# {result.paper.name} Author Packet

## Goal

Paper-faithful reimplementation used as a research subject for extreme-motion computer vision: inverse poses, severe articulation, occlusion, and audio-motion signature quality.

## Official Sources

{sources}

## Run Summary

{_score_summary(result)}

## Dataset Slice

- Canonical: `{result.paper.dataset_slice['canonical']}`
- Applied: `{result.paper.dataset_slice['applied']}`

## Metric Comparison

{_format_metric_table(result)}

## Notes

{result.reimplementation.notes or "No extra notes provided."}

## Artifacts

{artifacts}

## Questions For Authors

{questions_md}

## Outreach Draft

Hello {result.paper.name} authors,

I built a paper-faithful reimplementation focused on extreme-motion use cases: breakdancing inversions, severe articulation, occlusion, and downstream audio-motion signature quality. I compared it against your public implementation on a small canonical slice plus an applied clip bank.

The current run scored utility `{result.decision.utility_score:.3f}`, parity `{result.decision.parity_score:.3f}`, and overall surrogate `{result.decision.surrogate_score:.3f}`. I would value a quick sanity check on whether the correspondence and failure modes below match the intended behavior of the paper.

Attached are the metric table, artifacts, and the questions above.
"""


def render_analysis(summary: RunSummary) -> str:
    lines = [
        "# Extreme Motion Reimplementation Analysis",
        "",
        f"- Run ID: `{summary.run_id}`",
        f"- Papers evaluated: `{len(summary.results)}`",
        f"- Scenarios loaded: `{len(summary.scenarios)}`",
        f"- Promoted papers: `{summary.promoted_count}`",
        f"- Stopped early: `{'yes' if summary.stopped_early else 'no'}`",
        "",
        "## Ranking",
        "",
        "| Paper | Surrogate | Utility | Parity | Economy | Promoted |",
        "|---|---:|---:|---:|---:|---|",
    ]

    ranked = sorted(summary.results, key=lambda item: item.decision.surrogate_score, reverse=True)
    for result in ranked:
        lines.append(
            f"| `{result.paper.id}` | {result.decision.surrogate_score:.4f} | "
            f"{result.decision.utility_score:.4f} | {result.decision.parity_score:.4f} | "
            f"{result.decision.code_economy_score:.4f} | {'yes' if result.decision.promoted else 'no'} |"
        )

    lines.extend(["", "## Per Paper", ""])
    for result in summary.results:
        lines.append(f"### {result.paper.name}")
        lines.append("")
        lines.append(result.paper.summary)
        lines.append("")
        lines.append(_score_summary(result))
        lines.append("")
        lines.append("Canonical gate:")
        lines.append(json.dumps(result.decision.canonical_gate.per_metric, indent=2))
        lines.append("")
        lines.append("Applied gate:")
        lines.append(json.dumps(result.decision.applied_gate.per_metric, indent=2))
        lines.append("")

    lines.extend(
        [
            "## Integrated View",
            "",
            f"Promoted ladder: `{', '.join(summary.promoted_ids) if summary.promoted_ids else 'none'}`",
            "",
            "The ladder is considered ready for downstream breakdance utility once `cotracker3` and `motionbert` are both promoted. `sam3d` and `sam4d` extend the integrated stack toward 3D and cross-modal segmentation rather than blocking the core audio-motion analysis path.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(summary: RunSummary) -> None:
    paper_dir = summary.output_dir / "paper_results"
    packet_dir = summary.output_dir / "author_packets"
    paper_dir.mkdir(parents=True, exist_ok=True)
    packet_dir.mkdir(parents=True, exist_ok=True)

    for result in summary.results:
        (paper_dir / f"{result.paper.id}.json").write_text(json.dumps(result.as_dict(), indent=2))
        (packet_dir / f"{result.paper.id}.md").write_text(render_author_packet(result))

    analysis = render_analysis(summary)
    (summary.output_dir / "ANALYSIS.md").write_text(analysis)

    manifest = {
        "run_id": summary.run_id,
        "output_dir": str(summary.output_dir),
        "promoted_ids": summary.promoted_ids,
        "stopped_early": summary.stopped_early,
        "results": [result.as_dict() for result in summary.results],
    }
    (summary.output_dir / "summary.json").write_text(json.dumps(manifest, indent=2))
