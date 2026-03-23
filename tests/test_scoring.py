from __future__ import annotations

import json
from pathlib import Path

from extreme_motion_reimpl.manifest import load_papers
from extreme_motion_reimpl.models import CommandOutput
from extreme_motion_reimpl.scoring import score_paper


ROOT = Path(__file__).resolve().parents[1]


def _fixture(name: str) -> CommandOutput:
    payload = json.loads((ROOT / "fixtures" / name).read_text())
    return CommandOutput(
        paper_id=payload["paper_id"],
        mode=payload["mode"],
        canonical_metrics=payload["canonical_metrics"],
        applied_metrics=payload["applied_metrics"],
        runtime_cost=payload["runtime_cost"],
        code_stats=payload["code_stats"],
        artifacts=payload["artifacts"],
        open_questions=payload["open_questions"],
        notes=payload["notes"],
    )


def test_cotracker3_promotion_passes_with_gap_threshold() -> None:
    papers = {paper.id: paper for paper in load_papers(ROOT / "papers.yaml")}
    decision = score_paper(
        papers["cotracker3"],
        _fixture("cotracker3_oracle.json"),
        _fixture("cotracker3_reimpl.json"),
    )

    assert decision.canonical_gate.passed is True
    assert decision.applied_gate.passed is True
    assert decision.promoted is True
    assert decision.parity_score > 0.0


def test_motionbert_fails_when_mpjpe_gap_exceeds_limit() -> None:
    papers = {paper.id: paper for paper in load_papers(ROOT / "papers.yaml")}
    oracle = _fixture("motionbert_oracle.json")
    reimpl = _fixture("motionbert_reimpl.json")
    degraded = CommandOutput(
        paper_id=reimpl.paper_id,
        mode=reimpl.mode,
        canonical_metrics={**reimpl.canonical_metrics, "mpjpe_mm": 60.5},
        applied_metrics=reimpl.applied_metrics,
        runtime_cost=reimpl.runtime_cost,
        code_stats=reimpl.code_stats,
        artifacts=reimpl.artifacts,
        open_questions=reimpl.open_questions,
        notes=reimpl.notes,
    )

    decision = score_paper(papers["motionbert"], oracle, degraded)

    assert decision.canonical_gate.passed is False
    assert decision.promoted is False
