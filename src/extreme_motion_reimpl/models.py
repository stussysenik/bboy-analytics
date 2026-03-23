from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SourceLink:
    label: str
    url: str


@dataclass(frozen=True)
class MetricTarget:
    name: str
    comparator: str
    target: float
    weight: float = 1.0
    description: str = ""
    oracle_metric: str | None = None


@dataclass(frozen=True)
class CommandSpec:
    cmd: str
    workdir: str = "."
    executor: str = "local"
    target: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperSpec:
    id: str
    name: str
    summary: str
    official_sources: list[SourceLink]
    dataset_slice: dict[str, str]
    research_subject: str
    author_packet_targets: list[str]
    canonical_targets: list[MetricTarget]
    applied_targets: list[MetricTarget]
    oracle_cmd: CommandSpec
    reimpl_cmd: CommandSpec


@dataclass(frozen=True)
class Scenario:
    id: str
    video_path: str
    audio_path: str
    tags: list[str]
    notes: str


@dataclass(frozen=True)
class CommandOutput:
    paper_id: str
    mode: str
    canonical_metrics: dict[str, float]
    applied_metrics: dict[str, float]
    runtime_cost: dict[str, float]
    code_stats: dict[str, float]
    artifacts: list[str]
    open_questions: list[str]
    notes: str = ""


@dataclass(frozen=True)
class GateEvaluation:
    passed: bool
    attainment: float
    per_metric: dict[str, float]


@dataclass(frozen=True)
class PaperDecision:
    utility_score: float
    parity_score: float
    code_economy_score: float
    surrogate_score: float
    canonical_gate: GateEvaluation
    applied_gate: GateEvaluation
    promoted: bool


@dataclass(frozen=True)
class PaperResult:
    paper: PaperSpec
    oracle: CommandOutput
    reimplementation: CommandOutput
    decision: PaperDecision

    def as_dict(self) -> dict:
        return {
            "paper": {
                "id": self.paper.id,
                "name": self.paper.name,
                "summary": self.paper.summary,
                "dataset_slice": self.paper.dataset_slice,
                "research_subject": self.paper.research_subject,
                "author_packet_targets": self.paper.author_packet_targets,
            },
            "oracle": {
                "paper_id": self.oracle.paper_id,
                "mode": self.oracle.mode,
                "canonical_metrics": self.oracle.canonical_metrics,
                "applied_metrics": self.oracle.applied_metrics,
                "runtime_cost": self.oracle.runtime_cost,
                "code_stats": self.oracle.code_stats,
                "artifacts": self.oracle.artifacts,
                "open_questions": self.oracle.open_questions,
                "notes": self.oracle.notes,
            },
            "reimplementation": {
                "paper_id": self.reimplementation.paper_id,
                "mode": self.reimplementation.mode,
                "canonical_metrics": self.reimplementation.canonical_metrics,
                "applied_metrics": self.reimplementation.applied_metrics,
                "runtime_cost": self.reimplementation.runtime_cost,
                "code_stats": self.reimplementation.code_stats,
                "artifacts": self.reimplementation.artifacts,
                "open_questions": self.reimplementation.open_questions,
                "notes": self.reimplementation.notes,
            },
            "decision": {
                "utility_score": self.decision.utility_score,
                "parity_score": self.decision.parity_score,
                "code_economy_score": self.decision.code_economy_score,
                "surrogate_score": self.decision.surrogate_score,
                "canonical_gate": {
                    "passed": self.decision.canonical_gate.passed,
                    "attainment": self.decision.canonical_gate.attainment,
                    "per_metric": self.decision.canonical_gate.per_metric,
                },
                "applied_gate": {
                    "passed": self.decision.applied_gate.passed,
                    "attainment": self.decision.applied_gate.attainment,
                    "per_metric": self.decision.applied_gate.per_metric,
                },
                "promoted": self.decision.promoted,
            },
        }


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    output_dir: Path
    scenarios: list[Scenario]
    results: list[PaperResult]
    stopped_early: bool

    @property
    def promoted_ids(self) -> list[str]:
        return [result.paper.id for result in self.results if result.decision.promoted]

    @property
    def promoted_count(self) -> int:
        return len(self.promoted_ids)
