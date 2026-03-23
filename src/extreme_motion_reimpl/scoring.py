from __future__ import annotations

from dataclasses import dataclass

from .models import CommandOutput, GateEvaluation, MetricTarget, PaperDecision, PaperSpec


@dataclass(frozen=True)
class ScoreWeights:
    applied_utility: float = 0.50
    canonical_parity: float = 0.30
    code_economy: float = 0.20


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _metric_attainment(
    target: MetricTarget,
    candidate_metrics: dict[str, float],
    oracle_metrics: dict[str, float] | None = None,
) -> float:
    if target.name not in candidate_metrics:
        return 0.0

    candidate = float(candidate_metrics[target.name])
    comparator = target.comparator

    if comparator == "gte":
        if target.target == 0:
            return 1.0
        return _clamp(candidate / target.target)

    if comparator == "lte":
        if candidate <= 0:
            return 1.0
        return _clamp(target.target / candidate)

    if comparator == "max_gap":
        if not oracle_metrics:
            return 0.0
        oracle_name = target.oracle_metric or target.name
        if oracle_name not in oracle_metrics:
            return 0.0
        gap = abs(candidate - float(oracle_metrics[oracle_name]))
        if target.target == 0:
            return 1.0 if gap == 0 else 0.0
        return _clamp(1.0 - (gap / target.target))

    raise ValueError(f"Unsupported comparator: {comparator}")


def _metric_passed(
    target: MetricTarget,
    candidate_metrics: dict[str, float],
    oracle_metrics: dict[str, float] | None = None,
) -> bool:
    if target.name not in candidate_metrics:
        return False

    candidate = float(candidate_metrics[target.name])

    if target.comparator == "gte":
        return candidate >= target.target

    if target.comparator == "lte":
        return candidate <= target.target

    if target.comparator == "max_gap":
        if not oracle_metrics:
            return False
        oracle_name = target.oracle_metric or target.name
        if oracle_name not in oracle_metrics:
            return False
        gap = abs(candidate - float(oracle_metrics[oracle_name]))
        return gap <= target.target

    raise ValueError(f"Unsupported comparator: {target.comparator}")


def evaluate_gate(
    targets: list[MetricTarget],
    candidate_metrics: dict[str, float],
    oracle_metrics: dict[str, float] | None = None,
) -> GateEvaluation:
    if not targets:
        return GateEvaluation(passed=True, attainment=1.0, per_metric={})

    weighted_sum = 0.0
    total_weight = 0.0
    per_metric: dict[str, float] = {}
    passes: list[bool] = []

    for target in targets:
        attainment = _metric_attainment(target, candidate_metrics, oracle_metrics)
        per_metric[target.name] = round(attainment, 4)
        passes.append(_metric_passed(target, candidate_metrics, oracle_metrics))
        weighted_sum += attainment * target.weight
        total_weight += target.weight

    passed = all(passes)
    aggregate = weighted_sum / total_weight if total_weight else 1.0
    return GateEvaluation(
        passed=passed,
        attainment=round(aggregate, 4),
        per_metric=per_metric,
    )


def code_economy_score(code_stats: dict[str, float]) -> float:
    implementation_loc = max(float(code_stats.get("implementation_loc", 0.0)), 1.0)
    files_touched = max(float(code_stats.get("files_touched", 0.0)), 1.0)
    shared_ratio = float(code_stats.get("shared_module_ratio", 0.0))

    loc_score = _clamp(1400.0 / implementation_loc)
    files_score = _clamp(10.0 / files_touched)
    shared_score = _clamp(shared_ratio / 0.65)

    return round((0.5 * loc_score) + (0.2 * files_score) + (0.3 * shared_score), 4)


def score_paper(
    paper: PaperSpec,
    oracle: CommandOutput,
    reimplementation: CommandOutput,
    weights: ScoreWeights | None = None,
) -> PaperDecision:
    weights = weights or ScoreWeights()

    canonical_gate = evaluate_gate(
        paper.canonical_targets,
        reimplementation.canonical_metrics,
        oracle.canonical_metrics,
    )
    applied_gate = evaluate_gate(
        paper.applied_targets,
        reimplementation.applied_metrics,
        None,
    )

    utility_score = applied_gate.attainment
    parity_score = canonical_gate.attainment
    economy_score = code_economy_score(reimplementation.code_stats)

    surrogate = round(
        (utility_score * weights.applied_utility)
        + (parity_score * weights.canonical_parity)
        + (economy_score * weights.code_economy),
        4,
    )

    return PaperDecision(
        utility_score=utility_score,
        parity_score=parity_score,
        code_economy_score=economy_score,
        surrogate_score=surrogate,
        canonical_gate=canonical_gate,
        applied_gate=applied_gate,
        promoted=canonical_gate.passed and applied_gate.passed,
    )
