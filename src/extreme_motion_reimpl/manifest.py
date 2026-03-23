from __future__ import annotations

import json
from pathlib import Path

import yaml

from .models import CommandSpec, MetricTarget, PaperSpec, Scenario, SourceLink


def _source_links(items: list[dict]) -> list[SourceLink]:
    return [SourceLink(label=item["label"], url=item["url"]) for item in items]


def _targets(items: list[dict]) -> list[MetricTarget]:
    return [
        MetricTarget(
            name=item["name"],
            comparator=item["comparator"],
            target=float(item["target"]),
            weight=float(item.get("weight", 1.0)),
            description=item.get("description", ""),
            oracle_metric=item.get("oracle_metric"),
        )
        for item in items
    ]


def _command(spec: dict) -> CommandSpec:
    return CommandSpec(
        cmd=spec["cmd"],
        workdir=spec.get("workdir", "."),
        executor=spec.get("executor", "local"),
        target=spec.get("target"),
        env={key: str(value) for key, value in spec.get("env", {}).items()},
    )


def load_papers(path: str | Path) -> list[PaperSpec]:
    payload = yaml.safe_load(Path(path).read_text())
    return [
        PaperSpec(
            id=item["id"],
            name=item["name"],
            summary=item["summary"],
            official_sources=_source_links(item["official_sources"]),
            dataset_slice={key: str(value) for key, value in item["dataset_slice"].items()},
            research_subject=item["research_subject"],
            author_packet_targets=list(item.get("author_packet_targets", [])),
            canonical_targets=_targets(item["canonical_targets"]),
            applied_targets=_targets(item["applied_targets"]),
            oracle_cmd=_command(item["oracle_cmd"]),
            reimpl_cmd=_command(item["reimpl_cmd"]),
        )
        for item in payload["papers"]
    ]


def load_scenarios(path: str | Path) -> list[Scenario]:
    payload = json.loads(Path(path).read_text())
    return [
        Scenario(
            id=item["id"],
            video_path=item["video_path"],
            audio_path=item["audio_path"],
            tags=list(item["tags"]),
            notes=item["notes"],
        )
        for item in payload["scenarios"]
    ]
