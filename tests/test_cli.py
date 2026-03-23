from __future__ import annotations

import json
from pathlib import Path

from extreme_motion_reimpl.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_run_cli_generates_summary_and_packets(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    output_root = tmp_path / "runs"
    exit_code = main(["run", "--continue-on-fail", "--output-root", str(output_root)])

    assert exit_code == 0

    latest_dir = Path((output_root / "latest.txt").read_text().strip())
    summary = json.loads((latest_dir / "summary.json").read_text())

    assert summary["promoted_ids"] == ["cotracker3", "motionbert", "sam3d", "sam4d"]
    assert (latest_dir / "ANALYSIS.md").exists()
    assert (latest_dir / "author_packets" / "cotracker3.md").exists()


def test_score_audio_motion_synthetic_cli(capsys) -> None:
    exit_code = main(["score-audio-motion", "--synthetic", "--samples", "128"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["alignment_peak"] > 0.7
