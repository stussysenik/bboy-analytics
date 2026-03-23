PYTHON ?= uv run

.PHONY: sync dry-run test score-synthetic research-overnight research-status research-resume

sync:
	uv sync --extra dev

dry-run:
	$(PYTHON) extreme-motion-reimpl run --continue-on-fail

test:
	uv run --extra dev pytest

score-synthetic:
	$(PYTHON) extreme-motion-reimpl score-audio-motion --synthetic --samples 256

research-overnight:
	cd research && bun install && bun run research-runner.ts 2>&1 | tee ../data/research/research.log

research-status:
	cd research && bun run checkpoint.ts --status

research-resume:
	cd research && bun run research-runner.ts --resume 2>&1 | tee -a ../data/research/research.log
