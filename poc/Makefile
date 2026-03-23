.PHONY: setup inference analyze poc clean

# Full POC pipeline (run on GPU instance)
poc: setup inference analyze

# Step 1: Install GVHMR + dependencies
setup:
	bash remote/setup-gvhmr.sh

# Step 2: Run GVHMR inference (pass VIDEO=path/to/video.mp4)
inference:
	bash remote/run-inference.sh $(VIDEO)

# Step 2b: Static camera mode (skip visual odometry)
inference-static:
	bash remote/run-inference.sh $(VIDEO) --static

# Step 3: Compute musicality score
analyze:
	pip install -q scipy librosa matplotlib 2>/dev/null; \
	python analyze.py --video=$(VIDEO) --fps=$(or $(FPS),30)

# Analyze without audio (synthetic 120 BPM beats)
analyze-synthetic:
	pip install -q scipy matplotlib 2>/dev/null; \
	python analyze.py --no-audio --fps=$(or $(FPS),30)

# Clean results
clean:
	rm -rf results/*.npy results/*.json results/*.png results/*.wav
