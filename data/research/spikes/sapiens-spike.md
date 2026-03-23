# Sapiens Viability Spike

**Date:** 2026-03-23
**Verdict:** Best multi-task human model, strong for offline analysis. NOT real-time, NOT iPhone-compatible.

## Key Findings

1. **Paper:** arXiv:2408.12569, ECCV 2024 Oral. Meta Research. ViT pretrained on 300M human images.
2. **Tasks:** Pose (17/133/308 kpts) + Segmentation (28 classes) + Depth + Surface Normals — all from one backbone.
3. **133-keypoint mode is the sweet spot** for bboy: full hand articulation + foot detail (big toe, small toe, heel).
4. **Hand/foot accuracy is best available** — +10 AP over ViTPose on hands. Critical for freeze and footwork scoring.
5. **Inverted poses explicitly listed as a limitation** in the paper. 300M pretraining helps but pose finetuning is on indoor studio data (upright).
6. **M1 Max:** 0.3B model ~0.5-1 FPS via MPS. 1B model impractical.
7. **iPhone:** Not possible. No CoreML/ONNX export.
8. **License:** CC-BY-NC-4.0 — non-commercial only!
9. **All weights released** on HuggingFace.

## Benchmark Numbers (Whole-body AP, Humans-5K)

| Model | Body | Foot | Hand | Whole |
|-------|------|------|------|-------|
| DWPose-l | 57.9 | 56.5 | 49.3 | 53.1 |
| Sapiens 0.3B | 58.1 | 56.8 | 49.6 | 53.4 |
| Sapiens 0.6B | 59.8 | 64.7 | 52.1 | 56.2 |
| Sapiens 1B | 62.9 | 68.3 | 55.9 | 59.4 |

## Hardware Requirements

| Variant | VRAM | Speed (RTX 4090) |
|---------|------|-----------------|
| 0.3B | 8+ GB | ~4-8 FPS |
| 0.6B | 12+ GB | ~2-4 FPS |
| 1B | 16+ GB | ~0.7-4 FPS |
| 2B | 24-40+ GB | <1 FPS |

## Recommendation
Use Sapiens 1B (133-kpt) for offline analysis on cloud GPU. Use RTMPose-x for real-time preview.
