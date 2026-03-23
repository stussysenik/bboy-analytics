# WHAM Viability Spike

**Date:** 2026-03-23
**Verdict:** Superseded by GVHMR. None of these solve inversions.

## Key Findings

1. **WHAM** (CVPR 2024): W-MPJPE 354mm, 4.4 FPS, **12x more sensitive to camera noise than GVHMR**.
2. **GVHMR** (SIGGRAPH Asia 2024): W-MPJPE 274mm, network 5100 FPS, only 1.6mm drop with estimated camera.
3. **TRAM** (ECCV 2024): W-MPJPE 222mm, doesn't use MoCap prior for trajectory — may handle unusual motions better.
4. **JOSH3** (ICLR 2026): W-MPJPE **175mm**, joint scene+human optimization. Best accuracy but 0.8 FPS.
5. **ALL fail on inversions** — trained on AMASS (walking, sitting, sports). Zero breakdancing data.
6. **M1 Max:** No model works — all require CUDA (DPVO dependency).
7. **GH5 vs iPhone:** Resolution irrelevant (all crop to 256x192). GH5 wins on tripod stability = better SLAM.
8. **Multi-camera:** All strictly monocular. Multi-view fusion requires custom engineering.

## World Trajectory Accuracy (EMDB-2)

| Method | W-MPJPE (mm) | Drift over 3.3s | Camera Robustness |
|--------|-------------|-----------------|-------------------|
| WHAM | 354 | ~35cm | Poor (12x sensitive) |
| GVHMR | 274 | ~27cm | Good (1.6mm drop) |
| TRAM | 222 | ~22cm | Good (uses SLAM) |
| JOSH3 | 175 | ~17cm | Best (scene optimization) |

## GH5 Capture Guidelines
- Shoot 1080p 120fps in Rec.709 (NOT V-Log — washed out frames degrade detection)
- Lock focal length (zoom changes break SLAM)
- Use tripod (eliminates camera shake, the biggest error source)
- Fixed exposure if possible

## Recommendation
Use GVHMR for toprock/footwork/freezes. Accept power moves need a different approach.
Use GH5 on tripod for best results. iPhone handheld is the worst case for world trajectory.
