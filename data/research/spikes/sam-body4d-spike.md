# SAM-Body4D Viability Spike

**Date:** 2026-03-23
**Verdict:** Real but overhyped. CUDA-only, 1-2s/frame on H800, zero inversion evidence.

## Key Findings

1. **Paper exists** — arXiv:2512.08406, Dec 2025, Gao/Miao/Han. MIT licensed. Working Gradio demo on GitHub.
2. **"Training-free" is accurate** — orchestrates 5 pretrained models (SAM 3, SAM 3D Body, Diffusion-VAS, MoGe-2, Depth Anything V2).
3. **Zero quantitative benchmarks** — no MPJPE, no PA-MPJPE, no numerical comparisons. Only visual comparisons. Major red flag.
4. **Hardware massively understated** — TECH_STACK claimed ~200ms/frame. Actual: 1-2s/frame WITHOUT occlusion refinement on H800 (80GB), 17-18s WITH. Peak VRAM: 53GB.
5. **Cannot run on M1 Max** — SAM 3 has hard CUDA/Triton dependency. Minimum: RTX 4090 (24GB) without occlusion refinement.
6. **No inversion evidence** — paper never mentions inversions, dance, or acrobatics. The claim in TECH_STACK_REEVALUATION.md was an unsubstantiated inference.

## Alternatives Found
- **HSMR** (CVPR 2025): Biomechanical joint constraints, beats HMR 2.0 by >10mm on extreme yoga poses (MOYO dataset)
- **GenHMR** (AAAI 2025): Generative uncertainty modeling, 25-30% MPJPE reduction on SOTA

## Recommendation
Rent an A100 (~$5), run SAM-Body4D + HSMR + GenHMR on 5 BRACE dataset clips. Replace speculation with evidence.
