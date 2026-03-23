# cotracker3 — Minimal Implementation Audit

_Generated: 2026-03-23T03:58:46.630Z_

---

---

## Summary

The core insight: **CoTracker3 is architecturally simple.** It's a correlation-sampling + iterative-refinement transformer. The "3" innovation is entirely about training data (pseudo-labeling real videos), not architecture. For your bboy pipeline:

1. **~460 LOC** gets you the full inference pipeline
2. **Use pretrained weights** — training from scratch requires the pseudo-label infrastructure and GPU days
3. **The sliding window tracker is essential** — real bboy footage is 30-120 seconds, not 24 frames
4. **The correlation block is the most critical component** — get the bilinear sampling and grid construction exactly right, or nothing works
5. **The transformer is vanilla** — alternating time/point attention with standard pre-norm residuals

For the breakdancing use case specifically, the 16px search radius (`S=4, stride=4`) is the main bottleneck — extremity points at 13+ px/frame displacement are marginal. The fix is either higher FPS capture (reducing per-frame displacement) or increasing `S` to 6 at inference time (cheap, ~2x correlation compute).
