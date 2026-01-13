# knobs_v2.py
from __future__ import annotations

# ----------------------------
# Confidence / Gating Knobs
# ----------------------------

# Minimum score for accepting top result (scale depends on route)
MIN_SCORE = -0.5

# Dominance gap: top1 - top2 must be >= this
MIN_DOMINANCE_GAP = 0.05

# If you have rerank logits (often uncalibrated), dominance is more reliable
RERANK_MIN_DOMINANCE_GAP = 0.35

# Optional: debug mode
DEBUG_KNOBS = False
