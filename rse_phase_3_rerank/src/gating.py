from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GateConfig:
    min_rerank_score: float = 0.20
    min_dominance_gap: float = 0.08
    min_retr_score: float = 0.0  # set >0 if you want a retrieval floor


@dataclass
class GateResult:
    decision: str  # "ANSWER" | "ABSTAIN"
    reason: str
    top1_score: float
    top2_score: Optional[float]
    dominance_gap: Optional[float]


def gate(top1_rerank: float, top1_retr: float, top2_rerank: Optional[float], cfg: GateConfig) -> GateResult:
    if top1_rerank < cfg.min_rerank_score:
        return GateResult(
            decision="ABSTAIN",
            reason=f"FAIL_LOW_RERANK_SCORE (< {cfg.min_rerank_score})",
            top1_score=top1_rerank,
            top2_score=top2_rerank,
            dominance_gap=(top1_rerank - top2_rerank) if top2_rerank is not None else None,
        )

    if top1_retr < cfg.min_retr_score:
        return GateResult(
            decision="ABSTAIN",
            reason=f"FAIL_LOW_RETR_SCORE (< {cfg.min_retr_score})",
            top1_score=top1_rerank,
            top2_score=top2_rerank,
            dominance_gap=(top1_rerank - top2_rerank) if top2_rerank is not None else None,
        )

    if top2_rerank is not None:
        gap = top1_rerank - top2_rerank
        if gap < cfg.min_dominance_gap:
            return GateResult(
                decision="ABSTAIN",
                reason=f"FAIL_AMBIGUOUS (gap {gap:.4f} < {cfg.min_dominance_gap})",
                top1_score=top1_rerank,
                top2_score=top2_rerank,
                dominance_gap=gap,
            )

    return GateResult(
        decision="ANSWER",
        reason="PASS_CONFIDENCE_GATE",
        top1_score=top1_rerank,
        top2_score=top2_rerank,
        dominance_gap=(top1_rerank - top2_rerank) if top2_rerank is not None else None,
    )
