from __future__ import annotations
from typing import Any, Dict, List


def _as_score(x: Any) -> float:
    """Best-effort score extractor (supports 'score', 'score_rerank', etc.)."""
    if x is None:
        return float("-inf")
    if isinstance(x, (int, float)):
        return float(x)
    return float(x)


def gate_results(
    results: List[Dict[str, Any]],
    *,
    min_score: float = -12.0,
    min_gap: float = 0.25,
    # Day 73: lightweight semantic-absence check
    anchor_terms: List[str] | None = None,
    require_anchor: bool = False,
) -> Dict[str, Any]:
    """
    Returns a gate dict:
      {
        "pass": bool,
        "reason": str,
        "score1": float,
        "score2": float,
        "gap": float
      }

    Rules:
      1) If no results -> FAIL(no_results)
      2) If top1 score < min_score -> FAIL(low_top1_score)
      3) If (top1 - top2) < min_gap -> FAIL(low_gap)
      4) Optional: if require_anchor=True and none of anchor_terms appear in top text -> FAIL(semantic_absence)
    """

    def pass_gate(score1: float, score2: float, gap: float) -> Dict[str, Any]:
        return {
            "pass": True,
            "reason": "pass",
            "score1": float(score1),
            "score2": float(score2),
            "gap": float(gap),
        }

    def fail_gate(reason: str, score1: float, score2: float, gap: float) -> Dict[str, Any]:
        return {
            "pass": False,
            "reason": reason,
            "score1": float(score1),
            "score2": float(score2),
            "gap": float(gap),
        }

    if not results:
        return fail_gate("no_results", float("-inf"), float("-inf"), 0.0)

    # score keys: your pipeline sets universal key "score"
    score1 = _as_score(results[0].get("score", results[0].get("score_rerank", float("-inf"))))
    score2 = _as_score(results[1].get("score", results[1].get("score_rerank", float("-inf")))) if len(results) > 1 else float("-inf")
    gap = score1 - score2 if score2 != float("-inf") else float("inf")

    # 1) floor
    if score1 < float(min_score):
        return fail_gate(f"low_top1_score(<{min_score})", score1, score2, gap)

    # 2) dominance gap
    if gap < float(min_gap):
        return fail_gate(f"low_gap(<{min_gap})", score1, score2, gap)

    # 3) optional semantic absence (OFF by default)
    if anchor_terms is None:
        anchor_terms = ["policy", "allowed", "days", "period", "notice", "probation"]

    if require_anchor:
        top_text = (results[0].get("text") or "").lower()
        if not any(t.lower() in top_text for t in anchor_terms):
            return fail_gate("semantic_absence", score1, score2, gap)

    return pass_gate(score1, score2, gap)
