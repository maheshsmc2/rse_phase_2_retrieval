from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List

from .phase2_adapter import get_phase2_topk
from .reranker import CrossEncoderReranker, DEFAULT_RERANK_MODEL
from .gating import GateConfig, gate
from .trace_helpers import init_trace, add_timing, save_trace


def _preview(text: str, n: int = 160) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


def run_query_v3(
    query: str,
    *,
    top_k: int = 10,
    route: str = "dense",
    rerank_model: str = DEFAULT_RERANK_MODEL,
    gate_cfg: GateConfig = GateConfig(),
    keep_n: int = 3,
) -> Dict[str, Any]:
    trace = init_trace(query, meta={"route": route, "top_k": top_k})

    t0 = time.time()
    hits = get_phase2_topk(query, top_k=top_k, route=route)
    add_timing(trace, "phase2_retrieval", (time.time() - t0) * 1000)

    trace["phase2"]["topk"] = [
        {"chunk_id": h.chunk_id, "retr_score": h.score, "text_preview": _preview(h.text)}
        for h in hits
    ]

    if not hits:
        trace["final"]["decision"] = "ABSTAIN"
        trace["final"]["reason"] = "FAIL_NO_RETRIEVAL"
        save_trace(trace, tag="abstain")
        return trace

    # Rerank
    t1 = time.time()
    rr = CrossEncoderReranker(rerank_model)
    candidates = [(h.chunk_id, h.text, h.score) for h in hits]
    reranked = rr.rerank(query, candidates)
    add_timing(trace, "rerank", (time.time() - t1) * 1000)

    trace["rerank"]["model"] = rerank_model
    trace["rerank"]["candidates"] = [
        {
            "chunk_id": r.chunk_id,
            "retr_score": r.retr_score,
            "rerank_score": r.rerank_score,
            "text_preview": _preview(r.text),
        }
        for r in reranked
    ]

    top1 = reranked[0]
    top2 = reranked[1] if len(reranked) > 1 else None

    g = gate(
        top1_rerank=top1.rerank_score,
        top1_retr=top1.retr_score,
        top2_rerank=(top2.rerank_score if top2 else None),
        cfg=gate_cfg,
    )

    trace["gate"] = {
        "min_rerank_score": gate_cfg.min_rerank_score,
        "min_dominance_gap": gate_cfg.min_dominance_gap,
        "min_retr_score": gate_cfg.min_retr_score,
        "top1_rerank_score": g.top1_score,
        "top2_rerank_score": g.top2_score,
        "dominance_gap": g.dominance_gap,
        "decision": g.decision,
        "reason": g.reason,
    }

    if g.decision == "ABSTAIN":
        trace["final"]["decision"] = "ABSTAIN"
        trace["final"]["reason"] = g.reason
        save_trace(trace, tag="abstain")
        return trace

    selected = reranked[: max(1, keep_n)]
    trace["final"]["decision"] = "ANSWER"
    trace["final"]["reason"] = g.reason
    trace["final"]["selected"] = [
        {
            "chunk_id": s.chunk_id,
            "retr_score": s.retr_score,
            "rerank_score": s.rerank_score,
            "text": s.text,
        }
        for s in selected
    ]

    save_trace(trace, tag="answer")
    return trace


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: rerank + confidence gating over Phase 2 top-k hits")
    ap.add_argument("query", type=str, help="User query")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--route", type=str, default="dense", help="Passed through to Phase 2")
    ap.add_argument("--rerank_model", type=str, default=DEFAULT_RERANK_MODEL)

    ap.add_argument("--min_rerank_score", type=float, default=0.20)
    ap.add_argument("--min_gap", type=float, default=0.08)
    ap.add_argument("--min_retr_score", type=float, default=0.0)

    ap.add_argument("--keep_n", type=int, default=3)
    args = ap.parse_args()

    cfg = GateConfig(
        min_rerank_score=args.min_rerank_score,
        min_dominance_gap=args.min_gap,
        min_retr_score=args.min_retr_score,
    )

    trace = run_query_v3(
        args.query,
        top_k=args.top_k,
        route=args.route,
        rerank_model=args.rerank_model,
        gate_cfg=cfg,
        keep_n=args.keep_n,
    )

    print("\n=== PHASE 3 DECISION ===")
    print("Decision:", trace["final"]["decision"])
    print("Reason  :", trace["final"]["reason"])

    if trace["final"]["decision"] == "ANSWER":
        sel0 = trace["final"]["selected"][0]
        print("\nTop selected chunk:")
        print(" - chunk_id     :", sel0["chunk_id"])
        print(" - retr_score   :", sel0["retr_score"])
        print(" - rerank_score :", sel0["rerank_score"])
        print(" - preview      :", _preview(sel0["text"], 240))

    print("\nTrace saved in ./trace_phase3/ (see latest file).")


if __name__ == "__main__":
    main()
