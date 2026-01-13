# ragcore_v2/src/run_query_v2.py
from __future__ import annotations

import time
from typing import Dict, Any, List
from .knobs_v2 import MIN_SCORE, MIN_DOMINANCE_GAP, RERANK_MIN_DOMINANCE_GAP


from .retriever_v2 import dense_search
from .trace_helpers import (
    init_trace,
    add_timing,
    add_hits,
    set_final,
    save_trace,
)



def _preview(text: str, n: int = 160) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")

def _log_stage(trace: dict, stage: str, results: list, *, top_k: int = 5) -> None:
    """
    Read-only logger: does NOT modify results.
    Expects each item in `results` to have at least:
      - id (or chunk_id)
      - text (or chunk text field)
      - score (float)
    """
    logged = []
    for r in (results or [])[:top_k]:
        chunk_id = r.get("id") or r.get("chunk_id") or r.get("doc_id")
        score = r.get("score")
        text = r.get("text") or r.get("chunk") or r.get("content") or ""
        logged.append(
            {
                "chunk_id": chunk_id,
                "score": score,
                "preview": _preview(text),
            }
        )
    trace["stages"][stage] = logged

def _hit_view(h: dict, max_chars: int = 260) -> dict:
    text = h.get("text") or h.get("chunk") or h.get("content") or ""
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return {
        "id": h.get("id") or h.get("doc_id") or h.get("chunk_id") or h.get("chunk_i"),
        "score": h.get("score"),
        "text": text,
        "meta": {
            "source": h.get("source"),
            "chunk_i": h.get("chunk_i"),
            **(h.get("meta") or {}),
        },
    }


def run_query_v2(query: str, route: str = "dense", top_k: int = 5) -> Dict[str, Any]:
    """
    v2 currently supports dense retrieval only (because retriever_v2.py exposes dense_search + index_info).
    We'll add hybrid + rerank later once those functions exist.
    """
    trace = init_trace(query, meta={"route": route})
    t0 = time.perf_counter()


    t = time.perf_counter()
    hits: List[dict] = dense_search(query, top_k=top_k)
    add_timing(trace, "dense_ms", (time.perf_counter() - t) * 1000)
    add_hits(trace, "dense", [_hit_view(h) for h in hits])

    selected = []
    reason = "no_hits"

    if hits:
        top1 = hits[0]
        top2 = hits[1] if len(hits) > 1 else None

        score_ok = top1["score"] >= MIN_SCORE
        gap_ok = (
            True if top2 is None
            else (top1["score"] - top2["score"]) >= MIN_DOMINANCE_GAP
        )

        if score_ok and gap_ok:
            selected = [top1]
            reason = f"dense: score_ok={score_ok}, dominance_gap_ok={gap_ok}"
        else:
            reason = f"dense: rejected (score_ok={score_ok}, dominance_gap_ok={gap_ok})"

    set_final(
        trace,
        selected=[_hit_view(h) for h in selected],
        reason=reason
    )

    add_timing(trace, "total_ms", (time.perf_counter() - t0) * 1000)
    trace_path = save_trace(trace)

    return {
        "query": query,
        "route": "dense",
        "selected": selected,
        "trace_path": str(trace_path),
    }
if __name__ == "__main__":
    import json

    q = input("Query> ").strip()
    if not q:
        raise SystemExit("Empty query")

    # Try a few common function names used in this repo style.
    # One of these should exist in the file.
    if "run_query_v2" in globals():
        out = run_query_v2(q)
    elif "run_query" in globals():
        out = run_query(q)
    else:
        raise SystemExit("No runnable function found (expected run_query_v2 or run_query).")

    print("\n--- RESULT ---")
    if isinstance(out, (dict, list)):
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(out)
