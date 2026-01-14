from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path so package imports work when running this file directly
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rse_phase_2_retrieval.src.retriever import dense_search  # noqa: E402


def _preview(t: str, n: int = 300) -> str:
    t = (t or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


def save_trace(query: str, hits: List[Dict[str, Any]]) -> Path:
    trace_dir = ROOT / "rse_phase_2_retrieval" / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "query": query,
        "top_k": len(hits),
        "hits": [
            {
                "id": h.get("id"),
                "score": float(h.get("score", 0.0)),
                "text_preview": _preview(h.get("text", ""), 220),
            }
            for h in hits
        ],
    }

    out = trace_dir / "last_trace.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    print("RUN_QUERY STARTED ✅")

    if len(sys.argv) < 2:
        print('Usage: python rse_phase_2_retrieval/src/run_query.py "your question"')
        raise SystemExit(2)

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}")

    hits = dense_search(query, top_k=5)
    print(f"Retrieved hits: {len(hits)}")

    print("\n=== TOP HITS ===")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] score={h['score']:.4f}  id={h['id']}")
        print(_preview(h["text"], 500))

    trace_path = save_trace(query, hits)
    print(f"\nTrace saved: {trace_path}")


if __name__ == "__main__":
    main()
