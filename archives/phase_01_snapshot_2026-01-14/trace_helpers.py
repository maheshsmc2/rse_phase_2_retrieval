# trace_helpers.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import time

TRACE_DIR = Path("trace")
TRACE_DIR.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def init_trace(query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Trace envelope for one query.
    Keep it stable so old traces remain readable.
    """
    return {
        "query": query,
        "meta": meta or {},
        "created_at_unix": time.time(),
        "timing_ms": {},
        "routes": {
            "dense": [],
            "bm25": [],
            "hybrid": [],
            "rerank": [],
        },
        "stages": {},  # optional debug previews (safe even if unused)
        "final": {
            "selected": [],   # list of chosen chunks/docs
            "answer": None,   # if you later add an answer builder
            "reason": None,   # why selection won (dominance, threshold, etc.)
        },
    }


def add_timing(trace: Dict[str, Any], key: str, ms: float) -> None:
    trace["timing_ms"][key] = round(float(ms), 3)


def add_hits(
    trace: Dict[str, Any],
    route: str,
    hits: list[dict],
) -> None:
    """
    hits: list of dicts like:
      {"id": "...", "score": 0.123, "text": "...", "meta": {...}}
    Keep the shape simple and JSON-safe.
    """
    if route not in trace["routes"]:
        raise ValueError(f"Unknown route '{route}'. Expected one of: {list(trace['routes'].keys())}")
    trace["routes"][route] = hits


def set_final(
    trace: Dict[str, Any],
    selected: list[dict],
    reason: str,
    answer: Optional[str] = None,
) -> None:
    trace["final"]["selected"] = selected
    trace["final"]["reason"] = reason
    trace["final"]["answer"] = answer


def save_trace(trace: Dict[str, Any], filename: Optional[str] = None) -> Path:
    if filename is None:
        filename = f"trace_{now_ts()}.json"
    out_path = TRACE_DIR / filename
    out_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
