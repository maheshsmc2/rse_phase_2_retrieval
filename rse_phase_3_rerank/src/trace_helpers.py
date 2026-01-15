from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

TRACE_DIR = Path("trace_phase3")
TRACE_DIR.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def init_trace(query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "query": query,
        "meta": meta or {},
        "created_at_unix": time.time(),
        "timing_ms": {},
        "phase2": {
            "topk": [],
        },
        "rerank": {
            "model": None,
            "candidates": [],
        },
        "gate": {},
        "final": {
            "decision": None,   # ANSWER | ABSTAIN
            "reason": None,
            "selected": [],
            "answer": None,
        },
    }


def add_timing(trace: Dict[str, Any], key: str, ms: float) -> None:
    trace["timing_ms"][key] = round(float(ms), 3)


def save_trace(trace: Dict[str, Any], *, tag: str = "query") -> str:
    path = TRACE_DIR / f"{now_ts()}_{tag}.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
