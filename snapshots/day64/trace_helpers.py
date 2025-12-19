# trace_helpers.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import time

TRACE_DIR = Path("trace")
TRACE_DIR.mkdir(parents=True, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def init_trace(query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "query": query,
        "meta": meta or {},
        "stages": {"dense": [], "bm25": [], "hybrid": [], "rerank": []},
        "final": {"selected": []},
        "timing_ms": {},
        "created_at": time.time(),
    }

def add_timing(trace: Dict[str, Any], key: str, ms: float) -> None:
    trace["timing_ms"][key] = round(ms, 3)

def save_trace(trace: Dict[str, Any], filename: Optional[str] = None) -> Path:
    if filename is None:
        filename = f"trace_{now_ts()}.json"
    out = TRACE_DIR / filename
    out.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def clip_text(t: str, n: int = 220) -> str:
    t = (t or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")
