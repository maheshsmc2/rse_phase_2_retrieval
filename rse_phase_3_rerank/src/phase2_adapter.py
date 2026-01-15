from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import importlib.util


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    meta: Dict[str, Any]


def _coerce_hit(h: Dict[str, Any]) -> Hit:
    chunk_id = str(h.get("chunk_id") or h.get("id") or h.get("doc_id") or h.get("key") or "").strip()
    if not chunk_id:
        chunk_id = "noid_" + (str(h.get("text", ""))[:24].replace(" ", "_"))

    text = str(h.get("text") or h.get("chunk") or h.get("content") or "")
    score = float(h.get("score") or h.get("sim") or 0.0)

    meta = dict(h.get("meta") or {})
    # preserve extra fields
    for k, v in h.items():
        if k not in ("chunk_id","id","doc_id","key","text","chunk","content","score","sim","meta"):
            meta[k] = v

    return Hit(chunk_id=chunk_id, text=text, score=score, meta=meta)


def _load_phase2_dense_search():
    repo_root = Path(__file__).resolve().parents[2]
    phase2_file = repo_root / "ragcore_v2" / "src" / "retriever_v2.py"

    if not phase2_file.exists():
        raise FileNotFoundError(f"Could not find Phase 2 retriever at: {phase2_file}")

    spec = importlib.util.spec_from_file_location("phase2_retriever_v2", str(phase2_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {phase2_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "dense_search"):
        raise AttributeError("Phase 2 retriever does not define dense_search")

    return mod.dense_search



def get_phase2_topk(query: str, *, top_k: int = 10, route: str = "dense") -> List[Hit]:
    dense_search = _load_phase2_dense_search()

    # route kept for future hybrid; for now we map to dense_search
    raw_hits = dense_search(query, top_k=top_k)

    if not isinstance(raw_hits, list):
        raise TypeError(f"dense_search expected list of hits, got: {type(raw_hits)}")

    return [_coerce_hit(h) for h in raw_hits]
