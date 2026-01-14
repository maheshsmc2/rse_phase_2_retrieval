# rse_phase_2_retrieval/indexing/load_index.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import faiss

# Reuse Phase-1 index for now (fastest path to "Phase-2 alive")
DEFAULT_INDEX_DIR = Path("rse_phase_2_retrieval/data/indexes")


def load_faiss_bundle(index_dir: Path = DEFAULT_INDEX_DIR) -> Tuple[faiss.Index, Dict[str, Any]]:
    """
    Loads:
      - faiss.index
      - faiss_meta.json (must contain: 'ids', 'texts')
    Returns:
      (faiss_index, meta_dict)
    """
    index_dir = Path(index_dir)

    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "faiss_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index file: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing FAISS meta file: {meta_path}")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if "ids" not in meta or "texts" not in meta:
        raise ValueError("faiss_meta.json must contain keys: 'ids' and 'texts'")
    if len(meta["ids"]) != len(meta["texts"]):
        raise ValueError("Meta mismatch: len(ids) must equal len(texts)")

    return index, meta
