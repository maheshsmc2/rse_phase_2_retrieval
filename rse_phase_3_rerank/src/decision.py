from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Decision:
    decision: str  # "ANSWER" | "ABSTAIN"
    reason: str
    selected: List[Dict[str, Any]]  # usually 1-3 chunks
    answer: Optional[str] = None  # Phase 3 does not generate answers; keep hook
