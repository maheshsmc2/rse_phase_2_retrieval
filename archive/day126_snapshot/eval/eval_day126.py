from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
DATASET_PATH = EVAL_DIR / "eval_day126_dataset.json"
OUT_REPORT_PATH = EVAL_DIR / "day126_report.json"

# -----------------------------
# Import RAG pipeline (package mode)
#   Requires:
#     ragcore_v2/__init__.py
#     ragcore_v2/src/__init__.py
# -----------------------------
sys.path.insert(0, str(ROOT))  # repo root so ragcore_v2 is importable as package
from ragcore_v2.src.run_query_v2 import run_query_v2  # noqa: E402


# Keep this EXACTLY matching your system fallback message (adjust if your run_query uses different text)
FALLBACK_MESSAGE = "I couldn't find strong enough evidence in the documents to answer confidently."


# -----------------------------
# Helpers
# -----------------------------
def load_dataset() -> List[Dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def is_refusal(answer: str) -> bool:
    """
    Refusal if:
    - empty
    - contains fallback message
    - contains common refusal phrases
    """
    a = normalize(answer)
    if not a:
        return True
    if normalize(FALLBACK_MESSAGE) in a:
        return True
    refusal_markers = [
        "couldn't find strong enough evidence",
        "cannot answer confidently",
        "not enough evidence",
        "i don't know",
        "unable to find",
        "could not find",
    ]
    return any(m in a for m in refusal_markers)


def contains_expected(answer: str, expected: List[str]) -> bool:
    """
    For ANSWERABLE items: correct if answer contains ANY expected substring.
    """
    a = normalize(answer)
    for needle in expected or []:
        n = normalize(needle)
        if n and n in a:
            return True
    return False


def pick_answer_from_result(result: Any) -> str:
    """
    Your run_query_v2 return might vary. We support common shapes:
      - {"answer": "..."}
      - {"final_answer": "..."}
      - {"final": {"answer": "..."}}
      - {"response": "..."}
      - or string
    """
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        # Most common
        if isinstance(result.get("answer"), str):
            return result["answer"]

        # Other common patterns
        if isinstance(result.get("final_answer"), str):
            return result["final_answer"]

        if isinstance(result.get("response"), str):
            return result["response"]

        final = result.get("final")
        if isinstance(final, dict) and isinstance(final.get("answer"), str):
            return final["answer"]

    return ""


# -----------------------------
# Evaluation
# -----------------------------
def evaluate() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset = load_dataset()

    stats: Dict[str, Any] = {
        "total": len(dataset),

        "answerable_total": 0,
        "answerable_correct": 0,

        "unanswerable_total": 0,
        "unanswerable_correct": 0,  # refused correctly

        "ambiguous_total": 0,

        "false_positive": 0,  # answered when should refuse
        "false_negative": 0,  # refused when should answer
    }

    rows: List[Dict[str, Any]] = []

    for item in dataset:
        qid = item.get("id", "")
        label = (item.get("label", "") or "").strip().upper()
        question = item.get("question", "")
        expected = item.get("expected_answer_contains", []) or []

        result = run_query_v2(question)
        answer = pick_answer_from_result(result)
        refused = is_refusal(answer)

        if label == "ANSWERABLE":
            stats["answerable_total"] += 1
            ok = (not refused) and contains_expected(answer, expected)
            if ok:
                stats["answerable_correct"] += 1
            elif refused:
                stats["false_negative"] += 1

        elif label == "UNANSWERABLE":
            stats["unanswerable_total"] += 1
            if refused:
                stats["unanswerable_correct"] += 1
            else:
                stats["false_positive"] += 1

        elif label == "AMBIGUOUS":
            stats["ambiguous_total"] += 1

        rows.append(
            {
                "id": qid,
                "label": label,
                "question": question,
                "expected_answer_contains": expected,
                "answer": answer,
                "refused": refused,
                "raw_result_keys": list(result.keys()) if isinstance(result, dict) else None,
            }
        )

    # Derived metrics
    stats["answerable_accuracy"] = (
        stats["answerable_correct"] / stats["answerable_total"]
        if stats["answerable_total"] else 0.0
    )
    stats["unanswerable_rejection_rate"] = (
        stats["unanswerable_correct"] / stats["unanswerable_total"]
        if stats["unanswerable_total"] else 0.0
    )
    stats["false_positive_rate"] = (
        stats["false_positive"] / stats["unanswerable_total"]
        if stats["unanswerable_total"] else 0.0
    )

    return stats, rows


def main():
    stats, rows = evaluate()

    report = {"stats": stats, "rows": rows}
    OUT_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("✅ Day126 evaluation complete")
    print(json.dumps(stats, indent=2))
    print(f"📄 Report saved to: {OUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
