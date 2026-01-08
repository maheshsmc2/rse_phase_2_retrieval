from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------
# Path fix: locate repo root (where retriever.py exists)
# ---------------------------------------------------------
_this = Path(__file__).resolve()
for parent in [_this.parent] + list(_this.parents):
    if (parent / "retriever.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not locate repo root")

from run_query import run_query

DATASET_PATH = Path("eval/eval_dataset.json")
OUT_PATH = Path("eval/day112_results.json")


def load_dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def contains_any(text: str, needles):
    t = (text or "").lower()
    return any(n.lower() in t for n in needles)


def main():
    data = load_dataset()
    results = []
    counts = Counter()

    for row in data:
        out = run_query(row["query"])

        decision = out.get("decision")
        passed_gate = out.get("passed_confidence_gate", False)

        retrieved_text = ""
        if out.get("results"):
            retrieved_text = out["results"][0].get("text", "")

        decision_ok = decision == row["expected"]

        content_ok = True
        if row["expected"] == "ANSWER":
            content_ok = contains_any(
                retrieved_text, row.get("gold_contains", [])
            )

        row_ok = decision_ok and content_ok

        counts["total"] += 1
        counts["decision_ok"] += int(decision_ok)
        counts["row_ok"] += int(row_ok)

        if row["expected"] == "ABSTAIN" and decision == "ANSWER":
            counts["false_pass"] += 1
        if row["expected"] == "ANSWER" and decision == "ABSTAIN":
            counts["false_abstain"] += 1

        results.append({
            "id": row["id"],
            "query": row["query"],
            "expected": row["expected"],
            "predicted": decision,
            "passed_gate": passed_gate,
            "row_ok": row_ok,
            "answer_snippet": retrieved_text[:160],
        })

    report = {
        "total": counts["total"],
        "decision_accuracy": counts["decision_ok"] / counts["total"],
        "overall_accuracy": counts["row_ok"] / counts["total"],
        "false_pass": counts["false_pass"],
        "false_abstain": counts["false_abstain"],
        "results": results,
    }

    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
