from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# ----------------------------
# Day 72 knobs (confidence gating)
# ----------------------------
MIN_SCORE = -12.0   # CrossEncoder scores can be negative
MIN_GAP = 0.25      # dominance gap: score1 - score2

# Optional Day 73 semantic-absence knobs (only used for unanswerable)
ANCHOR_TERMS = ["policy", "allowed", "leave", "probation", "notice", "days", "period", "shall", "must"]

# ---------------------------------------------------------
# Path fix: find repo root (folder that contains retriever.py)
# ---------------------------------------------------------
_this = Path(__file__).resolve()
for parent in [_this.parent] + list(_this.parents):
    if (parent / "retriever.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError(
        "Could not locate repo root. Expected to find retriever.py in a parent directory."
    )

# ---------------------------------------------------------
# Imports (now local modules can be imported reliably)
# ---------------------------------------------------------
from retriever import hybrid_then_rerank  # Day 56 pipeline
from metrics import hit_at_k, recall_at_k, mean_rank  # eval/metrics.py
from gating import gate_results  # eval/gating.py

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DATASET_PATH = Path("eval/eval_dataset.json")

FINAL_K = 5
RETRIEVE_K = 20
ALPHA = 0.1  # keep aligned with DEFAULT_ALPHA for baseline run


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(query: str) -> List[Dict[str, Any]]:
    return hybrid_then_rerank(
        query,
        retrieve_k=RETRIEVE_K,
        final_k=FINAL_K,
        alpha=ALPHA,
    )


def main() -> None:
    data = load_dataset(DATASET_PATH)

    # ----------------------------
    # Retrieval metrics
    # ----------------------------
    hit_total = 0
    recall_total = 0.0
    ranks: List[float] = []

    worst_cases: List[Dict[str, Any]] = []
    gated_failures: List[Dict[str, Any]] = []

    # ----------------------------
    # Day 73 safety metrics
    # ----------------------------
    abstained = 0
    false_pass = 0
    should_abstain = 0

    for row in data:
        qid = row.get("id", "NA")
        query = (row.get("query") or "").strip()
        qtype = row.get("type", "normal")

        # For unanswerable, expected_ids can be empty
        expected = row.get("expected_ids")
        if expected is None:
            expected = row.get("expected_docs") or []

        if not query:
            raise ValueError(f"Row {qid} missing query")

        # Only enforce expected_ids for non-unanswerable
        if qtype != "unanswerable" and not expected:
            raise ValueError(f"Row {qid} missing expected_ids (type={qtype})")

        results = run_one(query)
        retrieved_ids = [r.get("id") for r in results if r.get("id") is not None]

        # ----------------------------
        # Gating
        # - normal: score+gap only
        # - unanswerable: score+gap + semantic absence check (require_anchor=True)
        # ----------------------------
        gate = gate_results(
            results,
            min_score=MIN_SCORE,
            min_gap=MIN_GAP,
            anchor_terms=ANCHOR_TERMS,
            require_anchor=(qtype == "unanswerable"),
        )

        # Track gated failures (doctor view)
        if not gate["pass"]:
            gated_failures.append(
                {
                    "id": qid,
                    "query": query,
                    "type": qtype,
                    "reason": gate["reason"],
                    "score1": gate["score1"],
                    "score2": gate["score2"],
                    "gap": gate["gap"],
                    "retrieved": retrieved_ids,
                }
            )

        # ----------------------------
        # Day 73 safety accounting
        # ----------------------------
        if qtype == "unanswerable":
            should_abstain += 1
            if gate["pass"]:
                false_pass += 1
            else:
                abstained += 1

        # ----------------------------
        # Retrieval scoring (skip for unanswerable)
        # ----------------------------
        if qtype != "unanswerable":
            h = hit_at_k(retrieved_ids, expected, FINAL_K)
            rec = recall_at_k(retrieved_ids, expected, FINAL_K)
            mr = mean_rank(retrieved_ids, expected)

            hit_total += h
            recall_total += rec
            if mr is not None:
                ranks.append(mr)

            if h == 0:
                worst_cases.append(
                    {
                        "id": qid,
                        "query": query,
                        "expected": expected,
                        "retrieved": retrieved_ids,
                    }
                )
        else:
            h, rec, mr = 0, 0.0, None  # placeholder for printing

        # ----------------------------
        # Print per query
        # ----------------------------
        print("\n" + "=" * 70)
        print(f"[{qid}] ({qtype}) {query}")
        print("Expected:", expected)
        print("Retrieved:", retrieved_ids)
        if qtype != "unanswerable":
            print(f"Hit@{FINAL_K}: {h} | Recall@{FINAL_K}: {rec:.2f} | MeanRank: {mr}")
        print(
            f"GATE: {gate['pass']} | reason={gate['reason']} | "
            f"score1={gate['score1']} | score2={gate['score2']} | gap={gate['gap']}"
        )

    # ----------------------------
    # Final reporting
    # ----------------------------
    n_scored = sum(1 for r in data if r.get("type", "normal") != "unanswerable")
    hit_at_k_avg = hit_total / float(n_scored) if n_scored else 0.0
    recall_at_k_avg = recall_total / float(n_scored) if n_scored else 0.0
    mean_rank_avg = (sum(ranks) / len(ranks)) if ranks else None

    print("\n" + "#" * 70)
    print("DAY 73 — FINAL METRICS (Retrieval + Gating + Safety)")
    print(f"Scored queries (non-unanswerable): {n_scored}/{len(data)}")
    print(f"Hit@{FINAL_K}: {hit_at_k_avg:.3f}")
    print(f"Recall@{FINAL_K}: {recall_at_k_avg:.3f}")
    print(f"Mean Rank: {mean_rank_avg}")
    print(f"Gating failures: {len(gated_failures)}/{len(data)}")
    print(f"Gating knobs: MIN_SCORE={MIN_SCORE}, MIN_GAP={MIN_GAP}")
    print("#" * 70)

    print("\n--- DAY 73 SAFETY METRICS ---")
    print(f"Unanswerable queries: {should_abstain}")
    print(f"Correct abstentions: {abstained}")
    print(f"False passes (should have abstained): {false_pass}")
    if should_abstain:
        print(f"Abstention accuracy: {abstained / should_abstain:.2f}")

    if gated_failures:
        print("\nGated FAIL cases (confidence rejected):")
        for g in gated_failures[:10]:
            print("-" * 70)
            print(g["id"], f"({g['type']})", g["query"])
            print("reason:", g["reason"])
            print("score1:", g["score1"], "score2:", g["score2"], "gap:", g["gap"])
            print("retrieved:", g["retrieved"])

    if worst_cases:
        print("\nRetrieval misses (Hit@K failed):")
        for w in worst_cases[:10]:
            print("-" * 70)
            print(w["id"], w["query"])
            print("expected:", w["expected"])
            print("retrieved:", w["retrieved"])
    else:
        print("\n✅ No misses in Hit@K on scored queries.")


if __name__ == "__main__":
    main()
