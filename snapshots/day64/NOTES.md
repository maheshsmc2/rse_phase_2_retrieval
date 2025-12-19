# Day 64 — Confidence Gating + Score Unification

## What changed
- Added universal `score` field in dense / hybrid / rerank outputs
- Fixed run_query routing to call hybrid_then_rerank correctly
- Added confidence gate (min_score) to refuse weak matches

## Why
- Prevent hallucinations
- Production-grade refusal behavior
- Simplified evaluation and UI logic

## LOC (Location of Change)
- retriever.py:
  - Added `score` key to dense, hybrid, rerank outputs
- run_query.py:
  - Added min_score gate
  - Fixed rerank parameter wiring

## Danger Flags
- High min_score ⇒ frequent NO_ANSWER
- Corpus-topic mismatch ⇒ NO_ANSWER is correct behavior
