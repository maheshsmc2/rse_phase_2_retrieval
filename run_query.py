# run_query.py

from retriever import dense_search, hybrid_search, hybrid_then_rerank


# ----------------------------
# Day 67: Answer Builder Layer (No-LLM) + Stable Envelope
# ----------------------------
DEFAULT_MIN_SCORE = 0.0  # inspect score scale first; then set per route if needed
MIN_SCORE_MARGIN = 0.05  # Day 68: dominance gap (top1 - top2)

# Day 69: reranker scores (logits) are often uncalibrated/negative → use dominance more
RERANK_MIN_MARGIN = 0.35 # stronger dominance for rerank routes

FALLBACK_MESSAGE = (
    "I couldn't find strong enough evidence in the documents to answer confidently."
)

HOW_TO_IMPROVE = [
    "Try adding a policy name (e.g., 'Leave Policy', 'Attendance Policy').",
    "Add a time/constraint (e.g., 'during probation', 'after confirmation').",
    "Use 1–2 strong keywords from the document wording (e.g., 'medical certificate', 'notice period').",
    "Ask a narrower question (one rule at a time).",
]


# ----------------------------
# Day 65: Query Router
# ----------------------------
def classify_query(query: str) -> str:
    q = query.lower().strip()

    # Keep definition strict (remove "explain" to avoid mis-routing)
    if any(p in q for p in ["what is", "define", "meaning of"]):
        return "definition"

    if any(p in q for p in ["policy", "rule", "leave", "probation", "attendance", "salary"]):
        return "policy"

    return "general"


# ----------------------------
# Day 66: Score + Envelope
# ----------------------------
def _unwrap_results(result):
    """
    Normalize result into a list[dict] if possible.
    Supports dict wrappers: hits/results/documents/docs.
    """
    if result is None:
        return None

    if isinstance(result, list):
        return result

    if isinstance(result, dict):
        for k in ("hits", "results", "documents", "docs"):
            v = result.get(k)
            if isinstance(v, list):
                return v
        # If dict itself looks like a single hit, treat as 1-item list
        if "score" in result or "text" in result or "chunk" in result or "content" in result:
            return [result]

    return None


def _extract_top_score(result):
    """
    Works with list[dict] with 'score'.
    Also supports dict wrappers with keys like 'hits'/'results'/'documents'/'docs'.
    """
    items = _unwrap_results(result)
    if not items or not isinstance(items, list):
        return None
    if not items:
        return None
    if isinstance(items[0], dict):
        return items[0].get("score", None)
    return None


def _extract_top2_scores(result):
    """
    Returns (top1, top2) float scores if available, else (None, None).
    Supports list output and dict wrappers.
    """
    items = _unwrap_results(result)
    if not isinstance(items, list) or len(items) < 2:
        return None, None
    if not isinstance(items[0], dict) or not isinstance(items[1], dict):
        return None, None

    s1 = items[0].get("score")
    s2 = items[1].get("score")

    try:
        return float(s1), float(s2)
    except Exception:
        return None, None


def _is_rerank_route(route_name: str) -> bool:
    return "rerank" in (route_name or "").lower()


def _wrap(query: str, route_name: str, out, passed: bool, best_score, min_score):
    """
    Stable return envelope so downstream code (API / evaluator) never breaks.
    """
    return {
        "query": query,
        "route": route_name,
        "decision": "ANSWER" if passed else "ABSTAIN",
        "passed_confidence_gate": passed,
        "best_score": best_score,
        "min_score": min_score,
        "results": out,
        "answer": None,
        # Day 67 fields
        "evidence_preview": [],
        "suggested_queries": [],
        "how_to_improve": [],
        "reason": None,
        # Day 68 field
        "score_margin": None,
    }


# ----------------------------
# Day 67: No-LLM Answer Builder Helpers
# ----------------------------
def _suggested_queries(query: str):
    """
    Rule-based query rewrites (no LLM).
    Gives the user better retrieval chances.
    """
    q = query.strip()
    q_low = q.lower()

    suggestions = []

    # Add "policy" anchors if missing
    if "policy" not in q_low:
        suggestions.append(f"{q} policy")

    # If user mentions probation, make it explicit
    if "probation" in q_low and "period" not in q_low:
        suggestions.append(f"{q} probation period rules")

    # If short/ambiguous, add HR framing
    if len(q.split()) <= 4:
        suggestions.append(f"HR {q} rule")

    # Generic tightening
    suggestions.append(f"{q} eligibility criteria")
    suggestions.append(f"{q} exceptions")

    # Deduplicate while preserving order
    out = []
    seen = set()
    for s in suggestions:
        s2 = " ".join(s.split())
        if s2.lower() not in seen:
            seen.add(s2.lower())
            out.append(s2)

    return out[:3]


def _extract_evidence_preview(results, n: int = 2):
    """
    Pulls a small preview from the top-N results.
    Defensive across different retriever keys.
    """
    items = _unwrap_results(results)
    if not isinstance(items, list):
        return []

    previews = []
    for item in items[:n]:
        if not isinstance(item, dict):
            continue

        previews.append(
            {
                "score": item.get("score"),
                "doc_id": item.get("doc_id") or item.get("id") or item.get("source_id"),
                "title": item.get("title") or item.get("source") or item.get("file"),
                "text_snippet": (item.get("text") or item.get("chunk") or item.get("content") or "")[:240],
            }
        )

    return previews
def _has_required_anchor(query: str, evidence_text: str) -> bool:
    q = query.lower()
    text = evidence_text.lower()

    # Define anchors dynamically
    anchors = []
    if "remote" in q:
        anchors += ["remote", "work from home", "wfh"]
    if "probation" in q:
        anchors += ["probation"]
    if "leave" in q:
        anchors += ["leave", "medical", "sick"]
    if "notice" in q:
        anchors += ["notice", "days", "period"]

    # If no anchors inferred, do not block
    if not anchors:
        return True

    return any(a in text for a in anchors)


def _build_answer_no_llm(results, max_chars: int = 600) -> str:
    """
    Deterministic answer: return the top chunk text (trimmed) + source id.
    This is enough for Day 104 wiring and will make Precision@1 non-zero if
    your expected_answer is actually present in the top chunk.
    """
    items = _unwrap_results(results)
    if not isinstance(items, list) or not items:
        return ""

    top = items[0] if isinstance(items[0], dict) else None
    if not top:
        return ""

    text = (top.get("text") or top.get("chunk") or top.get("content") or "").strip()
    doc_id = top.get("doc_id") or top.get("id") or top.get("source_id") or "unknown"

    if not text:
        return ""
        # Day 113: CONTENT_MIXING fix (intern vs employee)
    low = text.lower().replace("\n", " ")

    if "intern" in low:
        # Try to extract the intern-specific limit (e.g., "up to 1 day per week")
        import re

        m = re.search(r"interns?[^.]*?(up to\s+\d+\s+day[s]?\s+per\s+week)", low)
        if m:
            text = f"Interns may work from home {m.group(1)}."
        else:
            # Fallback: keep only sentences that mention interns
            sents = [s.strip() for s in low.split(".") if s.strip()]
            intern_sents = [s for s in sents if "intern" in s]
            if intern_sents:
                text = ". ".join(intern_sents).strip() + "."


    text = text[:max_chars]
    return f"{text}\n\nSource: {doc_id}"


# ----------------------------
# Day 67–69: Confidence Gate
# ----------------------------
def _gate_if_low_confidence(query: str, result, min_score: float, debug: bool, route_name: str):
    """
    Always returns a standard envelope dict.

    Gates:
    - Day 67: Absolute gate (best_score >= min_score) for non-rerank routes
    - Day 68: Margin gate (top1 - top2 >= MIN_SCORE_MARGIN)
    - Day 69: For rerank routes, skip absolute gate (scores are uncalibrated/negative) and
             apply a stronger margin requirement (RERANK_MIN_MARGIN).
    """
    best_score = _extract_top_score(result)
    is_rerank = _is_rerank_route(route_name)

    # Gate OFF (only for non-rerank routes) — treat as PASS, but still build answer
    if (min_score is None or float(min_score) <= 0) and (not is_rerank):
        if debug:
            print(f"[DAY67][{route_name}] Gate OFF (min_score={min_score}). Returning results without gating.")
        env = _wrap(query, route_name, result, True, best_score, min_score)
        env["evidence_preview"] = _extract_evidence_preview(result, n=2)

        ans = _build_answer_no_llm(result)
        evidence_text = ans.lower() if ans else ""

        ans = _build_answer_no_llm(result)

        # NEW: use raw evidence, not synthesized answer
        evidence_text = " ".join(
            r.get("text", "") for r in (result or [])
        ).lower()
        if True:

            print("\n[ANCHOR DEBUG]")
            print("QUERY:", query)
            print("EVIDENCE:", evidence_text[:300])
            print("ANCHOR_PASS:", _has_required_anchor(query, evidence_text))

        if ans and _has_required_anchor(query, evidence_text):
            env["decision"] = "ANSWER"
            env["answer"] = ans
        else:
            env["decision"] = "ABSTAIN"
            env["passed_confidence_gate"] = False
            env["reason"] = "Semantic absence: required concept not found in evidence"

            env["suggested_queries"] = _suggested_queries(query)
            env["how_to_improve"] = HOW_TO_IMPROVE

        return env

    # No score found => ABSTAIN
    if best_score is None:
        if debug:
            print(f"[DAY67][{route_name}] No 'score' found; treating as low confidence.")
        env = _wrap(query, route_name, result, False, best_score, float(min_score) if min_score is not None else None)
        env["decision"] = "ABSTAIN"
        env["answer"] = ""
        env["reason"] = "No score found in results"
        env["suggested_queries"] = _suggested_queries(query)
        env["how_to_improve"] = HOW_TO_IMPROVE
        env["evidence_preview"] = _extract_evidence_preview(result, n=1)
        return env

    # Non-numeric score => ABSTAIN
    try:
        best_score_f = float(best_score)
    except Exception:
        if debug:
            print(f"[DAY67][{route_name}] Non-numeric score={best_score}; treating as low confidence.")
        env = _wrap(query, route_name, result, False, best_score, float(min_score) if min_score is not None else None)
        env["decision"] = "ABSTAIN"
        env["answer"] = ""
        env["reason"] = "Non-numeric score in results"
        env["suggested_queries"] = _suggested_queries(query)
        env["how_to_improve"] = HOW_TO_IMPROVE
        env["evidence_preview"] = _extract_evidence_preview(result, n=1)
        return env

    # ----------------------------
    # Day 67: Absolute gate (non-rerank only)
    # ----------------------------
    if (not is_rerank) and (min_score is not None) and (best_score_f < float(min_score)):
        if debug:
            print(
                f"[DAY67][{route_name}] ABSTAIN. best_score={best_score_f:.4f} < min_score={float(min_score):.4f}"
            )
        env = _wrap(query, route_name, result, False, best_score_f, float(min_score))
        env["decision"] = "ABSTAIN"
        env["answer"] = ""
        env["reason"] = "Not enough information found"
        env["suggested_queries"] = _suggested_queries(query)
        env["how_to_improve"] = HOW_TO_IMPROVE
        env["evidence_preview"] = _extract_evidence_preview(result, n=1)
        return env

    # ----------------------------
    # Day 68/69: Margin gate (dominance)
    # ----------------------------
    top1, top2 = _extract_top2_scores(result)
    if top1 is not None and top2 is not None:
        margin = top1 - top2
        min_margin = float(RERANK_MIN_MARGIN) if is_rerank else float(MIN_SCORE_MARGIN)

        if margin < min_margin:
            if debug:
                print(
                    f"[DAY69][{route_name}] ABSTAIN. margin={margin:.4f} < min_margin={min_margin:.4f}"
                )
            env = _wrap(query, route_name, result, False, top1, float(min_score) if min_score is not None else None)
            env["decision"] = "ABSTAIN"
            env["answer"] = ""
            env["reason"] = "Low score dominance (ambiguous match)"
            env["suggested_queries"] = _suggested_queries(query)
            env["how_to_improve"] = HOW_TO_IMPROVE
            env["evidence_preview"] = _extract_evidence_preview(result, n=1)
            env["score_margin"] = margin
            return env

    # PASS → build deterministic answer
    if debug:
        if top1 is not None and top2 is not None:
            min_margin = float(RERANK_MIN_MARGIN) if is_rerank else float(MIN_SCORE_MARGIN)
            print(
                f"[DAY69][{route_name}] PASS. best_score={best_score_f:.4f} "
                f"| margin={(top1-top2):.4f} (min_margin={min_margin:.4f}) "
                f"| is_rerank={is_rerank}"
            )
        else:
            print(f"[DAY69][{route_name}] PASS. best_score={best_score_f:.4f} | is_rerank={is_rerank}")

    env = _wrap(
        query,
        route_name,
        result,
        True,
        best_score_f,
        float(min_score) if min_score is not None else None,
    )
    env["evidence_preview"] = _extract_evidence_preview(result, n=2)
    if top1 is not None and top2 is not None:
        env["score_margin"] = float(top1 - top2)

    ans = _build_answer_no_llm(result)
    if ans:
        env["decision"] = "ANSWER"
        env["answer"] = ans
    else:
        env["decision"] = "ABSTAIN"
        env["passed_confidence_gate"] = False
        env["reason"] = "No usable text in top evidence"
        env["suggested_queries"] = _suggested_queries(query)
        env["how_to_improve"] = HOW_TO_IMPROVE

    return env


def run_query(
    query: str,
    top_k: int = 5,
    alpha: float = 0.2,
    use_reranker: bool = True,
    min_score: float = DEFAULT_MIN_SCORE,
    debug: bool = False,
):
    q_type = classify_query(query)

    # ----------------------------
    # Day 65: Intent-aware knobs
    # ----------------------------
    if q_type == "definition":
        alpha = 0.0  # dense only
    elif q_type == "policy":
        alpha = max(alpha, 0.35)  # bias hybrid toward lexical
    # general keeps passed alpha

    if debug:
        print("\n" + "=" * 60)
        print(
            f"[DAY69] query='{query}' | q_type='{q_type}' | "
            f"top_k={top_k} | alpha={alpha} | use_reranker={use_reranker} | "
            f"min_score={min_score} | MIN_SCORE_MARGIN={MIN_SCORE_MARGIN} | RERANK_MIN_MARGIN={RERANK_MIN_MARGIN}"
        )

    # Route A: definition → dense only
    if q_type == "definition":
        out = dense_search(query, top_k=top_k)
        return _gate_if_low_confidence(query, out, min_score=min_score, debug=debug, route_name="definition:dense")

    # Route B: policy → hybrid (+ optional rerank)
    if q_type == "policy":
        if use_reranker:
            out = hybrid_then_rerank(query, retrieve_k=20, final_k=top_k, alpha=alpha)
            return _gate_if_low_confidence(
                query, out, min_score=min_score, debug=debug, route_name="policy:hybrid_then_rerank"
            )

        out = hybrid_search(query, top_k=top_k, alpha=alpha)
        return _gate_if_low_confidence(query, out, min_score=min_score, debug=debug, route_name="policy:hybrid")

    # Route C: general → hybrid (+ optional rerank)
    if use_reranker:
        out = hybrid_then_rerank(query, retrieve_k=20, final_k=top_k, alpha=alpha)
        return _gate_if_low_confidence(
            query, out, min_score=min_score, debug=debug, route_name="general:hybrid_then_rerank"
        )

    out = hybrid_search(query, top_k=top_k, alpha=alpha)
    return _gate_if_low_confidence(query, out, min_score=min_score, debug=debug, route_name="general:hybrid")


if __name__ == "__main__":
    tests = [
        "What is FAISS?",
        "probation leave policy",
        "explain attendance policy",
        "tell me about RAG",
        "random nonsense xyz123",
        "asdjkl qweoi zxcmn",
    ]

    for q in tests:
        print(q, "=>", classify_query(q))
        out = run_query(q, top_k=3, alpha=0.2, use_reranker=True, min_score=0.35, debug=True)
        print(
            "DECISION:",
            out.get("decision"),
            "| PASSED:",
            out.get("passed_confidence_gate"),
            "| BEST_SCORE:",
            out.get("best_score"),
            "| MARGIN:",
            out.get("score_margin"),
        )
        print("REASON:", out.get("reason"))
        print("SUGGESTED:", out.get("suggested_queries"))
        print("EVIDENCE:", out.get("evidence_preview"))
        print("ANSWER_SNIP:", (out.get("answer") or "")[:180])
        print("-" * 60)
