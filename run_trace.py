from retriever import retrieve_with_trace

out = retrieve_with_trace(
    "probation period policy",
    retrieve_k=20,
    final_k=5,
    alpha=0.2,
    use_reranker=True,
)

print("Saved:", out["trace"]["meta"]["trace_path"])
print("Top ids:", [x["id"] for x in out["final"]])
