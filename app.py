# app.py
# app.py

from fastapi import FastAPI
from pydantic import BaseModel

from retriever import answer_query  # import from retriever.py


app = FastAPI(title="Dummy-Book RAG API")


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    """
    RAG query endpoint.

    Request JSON:
        { "query": "your question here" }

    Response JSON:
        { "answer": "retriever output with debug info" }
    """
    result = answer_query(req.query)
    return {"answer": result}




