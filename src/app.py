# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import build_faiss_index, load_metadata, search_papers
from generation import generate_summary

app = FastAPI(title="MedRAG API", version="1.0")

# Load once at startup
print(" Initializing MedRAG backend...")
index = build_faiss_index()
meta = load_metadata()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def root():
    return {"message": "Welcome to MedRAG API — Multi-Modal Retrieval for Medicine"}

@app.post("/query")
def get_medical_answer(req: QueryRequest):
    papers = search_papers(req.query, index, meta, top_k=req.top_k)
    answer = generate_summary(papers, req.query)
    return {
        "query": req.query,
        "results": papers,
        "answer": answer
    }
