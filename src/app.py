import sys
import os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import (
    load_faiss_index,
    load_metadata,
    search_text,
    search_images,
    TEXT_INDEX_PATH, TEXT_EMB_PATH, TEXT_META_PATH,
    IMAGE_INDEX_PATH, IMAGE_EMB_PATH, IMAGE_META_PATH
)
from generation import generate_summary


app = FastAPI(title="MedRAG API", version="1.0")

print("🚀 Initializing MedRAG backend...")

# Load indexes and metadata once at startup
text_index = load_faiss_index(TEXT_INDEX_PATH, TEXT_EMB_PATH)
image_index = load_faiss_index(IMAGE_INDEX_PATH, IMAGE_EMB_PATH)
text_meta = load_metadata(TEXT_META_PATH)
image_meta = load_metadata(IMAGE_META_PATH)

class QueryRequest(BaseModel):
    query: str
    top_k_text: int = 5
    top_k_image: int = 3

@app.get("/")
def root():
    return {"message": "Welcome to MedRAG API — Multi-Modal Retrieval for Medicine"}

@app.post("/query")
def get_medical_answer(req: QueryRequest):
    papers = search_text(req.query, text_index, text_meta, top_k=req.top_k_text)
    images = search_images(req.query, image_index, image_meta, top_k=req.top_k_image)
    answer = generate_summary(papers, req.query)
    return {
        "query": req.query,
        "text_results": papers,
        "image_results": images,
        "answer": answer
    }
