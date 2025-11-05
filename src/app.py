import gradio as gr
import numpy as np
import faiss
import json
from pathlib import Path
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from src import retrieval, generation  # your existing code

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).parent
EMB_DIR = BASE_DIR / "data" / "embeddings"
FAISS_DIR = BASE_DIR / "data" / "FAISS_index"

TEXT_EMB_PATH = EMB_DIR / "text_embeddings_demo.npy"
TEXT_META_PATH = EMB_DIR / "text_meta_demo.json"
TEXT_INDEX_PATH = FAISS_DIR / "text_index_demo.faiss"

# ---------------- Load models ----------------
text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device).eval()

# ---------------- Load data ----------------
text_index = faiss.read_index(str(TEXT_INDEX_PATH))
with open(TEXT_META_PATH, "r") as f:
    text_meta = json.load(f)

# ---------------- Define search ----------------
def query_medrag(query: str):
    # Text search
    q_emb = text_model.encode([query], convert_to_numpy=True)
    q_emb = normalize(q_emb, norm="l2", axis=1).astype("float32")
    D, I = text_index.search(q_emb, 5)
    papers = [text_meta[i] for i in I[0]]

    # Summarize using your existing generation.py function
    summary = generation.generate_summary([p.get("abstract", "") for p in papers])

    # Return paper titles + summary
    titles = "\n".join([f"{i+1}. {p['title']}" for i, p in enumerate(papers)])
    return f"Top 5 Papers:\n{titles}\n\nAI Summary:\n{summary}"

# ---------------- Gradio Interface ----------------
iface = gr.Interface(
    fn=query_medrag,
    inputs=gr.Textbox(label="Enter medical query"),
    outputs=gr.Textbox(label="Results"),
    title="MedRAG Demo",
    description="Demo for Alzheimer's & Brain Tumor retrieval + AI summarization"
)

iface.launch()

