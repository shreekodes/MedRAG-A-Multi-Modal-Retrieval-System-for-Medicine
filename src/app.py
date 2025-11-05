# app.py - Streamlit MedRAG Demo (Full-Fledged)

import streamlit as st
import os
import json
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path
import time

# -------------------------------
# PATH CONFIGURATION (Relative Paths)
# -------------------------------

BASE_DIR = Path(__file__).parent
EMB_DIR = BASE_DIR / "data" / "embeddings"
FAISS_DIR = BASE_DIR / "data" / "FAISS_index"

TEXT_EMB_PATH = EMB_DIR / "text_embeddings.npy"
TEXT_META_PATH = EMB_DIR / "text_meta.json"
TEXT_INDEX_PATH = FAISS_DIR / "text_index.faiss"

IMAGE_EMB_PATH = EMB_DIR / "image_embeddings.npy"
IMAGE_META_PATH = EMB_DIR / "image_meta.json"
IMAGE_INDEX_PATH = FAISS_DIR / "image_index.faiss"

# -------------------------------
# LOAD MODELS
# -------------------------------

@st.cache_resource(show_spinner=True)
def load_models():
    st.info("🧠 Loading models...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device).eval()
    return text_model, clip_model, clip_processor, device

text_model, clip_model, clip_processor, device = load_models()

# -------------------------------
# LOAD FAISS INDEXES & METADATA
# -------------------------------

@st.cache_resource(show_spinner=True)
def load_faiss_index(index_path, emb_path):
    if index_path.exists():
        return faiss.read_index(str(index_path))
    else:
        emb = np.load(emb_path).astype("float32")
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return index

@st.cache_data(show_spinner=True)
def load_metadata(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

text_index = load_faiss_index(TEXT_INDEX_PATH, TEXT_EMB_PATH)
image_index = load_faiss_index(IMAGE_INDEX_PATH, IMAGE_EMB_PATH)
text_meta = load_metadata(TEXT_META_PATH)
image_meta = load_metadata(IMAGE_META_PATH)

# -------------------------------
# EMBEDDING FUNCTIONS
# -------------------------------

def embed_query_text(query: str):
    q_emb = text_model.encode([query], convert_to_numpy=True)
    q_emb = normalize(q_emb, norm="l2", axis=1).astype("float32")
    return q_emb

def embed_query_image_text(query: str):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        q_emb = clip_model.get_text_features(**inputs)
        q_emb = q_emb.cpu().numpy().astype("float32")
        q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
    return q_emb

# -------------------------------
# SEARCH FUNCTIONS
# -------------------------------

def search_text(query: str, top_k=5):
    q_emb = embed_query_text(query)
    D, I = text_index.search(q_emb, top_k)
    return [text_meta[i] for i in I[0] if i < len(text_meta)]

def search_images(query: str, top_k=3):
    q_emb = embed_query_image_text(query)
    D, I = image_index.search(q_emb, top_k)
    return [image_meta[i] for i in I[0] if i < len(image_meta)]

# -------------------------------
# FLAN-T5 SUMMARIZATION
# -------------------------------
from src.generation import generate_summary  # keep your existing function

# -------------------------------
# STREAMLIT UI
# -------------------------------

st.set_page_config(page_title="MedRAG Demo", layout="wide")
st.title("🧠 MedRAG: Multi-Modal Medical Assistant")

st.markdown("Enter a query related to Alzheimer’s or Brain Tumor:")

query = st.text_input("Your Query:")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query!")
    else:
        with st.spinner("📚 Retrieving top 5 papers..."):
            text_results = search_text(query, top_k=5)
            time.sleep(0.5)

        st.subheader("📄 Top 5 Relevant Research Papers")
        for idx, doc in enumerate(text_results, 1):
            st.markdown(f"**{idx}. {doc.get('title', 'Untitled')}**")
            st.write(f"Topic: {doc.get('topic', 'N/A')}")
            st.write(doc.get("abstract", "No abstract available"))
            st.markdown("---")

        with st.spinner("🖼️ Retrieving top 3 images..."):
            image_results = search_images(query, top_k=3)
            time.sleep(0.5)

        st.subheader("🩻 Top 3 Relevant MRI Images")
        for img in image_results:
            st.image(img.get("filepath", ""), width=300, caption=img.get("image_id", "Image"))

        with st.spinner("🩺 Generating AI summary..."):
            summary = generate_summary([r.get("abstract", "") for r in text_results])
            time.sleep(0.5)

        st.subheader("🩺 AI-Summarized Answer")
        st.write(summary)
