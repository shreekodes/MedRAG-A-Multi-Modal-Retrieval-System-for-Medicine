import os
import json
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch

# PATH CONFIGURATION

EMB_DIR = r"C:\Users\aditi\Med_RAG\MedRAG-A-Multi-Modal-Retrieval-System-for-Medicine\data\embeddings"
FAISS_DIR = r"C:\Users\aditi\Med_RAG\MedRAG-A-Multi-Modal-Retrieval-System-for-Medicine\data\FAISS_index"

TEXT_EMB_PATH = os.path.join(EMB_DIR, "text_embeddings.npy")
TEXT_META_PATH = os.path.join(EMB_DIR, "text_meta.json")
TEXT_INDEX_PATH = os.path.join(FAISS_DIR, "text_index.faiss")

IMAGE_EMB_PATH = os.path.join(EMB_DIR, "image_embeddings.npy")
IMAGE_META_PATH = os.path.join(EMB_DIR, "image_meta.json")
IMAGE_INDEX_PATH = os.path.join(FAISS_DIR, "image_index.faiss")

# LOAD MODELS

print("Loading SentenceTransformer model for text retrieval...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading CLIP model for image retrieval...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device).eval()

# LOAD METADATA AND FAISS INDEXES

def load_faiss_index(index_path, emb_path):
    if os.path.exists(index_path):
        print(f"Loading FAISS index from {index_path}")
        return faiss.read_index(index_path)
    else:
        print(f"{index_path} not found, rebuilding from {emb_path}")
        emb = np.load(emb_path).astype("float32")
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return index

def load_metadata(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# EMBEDDING FUNCTIONS

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

# SEARCH FUNCTIONS

def search_text(query: str, text_index, text_meta, top_k=5):
    q_emb = embed_query_text(query)
    D, I = text_index.search(q_emb, top_k)
    return [text_meta[i] for i in I[0] if i < len(text_meta)]

def search_images(query: str, image_index, image_meta, top_k=3):
    q_emb = embed_query_image_text(query)
    D, I = image_index.search(q_emb, top_k)
    return [image_meta[i] for i in I[0] if i < len(image_meta)]

# MAIN DEMO

if __name__ == "__main__":
    text_index = load_faiss_index(TEXT_INDEX_PATH, TEXT_EMB_PATH)
    image_index = load_faiss_index(IMAGE_INDEX_PATH, IMAGE_EMB_PATH)
    text_meta = load_metadata(TEXT_META_PATH)
    image_meta = load_metadata(IMAGE_META_PATH)

    print("\n Ready for multimodal retrieval!")

    while True:
        query = input("\n Enter a medical query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        # ---- TEXT SEARCH ----
        print("\n Top 5 Relevant Research Papers:")
        text_results = search_text(query, text_index, text_meta, top_k=5)
        for i, doc in enumerate(text_results, 1):
            print(f"{i}.  {doc.get('title', 'Untitled')}")
            print(f"   Topic: {doc.get('topic', 'N/A')}\n")

        # ---- IMAGE SEARCH ----
        print("Top 3 Relevant MRI Images:")
        image_results = search_images(query, image_index, image_meta, top_k=3)
        for i, img in enumerate(image_results, 1):
            print(f"{i}.  File: {img.get('filepath', 'Unknown')}")
            print(f"   ID: {img.get('image_id', 'N/A')}\n")

    print("Exiting retrieval demo.")
