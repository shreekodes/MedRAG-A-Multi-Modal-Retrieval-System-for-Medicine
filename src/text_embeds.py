import json, os, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

MODEL_NAME = "all-MiniLM-L6-v2" 

def load_docs(path="data/text_samples/abstracts_neuro.json"):
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    texts = []
    metas = []
    for d in docs:
        text = (d.get("title","") + ". " + d.get("abstract","")).strip()
        if not text:
            continue
        texts.append(text)
        metas.append({"pmid": d.get("pmid",""), "title": d.get("title",""), "topic": d.get("topic","")})
    return texts, metas

def embed_and_save():
    texts, metas = load_docs()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize(embeddings, norm='l2', axis=1)
    os.makedirs("data/embeddings", exist_ok=True)
    np.save("data/embeddings/text_embeddings.npy", embeddings)
    with open("data/embeddings/text_meta.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False)
    print("Saved embeddings and metadata.")

if __name__ == "__main__":
    embed_and_save()