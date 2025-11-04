import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

# ----------------------------
# PATH CONFIGURATION
# ----------------------------
RAW_DIR = r"C:\Users\aditi\Med_RAG\MedRAG-A-Multi-Modal-Retrieval-System-for-Medicine\data\image_samples\raw"
PROCESSED_DIR = r"C:\Users\aditi\Med_RAG\MedRAG-A-Multi-Modal-Retrieval-System-for-Medicine\data\image_samples\processed"
EMB_DIR = r"C:\Users\aditi\Med_RAG\MedRAG-A-Multi-Modal-Retrieval-System-for-Medicine\data\embeddings"

METADATA_JSON = os.path.join(EMB_DIR, "metadata_images.json")
IMAGE_META_JSON = os.path.join(EMB_DIR, "image_meta.json")
IMAGE_EMB_NPY = os.path.join(EMB_DIR, "image_embeddings.npy")

BATCH_SIZE = 4

# ----------------------------
# LOAD MODEL
# ----------------------------
print("🔄 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
torch.manual_seed(0)
print(f"✅ Model loaded on {device.upper()}")

# ----------------------------
# LOAD IMAGE METADATA
# ----------------------------
if not os.path.exists(METADATA_JSON):
    raise FileNotFoundError(f"❌ Metadata file not found: {METADATA_JSON}")

with open(METADATA_JSON, "r") as f:
    meta = json.load(f)

image_paths = [item["filepath"] for item in meta]
image_ids = [item["image_id"] for item in meta]
print(f"📂 Found {len(image_paths)} images for embedding.")

# ----------------------------
# EMBEDDING FUNCTION
# ----------------------------
def generate_embeddings(paths):
    valid_images, valid_idx = [], []
    for idx, p in enumerate(paths):
        try:
            valid_images.append(Image.open(p).convert("RGB"))
            valid_idx.append(idx)
        except Exception as e:
            print(f"⚠️ Skipped {p} ({e})")

    if not valid_images:
        return np.zeros((0, 512), dtype=np.float32), []

    inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    features = features.cpu().numpy()
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-8, None)
    return features, valid_idx

# ----------------------------
# MAIN LOOP
# ----------------------------
all_records, all_embeds = [], []
embed_index = 0

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    batch_ids = image_ids[i:i + BATCH_SIZE]

    embeds, valid_idx = generate_embeddings(batch_paths)
    for j, idx in enumerate(valid_idx):
        rec = {
            "image_id": batch_ids[idx],
            "filepath": str(batch_paths[idx]),
            "embedding_index": embed_index,
        }
        all_records.append(rec)
        all_embeds.append(embeds[j])
        embed_index += 1

# ----------------------------
# SAVE OUTPUT
# ----------------------------
os.makedirs(EMB_DIR, exist_ok=True)
np.save(IMAGE_EMB_NPY, np.array(all_embeds, dtype=np.float32))
with open(IMAGE_META_JSON, "w") as f:
    json.dump(all_records, f, indent=2)

print(f"💾 Saved embeddings → {IMAGE_EMB_NPY}")
print(f"💾 Saved metadata → {IMAGE_META_JSON}")
print(f"✅ Done! Generated {len(all_records)} image embeddings.")

