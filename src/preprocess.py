from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import json

# optional: medical formats
import pydicom, nibabel as nib

def load_image(path: Path):
    s = str(path).lower()
    if s.endswith(".dcm"):
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
    elif (s.endswith(".nii") or s.endswith(".nii.gz")):
        img = nib.load(str(path))
        data = img.get_fdata()
        idx = data.shape[2] // 2 if data.ndim == 3 else 0
        arr = data[:, :, idx].astype(np.float32)
    else:
        im = Image.open(str(path)).convert("L")
        arr = np.array(im).astype(np.float32)

    arr = arr - np.min(arr)
    if np.max(arr) > 0:
        arr = arr / np.max(arr)
    arr = (arr * 255).astype(np.uint8)
    return arr

def to_rgb_and_resize(arr, size=(224,224)):
    im = Image.fromarray(arr)
    im = im.resize(size, Image.BILINEAR)
    im = im.convert("RGB")
    return im

def process_folder(input_dir, output_dir, label):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in tqdm(list(Path(input_dir).rglob("*"))):
        if f.is_file():
            try:
                arr = load_image(f)
                im = to_rgb_and_resize(arr)
                img_id = f"{label}_{f.stem}"
                out_path = output_dir / f"{img_id}.png"
                im.save(out_path)
                rows.append({
                    "image_id": img_id,
                    "filepath": str(out_path),
                    "label": label,
                    "orig_file": str(f)
                })
            except Exception as e:
                print(" Skipped:", f, "Error:", e)
    return rows

def run_preprocessing(input_dir, output_dir, master_meta="metadata_images.json"):
    input_root = Path(input_dir)
    out_root = Path(output_dir)
    all_rows = []

    for child in input_root.iterdir():
        if child.is_dir():
            label = child.name
            out_dir = out_root / label
            rows = process_folder(child, out_dir, label)
            all_rows.extend(rows)

    if all_rows:
        master_path = out_root.parent / master_meta

        # If file exists, load and append
        if master_path.exists():
            with open(master_path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        # Append new rows
        existing_data.extend(all_rows)

        # Save back to JSON
        with open(master_path, "w") as f:
            json.dump(existing_data, f, indent=2)

        print(f"Updated metadata at {master_path} (total records: {len(existing_data)})")
