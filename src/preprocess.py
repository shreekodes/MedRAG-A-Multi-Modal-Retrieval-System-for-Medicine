# -*- coding: utf-8 -*-
"""
Script to preprocess MRI datasets stored locally in the repository.
- Reads images from /data/raw/
- Normalizes and resizes them to 224x224 RGB
- Saves processed images and metadata.csv

"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

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
                print("⚠️ Skipped:", f, "Error:", e)
    return rows

def run_preprocessing(input_dir, output_dir):
    input_root = Path(input_dir)
    out_root = Path(output_dir)
    all_rows = []
    for child in input_root.iterdir():
        if child.is_dir():
            label = child.name
            out_dir = out_root / label
            rows = process_folder(child, out_dir, label)
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(out_root.parent / "metadata.csv", index=False)
        print("✅ Saved metadata:", out_root.parent / "metadata.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to raw dataset folder")
    parser.add_argument("--output_dir", required=True, help="Path to save processed images")
    args = parser.parse_args()
    run_preprocessing(args.input_dir, args.output_dir)
