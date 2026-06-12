"""Crop mastitis bounding boxes from Roboflow YOLO dataset into classification images."""
import os
from pathlib import Path
from PIL import Image

ROBOFLOW_DIR = Path("data/roboflow_mastitis")
OUTPUT_DIR = Path("data/mastitis_cropped")
MASTITIS_CLASS = 6  # Mastitis_infected_udder
MIN_SIZE = 64  # skip crops smaller than 64x64


def crop_from_split(split: str):
    img_dir = ROBOFLOW_DIR / split / "images"
    lbl_dir = ROBOFLOW_DIR / split / "labels"
    out_dir = OUTPUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        img_name = lbl_file.stem
        # find matching image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = img_dir / (img_name + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        with open(lbl_file) as f:
            lines = f.readlines()

        img = None
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id != MASTITIS_CLASS:
                continue

            if img is None:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size

            # YOLO format: cx cy bw bh (normalized)
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = max(0, int((cx - bw / 2) * w))
            y1 = max(0, int((cy - bh / 2) * h))
            x2 = min(w, int((cx + bw / 2) * w))
            y2 = min(h, int((cy + bh / 2) * h))

            if (x2 - x1) < MIN_SIZE or (y2 - y1) < MIN_SIZE:
                continue

            crop = img.crop((x1, y1, x2, y2))
            out_name = f"{img_name}_mastitis_{i}.jpg"
            crop.save(out_dir / out_name, quality=95)
            count += 1

    return count


if __name__ == "__main__":
    total = 0
    for split in ["train", "valid", "test"]:
        n = crop_from_split(split)
        print(f"{split}: {n} mastitis crops")
        total += n
    print(f"Total: {total} mastitis crops saved to {OUTPUT_DIR}")
