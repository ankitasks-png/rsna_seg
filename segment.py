import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
import sys

IMAGE_DIR   = Path("data/images")
MASK_DIR    = Path("masks")
OVERLAY_DIR = Path("overlays")
SAM2_CKPT   = Path("checkpoints/sam2_large.pt")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

MASK_DIR.mkdir(exist_ok=True)
OVERLAY_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Loading SAM 2...")

sys.path.insert(0, str(Path("C:/Users/ankit/Downloads/Bone+Age+Training+Set+Annotations/sam2")))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2 = build_sam2(
    "configs/sam2.1/sam2.1_hiera_large.yaml",
    str(SAM2_CKPT),
    device=DEVICE
)
generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    min_mask_region_area=500,
)
print("SAM 2 loaded.\n")

all_images = sorted(IMAGE_DIR.glob("*.png"))[:200]
print(f"Processing {len(all_images)} images...\n")

for img_path in tqdm(all_images):
    stem = img_path.stem
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        continue
    img_resized = cv2.resize(img_gray, (512, 512))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_resized)
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    try:
        all_masks = generator.generate(img_rgb)
    except Exception as e:
        print(f"Failed on {stem}: {e}")
        continue

    bone_masks = [m for m in all_masks
                  if 500 < m["area"] < 512*512*0.15
                  and m["predicted_iou"] > 0.88]

    if not bone_masks:
        continue

    label_map = np.zeros((512, 512), dtype=np.uint8)
    for i, m in enumerate(bone_masks):
        label_map[m["segmentation"]] = (i + 1) * 10

    cv2.imwrite(str(MASK_DIR / f"{stem}_mask.png"), label_map)

    overlay = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    colours = [(255,100,100),(100,255,100),(100,100,255),(255,255,100),(255,100,255),(100,255,255)]
    for i, m in enumerate(bone_masks):
        c = colours[i % len(colours)]
        coloured = np.zeros_like(overlay)
        coloured[m["segmentation"]] = c
        overlay = cv2.addWeighted(overlay, 1.0, coloured, 0.45, 0)
    cv2.imwrite(str(OVERLAY_DIR / f"{stem}_overlay.png"), overlay)

    with open(MASK_DIR / f"{stem}_meta.json", "w") as f:
        json.dump([{"bone_index": i+1, "area": m["area"],
                    "confidence": round(m["predicted_iou"], 3),
                    "bbox": [int(x) for x in m["bbox"]]}
                   for i, m in enumerate(bone_masks)], f, indent=2)

print(f"\nDone! Masks saved to masks/  Overlays saved to overlays/")
print(f"Open overlays/ folder to visually check results.")