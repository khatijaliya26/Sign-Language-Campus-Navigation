# scripts/segment_landmarks_augmented.py

import os
import numpy as np

INPUT_ROOT = "augmented_landmarks"
OUTPUT_ROOT = "segmented_landmarks_augmented"

SEGMENT_LENGTH = 30
STRIDE = 10

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for cls in os.listdir(INPUT_ROOT):
    cls_path = os.path.join(INPUT_ROOT, cls)
    if not os.path.isdir(cls_path):
        continue

    out_cls_path = os.path.join(OUTPUT_ROOT, cls)
    os.makedirs(out_cls_path, exist_ok=True)

    count = 0

    for file in os.listdir(cls_path):
        if not file.endswith(".npy"):
            continue

        data = np.load(os.path.join(cls_path, file))
        total_frames = data.shape[0]

        for start in range(0, total_frames - SEGMENT_LENGTH + 1, STRIDE):
            segment = data[start:start + SEGMENT_LENGTH]
            save_path = os.path.join(
                out_cls_path,
                f"segment_{count}.npy"
            )
            np.save(save_path, segment)
            count += 1

    print(f"{cls}: {count} segments created")

print("\nAugmented segmentation completed.")
