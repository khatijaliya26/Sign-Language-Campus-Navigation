# scripts/add_velocity_features.py

import os
import numpy as np

INPUT_ROOT = "segmented_landmarks_with_angles"
OUTPUT_ROOT = "segmented_landmarks_with_angles_velocity"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for cls in os.listdir(INPUT_ROOT):
    cls_path = os.path.join(INPUT_ROOT, cls)
    if not os.path.isdir(cls_path):
        continue

    out_cls_path = os.path.join(OUTPUT_ROOT, cls)
    os.makedirs(out_cls_path, exist_ok=True)

    for file in os.listdir(cls_path):
        if not file.endswith(".npy"):
            continue

        data = np.load(os.path.join(cls_path, file))  # (T, F)
        T, F = data.shape

        # Velocity = difference between consecutive frames
        velocity = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]

        # Concatenate static + velocity
        combined = np.concatenate([data, velocity], axis=1)

        np.save(
            os.path.join(out_cls_path, file),
            combined
        )

print("Velocity feature integration completed.")
