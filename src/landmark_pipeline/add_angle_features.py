import os
import numpy as np
from compute_angle_features import compute_angles

INPUT_ROOT = "segmented_landmarks_augmented"
OUTPUT_ROOT = "segmented_landmarks_with_angles"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for cls in os.listdir(INPUT_ROOT):
    in_cls = os.path.join(INPUT_ROOT, cls)
    out_cls = os.path.join(OUTPUT_ROOT, cls)

    if not os.path.isdir(in_cls):
        continue

    os.makedirs(out_cls, exist_ok=True)

    for file in os.listdir(in_cls):
        if not file.endswith(".npy"):
            continue

        data = np.load(os.path.join(in_cls, file))  # (30, 63)

        angle_seq = []
        for frame in data:
            angle_seq.append(compute_angles(frame))

        angle_seq = np.array(angle_seq)  # (30, A)

        combined = np.concatenate([data, angle_seq], axis=1)

        np.save(os.path.join(out_cls, file), combined)

print("Angle feature integration completed.")
