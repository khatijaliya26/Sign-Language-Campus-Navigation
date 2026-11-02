import os
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
WINDOW_SIZE = 30
STRIDE = 15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARK_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "landmarks"))
OUTPUT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "segmented_landmarks"))

os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("Landmark root:", LANDMARK_ROOT)
print("Segmented output root:", OUTPUT_ROOT)

# -----------------------------
# HELPERS
# -----------------------------

def pad_sequence(seq, target_len):
    """
    Pads a sequence by repeating the last frame.
    """
    if len(seq) >= target_len:
        return seq[:target_len]

    pad_len = target_len - len(seq)
    last_frame = seq[-1]
    padding = np.repeat(last_frame[np.newaxis, :], pad_len, axis=0)
    return np.vstack([seq, padding])


def segment_sequence(seq):
    """
    Segments a landmark sequence into fixed-length windows.
    """
    segments = []
    length = len(seq)

    if length < WINDOW_SIZE:
        segments.append(pad_sequence(seq, WINDOW_SIZE))
        return segments

    for start in range(0, length - WINDOW_SIZE + 1, STRIDE):
        window = seq[start:start + WINDOW_SIZE]
        segments.append(window)

    return segments

# -----------------------------
# MAIN LOOP
# -----------------------------

for block_name in os.listdir(LANDMARK_ROOT):
    block_path = os.path.join(LANDMARK_ROOT, block_name)

    if not os.path.isdir(block_path):
        continue

    print(f"\nProcessing block: {block_name}")

    output_block_dir = os.path.join(OUTPUT_ROOT, block_name)
    os.makedirs(output_block_dir, exist_ok=True)

    segment_count = 0

    for file in os.listdir(block_path):
        if not file.endswith(".npy"):
            continue

        seq = np.load(os.path.join(block_path, file))

        segments = segment_sequence(seq)

        for seg in segments:
            segment_count += 1
            save_path = os.path.join(
                output_block_dir,
                f"segment_{segment_count}.npy"
            )
            np.save(save_path, seg)

    print(f"  Total segments created: {segment_count}")

print("\n Segmentation completed.")

import numpy as np
x = np.load("segmented_landmarks/AudiBlock/segment_1.npy")
print("Audi Block:", x.shape)
