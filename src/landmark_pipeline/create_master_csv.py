import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "segmented_landmarks"))
OUTPUT_CSV = os.path.abspath(os.path.join(BASE_DIR, "..", "master_landmarks.csv"))

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Create column headers
columns = ["BLOCK_NAME", "SEGMENT_ID", "FRAME_ID"]
for name in LANDMARK_NAMES:
    columns.extend([f"{name}_X", f"{name}_Y", f"{name}_Z"])

rows = []

for block_name in os.listdir(SEGMENT_ROOT):
    block_path = os.path.join(SEGMENT_ROOT, block_name)
    if not os.path.isdir(block_path):
        continue

    for segment_file in os.listdir(block_path):
        if not segment_file.endswith(".npy"):
            continue

        segment_path = os.path.join(block_path, segment_file)
        data = np.load(segment_path)  # (30, 63)

        for frame_id, frame in enumerate(data):
            row = [block_name, segment_file, frame_id]
            row.extend(frame.tolist())
            rows.append(row)

# Create DataFrame and save
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Master CSV created at: {OUTPUT_CSV}")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
