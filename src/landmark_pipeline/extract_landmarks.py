import os
import cv2
import numpy as np
import mediapipe as mp

# -----------------------------
# CONFIGURATION
# -----------------------------

# Absolute paths (robust on Windows)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_ROOT = os.path.join(BASE_DIR, "..", "videos")
OUTPUT_ROOT = os.path.join(BASE_DIR, "..", "landmarks")

VIDEO_ROOT = os.path.abspath(VIDEO_ROOT)
OUTPUT_ROOT = os.path.abspath(OUTPUT_ROOT)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("Video root:", VIDEO_ROOT)
print("Output root:", OUTPUT_ROOT)

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,            # dominant hand only
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# FUNCTION: Extract landmarks from ONE video
# -----------------------------

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_frames_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # dominant hand
            frame_landmarks = []

            for lm in hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])

            all_frames_landmarks.append(frame_landmarks)

    cap.release()
    return np.array(all_frames_landmarks)  # shape: (frames, 63)

# -----------------------------
# MAIN LOOP: Process all folders
# -----------------------------

for block_name in os.listdir(VIDEO_ROOT):
    block_path = os.path.join(VIDEO_ROOT, block_name)

    if not os.path.isdir(block_path):
        continue

    print(f"\nProcessing block: {block_name}")

    output_block_dir = os.path.join(OUTPUT_ROOT, block_name)
    os.makedirs(output_block_dir, exist_ok=True)

    sample_count = 0

    for video_file in os.listdir(block_path):
        if not video_file.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(block_path, video_file)
        print(f"  Reading video: {video_file}")

        landmarks = extract_landmarks_from_video(video_path)

        if landmarks.shape[0] < 10:
            print(f"  Skipping (too few frames): {video_file}")
            continue

        sample_count += 1
        save_path = os.path.join(
            output_block_dir,
            f"sample_{sample_count}.npy"
        )

        np.save(save_path, landmarks)
        print(f"  ✔ Saved {save_path} | shape: {landmarks.shape}")

print("\nLandmark extraction completed for all blocks.")
