import os
import numpy as np
import random

# -----------------------------
# CONFIG
# -----------------------------
SEGMENT_ROOT = "segmented_landmarks"
AUGMENTED_ROOT = "augmented_landmarks"

TARGET_SAMPLES_PER_CLASS = 40
NOISE_STD = 0.01        # small Gaussian noise
DROP_FRAMES = 2         # max frames to drop

os.makedirs(AUGMENTED_ROOT, exist_ok=True)

# -----------------------------
# AUGMENTATION FUNCTIONS
# -----------------------------

def add_gaussian_noise(sequence, std=NOISE_STD):
    noise = np.random.normal(0, std, sequence.shape)
    return sequence + noise

def temporal_crop(sequence, drop_frames=DROP_FRAMES):
    if sequence.shape[0] <= drop_frames:
        return sequence
    start = random.randint(0, drop_frames)
    end = sequence.shape[0] - (drop_frames - start)
    return sequence[start:end]

def pad_sequence(sequence, target_len=30):
    if sequence.shape[0] >= target_len:
        return sequence[:target_len]
    pad_len = target_len - sequence.shape[0]
    pad = np.zeros((pad_len, sequence.shape[1]))
    return np.vstack([sequence, pad])

# -----------------------------
# MAIN AUGMENTATION LOOP
# -----------------------------

for class_name in os.listdir(SEGMENT_ROOT):
    class_path = os.path.join(SEGMENT_ROOT, class_name)
    if not os.path.isdir(class_path):
        continue

    files = [f for f in os.listdir(class_path) if f.endswith(".npy")]
    current_count = len(files)

    print(f"\nClass: {class_name} | Original samples: {current_count}")

    # Copy original samples
    target_class_dir = os.path.join(AUGMENTED_ROOT, class_name)
    os.makedirs(target_class_dir, exist_ok=True)

    for f in files:
        src = os.path.join(class_path, f)
        dst = os.path.join(target_class_dir, f)
        np.save(dst, np.load(src))

    if current_count >= TARGET_SAMPLES_PER_CLASS:
        print("  ✔ No augmentation needed")
        continue

    needed = TARGET_SAMPLES_PER_CLASS - current_count
    print(f"  ➜ Augmenting {needed} samples")

    for i in range(needed):
        base_file = random.choice(files)
        seq = np.load(os.path.join(class_path, base_file))

        aug_seq = add_gaussian_noise(seq)
        aug_seq = temporal_crop(aug_seq)
        aug_seq = pad_sequence(aug_seq)

        save_name = f"aug_{i+1}.npy"
        save_path = os.path.join(target_class_dir, save_name)
        np.save(save_path, aug_seq)

    print(f"  ✔ Augmented to {TARGET_SAMPLES_PER_CLASS} samples")

print("\nTargeted augmentation completed.")
