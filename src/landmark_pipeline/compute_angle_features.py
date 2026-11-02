import numpy as np

def angle_between(a, b, c):
    """Compute angle at point b given points a, b, c"""
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))


def compute_angles(frame):
    """
    frame: (63,) flattened landmark array
    returns: angle feature vector
    """

    lm = frame.reshape(21, 3)
    angles = []

    # ---- Finger joint angles ----
    finger_joints = [
        (1, 2, 3), (2, 3, 4),            # Thumb
        (0, 5, 6), (5, 6, 7), (6, 7, 8), # Index
        (0, 9, 10), (9, 10, 11), (10, 11, 12), # Middle
        (0, 13, 14), (13, 14, 15), (14, 15, 16), # Ring
        (0, 17, 18), (17, 18, 19), (18, 19, 20) # Pinky
    ]

    for a, b, c in finger_joints:
        angles.append(angle_between(lm[a], lm[b], lm[c]))

    # ---- Inter-finger spread angles ----
    finger_pairs = [
        (5, 9), (9, 13), (13, 17), (2, 5)
    ]

    wrist = lm[0]
    for f1, f2 in finger_pairs:
        v1 = lm[f1] - wrist
        v2 = lm[f2] - wrist
        angle = angle_between(v1, np.zeros(3), v2)
        angles.append(angle)

    # ---- Palm orientation ----
    palm_normal = np.cross(lm[5] - lm[0], lm[17] - lm[0])
    palm_normal /= np.linalg.norm(palm_normal) + 1e-6

    for axis in np.eye(3):
        angles.append(
            np.arccos(np.clip(np.dot(palm_normal, axis), -1.0, 1.0))
        )

    return np.array(angles)
