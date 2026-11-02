"""
FINAL REAL-TIME INFERENCE SCRIPT (OPTIMIZED & SAFE)
YOLO → ROI → ResNet50 → BiLSTM
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from collections import deque
import time

# ─────────────────────────────────────────────
# ① CONFIG
# ─────────────────────────────────────────────
YOLO_WEIGHTS   = "weights/best.pt"
BILSTM_WEIGHTS = "lstm_model/best_bilstm.pt"

CLASS_NAMES = ['Audi Block', 'Block 1', 'Block 2', 'Block 3', 'Block 4', 'Examination',
               'Office', 'R&D Block', 'Registrar']

BUFFER_SIZE   = 25
LOCK_SECONDS  = 4
YOLO_CONF     = 0.35
MIN_ROI_SIZE  = 80
PAD_RATIO     = 0.15

# ─────────────────────────────────────────────
# ② MODEL
# ─────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=9):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        hn = self.dropout(hn)
        return self.fc(hn)

# ─────────────────────────────────────────────
# ③ LOAD MODELS
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# YOLO
yolo_model = YOLO(YOLO_WEIGHTS)
yolo_model.to(device)   # ✅ FIX: ensure GPU usage

# ResNet
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()

# BiLSTM
bilstm = BiLSTM(2048, 256, len(CLASS_NAMES))
bilstm.load_state_dict(torch.load(BILSTM_WEIGHTS, map_location=device))
bilstm = bilstm.to(device).eval()

print("[INFO] Models loaded successfully")

# ─────────────────────────────────────────────
# ④ TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# ⑤ FUNCTIONS
# ─────────────────────────────────────────────
def extract_roi(frame):
    h_img, w_img = frame.shape[:2]
    results = yolo_model(frame, conf=YOLO_CONF, verbose=False)

    # ✅ FIX: safe check
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    best = int(np.argmax(areas))

    x1,y1,x2,y2 = boxes[best].astype(int)
    bw, bh = x2-x1, y2-y1

    x1 = max(0, int(x1 - PAD_RATIO*bw))
    y1 = max(0, int(y1 - PAD_RATIO*bh))
    x2 = min(w_img, int(x2 + PAD_RATIO*bw))
    y2 = min(h_img, int(y2 + PAD_RATIO*bh))

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0 or roi.shape[0] < MIN_ROI_SIZE or roi.shape[1] < MIN_ROI_SIZE:
        return None, None

    roi = cv2.resize(roi, (224,224))
    return roi, (x1,y1,x2,y2)


def extract_features(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = transform(roi).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(tensor).squeeze().cpu()
        feat = feat / (feat.norm() + 1e-6)  # L2 normalization
    return feat


def run_bilstm(buffer):
    seq = torch.stack(buffer).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = bilstm(seq)
        probs = torch.softmax(logits, dim=1).squeeze()

    print("PROBS:", probs.cpu().numpy())   # 👈 ADD THIS LINE

    idx = int(torch.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])


# ─────────────────────────────────────────────
# ⑥ MAIN LOOP
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce lag

    feature_buffer = deque(maxlen=BUFFER_SIZE)
    pred_history = deque(maxlen=5)

    locked = False
    lock_end = 0
    prediction = None
    confidence = 0

    frame_count = 0

    print("[INFO] Starting webcam... Press Q to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()

        # --------------------------
        # 🔓 UNLOCK LOGIC
        # --------------------------
        if locked and now >= lock_end:
            locked = False
            feature_buffer = deque(maxlen=BUFFER_SIZE)
            pred_history.clear()

        # --------------------------
        # 🔍 PROCESSING (YOLO + CNN + BUFFER)
        # --------------------------
        if not locked:
            frame_count += 1

            # YOLO every 2 frames (speed optimization)
            if frame_count % 2 == 0:
                roi, bbox = extract_roi(frame)

                if roi is not None:
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    feat = extract_features(roi)
                    feature_buffer.append(feat)

            # --------------------------
            # 🔄 PREVIEW PREDICTION
            # --------------------------
            if len(feature_buffer) >= 10:
                temp_pred, _ = run_bilstm(list(feature_buffer))
                cv2.putText(frame, f"Preview: {temp_pred}",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2)

            # --------------------------
            # 🎯 FINAL PREDICTION
            # --------------------------
            if len(feature_buffer) == BUFFER_SIZE:
                prediction, confidence = run_bilstm(list(feature_buffer))

                pred_history.append(prediction)
                prediction = max(set(pred_history), key=pred_history.count)

                locked = True
                lock_end = now + LOCK_SECONDS

        # --------------------------
        # 🖥️ UI OVERLAY (USER FRIENDLY)
        # --------------------------

        # Top bar background
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)

        # Controls
        cv2.putText(frame,
                    "[R] Reset   |   [SPACE] Predict   |   [Q] Quit",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

        # Buffer status
        cv2.putText(frame,
                    f"Frames: {len(feature_buffer)}/{BUFFER_SIZE}",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1)

        # Collecting state
        if len(feature_buffer) < BUFFER_SIZE and not prediction:
            cv2.putText(frame,
                        "Collecting gesture...",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2)

        # Final prediction display
        if prediction:
            cv2.putText(frame,
                        f"{prediction} ({confidence:.0%})",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

        # Locked indicator
        if locked:
            cv2.putText(frame,
                        "Prediction Locked",
                        (400, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1)

        # --------------------------
        # 🎥 SHOW FRAME
        # --------------------------
        cv2.imshow("Gesture System", frame)

        # --------------------------
        # 🎮 CONTROLS
        # --------------------------
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):   # Quit
            break

        elif key == ord('r'):       # Reset
            feature_buffer = deque(maxlen=BUFFER_SIZE)
            pred_history.clear()
            locked = False
            prediction = None

        elif key == ord(' '):       # Force predict
            if len(feature_buffer) >= 2:
                prediction, confidence = run_bilstm(list(feature_buffer))
                locked = True
                lock_end = now + LOCK_SECONDS

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")

if __name__ == "__main__":
    main()