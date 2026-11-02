"""
Sign Language Based Campus Navigation System
=============================================
Streamlit web app wrapping the YOLOv8 → ROI → ResNet50 → BiLSTM pipeline.

Run:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from collections import deque, Counter
import time
from PIL import Image

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Campus Sign Navigation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F5F7FB;   /* ✅ soft white */
    color: #1A1A1A;
}
.stApp {
    background: linear-gradient(180deg, #F5F7FB 0%, #EEF2F8 100%);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Landing hero ── */
.hero-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 3rem 1rem 2rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 5vw, 3.6rem);
    color: #1E3A8A;   /* ✅ strong blue */
    margin-bottom: 1.2rem;
}

.hero-sub {
    font-size: 1.05rem;
    color: #4A5568;
    max-width: 540px;
    line-height: 1.7;
    margin-bottom: 2.5rem;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1E5AA8 0%, #3A7BD5 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    border: none !important;
    border-radius: 12px !important;
}

/* ── Section heading ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #2D3748;
    margin-bottom: 0.5rem;
}

/* ── Video panel ── */
.video-panel {
    background: #FFFFFF;
    border: 1px solid #E3E6ED;
    border-radius: 18px;
}

/* ── Status chip ── */
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0.5rem 1.1rem;
    border-radius: 100px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

.status-collecting {
    background: #FFF4E5;
    border: 1px solid #FFD699;
    color: #B26A00;
}

.status-locked {
    background: #E6F0FF;
    border: 1px solid #99BBFF;
    color: #1E5AA8;
}

.status-idle {
    background: #F0F2F7;
    border: 1px solid #DADDE5;
    color: #6A7190;
}

/* ── Prediction card ── */
.pred-card {
    background: #FFFFFF;
    border: 1px solid #E3E6ED;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

.pred-class {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #1E3A8A;
}

.pred-conf {
    font-size: 0.9rem;
    color: #4A5568;
}

/* ── Navigation card ── */
.nav-card {
    background: #FFFFFF;
    border: 1px solid #E3E6ED;
    border-left: 4px solid #1E5AA8;
    border-radius: 12px;
    padding: 1.2rem;
    font-size: 0.92rem;
    color: #1A1A1A;
    line-height: 1.7;
}

/* ── Buffer bar ── */
.buf-track {
    background: #E3E6ED;
    border-radius: 100px;
    height: 6px;
}
.buf-fill {
    background: linear-gradient(90deg, #1E5AA8, #3A7BD5);
}

/* ── Reset button ── */
.reset-btn > button {
    background: #FFFFFF !important;
    border: 1px solid #DADDE5 !important;
    color: #6A7190 !important;
    border-radius: 10px !important;
}
.reset-btn > button:hover {
    border-color: #1E5AA8 !important;
    color: #1E5AA8 !important;
}

/* ── How-it-works cards ── */
.how-card {
    background: #FFFFFF;
    border: 1px solid #E3E6ED;
    border-radius: 14px;
    padding: 1.2rem;
}

.how-title {
    font-family: 'Syne', sans-serif;
    color: #1A1A1A;
}

.how-desc {
    color: #6A7190;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #E3E6ED, transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
YOLO_WEIGHTS   = "weights/best.pt"
BILSTM_WEIGHTS = "lstm_model/best_bilstm.pt"

CLASS_NAMES = ["AudiBlock", "Block1", "Block2", "Block3",
               "Block4", "Exam", "R-DBlock", "Registrar", "office"]

BUFFER_SIZE       = 30
LOCK_SECONDS      = 7
YOLO_CONF         = 0.35
MIN_ROI_SIZE      = 80
PAD_RATIO         = 0.15
CONFIDENCE_THRESH = 0.50    # minimum confidence to accept a prediction
SMOOTHING_WINDOW  = 5       # majority-vote over last N predictions

NAVIGATION_MAP = {
    "AudiBlock":  "Audi Block:\nRight in front of the main gate.",
    "Block1":     "Block 1:\nGo straight from the main gate.\nOn the left, opposite the Central Block.",
    "Block2":     "Block 2:\nWalk straight from the main gate.\nPass the Central Block, Block 1, and the football field.\nLocated near the red benches.",
    "Block3":     "Block 3:\nWalk straight from the main gate.\nPass Central Block, Block 1, football field, and Block 2.\nNear the Birds Park (cage area), inside the alley.",
    "Block4":     "Block 4:\nWalk straight from the main gate.\nPass Central Block, Block 1, football field, and Block 2.\nOpposite the Birds Park, close to R&D Block.",
    "Exam":       "Examination Section:\nWalk straight from the main gate.\nCentral Block will be on your right.\nEnquire at the desk for exam locations.",
    "office":     "Office:\nWalk straight from the main gate.\nCentral Block is on your right.\nGo upstairs and ask for the office.",
    "R-DBlock":   "R&D Block:\nWalk straight from the main gate.\nPass Central Block, Block 1, football field, Block 2, Birds Park, and Block 4.\nLocated near the back gate.",
    "Registrar":  "Registrar:\nWalk straight from the main gate.\nCentral Block is on your right.\nGo upstairs and ask for the Registrar's office.",
}

DISPLAY_NAMES = {
    "AudiBlock": "Audi Block",  "Block1": "Block 1",
    "Block2":    "Block 2",     "Block3": "Block 3",
    "Block4":    "Block 4",     "Exam":   "Examination",
    "office":    "Office",      "R-DBlock": "R&D Block",
    "Registrar": "Registrar",
}


# ──────────────────────────────────────────────────────────────
# MODEL LOADING  (cached so weights load only once)
# ──────────────────────────────────────────────────────────────

class BiLSTM(nn.Module):
    """Exact architecture — do NOT modify."""
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=9):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(hn)


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YOLO(YOLO_WEIGHTS)

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()

    bilstm = BiLSTM(input_dim=2048, hidden_dim=256, num_classes=len(CLASS_NAMES))
    bilstm.load_state_dict(torch.load(BILSTM_WEIGHTS, map_location=device))
    bilstm = bilstm.to(device).eval()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return yolo, resnet, bilstm, tfm, device


# ──────────────────────────────────────────────────────────────
# INFERENCE HELPERS  (same logic as original script)
# ──────────────────────────────────────────────────────────────

def extract_roi(frame_bgr, yolo_model):
    h_img, w_img = frame_bgr.shape[:2]
    results = yolo_model(frame_bgr, conf=YOLO_CONF, verbose=False)
    if len(results[0].boxes) == 0:
        return None, None, frame_bgr

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best  = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[best].astype(int)
    bw, bh = x2 - x1, y2 - y1

    x1p = max(0,     int(x1 - PAD_RATIO * bw))
    y1p = max(0,     int(y1 - PAD_RATIO * bh))
    x2p = min(w_img, int(x2 + PAD_RATIO * bw))
    y2p = min(h_img, int(y2 + PAD_RATIO * bh))

    roi = frame_bgr[y1p:y2p, x1p:x2p]
    if roi.size == 0 or roi.shape[0] < MIN_ROI_SIZE or roi.shape[1] < MIN_ROI_SIZE:
        return None, None, frame_bgr

    roi_resized = cv2.resize(roi, (224, 224))

    # Draw bounding box on frame for visual feedback
    annotated = frame_bgr.copy()
    cv2.rectangle(annotated, (x1p, y1p), (x2p, y2p), (0, 255, 178), 2)
    cv2.putText(annotated, "Hand", (x1p, y1p - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 178), 2)
    return roi_resized, (x1p, y1p, x2p, y2p), annotated


def extract_features(roi_bgr, resnet, tfm, device):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor  = tfm(roi_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(tensor).squeeze().cpu()
    return feat


def run_bilstm(buffer, bilstm_model, device):
    seq = torch.stack(list(buffer)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = bilstm_model(seq)
        probs  = torch.softmax(logits, dim=1).squeeze()
    idx  = int(torch.argmax(probs).item())
    conf = float(probs[idx].item())
    return CLASS_NAMES[idx], conf


# ──────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "page":             "landing",   # "landing" | "camera"
        "feature_buffer":   deque(maxlen=BUFFER_SIZE),
        "pred_history":     deque(maxlen=SMOOTHING_WINDOW),
        "locked":           False,
        "lock_end_time":    0.0,
        "prediction":       None,
        "confidence":       0.0,
        "nav_text":         None,
        "camera_running":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def reset_state():
    st.session_state.feature_buffer  = deque(maxlen=BUFFER_SIZE)
    st.session_state.pred_history     = deque(maxlen=SMOOTHING_WINDOW)
    st.session_state.locked           = False
    st.session_state.lock_end_time    = 0.0
    st.session_state.prediction       = None
    st.session_state.confidence       = 0.0
    st.session_state.nav_text         = None


# ──────────────────────────────────────────────────────────────
# PAGE: LANDING
# ──────────────────────────────────────────────────────────────

def render_landing():
    # Hero section
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-title">
            Sign Language Based<br>Campus Navigation System
        </div>
        <div class="hero-sub">
            An intelligent real-time navigation assistant that recognizes hand gestures
            using deep learning and provides step-by-step directions across campus.
            Powered by a multi-stage pipeline combining YOLOv8 for hand detection,
            ResNet50 for feature extraction, and a BiLSTM network for temporal gesture recognition.
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Start button — centred
    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        if st.button(" Start Camera", use_container_width=True):
            st.session_state.page = "camera"
            st.rerun()

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

   

# ──────────────────────────────────────────────────────────────
# PAGE: CAMERA
# ──────────────────────────────────────────────────────────────

def render_camera():
    # ── Load models once ──────────────────────────────────────
    try:
        yolo_model, resnet, bilstm_model, tfm, device = load_models()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.info("Update `YOLO_WEIGHTS` and `BILSTM_WEIGHTS` paths at the top of app.py.")
        if st.button("← Back"):
            st.session_state.page = "landing"
            st.rerun()
        return

    # ── Header ────────────────────────────────────────────────
    hcol1, hcol2 = st.columns([5, 1])

    with hcol1:
        st.markdown("""
        <div style="
            display:flex;
            align-items:center;
            gap:20px;
        ">
            <div style="
                font-family:'Syne',sans-serif;
                font-size:1.6rem;
                font-weight:800;
                color:#1E3A8A;">
                Campus Navigation
            </div>
            <div style="
                font-size:0.95rem;
                color:#4A5568;">
                Live Gesture Recognition
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hcol2:
        if st.button("Home"):
            reset_state()
            st.session_state.page = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider" style="margin:0.75rem 0 1.25rem;"></div>',
                unsafe_allow_html=True)

    # ── Layout: video (left) + panel (right) ──────────────────
    vid_col, info_col = st.columns([3, 2], gap="large")

    with vid_col:
        st.markdown('<div class="section-label"> Live Feed</div>',
                    unsafe_allow_html=True)
        frame_slot = st.empty()

        # Buffer progress
        buf_pct_slot  = st.empty()
        buf_bar_slot  = st.empty()

        # Reset button below video
        st.markdown("")
        rcol1, rcol2 = st.columns([1, 1])
        with rcol1:
            st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
            reset_clicked = st.button("Reset Buffer", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with rcol2:
            st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
            force_clicked = st.button("Predict Now", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="section-label"> Recognition Output</div>',
                    unsafe_allow_html=True)
        status_slot = st.empty()
        pred_slot   = st.empty()
        nav_slot    = st.empty()

        st.markdown('<div class="divider" style="margin:1rem 0;"></div>',
                    unsafe_allow_html=True)

    # ── Webcam loop ───────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Could not open webcam. Make sure it is connected and not in use.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()

            # ── Handle manual button actions ──────────────────
            if reset_clicked:
                reset_state()

            # ── Lock expiry ───────────────────────────────────
            if st.session_state.locked and now >= st.session_state.lock_end_time:
                st.session_state.locked       = False
                st.session_state.feature_buffer = deque(maxlen=BUFFER_SIZE)
                st.session_state.pred_history    = deque(maxlen=SMOOTHING_WINDOW)

            # ── Detection & feature extraction ───────────────
            annotated_frame = frame.copy()
            hand_detected   = False

            if not st.session_state.locked:
                roi, bbox, annotated_frame = extract_roi(frame, yolo_model)

                if roi is not None:
                    hand_detected = True
                    feat = extract_features(roi, resnet, tfm, device)
                    st.session_state.feature_buffer.append(feat)

                # ── Force predict (SPACE equivalent) ─────────
                if force_clicked and len(st.session_state.feature_buffer) >= 2:
                    raw_cls, conf = run_bilstm(
                        st.session_state.feature_buffer, bilstm_model, device)
                    if conf >= CONFIDENCE_THRESH:
                        st.session_state.pred_history.append(raw_cls)
                        majority = Counter(st.session_state.pred_history).most_common(1)[0][0]
                        st.session_state.prediction  = majority
                        st.session_state.confidence  = conf
                        st.session_state.nav_text    = NAVIGATION_MAP.get(majority, "")
                        st.session_state.locked      = True
                        st.session_state.lock_end_time = now + LOCK_SECONDS

                # ── Auto predict when buffer is full ─────────
                elif len(st.session_state.feature_buffer) == BUFFER_SIZE:
                    raw_cls, conf = run_bilstm(
                        st.session_state.feature_buffer, bilstm_model, device)
                    st.session_state.pred_history.append(raw_cls)
                    majority = Counter(st.session_state.pred_history).most_common(1)[0][0]

                    if conf >= CONFIDENCE_THRESH:
                        st.session_state.prediction  = majority
                        st.session_state.confidence  = conf
                        st.session_state.nav_text    = NAVIGATION_MAP.get(majority, "")
                        st.session_state.locked      = True
                        st.session_state.lock_end_time = now + LOCK_SECONDS

            # ── Render video frame ────────────────────────────
            rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(rgb, channels="RGB", use_container_width=True)

            # ── Buffer bar ────────────────────────────────────
            buf_len = len(st.session_state.feature_buffer)
            buf_pct = int(buf_len / BUFFER_SIZE * 100)
            buf_pct_slot.markdown(
                f'<div style="font-size:0.78rem; color:#6A7190; margin-top:0.5rem;">'
                f'Buffer &nbsp;<span style="color:#E8EAF0; font-weight:600;">'
                f'{buf_len}/{BUFFER_SIZE}</span></div>',
                unsafe_allow_html=True)
            buf_bar_slot.markdown(
                f'<div class="buf-track"><div class="buf-fill" style="width:{buf_pct}%;"></div></div>',
                unsafe_allow_html=True)

            # ── Status chip ───────────────────────────────────
            if st.session_state.locked:
                remaining = max(0, st.session_state.lock_end_time - now)
                status_slot.markdown(
                    f'<div class="status-chip status-locked">'
                    f'<div class="pulse-dot"></div>'
                    f'Prediction Locked &nbsp;·&nbsp; {remaining:.1f}s</div>',
                    unsafe_allow_html=True)
            elif hand_detected:
                status_slot.markdown(
                    '<div class="status-chip status-collecting">'
                    '<div class="pulse-dot"></div>Collecting gesture…</div>',
                    unsafe_allow_html=True)
            else:
                status_slot.markdown(
                    '<div class="status-chip status-idle">'
                    '<div class="pulse-dot"></div>Waiting for hand…</div>',
                    unsafe_allow_html=True)

            # ── Prediction card ───────────────────────────────
            if st.session_state.prediction:
                display_name = DISPLAY_NAMES.get(
                    st.session_state.prediction, st.session_state.prediction)
                conf_pct = int(st.session_state.confidence * 100)
                pred_slot.markdown(f"""
                <div class="pred-card">
                    <div style="font-size:0.7rem; color:#6A7190;
                                letter-spacing:0.12em; text-transform:uppercase;
                                margin-bottom:0.5rem;">Detected Location</div>
                    <div class="pred-class">{display_name}</div>
                    <div class="pred-conf">
                        Confidence &nbsp;
                        <span style="color:#1E5AA8; font-weight:600;">{conf_pct}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                if st.session_state.nav_text:
                    nav_slot.markdown(
                        f'<div class="nav-card">{st.session_state.nav_text}</div>',
                        unsafe_allow_html=True)
            else:
                pred_slot.markdown("""
                <div class="pred-card" style="opacity:0.5;">
                    <div style="font-size:0.7rem; color:#6A7190;
                                letter-spacing:0.12em; text-transform:uppercase;
                                margin-bottom:0.5rem;">Detected Location</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.4rem;
                                color:#2A3050;">Awaiting gesture…</div>
                </div>""", unsafe_allow_html=True)

            # Small sleep to reduce CPU spin
            time.sleep(0.03)

    finally:
        cap.release()


# ──────────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────────
if st.session_state.page == "landing":
    render_landing()
else:
    render_camera()
