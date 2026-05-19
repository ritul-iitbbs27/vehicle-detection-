import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Detection & Segmentation",
    page_icon="🚗",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0e0e0e;
    color: #f0f0f0;
}

.stApp {
    background: #0e0e0e;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    color: #ffffff;
    margin-bottom: 0.25rem;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

.accent { color: #00e5a0; }

.stat-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00e5a0;
}

.stat-label {
    font-size: 0.75rem;
    color: #666;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.detection-tag {
    display: inline-block;
    background: #1a2e25;
    color: #00e5a0;
    border: 1px solid #00e5a044;
    border-radius: 6px;
    padding: 4px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    margin: 3px;
}

.info-box {
    background: #111;
    border-left: 3px solid #00e5a0;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #aaa;
}

.stButton > button {
    background: #00e5a0;
    color: #000;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #00ffb3;
    transform: translateY(-1px);
}

div[data-testid="stFileUploader"] {
    background: #141414;
    border: 2px dashed #2a2a2a;
    border-radius: 12px;
    padding: 1rem;
}

.stSlider > div > div > div { background: #00e5a0 !important; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


# ── Helper: process image ──────────────────────────────────────────────────────
def process_image(model, image_array, conf):
    results = model.predict(image_array, conf=conf, task="segment", verbose=False)
    result = results[0]
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]
            detections.append({"class": class_name, "confidence": confidence})

    return annotated_rgb, detections


# ── Helper: process video ──────────────────────────────────────────────────────
def process_video(model, video_path, conf, output_path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    progress = st.progress(0, text="Processing video...")
    frame_count = 0
    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = model.predict(frame, conf=conf, task="segment", verbose=False)[0]
        out.write(result.plot())

        if result.boxes is not None:
            for box in result.boxes:
                all_detections.append(model.names[int(box.cls[0])])

        frame_count += 1
        pct = min(int((frame_count / max(total_frames, 1)) * 100), 100)
        progress.progress(pct, text=f"Processing frame {frame_count} of {total_frames}...")

    cap.release()
    out.release()
    progress.empty()
    return all_detections


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

# Hero header
st.markdown("""
<div class="hero-title">Vehicle <span class="accent">Detection</span><br>& Segmentation</div>
<div class="hero-sub">// powered by YOLO26 · instance segmentation · real-time</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    model_path = st.text_input(
        "Model weights path",
        value=r"C:\Users\ritul\OneDrive\Desktop\yolo26\runs\segment\vehicle_run\weights\best.pt",
        help="Path to your trained best.pt or yolo26n-seg.pt"
    )

    conf_threshold = st.slider(
        "Confidence threshold", 0.1, 0.95, 0.40, 0.05,
        help="Detections below this score are ignored"
    )

    st.markdown("---")
    st.markdown("**Detectable classes**")
    classes = {"🚌": "bus", "🚗": "car", "🏍️": "motorcycle", "🛺": "tricycle", "🚛": "truck"}
    for icon, name in classes.items():
        st.markdown(f"{icon} `{name}`")

    st.markdown("---")
    st.markdown(
        '<div style="font-family:Space Mono,monospace;font-size:0.75rem;color:#555">'
        'YOLO26 · Ultralytics<br>Vehicle Segmentation v1</div>',
        unsafe_allow_html=True
    )

# ── Load model ─────────────────────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.warning(f"⚠️ Model not found at `{model_path}`. Using pretrained `yolo26n-seg.pt` instead.")
    model_path = "yolo26n-seg.pt"

try:
    model = load_model(model_path)
    st.markdown(
        f'<div class="info-box">✅ Model loaded — <b>{Path(model_path).name}</b> · '
        f'{len(model.names)} classes · ready</div>',
        unsafe_allow_html=True
    )
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷  Image Detection", "🎬  Video Detection"])

# ════════════════════════════════════
#  TAB 1 — IMAGE
# ════════════════════════════════════
with tab1:
    uploaded_image = st.file_uploader(
        "Upload a vehicle image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown("**Original image**")
            st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("🔍 Detect & Segment Vehicles", key="img_btn"):
            with st.spinner("Running detection..."):
                annotated, detections = process_image(model, image_array, conf_threshold)

            with col2:
                st.markdown("**Detection result**")
                st.image(annotated, use_container_width=True)

            # Stats
            st.markdown("---")
            total = len(detections)
            counts = {}
            for d in detections:
                counts[d["class"]] = counts.get(d["class"], 0) + 1
            avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="stat-box"><div class="stat-num">{total}</div><div class="stat-label">Total detections</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-box"><div class="stat-num">{len(counts)}</div><div class="stat-label">Vehicle types</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_conf:.0%}</div><div class="stat-label">Avg confidence</div></div>', unsafe_allow_html=True)

            if counts:
                st.markdown("**Detected vehicles**")
                tags_html = ""
                for cls, cnt in counts.items():
                    tags_html += f'<span class="detection-tag">{cls} × {cnt}</span>'
                st.markdown(tags_html, unsafe_allow_html=True)

            # Download
            result_pil = Image.fromarray(annotated)
            import io
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download result image",
                data=buf.getvalue(),
                file_name="detection_result.png",
                mime="image/png"
            )

# ════════════════════════════════════
#  TAB 2 — VIDEO
# ════════════════════════════════════
with tab2:
    uploaded_video = st.file_uploader(
        "Upload a vehicle video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:
        st.video(uploaded_video)

        if st.button("🎬 Process Video", key="vid_btn"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(uploaded_video.read())
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path.replace(".mp4", "_output.mp4")

            with st.spinner("Processing video — this may take a few minutes..."):
                all_detections = process_video(model, tmp_in_path, conf_threshold, tmp_out_path)

            st.success("✅ Video processed!")

            # Stats
            total = len(all_detections)
            counts = {}
            for cls in all_detections:
                counts[cls] = counts.get(cls, 0) + 1

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="stat-box"><div class="stat-num">{total}</div><div class="stat-label">Total detections</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-box"><div class="stat-num">{len(counts)}</div><div class="stat-label">Vehicle types found</div></div>', unsafe_allow_html=True)

            if counts:
                st.markdown("**Vehicles detected across all frames**")
                tags_html = ""
                for cls, cnt in counts.items():
                    tags_html += f'<span class="detection-tag">{cls} × {cnt} detections</span>'
                st.markdown(tags_html, unsafe_allow_html=True)

            # Download output video
            with open(tmp_out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download output video",
                    data=f.read(),
                    file_name="vehicle_detection_output.mp4",
                    mime="video/mp4"
                )

            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)
