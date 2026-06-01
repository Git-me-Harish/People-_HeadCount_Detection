"""PeopleSense — Streamlit demo app.

This is a self-contained quick-start demo that wraps the same YOLOv8 backbone
used by the main FastAPI/React application. For the full product, see the
`backend/` and `frontend/` directories.

Run with:
    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    st.error(f"ultralytics is not installed: {exc}")
    raise

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "yolov8n.pt"
PERSON_CLASS_ID = 0


@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def annotate(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{det['name']} {det['conf']:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(out, (0, 0), (280, 42), (0, 0, 0), -1)
    cv2.putText(
        out,
        f"People: {len(detections)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def predict(model: YOLO, image: np.ndarray, conf: float) -> list[dict]:
    result = model.predict(source=image, conf=conf, iou=0.5, verbose=False)[0]
    detections: list[dict] = []
    if result.boxes is None:
        return detections
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names or {0: "person"}
    for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids, strict=False):
        if cid != PERSON_CLASS_ID:
            continue
        detections.append(
            {
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "conf": float(c),
                "name": names.get(int(cid), "person"),
            }
        )
    return detections


def image_mode(model: YOLO, conf: float) -> None:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        return
    image = np.array(Image.open(uploaded).convert("RGB"))[:, :, ::-1].copy()
    detections = predict(model, image, conf)
    annotated = annotate(image, detections)
    col1, col2 = st.columns(2)
    col1.metric("People detected", len(detections))
    col2.metric(
        "Avg. confidence",
        f"{np.mean([d['conf'] for d in detections]):.2f}" if detections else "—",
    )
    st.image(annotated[:, :, ::-1], caption="Annotated", use_column_width=True)


def video_mode(model: YOLO, conf: float) -> None:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if not uploaded:
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
        tf.write(uploaded.read())
        input_path = tf.name
    output_path = os.path.splitext(input_path)[0] + "_annotated.mp4"

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    progress = st.progress(0.0, text="Processing…")
    counts: list[int] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = predict(model, frame, conf)
        writer.write(annotate(frame, detections))
        counts.append(len(detections))
        idx += 1
        if total:
            progress.progress(min(idx / total, 1.0))
    cap.release()
    writer.release()
    progress.empty()

    col1, col2, col3 = st.columns(3)
    col1.metric("Frames", idx)
    col2.metric("Peak", max(counts) if counts else 0)
    col3.metric("Average", f"{np.mean(counts):.1f}" if counts else "—")
    st.video(output_path)
    st.line_chart({"people": counts})


def main() -> None:
    st.set_page_config(page_title="PeopleSense — Demo", page_icon="👥", layout="wide")
    st.title("PeopleSense — YOLOv8 Headcount Demo")
    st.caption(
        "Quick-start demo. For the full multi-user product (auth, analytics, alerts, live streams), "
        "see the FastAPI + React app in `backend/` and `frontend/`."
    )

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input(
            "Model weights",
            value=str(DEFAULT_MODEL),
            help="Path to YOLOv8 .pt weights",
        )
        conf = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
        mode = st.radio("Input", ["Image", "Video"])

    if not Path(model_path).exists():
        st.error(f"Model file not found: {model_path}")
        return

    model = load_model(model_path)
    if mode == "Image":
        image_mode(model, conf)
    else:
        video_mode(model, conf)


if __name__ == "__main__":
    main()
