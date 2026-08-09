"""
Pothole Detection & Counting — Streamlit demo
==============================================
Two-stage pipeline:
  Stage A  MobileNetV2 classifier  ->  is there a pothole in this frame?
  Stage B  YOLOv8n detector        ->  where exactly, and how many?
  Stage C  ByteTrack tracker       ->  how many UNIQUE potholes along the road?

Run:  streamlit run app.py
Needs: best.pt (detector) and pothole_classifier.pt (classifier) in the same folder.
"""

import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

# --------------------------------------------------------------------------- config
st.set_page_config(page_title="Pothole Detection & Counting", page_icon="🕳️", layout="wide")

HERE = Path(__file__).parent
DET_WEIGHTS = HERE / "best.pt"
CLS_WEIGHTS = HERE / "pothole_classifier.pt"

# Streamlit Community Cloud mounts the repo under /mount/src — used to hide the
# webcam tab, which cannot work on a remote server (no access to your camera).
IS_CLOUD = Path("/mount/src").exists() or os.environ.get("STREAMLIT_CLOUD") == "1"
CLASSES = ["normal", "potholes"]
POS = 1
IMSIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMSIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# --------------------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading detector…")
def load_detector():
    if not DET_WEIGHTS.exists():
        st.error(
            f"**Detector weights `{DET_WEIGHTS.name}` not found.**\n\n"
            "This file is produced by the training notebook. Download it from Colab "
            "and commit it to the repository root, next to `app.py`."
        )
        st.stop()
    return YOLO(str(DET_WEIGHTS))


@st.cache_resource(show_spinner="Loading classifier…")
def load_classifier():
    if not CLS_WEIGHTS.exists():
        return None
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(CLASSES))
    m.load_state_dict(torch.load(CLS_WEIGHTS, map_location=DEV))
    return m.to(DEV).eval()


def classify(bgr, clf):
    """Return p(pothole) for a BGR frame. 1.0 if no classifier is loaded."""
    if clf is None:
        return 1.0
    x = eval_tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEV)
    with torch.no_grad():
        return float(torch.softmax(clf(x), 1)[0, POS])


def severity(area_frac):
    if area_frac > 0.05:
        return "SEVERE", (0, 0, 255)
    if area_frac > 0.015:
        return "MODERATE", (0, 140, 255)
    return "MINOR", (0, 215, 255)


def draw(frame, xyxy, confs, ids, in_frame, total_unique, p_pot, show_hud=True):
    H, W = frame.shape[:2]
    rows = []
    for j, b in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, b)
        frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
        sev, col = severity(frac)
        tid = ids[j] if j < len(ids) else -1
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = (f"ID{tid} " if tid >= 0 else "") + f"{sev} {confs[j]:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        rows.append({"track_id": tid, "conf": round(float(confs[j]), 3),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "area_frac": round(frac, 5), "severity": sev})
    if show_hud:
        cv2.rectangle(frame, (0, 0), (445, 118), (0, 0, 0), -1)
        cv2.putText(frame, f"In frame     : {in_frame}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Total unique : {total_unique}", (12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"p(pothole)   : {p_pot:.2f}", (12, 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame, rows


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
conf_thr = st.sidebar.slider("Detector confidence", 0.05, 0.90, 0.35, 0.05,
                             help="Lower = higher recall (fewer missed potholes), more false alarms.")
iou_thr = st.sidebar.slider("NMS IoU", 0.1, 0.9, 0.5, 0.05)
use_gate = st.sidebar.checkbox("Use classifier as a gate", value=True,
                               help="Skip the detector on frames the classifier says are clean road. Saves compute.")
cls_thr = st.sidebar.slider("Classifier threshold", 0.1, 0.9, 0.5, 0.05, disabled=not use_gate)
min_frames = st.sidebar.slider("Confirm after N frames", 1, 20, 5,
                               help="A track must survive this many frames to be counted — removes flicker false positives.")
frame_skip = st.sidebar.slider("Process every Nth frame", 1, 5, 1,
                               help="Increase to speed up long videos.")

detector = load_detector()
classifier = load_classifier()

st.sidebar.markdown("---")
st.sidebar.write(f"**Device:** `{DEV}`")
st.sidebar.write(f"**Detector:** {'✅ best.pt' if DET_WEIGHTS.exists() else '❌ missing'}")
st.sidebar.write(f"**Classifier:** {'✅ pothole_classifier.pt' if classifier else '⚠️ not found (gate disabled)'}")

# --------------------------------------------------------------------------- header
st.title("🕳️ Pothole Detection and Counting")
st.caption("MobileNetV2 screening → YOLOv8n detection → ByteTrack counting | "
           "Image and Video Analytics (MDS507D-4)")

if IS_CLOUD:
    st.info("Running on Streamlit Community Cloud (CPU only). Video processing is slower here "
            "than on a local machine — keep clips under ~30 seconds, or raise "
            "*Process every Nth frame* in the sidebar.")

tab_img, tab_vid, tab_cam, tab_about = st.tabs(
    ["🖼️ Image", "🎬 Video", "📷 Live Webcam", "ℹ️ How it works"])

# --------------------------------------------------------------------------- IMAGE
with tab_img:
    st.subheader("Single image analysis")
    up = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"], key="img")

    if up:
        raw = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        p_pot = classify(raw, classifier)

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)

        if use_gate and classifier is not None and p_pot < cls_thr:
            c2.success("**Stage A verdict: clean road.** Detector skipped.")
            c2.metric("p(pothole)", f"{p_pot:.3f}")
        else:
            t0 = time.time()
            r = detector.predict(raw, conf=conf_thr, iou=iou_thr, imgsz=640, verbose=False)[0]
            dt = time.time() - t0
            xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.empty((0,))
            annotated, rows = draw(raw.copy(), xyxy, confs, [], len(xyxy), len(xyxy),
                                   p_pot, show_hud=False)
            c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detections", use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Potholes detected", len(xyxy))
            m2.metric("p(pothole) — Stage A", f"{p_pot:.3f}")
            m3.metric("Inference time", f"{dt*1000:.0f} ms")

            if rows:
                df = pd.DataFrame(rows).drop(columns=["track_id"])
                st.dataframe(df, use_container_width=True)
                st.download_button("⬇️ Download detections (CSV)",
                                   df.to_csv(index=False).encode(),
                                   "detections.csv", "text/csv")
                sev_counts = df.severity.value_counts()
                st.bar_chart(sev_counts)
            else:
                st.info("No potholes detected above the confidence threshold. "
                        "Try lowering it in the sidebar.")

# --------------------------------------------------------------------------- VIDEO
with tab_vid:
    st.subheader("Video analysis with unique-pothole counting")
    st.caption("Each pothole gets a persistent track ID, so it is counted once no matter how "
               "many frames it appears in.")
    upv = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"], key="vid")

    if upv and st.button("▶️ Run analysis", type="primary"):
        tmp_in = Path(tempfile.gettempdir()) / f"in_{int(time.time())}.mp4"
        tmp_in.write_bytes(upv.read())

        cap = cv2.VideoCapture(str(tmp_in))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        out_path = Path(tempfile.gettempdir()) / f"out_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        preview = st.empty()
        bar = st.progress(0.0)
        stat = st.empty()

        seen, log, skipped = set(), [], 0
        t0 = time.time()

        stream = detector.track(source=str(tmp_in), stream=True, persist=True,
                                tracker="bytetrack.yaml", conf=conf_thr, iou=iou_thr,
                                imgsz=640, vid_stride=frame_skip, verbose=False)

        i = -1
        for i, r in enumerate(stream):
            frame = r.orig_img.copy()
            p_pot = classify(frame, classifier) if (use_gate and classifier) else 1.0
            b = r.boxes

            if use_gate and classifier and p_pot < cls_thr:
                xyxy, confs, ids = np.empty((0, 4)), np.empty((0,)), []
                skipped += 1
            else:
                xyxy = b.xyxy.cpu().numpy() if b is not None else np.empty((0, 4))
                confs = b.conf.cpu().numpy() if b is not None else np.empty((0,))
                ids = b.id.int().cpu().tolist() if (b is not None and b.id is not None) else []
            seen.update(ids)

            annotated, rows = draw(frame, xyxy, confs, ids, len(xyxy), len(seen), p_pot)
            for row in rows:
                row.update({"frame": i, "time_s": round(i / fps, 2)})
                log.append(row)

            writer.write(annotated)

            if i % 5 == 0:
                preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                              caption=f"Frame {i}", use_container_width=True)
                if n_frames:
                    bar.progress(min(1.0, (i * frame_skip) / n_frames))
                stat.write(f"Frame {i} · in frame **{len(xyxy)}** · unique so far **{len(seen)}** "
                           f"· {(i+1)/(time.time()-t0):.1f} FPS")

        writer.release()
      # Convert MP4V output to H.264 for browser/Streamlit compatibility
        h264_path = out_path.with_name(out_path.stem + "_h264.mp4")

        os.system(
        f'ffmpeg -y -i "{out_path}" '
        f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart '
        f'"{h264_path}" > /dev/null 2>&1'
    )

        if h264_path.exists():
          out_path = h264_path
        bar.progress(1.0)
        n_done = i + 1

        log_df = pd.DataFrame(log)
        if len(log_df):
            per_id = (log_df.groupby("track_id")
                      .agg(frames_visible=("frame", "count"),
                           first_seen_s=("time_s", "min"),
                           last_seen_s=("time_s", "max"),
                           max_conf=("conf", "max"),
                           severity=("severity", lambda s: s.mode().iat[0]))
                      .reset_index())
            per_id["confirmed"] = per_id.frames_visible >= min_frames
            confirmed = int(per_id.confirmed.sum())
        else:
            per_id, confirmed = pd.DataFrame(), 0

        st.success("Analysis complete")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🕳️ Confirmed potholes", confirmed,
                  help=f"Unique track IDs that persisted ≥ {min_frames} frames")
        m2.metric("Raw unique IDs", len(per_id))
        m3.metric("Frames processed", n_done)
        m4.metric("Throughput", f"{n_done/(time.time()-t0):.1f} FPS")

        if use_gate and classifier:
            st.info(f"Classifier gate skipped the detector on **{skipped}** frames "
                    f"({skipped/max(n_done,1):.0%} of detector calls saved).")

        st.video(str(out_path))

        if len(per_id):
            st.markdown("#### Per-pothole report")
            st.dataframe(per_id, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.markdown("**Potholes detected per frame**")
            c1.line_chart(log_df.groupby("frame").size())
            c2.markdown("**Confirmed potholes by severity**")
            c2.bar_chart(per_id[per_id.confirmed].severity.value_counts())

            st.download_button("⬇️ Per-pothole report (CSV)",
                               per_id.to_csv(index=False).encode(),
                               "pothole_report.csv", "text/csv")
            st.download_button("⬇️ Full frame-by-frame log (CSV)",
                               log_df.to_csv(index=False).encode(),
                               "pothole_log.csv", "text/csv")
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Annotated video (MP4)", f.read(),
                               "annotated.mp4", "video/mp4")

# --------------------------------------------------------------------------- WEBCAM
with tab_cam:
    st.subheader("Live camera feed")

    if IS_CLOUD:
        st.warning(
            "**Not available on the hosted version.**\n\n"
            "This app is running on a server in a data centre, which has no camera and no access "
            "to yours. OpenCV's `VideoCapture(0)` would open a camera *on the server*, not on your "
            "laptop.\n\n"
            "To see this tab working, clone the repository and run it locally:\n"
            "```\ngit clone <this repo>\npip install -r requirements.txt\nstreamlit run app.py\n```\n"
            "Meanwhile, the **Video** tab demonstrates exactly the same tracking and counting "
            "logic on an uploaded clip."
        )
    else:
        st.caption("Runs on the machine Streamlit is running on. Use the checkbox to start/stop.")
        cam_index = st.number_input("Camera index", 0, 4, 0)
        run_cam = st.checkbox("🔴 Start camera")

        frame_slot = st.empty()
        metric_slot = st.empty()

        if run_cam:
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error("Could not open the camera. Check the index, or that no other app is using it.")
            else:
                seen = set()
                counts = {}
                i = 0
                while run_cam and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    p_pot = classify(frame, classifier) if (use_gate and classifier) else 1.0

                    if use_gate and classifier and p_pot < cls_thr:
                        xyxy, confs, ids = np.empty((0, 4)), np.empty((0,)), []
                    else:
                        r = detector.track(frame, persist=True, tracker="bytetrack.yaml",
                                           conf=conf_thr, iou=iou_thr, imgsz=640, verbose=False)[0]
                        b = r.boxes
                        xyxy = b.xyxy.cpu().numpy() if b is not None else np.empty((0, 4))
                        confs = b.conf.cpu().numpy() if b is not None else np.empty((0,))
                        ids = b.id.int().cpu().tolist() if (b is not None and b.id is not None) else []

                    for t in ids:
                        counts[t] = counts.get(t, 0) + 1
                    seen.update(t for t, c in counts.items() if c >= min_frames)

                    annotated, _ = draw(frame, xyxy, confs, ids, len(xyxy), len(seen), p_pot)
                    frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                     channels="RGB", use_container_width=True)
                    metric_slot.write(f"**In frame:** {len(xyxy)}  ·  "
                                      f"**Confirmed unique:** {len(seen)}  ·  "
                                      f"**p(pothole):** {p_pot:.2f}")
                    i += 1
                cap.release()
        else:
            st.info("Tick **Start camera** to begin. If you are running Streamlit on a remote server, "
                    "use the Video tab instead — the server has no access to your local camera.")


# --------------------------------------------------------------------------- ABOUT
with tab_about:
    st.markdown("""
### Pipeline

```
frame ──► MobileNetV2 classifier ──► p(pothole) < threshold ? ──► clean road, skip detector
                                            │ no
                                            ▼
                                    YOLOv8n detector ──► bounding boxes
                                            ▼
                                      ByteTrack ──► persistent IDs ──► unique count
```

**Stage A — screening classifier.** MobileNetV2 fine-tuned on 681 road images
(352 `normal` / 329 `potholes`) using two-phase transfer learning: the head is trained first with
the backbone frozen, then the whole network is fine-tuned at a 10× lower learning rate. At 3.5 M
parameters it runs in a few milliseconds, so it can gate the far heavier detector.

**Stage B — YOLOv8n detector.** Anchor-free single-stage detector fine-tuned from COCO weights on
a bounding-box pothole dataset. CSPDarknet backbone, PANet neck (multi-scale, so both distant
small potholes and near large ones are found), decoupled head, CIoU + DFL + BCE loss.

**Stage C — counting.** Summing detections per frame massively over-counts, because one pothole is
visible for 30–60 frames at 30 FPS. ByteTrack runs a Kalman filter per object and matches
predictions to detections by IoU, giving each pothole a persistent ID. The **unique count** is the
number of distinct IDs that survived at least *N* frames — the persistence filter removes
single-frame flicker false positives.

**Severity** is estimated from box area as a fraction of the frame
(< 1.5 % minor, 1.5–5 % moderate, > 5 % severe). This ranks potholes within one video but is not a
physical depth measurement — real severity would need stereo or monocular depth estimation.

### Known failure modes
- Water-filled potholes read as dark reflective patches and are often missed.
- Shadows, tar patches and manhole covers are the dominant false positives.
- Fast camera motion or occlusion can cause ID switches, inflating the unique count.
""")
