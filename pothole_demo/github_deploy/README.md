# 🕳️ Pothole Detection and Counting from Images and Real-Time Video

A two-stage computer-vision system that detects potholes in road imagery and counts **unique**
potholes across a video stream.

**🔗 Live demo:** _add your Streamlit Cloud URL here after deploying_

> M.Sc. Data Science · CHRIST (Deemed to be University)
> MDS507D-4 Image and Video Analytics — CIA I Project

---

## What it does

| Stage | Model | Answers |
|---|---|---|
| **A — Screening** | MobileNetV2 (transfer learning) | Does this frame contain a pothole? |
| **B — Localisation** | YOLOv8n (fine-tuned from COCO) | Where exactly, and how many? |
| **C — Counting** | ByteTrack | How many **unique** potholes along this road? |

```
frame ──► MobileNetV2 classifier ──► p(pothole) < threshold ? ──► clean road, skip detector
                                            │ no
                                            ▼
                                    YOLOv8n detector ──► bounding boxes
                                            ▼
                                      ByteTrack ──► persistent IDs ──► unique count
```

### Why two stages

The primary dataset is **classification-only** — 681 road images labelled `normal` / `potholes`,
with no bounding boxes. A classifier can say *"there is a pothole"* but never *"there are three"*,
because counting requires per-object localisation. So the detector is trained separately on a
bounding-box dataset, and the classifier is repurposed as a **cheap high-recall gate**: at 3.5 M
parameters it runs on every frame in milliseconds, and the far heavier detector is invoked only on
frames it flags. Most road footage contains no potholes, so this saves the bulk of the compute.

### Why counting is the hard part

At 30 FPS a single pothole stays in view for 30–60 frames. Summing per-frame detections would
report ~40 potholes where there is one. **ByteTrack** runs a Kalman filter per object and matches
predictions to detections by IoU — using both high- and low-confidence detections, which keeps
identities alive through motion blur and partial occlusion. Each pothole gets a persistent ID, and
the unique count is the number of distinct IDs that survived at least *N* frames. That persistence
filter removes single-frame flicker false positives.

---

## Dataset

**Stage A (classification)** — 681 road photographs:

| Class | Images |
|---|---|
| `normal` | 352 |
| `potholes` | 329 |

462 distinct resolutions, varying angle, illumination and weather. The `normal` class deliberately
includes shadows, tar patches, road markings and manhole covers — the hard negatives.

**Stage B (detection)** — ~665 annotated road images, single class `pothole`, from Roboflow Universe.

### One EDA finding worth highlighting

The intuitive assumption is *"a pothole is a dark patch"*. Measured across all 681 images, that is
**false**:

| Feature | `normal` | `potholes` |
|---|---|---|
| Mean intensity | 119.9 | 120.2 |
| Canny edge density | 0.100 | **0.177** |

Global brightness carries essentially **no signal** — a shaded intact road is as dark as a sunlit
pothole, which rules out any hand-crafted thresholding approach. What separates the classes is
**texture**: 77 % higher edge density, from the broken rim, shadowed interior and debris. Grad-CAM
confirms the network learns exactly this cue.

---

## Results

| Stage A (classification) | Value |
|---|---|
| Accuracy | _fill in_ |
| Precision / Recall / F1 | _fill in_ |
| ROC-AUC | _fill in_ |

| Stage B (detection) | Value |
|---|---|
| mAP@0.5 | _fill in_ |
| mAP@0.5:0.95 | _fill in_ |
| Precision / Recall | _fill in_ |

**Recall is the metric that matters most here** — a missed pothole is an unreported safety hazard —
so the operating threshold is deliberately biased towards recall at some cost in precision.

---

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit application — image, video and webcam tabs |
| `Pothole_Detection_YOLOv8.ipynb` | Colab notebook that trains both models end to end |
| `best.pt` | Trained YOLOv8n detector weights |
| `pothole_classifier.pt` | Trained MobileNetV2 classifier weights |
| `requirements.txt` | Python dependencies (CPU-only PyTorch for cloud hosting) |
| `packages.txt` | System libraries required by Streamlit Cloud |

---

## Run locally

```bash
git clone https://github.com/<your-username>/pothole-detection.git
cd pothole-detection
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

> The **Live Webcam** tab only works locally. On the hosted version it is disabled, because the
> server has no camera and no access to yours.

## Retrain from scratch

Open `Pothole_Detection_YOLOv8.ipynb` in Google Colab, set **Runtime → T4 GPU**, and run top to
bottom. You will be prompted for the dataset zip, a free Roboflow API key, and a road video.
Roughly 30 minutes end to end.

---

## Limitations

1. **Small, narrow dataset** — 681 images, mostly daytime and dry; recall drops at night and in rain.
2. **Domain gap** — classifier and detector were trained on different image distributions.
3. **Water-filled potholes** read as dark reflective patches and are frequently missed.
4. **Shadows, tar patches and manhole covers** are the dominant false positives.
5. **Severity is 2-D** — box area scales with camera distance, so it ranks potholes within one video
   but is not a physical depth measurement.
6. **ID switches** under fast camera motion or occlusion can inflate the unique count.

## Future work

- Annotate the 329 own-dataset pothole images with bounding boxes and fine-tune the detector on
  them — this closes the domain gap and is the highest-value next step.
- Scale to **RDD2022** (47 k images, multi-country) with crack and rutting classes.
- **Monocular depth estimation** (MiDaS) to convert box area into real depth and repair cost.
- **GPS tagging** so each unique pothole ID becomes a pin on a municipal repair dashboard.
- **TensorRT export** for edge deployment on a Jetson fitted to a bus or garbage truck.

---

## Tech stack

Python · PyTorch · torchvision · Ultralytics YOLOv8 · OpenCV · ByteTrack · scikit-learn · Streamlit

## Acknowledgements

- Detection dataset: [Roboflow Universe](https://universe.roboflow.com)
- Detector: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Classifier backbone: MobileNetV2, Sandler et al. (2018)
