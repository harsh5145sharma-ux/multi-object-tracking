# 🎯 Multi-Object Detection & Persistent ID Tracking

**Assignment:** Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage  
**Type:** AI / Computer Vision / Data Science  
**Pipeline:** YOLOv8n → DeepSORT → Annotated Output Video

---

## 📌 Source Video

> **Video used:** [Paste your YouTube / public video link here]  
> *(e.g., https://www.youtube.com/watch?v=XXXXXXX)*

---

## 🧠 Approach Summary

This pipeline uses two proven, industry-standard components:

| Component | Role |
|---|---|
| **YOLOv8n** | Real-time person detection on every frame |
| **DeepSORT** | Multi-object tracking with persistent, unique IDs |

Each detected person is assigned a **unique ID that persists** across the entire video, even through partial occlusion and rapid motion.

---

## 🗂️ Project Structure

```
multi-object-tracking/
├── track.py           ← Main Python script (run locally)
├── notebook.ipynb     ← Google Colab notebook (recommended)
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
└── screenshots/       ← Sample output frames & charts
```

---

## ⚙️ Installation

### Option A — Google Colab (Recommended, free GPU)
1. Open `notebook.ipynb` in [Google Colab](https://colab.research.google.com)
2. Click **Runtime → Run All**
3. Upload your video when prompted

### Option B — Run Locally

**Requirements:** Python 3.8+, pip

```bash
# Clone / download this repo, then:
pip install -r requirements.txt

# Run tracking
python track.py --video your_video.mp4 --output output.mp4
```

**All CLI options:**
```
--video        Path to input video (required)
--output       Output file path (default: output.mp4)
--conf         YOLO confidence threshold (default: 0.4)
--skip_frames  Process every N-th frame for speed (default: 1)
--max_age      Frames to keep a lost track alive (default: 30)
```

---

## 📦 Dependencies

```
ultralytics>=8.0.0       # YOLOv8 detection
deep-sort-realtime>=1.3  # DeepSORT tracking
opencv-python>=4.8.0     # Video I/O and drawing
numpy>=1.24.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔍 How It Works

```
Input Video
    │
    ▼
[YOLOv8n] ──── detects all persons in frame
    │           (bounding box + confidence score)
    ▼
[DeepSORT] ─── matches detections to existing tracks
    │           using Kalman filter + appearance embedding
    ▼
[Annotated] ── draws bounding box, unique ID label,
    │           trajectory tail, and HUD overlay
    ▼
Output Video
```

### ID Consistency — How It's Maintained

1. **Kalman Filter** — predicts where each object will be next frame, even before a detection arrives
2. **Appearance Embedding (MobileNet)** — extracts a visual feature vector for each detected crop; used to re-identify the same person after brief occlusion
3. **Hungarian Algorithm** — optimally matches new detections to existing tracks using both position and appearance
4. **`n_init=3`** — a track is only confirmed (shown) after matching 3 consecutive frames → reduces false IDs
5. **`max_age=30`** — a lost track is kept alive for 30 frames, allowing re-association after short disappearances

---

## ⚠️ Assumptions

- Input video contains **people** as primary moving subjects (class 0 in COCO)
- Video is from a **publicly accessible source**
- Lighting and resolution are reasonable (not extremely dark or very low-res)
- Processing is done on every frame by default; `--skip_frames 2` halves compute for longer videos

---

## 🚧 Limitations

| Issue | Description |
|---|---|
| **ID switching** | Can occur when two people overlap for many frames and then separate |
| **Crowded scenes** | Very dense crowds (20+ overlapping people) reduce tracking accuracy |
| **Camera shake** | Fast panning cameras make Kalman predictions less accurate |
| **Re-entry** | If a person leaves the frame entirely and re-enters after `max_age` frames, they get a new ID |
| **Similar appearance** | Players in identical uniforms challenge the appearance embedder |

---

## 🚀 Possible Improvements

- Replace DeepSORT with **ByteTrack** or **BoT-SORT** for better multi-object handling in crowds
- Use **YOLOv8x** (larger model) instead of YOLOv8n for higher detection accuracy
- Fine-tune YOLO on a **sports-specific dataset** (e.g., SoccerNet, SportsMOT)
- Add **team/role clustering** by jersey colour using K-means on HSV histograms
- Add **speed estimation** by converting pixel displacement to real-world units using homography
- Use a **bird's-eye view projection** for top-down movement analysis

---

## 🏆 Model & Tracker Choices — Rationale

**Why YOLOv8n?**
- State-of-the-art single-stage detector; best speed/accuracy trade-off
- Nano variant runs in real-time even on CPU
- Pre-trained on COCO (includes "person" class with 80K+ examples)
- Simple 1-line inference API via `ultralytics`

**Why DeepSORT?**
- Industry-proven tracker used in production surveillance and sports analytics systems
- Combines motion prediction (Kalman) + appearance features → robust re-identification
- Handles occlusion far better than pure IoU-based trackers (SORT)
- Well-documented and easy to integrate

---

## 📸 Sample Output

Screenshots from the annotated output video are in the `screenshots/` folder.

Each tracked subject shows:
- Coloured bounding box unique to their ID
- `ID:N` label with filled background for readability  
- Trajectory tail showing recent movement path
- HUD overlay showing frame number and active track count

---

## 📄 Technical Report

See `technical_report.pdf` in the submission for the full 2-page technical report covering model choices, tracking design, challenges, and possible improvements.
