"""
Multi-Object Detection and Persistent ID Tracking
==================================================
Author: [Your Name]
Assignment: Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage

Pipeline:
  YOLOv8n (Detection) --> DeepSORT (Tracking) --> Annotated Output Video

Usage:
  python track.py --video input.mp4 --output output.mp4
  python track.py --video input.mp4 --output output.mp4 --skip_frames 2
"""

import argparse
import cv2
import os
import time
import numpy as np
from collections import defaultdict

# ── Install check ────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    raise ImportError("Run: pip install deep-sort-realtime")


# ── Colour palette for unique IDs ────────────────────────────────────────────
COLORS = [
    (0, 255, 0),   (255, 0, 0),   (0, 0, 255),   (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0),  (0, 128, 255),
    (255, 128, 0), (128, 0, 255), (0, 255, 128),  (255, 0, 128),
]

def get_color(track_id: int):
    """Return a consistent colour for a given track ID."""
    return COLORS[track_id % len(COLORS)]


# ── Detection helpers ─────────────────────────────────────────────────────────
def yolo_to_deepsort(boxes):
    """
    Convert YOLO xyxy boxes --> DeepSORT [left, top, width, height] format.
    Returns list of ([l,t,w,h], confidence, class_label).
    """
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))
    return detections


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_track(frame, track_id: int, ltrb, trajectory: list):
    """Draw bounding box, ID label, and trajectory tail for one track."""
    l, t, r, b = map(int, ltrb)
    color = get_color(track_id)

    # Bounding box
    cv2.rectangle(frame, (l, t), (r, b), color, 2)

    # ID label with filled background for readability
    label = f"ID:{track_id}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (l, t - lh - 10), (l + lw + 4, t), color, -1)
    cv2.putText(frame, label, (l + 2, t - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Trajectory tail (last 30 centres)
    if len(trajectory) > 1:
        pts = trajectory[-30:]
        for i in range(1, len(pts)):
            if pts[i - 1] and pts[i]:
                alpha = i / len(pts)          # fade older points
                t_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts[i - 1], pts[i], t_color, 2)


def draw_hud(frame, frame_no: int, fps: float, active_ids: int):
    """Overlay a heads-up display with frame info."""
    cv2.rectangle(frame, (0, 0), (260, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame : {frame_no}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS   : {fps:.1f}", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Tracked: {active_ids}", (8, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1)


# ── Core pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(video_path: str,
                 output_path: str,
                 conf_threshold: float = 0.4,
                 skip_frames: int = 1,
                 max_age: int = 30):
    """
    Full detection + tracking pipeline.

    Parameters
    ----------
    video_path     : path to input video
    output_path    : path for annotated output video
    conf_threshold : minimum YOLO confidence (0–1)
    skip_frames    : process every N-th frame (1 = every frame)
    max_age        : frames a track survives without a detection
    """

    # ── Load models ───────────────────────────────────────────────────────────
    print("[INFO] Loading YOLOv8n model ...")
    model = YOLO("yolov8n.pt")          # downloads automatically on first run

    print("[INFO] Initialising DeepSORT tracker ...")
    tracker = DeepSort(
        max_age=max_age,                # keep lost tracks for N frames
        n_init=3,                       # confirm a track after 3 matches
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,        # appearance similarity threshold
        nn_budget=None,
        embedder="mobilenet",           # lightweight appearance embedder
        half=True,
        bgr=True,
    )

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

    print(f"[INFO] Input : {video_path}  ({width}x{height} @ {src_fps:.1f} fps, {total_frames} frames)")
    print(f"[INFO] Output: {output_path}")

    # ── State ─────────────────────────────────────────────────────────────────
    trajectories   = defaultdict(list)   # track_id -> list of (cx, cy) centres
    count_over_time = []                 # (frame_no, active_count) log
    all_seen_ids   = set()
    frame_no = 0
    prev_time = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        # Optionally skip frames for speed
        if frame_no % skip_frames != 0:
            out.write(frame)
            continue

        # ── Detect ────────────────────────────────────────────────────────────
        results = model(frame, classes=[0], conf=conf_threshold, verbose=False)[0]
        detections = yolo_to_deepsort(results.boxes)

        # ── Track ─────────────────────────────────────────────────────────────
        tracks = tracker.update_tracks(detections, frame=frame)

        active_ids = 0
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid  = track.track_id
            ltrb = track.to_ltrb()
            all_seen_ids.add(tid)
            active_ids += 1

            # Update trajectory
            l, t, r, b = map(int, ltrb)
            cx, cy = (l + r) // 2, (t + b) // 2
            trajectories[tid].append((cx, cy))

            draw_track(frame, tid, ltrb, trajectories[tid])

        # ── HUD ───────────────────────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now
        draw_hud(frame, frame_no, fps, active_ids)

        count_over_time.append((frame_no, active_ids))
        out.write(frame)

        # Progress log every 50 frames
        if frame_no % 50 == 0:
            pct = (frame_no / total_frames * 100) if total_frames else 0
            print(f"  [{pct:5.1f}%] Frame {frame_no}/{total_frames}  |  "
                  f"Active IDs: {active_ids}  |  "
                  f"All-time IDs: {len(all_seen_ids)}")

    cap.release()
    out.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[DONE] ─────────────────────────────────────")
    print(f"  Total frames processed : {frame_no}")
    print(f"  Unique IDs assigned    : {len(all_seen_ids)}")
    print(f"  Output saved to        : {output_path}")

    return count_over_time, all_seen_ids


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Multi-Object Tracking Pipeline")
    p.add_argument("--video",  required=True,  help="Path to input video file")
    p.add_argument("--output", default="output.mp4", help="Path for annotated output video")
    p.add_argument("--conf",   type=float, default=0.4, help="YOLO confidence threshold (default: 0.4)")
    p.add_argument("--skip_frames", type=int, default=1, help="Process every N-th frame (default: 1 = all frames)")
    p.add_argument("--max_age",     type=int, default=30, help="DeepSORT max_age: frames to keep a lost track (default: 30)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        conf_threshold=args.conf,
        skip_frames=args.skip_frames,
        max_age=args.max_age,
    )
