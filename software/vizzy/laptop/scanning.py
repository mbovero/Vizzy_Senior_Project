# -----------------------------------------------------------------------------
# vizzy/laptop/scanning.py
#
# Purpose
#   Implements a short "scan window" using YOLO object detection to identify
#   objects in the camera feed while the robotic arm is held still at a
#   specific position in its search path.
#
# Why this exists
#   - The RPi instructs the laptop to scan at each pose in the search grid.
#   - The laptop runs YOLO on camera frames for a fixed period (`duration_s`)
#     without moving the servos, so detections remain stable for averaging.
#   - The scan collects confidence and location statistics for any objects
#     seen at this pose, then reports the best candidates back to the RPi.
#
# How it fits into the project
#   - The robotic arm systematically sweeps through a set of poses.
#   - At each pose, this function runs for `duration_s` to detect visible
#     objects and estimate their median positions and average confidences.
#   - The RPi uses this summarized data to decide which object to center on.
#
# Key points for understanding:
#   - The arm is stationary during each scan; no motion tracking is performed.
#   - YOLO runs on every frame captured during the scan window.
#   - For each object class, only the largest box per frame is considered.
#   - Confidence scores are averaged, and box centers are median-filtered
#     to reduce jitter from frame to frame.
#   - Classes that appear in too few frames are ignored.
# -----------------------------------------------------------------------------

from __future__ import annotations
import time, statistics, cv2
from typing import Dict, List, Tuple, Iterable, Optional, Set
from .hud import draw_wrapped_text
from .yolo_runner import infer_all

def run_scan_window(
    cap,
    model,
    duration_s: float,
    class_filter: int,
    exclude_ids: Optional[Iterable[int]],
    display_scale: float,
    get_name,
    min_frames_for_class: int = 4
) -> dict:
    """
    Perform a short, stationary scan at the current arm position and
    summarize object detections from YOLO.

    Args:
        cap                : OpenCV VideoCapture object for reading frames.
        model              : YOLO model instance.
        duration_s         : How long to scan (seconds) while stationary.
        class_filter       : Restrict detection to one class (-1 for all).
        exclude_ids        : Iterable of class IDs to ignore.
        display_scale      : Scaling factor for display window size.
        get_name           : Function mapping class ID → human-readable name.
        min_frames_for_class: Minimum appearances before keeping a class.

    Returns:
        Dictionary containing:
          - "frames": total frames processed
          - "objects": list of objects sorted by avg_conf desc, each with:
              cls_id, cls_name, avg_conf, median_center[x,y], frames
    """
    # Convert exclusion list to set for faster lookups
    exclude: Set[int] = set(int(x) for x in (exclude_ids or []))
    t0 = time.time()
    frames = 0

    # Per-class tracking of confidences and center positions
    per_class_conf: Dict[int, List[float]] = {}
    per_class_cx:   Dict[int, List[int]]   = {}
    per_class_cy:   Dict[int, List[int]]   = {}

    hud_text = f"SCANNING ~{int(duration_s*1000)} ms"

    # Loop until scan window duration is reached
    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok:
            break  # End scan if camera feed fails

        # Run YOLO inference (filter to single class if requested)
        results = infer_all(model, frame, None if class_filter == -1 else [class_filter])

        for result in results:
            frames += 1

            # Pick best (highest-confidence) detection per class this frame 
            # best_per_class[cid] = (conf, cx, cy)
            best_per_class: Dict[int, Tuple[float, int, int]] = {}

            if len(result.boxes) > 0:
                for box in result.boxes:
                    cid = int(box.cls)

                    # Respect filter and exclusion list
                    if class_filter != -1 and cid != class_filter:
                        continue
                    if cid in exclude:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    conf = float(box.conf[0])

                    prev = best_per_class.get(cid)
                    # Keep the highest-confidence detection for THIS CLASS in THIS FRAME
                    if (prev is None) or (conf > prev[0]):
                        best_per_class[cid] = (conf, cx, cy)

            # Accumulate per-class stats using the per-class bests for this frame
            for cid, (conf, cx, cy) in best_per_class.items():
                per_class_conf.setdefault(cid, []).append(conf)
                per_class_cx.setdefault(cid, []).append(cx)
                per_class_cy.setdefault(cid, []).append(cy)
                
            # Draw text overlay and display the annotated frame
            annotated = result.plot()
            draw_wrapped_text(
                annotated,
                hud_text,
                10,
                24,
                int(annotated.shape[1] * 0.92)
            )
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * display_scale), int(h * display_scale)))
            cv2.imshow("YOLO Detection", resized)
            cv2.waitKey(1)  # Keep OpenCV's window responsive

    # Build output list of detected objects
    objects = []
    for cls_id, confs in per_class_conf.items():
        if len(confs) < min_frames_for_class:
            continue  # Ignore objects that appeared too few times
        avg_conf = sum(confs) / len(confs)
        med_cx = statistics.median(per_class_cx[cls_id])
        med_cy = statistics.median(per_class_cy[cls_id])
        name = get_name(int(cls_id))
        objects.append({
            "cls_id": int(cls_id),
            "cls_name": name,
            "avg_conf": float(avg_conf),
            "median_center": [float(med_cx), float(med_cy)],
            "frames": int(len(confs))
        })

    # Sort by confidence (highest first)
    objects.sort(key=lambda r: r["avg_conf"], reverse=True)

    return {"frames": int(frames), "objects": objects}
