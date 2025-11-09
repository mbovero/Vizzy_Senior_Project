from __future__ import annotations
import time
import statistics
from typing import Dict, List, Iterable, Optional, Set, Any, Callable, Tuple

import cv2
import numpy as np

from .hud import draw_wrapped_text
from .centering import instance_center
from ..shared import config as C

# Always publish frames via the sink; main thread renders them.
FrameSink = Callable[[Any], None]


def _draw_scan_hud(frame, text: str, *, display_scale: float) -> Any:
    h, w = frame.shape[:2]
    # translucent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, int(28 * display_scale)), (0, 0, 0), -1)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    draw_wrapped_text(frame, text, 8, 8, int(w * 0.8))
    return frame

# ---------------------------------------------------------------------------
# Laptop-driven search path
# ---------------------------------------------------------------------------
def build_search_path() -> list[dict]:
    """
    Build a custom arch search path with 15 total points:
      - Start: (0, -300mm), (0, -400mm), (0, -500mm) - 3 points
      - Middle: 9 interpolated points forming an arch from (0, -500mm) to (0, 500mm)
      - End: (0, 500mm), (0, 400mm), (0, 300mm) - 3 points
    
    Constraints:
      - x >= 0 (non-negative, 0mm is fully extended)
      - y can be negative or positive
      - Magnitude of (x, y) must be >= 300mm: sqrt(x^2 + y^2) >= 300mm
      - Magnitude of (x, y) must be <= 500mm: sqrt(x^2 + y^2) <= 500mm
    """
    import math
    
    def is_valid_pose(x: float, y: float) -> bool:
        """Check if pose meets constraints."""
        # x must be non-negative (0mm is fully extended)
        if x < 0:
            return False
        
        # Calculate magnitude (distance from origin)
        magnitude = math.sqrt(x * x + y * y)
        
        # Magnitude must be between 300-500mm
        if magnitude < 300.0 or magnitude > 500.0:
            return False
        
        return True
    
    # Z is fixed at 275mm for the entire scan cycle
    z_fixed = getattr(C, 'SEARCH_Z_FIXED_MM', 275.0)
    pitch = float(C.SEARCH_PITCH_DEG)
    
    path: list[dict] = []
    pose_id = 0
    
    # Step 1: Start points (3 points)
    # (0, -300), (0, -400), (0, -500)
    start_points = [
        (0.0, -300.0),
        (0.0, -400.0),
        (0.0, -500.0),
    ]
    
    for x, y in start_points:
        if is_valid_pose(x, y):
            path.append({
                "pose_id": pose_id,
                "x": float(x),
                "y": float(y),
                "z": float(z_fixed),
                "pitch": pitch,
            })
            pose_id += 1
    
    # Step 2: Interpolate 9 points between (0, -500mm) and (0, 500mm)
    # Create points along an arch that maintains magnitude between 300-500mm
    # We'll create points along a circular arc with radius ~450mm
    # This gives us points that smoothly transition from negative y to positive y
    
    # Generate 9 intermediate points along an arch
    # Using angles from -90 degrees (pointing down) to +90 degrees (pointing up)
    # With radius ~450mm to stay within 300-500mm range
    radius = 450.0  # Approximate radius for the arch
    
    # 9 points means 10 intervals (from -90 to +90 degrees)
    # Angles: -90, -70, -50, -30, -10, 10, 30, 50, 70, 90 (9 intermediate points)
    # But we want to exclude the endpoints since they're in start/end, so:
    # We need 9 points between -90 and +90, so we use 10 intervals
    num_middle_points = 9
    start_angle = -90.0  # degrees (pointing down, y negative)
    end_angle = 90.0     # degrees (pointing up, y positive)
    angle_step = (end_angle - start_angle) / (num_middle_points + 1)  # +1 because we exclude endpoints
    
    for i in range(1, num_middle_points + 1):  # 1 to 9 (skip endpoints)
        angle_deg = start_angle + (angle_step * i)
        angle_rad = math.radians(angle_deg)
        
        # Calculate x and y along the arch
        # x = radius * cos(angle), y = radius * sin(angle)
        x = radius * math.cos(angle_rad)
        y = radius * math.sin(angle_rad)
        
        # Ensure x >= 0 (if angle is in second quadrant, x would be negative)
        # For angles from -90 to +90, cos is positive, so x should be positive
        if x < 0:
            x = 0.0  # Clamp to 0 if somehow negative
        
        # Adjust to maintain exact magnitude if needed
        magnitude = math.sqrt(x * x + y * y)
        if magnitude > 500.0:
            # Scale down to fit within 500mm
            scale = 500.0 / magnitude
            x *= scale
            y *= scale
        elif magnitude < 300.0:
            # Scale up to fit within 300mm minimum
            scale = 300.0 / magnitude
            x *= scale
            y *= scale
        
        if is_valid_pose(x, y):
            path.append({
                "pose_id": pose_id,
                "x": float(x),
                "y": float(y),
                "z": float(z_fixed),
                "pitch": pitch,
            })
            pose_id += 1
    
    # Step 3: End points (3 points)
    # (0, 300), (0, 400), (0, 500) - ending sequence
    end_points = [
        (0.0, 300.0),
        (0.0, 400.0),
        (0.0, 500.0),
    ]
    
    for x, y in end_points:
        if is_valid_pose(x, y):
            path.append({
                "pose_id": pose_id,
                "x": float(x),
                "y": float(y),
                "z": float(z_fixed),
                "pitch": pitch,
            })
            pose_id += 1
    
    print(f"[build_search_path] Created {len(path)} poses in custom arch path (should be 15)")
    for i, p in enumerate(path):
        mag = math.sqrt(p["x"]**2 + p["y"]**2)
        print(f"  Pose {i}: ({p['x']:.1f}, {p['y']:.1f}) mm, magnitude={mag:.1f} mm")
    
    if len(path) != 15:
        print(f"[build_search_path] WARNING: Expected 15 points, got {len(path)}")
    
    return path

def _result_iter(model_result, frame_w: int, frame_h: int):
    """
    Yield (cls_id:int, conf:float, cx:int, cy:int) for each detection.
    Uses mask center (segmentation) when available; otherwise bbox center.
    """
    boxes = getattr(model_result, "boxes", None)
    masks = getattr(model_result, "masks", None)

    if boxes is None or boxes.xyxy is None:
        return

    # Torch -> numpy for boxes; leave masks as *torch tensors* (centering.instance_center expects tensor)
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros((xyxy.shape[0],), dtype=float)
    clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)

    mask_list = None
    if masks is not None and getattr(masks, "data", None) is not None:
        # masks.data is a torch.Tensor of shape (N, Hm, Wm); keep as tensors
        mask_list = list(masks.data)

    for i in range(xyxy.shape[0]):
        cls_id = int(clss[i])
        conf = float(confs[i])

        if mask_list is not None and i < len(mask_list):
            cx, cy = instance_center(xyxy[i], mask_list[i], frame_w, frame_h)
            yield cls_id, conf, int(cx), int(cy)
        else:
            x1, y1, x2, y2 = xyxy[i]
            cx = int(0.5 * (x1 + x2))
            cy = int(0.5 * (y1 + y2))
            yield cls_id, conf, int(cx), int(cy)


def run_scan_window(
    cap,
    model,
    exclude_ids: Optional[Iterable[int]],
    get_name,
    min_frames_for_class: int = 4,
    *,
    frame_sink: FrameSink,                 # REQUIRED: main-thread renderer
    display_scale: float = None,
    allowed_class_ids: Optional[Iterable[int]] = None,
) -> dict:
    """
    Run a short scan window and aggregate per-class stats while continuously
    publishing YOLO-annotated frames to the main thread.
    """
    if display_scale is None:
        display_scale = float(C.DISPLAY_SCALE)

    exclude: Set[int] = set(int(e) for e in (exclude_ids or []))
    t0 = time.time()
    frames = 0

    per_class_conf: Dict[int, List[float]] = {}
    per_class_cx: Dict[int, List[int]] = {}
    per_class_cy: Dict[int, List[int]] = {}

    window_ms = int(C.SCAN_DURATION_MS)
    hud_text = f"SCANNING ~{window_ms} ms"
    duration_s = float(window_ms) / 1000.0

    allowed_ids = list(allowed_class_ids) if allowed_class_ids is not None else None

    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        h, w = frame.shape[:2]

        # Run YOLO and get an annotated image to display (faster inference)
        results = model(frame, classes=allowed_ids, verbose=False, half=False)  # Disable verbose for speed

        annotated = frame  # fallback if no results object behaves oddly
        for r in results:
            # Aggregate stats from detections
            for cls_id, conf, cx, cy in _result_iter(r, w, h):
                if cls_id in exclude:
                    continue
                per_class_conf.setdefault(cls_id, []).append(conf)
                per_class_cx.setdefault(cls_id, []).append(cx)
                per_class_cy.setdefault(cls_id, []).append(cy)

            # Use YOLO's plotted/annotated frame for display
            try:
                annotated = r.plot()
            except Exception:
                pass  # if plot fails, show the raw frame + HUD

        # Add a simple HUD ribbon and publish annotated frame (only publish once per loop)
        annotated = _draw_scan_hud(annotated, hud_text, display_scale=display_scale)
        rh, rw = annotated.shape[:2]
        resized = cv2.resize(annotated, (int(rw * display_scale), int(rh * display_scale)))
        frame_sink(resized)

    # Build summary
    objects: List[dict] = []
    for cls_id, confs in per_class_conf.items():
        if len(confs) < int(min_frames_for_class):
            continue
        name = str(get_name(cls_id))
        avg_conf = float(sum(confs) / max(1, len(confs)))
        med_cx = int(statistics.median(per_class_cx.get(cls_id, [0])))
        med_cy = int(statistics.median(per_class_cy.get(cls_id, [0])))
        objects.append(
            {
                "cls_id": int(cls_id),
                "cls_name": name,
                "avg_conf": float(avg_conf),
                "median_center": [float(med_cx), float(med_cy)],
                "frames": int(len(confs)),
            }
        )

    objects.sort(key=lambda r: r["avg_conf"], reverse=True)
    return {"frames": int(frames), "objects": objects}
