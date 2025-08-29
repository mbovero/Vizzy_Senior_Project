# -----------------------------------------------------------------------------
# vizzy/laptop/centering.py
#
# Purpose:
#   Runs a closed-loop centering routine for a single object class.
#   This is typically called while the robotic arm is stationary at a given
#   search position, to align the camera’s view so that the target object
#   is centered in frame before verification and memory update.
#
# Operation:
#   • Runs YOLO inference on live camera frames for a specified duration.
#   • Filters detections to only the target class (class ID).
#   • Picks the *largest visible instance* of the target as the primary target.
#   • Measures pixel error between target center and frame center.
#   • Normalizes error to [-1, 1] range for servo commands.
#   • Sends movement commands until:
#       - Object is within pixel epsilon (positional accuracy threshold)
#       - Movement magnitude is under the normalized epsilon (stable)
#       - Confidence exceeds the per-frame threshold
#       - These conditions hold for the required quota of frames
#   • Stops early if quota is met before time runs out.
#
# Quota-based verification:
#   Instead of requiring stability for an entire continuous period, the
#   verification succeeds when a certain number of frames (“good frames”)
#   meet the thresholds, even if they are non-consecutive.
#
# Debug mode:
#   When debug=True, collects diagnostic statistics (max confidence seen,
#   min pixel error seen, min normalized move seen) and returns them
#   alongside the success flag.
# -----------------------------------------------------------------------------

from __future__ import annotations
import time, cv2
from typing import Callable, Optional, Dict, Any, Tuple
from .hud import draw_wrapped_text
from .yolo_runner import infer_all, clear_class_filter
import numpy as np

def _contour_center(mask_u8: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (cx, cy) using contour moments of the largest blob in a binary mask."""
    # mask_u8 must be 0/255 uint8
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def instance_center(box_xyxy, mask_tensor, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """
    Return (cx, cy) for a single detection:
      - If a mask is provided, compute segmentation center (contour moments).
      - Else, return box center.
    """
    if mask_tensor is not None:
        # mask tensor is (Hmask, Wmask) float/bool; resize to frame dims, make binary u8
        mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
        mask_np = cv2.resize(mask_np, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        mask_u8 = (mask_np * 255).astype(np.uint8)
        c = _contour_center(mask_u8)
        if c is not None:
            return c

    # fallback: box center
    x1, y1, x2, y2 = map(int, box_xyxy)
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def center_on_class(cap,
                    model,
                    duration_s: float,
                    target_cls: int,
                    epsilon_px: int,
                    center_x: int, center_y: int,
                    thresholds: Dict[str, float],
                    send_move: Callable[[float, float], None],
                    display_scale: float,
                    label: str,
                    debug: bool) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Run closed-loop centering on a specific class for up to `duration_s` seconds.

    Args:
        cap: OpenCV VideoCapture providing frames.
        model: YOLO model instance (from yolo_runner.py).
        duration_s: Max time to attempt centering.
        target_cls: Class ID to track.
        epsilon_px: Max allowed pixel error (both x and y) for stability.
        center_x, center_y: Pixel coordinates of frame center.
        thresholds: Dict with keys:
            - "conf" : per-frame confidence minimum (float)
            - "move_norm_eps" : normalized motion epsilon (float)
            - "required_frames" : quota of frames meeting all thresholds (int)
        send_move: Callback taking normalized (dx, dy) to request servo movement.
        display_scale: Factor to scale display window for annotation.
        label: Text prefix for HUD overlay.
        debug: If True, collect diagnostics to return.

    Returns:
        (success, diag) where:
            success: True if quota reached within duration_s, else False.
            diag: Diagnostic dict if debug=True, else None.
    """
    # Extract thresholds
    CONF = float(thresholds["conf"])
    MOVE_EPS = float(thresholds["move_norm_eps"])
    REQUIRED = int(thresholds["required_frames"])

    # Initialize timing and counters
    t0 = time.time()
    good_frames = 0
    success = False

    # Diagnostic variables (only meaningful if debug=True)
    total_frames = 0
    max_conf_seen = 0.0
    min_err_px_seen = float('inf')
    min_move_norm_seen = float('inf')

    # Main loop — runs until time limit is reached
    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok:
            break

        # Run YOLO inference restricted to the target class
        results = infer_all(model, frame, [int(target_cls)])
        for result in results:
            total_frames += 1
            annotated = result.plot()

            # Select the highest-confidence instance of the target class in view
            best = None                 # (cx, cy)
            best_conf = 0.0
            best_area = 0

            if len(result.boxes) > 0:
                masks = getattr(result, "masks", None)
                mask_list = list(masks.data) if (masks is not None and masks.data is not None) else None

                n = len(result.boxes)
                for i in range(n):
                    cid = int(result.boxes.cls[i].item())
                    if cid != int(target_cls):
                        continue

                    box_xyxy = result.boxes.xyxy[i].detach().cpu().numpy()
                    conf     = float(result.boxes.conf[i].item())
                    x1, y1, x2, y2 = map(int, box_xyxy)
                    area = (x2 - x1) * (y2 - y1)

                    mask_tensor = mask_list[i] if mask_list is not None and i < len(mask_list) else None
                    cx, cy = instance_center(box_xyxy, mask_tensor, annotated.shape[1], annotated.shape[0])

                    if (conf > best_conf) or (conf == best_conf and area > best_area):
                        best_conf = conf
                        best_area = area
                        best = (cx, cy)

            if best is not None:
                bx, by = int(best[0]), int(best[1])

                # Pixel error from frame center
                dx = (center_x - bx)   # >0 → object is left of center
                dy = (by - center_y)   # >0 → object is below center
                err_px = max(abs(dx), abs(dy))

                # Normalize error to [-1, 1] range
                ndx = dx / center_x
                ndy = dy / center_y
                move_norm = max(abs(ndx), abs(ndy))

                # Threshold checks
                conf_ok = (best_conf >= CONF)
                err_ok  = (abs(dx) <= epsilon_px and abs(dy) <= epsilon_px)
                move_ok = (abs(ndx) < MOVE_EPS and abs(ndy) < MOVE_EPS)

                # If not stable, command movement toward center
                if not (err_ok and move_ok):
                    send_move(ndx, ndy)

                # Count good frames toward quota
                if conf_ok and err_ok and move_ok:
                    good_frames += 1
                    if good_frames >= REQUIRED:
                        success = True

                # --- NEW: draw object center, camera center, and connecting line ---
                # Object center (red)
                cv2.circle(annotated, (bx, by), 6, (0, 0, 255), -1)
                # Camera/frame center (blue)
                cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
                # Line between them (green)
                cv2.line(annotated, (bx, by), (center_x, center_y), (0, 255, 0), 2)

                # Build HUD overlay text with live metrics and centers
                hud = (
                    f"{label}  quota {good_frames}/{REQUIRED}  "
                    f"conf {best_conf:.2f}  err {err_px:.0f}px  move {move_norm:.3f}  "
                    f"obj ({bx},{by})  cam ({center_x},{center_y})"
                )

            else:
                # No detection this frame — still draw camera center as a reference
                cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
                hud = f"{label}  quota {good_frames}/{REQUIRED}"

            # Track diagnostic extremes if debug enabled
            if debug:
                max_conf_seen = max(max_conf_seen, best_conf if best is not None else 0.0)
                if best is not None:
                    min_err_px_seen = min(min_err_px_seen, err_px)
                    min_move_norm_seen = min(min_move_norm_seen, move_norm)

            # Draw HUD and show frame
            max_w = int(annotated.shape[1] * 0.92)
            draw_wrapped_text(annotated, hud, 10, 24, max_w)
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * display_scale), int(h * display_scale)))
            cv2.imshow("YOLO Detection", resized)
            cv2.waitKey(1)

    # Clear any class filter set during this centering attempt
    clear_class_filter(model)

    # Prepare diagnostic output if requested
    diag = None
    if debug:
        diag = {
            "thresholds": {
                "conf_per_frame": CONF,
                "pixel_epsilon": int(epsilon_px),
                "move_norm_eps": MOVE_EPS,
                "required_good_frames": int(REQUIRED),
            },
            "observed": {
                "total_frames": int(total_frames),
                "good_frames": int(good_frames),
                "max_conf_seen": round(max_conf_seen, 4),
                "min_err_px_seen": None if min_err_px_seen == float('inf') else int(min_err_px_seen),
                "min_move_norm_seen": None if min_move_norm_seen == float('inf') else round(min_move_norm_seen, 4),
            }
        }

    return success, diag
