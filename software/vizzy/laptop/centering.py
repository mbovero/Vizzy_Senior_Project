from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .hud import draw_wrapped_text
from .yolo_runner import clear_class_filter
from ..shared import config


# ------------------------------ helpers ------------------------------------- #

def _contour_center(mask_u8: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (cx, cy) using contour moments of the largest blob in a binary mask."""
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


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
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# ----------------------------- main routine --------------------------------- #

def center_on_class(
    cap,
    model,
    target_cls: int,
    center_x: int,
    center_y: int,
    send_move: Callable[[float, float], None],
    display_scale: float,
    label: str,
    frame_sink: Callable[[np.ndarray], None],
) -> bool:
    """
    Closed-loop centering on a class for up to config.CENTER_DURATION_MS.

    Args:
        cap: OpenCV VideoCapture.
        model: YOLO model instance (Ultralytics).
        target_cls: Class ID to center on.
        center_x, center_y: Pixel coordinates of frame center.
        send_move: Callback taking normalized (dx, dy) in [-1,1].
        display_scale: Scale factor for displayed frames.
        label: HUD label prefix.
        frame_sink: Callback to publish frames to main/UI thread.

    Returns:
        success (bool)
    """
    t0 = time.time()
    good_frames = 0
    success = False

    # Loop until time budget is exhausted
    while (time.time() - t0) < (config.CENTER_DURATION_MS / 1000.0):
        ok, frame = cap.read()
        if not ok:
            break

        # Model inference filtered to the target class
        results = model(frame, [int(target_cls)], verbose = config.YOLO_VERBOSE)
        for result in results:
            annotated = result.plot()

            # Choose highest-confidence instance (break ties by larger bbox area)
            best_conf = 0.0
            best_area = 0
            best_xy = None

            if len(result.boxes) > 0:
                masks = getattr(result, "masks", None)
                mask_list = list(masks.data) if (masks is not None and masks.data is not None) else None

                n = len(result.boxes)
                for i in range(n):
                    cid = int(result.boxes.cls[i].item())
                    if cid != int(target_cls):
                        continue

                    box_xyxy = result.boxes.xyxy[i].detach().cpu().numpy()
                    conf = float(result.boxes.conf[i].item())
                    x1, y1, x2, y2 = map(int, box_xyxy)
                    area = max(0, (x2 - x1)) * max(0, (y2 - y1))

                    mask_tensor = mask_list[i] if mask_list is not None and i < len(mask_list) else None
                    cx, cy = instance_center(box_xyxy, mask_tensor, annotated.shape[1], annotated.shape[0])

                    if (conf > best_conf) or (conf == best_conf and area > best_area):
                        best_conf = conf
                        best_area = area
                        best_xy = (int(cx), int(cy))

            if best_xy is not None:
                bx, by = best_xy

                # Pixel error relative to frame center
                dx_px = center_x - bx     # >0 → object left of center
                dy_px = by - center_y     # >0 → object below center
                err_px = max(abs(dx_px), abs(dy_px))

                # Normalized error to [-1, 1]
                ndx = dx_px / max(1, center_x)
                ndy = dy_px / max(1, center_y)
                move_norm = max(abs(ndx), abs(ndy))

                # Threshold checks
                conf_ok = (best_conf >= config.CENTER_CONF)
                err_ok = (abs(dx_px) <= config.CENTER_EPSILON_PX and abs(dy_px) <= config.CENTER_EPSILON_PX)
                move_ok = (abs(ndx) < config.CENTER_MOVE_NORM and abs(ndy) < config.CENTER_MOVE_NORM)

                # Command movement if not within stability thresholds
                if not (err_ok and move_ok):
                    send_move(ndx, ndy)

                # Count stable frames toward quota
                if conf_ok and err_ok and move_ok:
                    good_frames += 1
                    if good_frames >= config.CENTER_FRAMES:
                        success = True

                # ---------------- HUD ----------------
                # Object center (red)
                cv2.circle(annotated, (bx, by), 6, (0, 0, 255), -1)
                # Frame center (blue)
                cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
                # Line between them (green)
                cv2.line(annotated, (bx, by), (center_x, center_y), (0, 255, 0), 2)

                hud = (
                    f"{label}  quota {good_frames}/{config.CENTER_FRAMES}  "
                    f"conf {best_conf:.2f}  err {err_px:.0f}px  move {move_norm:.3f}"
                )
            else:
                # No detection — still draw camera center
                cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
                hud = f"{label}  quota {good_frames}/{config.CENTER_FRAMES}"

            # Draw HUD and publish frame
            max_w = int(annotated.shape[1] * 0.92)
            draw_wrapped_text(annotated, hud, 10, 24, max_w)
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * display_scale), int(h * display_scale)))
            frame_sink(resized)

    # Clear any class filter set during this centering attempt
    clear_class_filter(model)

    return success
