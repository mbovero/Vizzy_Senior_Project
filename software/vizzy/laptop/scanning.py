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
    Build the serpentine search path as a list of poses:
      [{"pose_id": int, "pwm_btm": int, "pwm_top": int, "slew_ms": int}, ...]

    Logic matches the RPi version:
      - Use [SERVO_MIN+SEARCH_MIN_OFFSET, SERVO_MAX-SEARCH_MAX_OFFSET] for both axes
      - Inclusive steps (SEARCH_H_STEP / SEARCH_V_STEP), step<=0 -> 1
      - Serpentine traversal across rows to avoid long reversals
    """
    # Bounds (trimmed)
    b_lo = int(C.SERVO_MIN + C.SEARCH_MIN_OFFSET)
    b_hi = int(C.SERVO_MAX - C.SEARCH_MAX_OFFSET)
    t_lo = int(C.SERVO_MIN + C.SEARCH_MIN_OFFSET)
    t_hi = int(C.SERVO_MAX - C.SEARCH_MAX_OFFSET)

    # Ensure lo <= hi on both axes
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo

    # Steps (guard against non-positive)
    h_step = int(C.SEARCH_H_STEP) if int(getattr(C, "SEARCH_H_STEP", 0)) > 0 else 1
    v_step = int(C.SEARCH_V_STEP) if int(getattr(C, "SEARCH_V_STEP", 0)) > 0 else 1

    # Inclusive ranges
    btms: list[int] = []
    x = b_lo
    while x <= b_hi:
        btms.append(int(x))
        x += h_step

    tops: list[int] = []
    y = t_lo
    while y <= t_hi:
        tops.append(int(y))
        y += v_step

    # Serpentine rows
    grid: list[tuple[int, int]] = []
    reverse = False
    for top in tops:
        row = btms[::-1] if reverse else btms
        grid.extend((b, top) for b in row)
        reverse = not reverse

    # Attach pose_id and per-pose slew
    slew_ms = int(C.GOTO_POSE_SLEW_MS)
    path: list[dict] = []
    for pid, (pwm_btm, pwm_top) in enumerate(grid):
        path.append({
            "pose_id": pid,
            "pwm_btm": int(pwm_btm),
            "pwm_top": int(pwm_top),
            "slew_ms": slew_ms,
        })
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
) -> dict:
    """
    Run a short scan window and aggregate per-class stats while continuously
    publishing YOLO-annotated frames to the main thread.
    """
    if display_scale is None:
        display_scale = float(getattr(C, "DISPLAY_SCALE", 1.0))

    exclude: Set[int] = set(int(e) for e in (exclude_ids or []))
    t0 = time.time()
    frames = 0

    per_class_conf: Dict[int, List[float]] = {}
    per_class_cx: Dict[int, List[int]] = {}
    per_class_cy: Dict[int, List[int]] = {}

    window_ms = int(getattr(C, "SCAN_DURATION_MS", 800))
    hud_text = f"SCANNING ~{window_ms} ms"
    duration_s = float(window_ms) / 1000.0

    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        h, w = frame.shape[:2]

        # Run YOLO and get an annotated image to display
        results = model(frame, verbose = C.YOLO_VERBOSE)

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

        # Add a simple HUD ribbon and publish to the main thread
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
