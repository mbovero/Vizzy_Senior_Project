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
    Build the serpentine search path as a list of Cartesian poses:
      [{"pose_id": int, "x": float, "y": float, "z": float, "pitch": float}, ...]

    Traversal mirrors the previous PWM grid:
      - Inclusive ranges between configured min/max values (step <= 0 -> treated as 1)
      - Serpentine traversal across X for each Y row to minimise long reversals
      - Iterate Z layers outermost, reusing the serpentine plan per layer
    """

    def inclusive_range(lo: float, hi: float, step: float) -> list[float]:
        """Return inclusive range from lo to hi with the provided step."""
        if step == 0:
            step = 1.0
        step = float(abs(step))
        values: list[float] = []
        if lo <= hi:
            x = float(lo)
            while x <= hi + 1e-6:
                values.append(round(x, 3))
                x += step
        else:
            x = float(lo)
            while x >= hi - 1e-6:
                values.append(round(x, 3))
                x -= step
        return values

    xs = inclusive_range(C.SEARCH_X_MIN_MM, C.SEARCH_X_MAX_MM, C.SEARCH_X_STEP_MM)
    ys = inclusive_range(C.SEARCH_Y_MIN_MM, C.SEARCH_Y_MAX_MM, C.SEARCH_Y_STEP_MM)
    zs = inclusive_range(C.SEARCH_Z_MIN_MM, C.SEARCH_Z_MAX_MM, C.SEARCH_Z_STEP_MM)

    pitch = float(C.SEARCH_PITCH_DEG)

    path: list[dict] = []
    pose_id = 0
    for z in zs:
        reverse = False
        for y in ys:
            row = list(reversed(xs)) if reverse else xs
            for x in row:
                path.append({
                    "pose_id": pose_id,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "pitch": pitch,
                })
                pose_id += 1
            reverse = not reverse
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

        # Run YOLO and get an annotated image to display
        results = model(frame, classes=allowed_ids, verbose = C.YOLO_VERBOSE)

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
