# vizzy/laptop/centering.py
from __future__ import annotations
import time, cv2
from typing import Callable, Optional, Dict, Any
from .hud import draw_wrapped_text
from .yolo_runner import infer_all, clear_class_filter

def center_on_class(cap, model,
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
    Closed-loop centering with quota-based verification.
    thresholds keys: conf, move_norm_eps, required_frames
    Returns (success, diag?)  diag included only if debug=True
    """
    CONF = float(thresholds["conf"])
    MOVE_EPS = float(thresholds["move_norm_eps"])
    REQUIRED = int(thresholds["required_frames"])

    t0 = time.time()
    good_frames = 0
    success = False

    # Diagnostics
    total_frames = 0
    max_conf_seen = 0.0
    min_err_px_seen = float('inf')
    min_move_norm_seen = float('inf')

    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok:
            break

        results = infer_all(model, frame, [int(target_cls)])
        for result in results:
            total_frames += 1
            annotated = result.plot()

            best = None
            best_conf = 0.0
            max_area = 0
            if len(result.boxes) > 0:
                for box in result.boxes:
                    if int(box.cls) != int(target_cls):
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best = ((x1 + x2) // 2, (y1 + y2) // 2)
                        best_conf = float(box.conf[0])

            if best is not None:
                dx = (center_x - best[0])
                dy = (best[1] - center_y)
                err_px = max(abs(dx), abs(dy))
                ndx = dx / center_x
                ndy = dy / center_y
                move_norm = max(abs(ndx), abs(ndy))

                conf_ok = (best_conf >= CONF)
                err_ok  = (abs(dx) <= epsilon_px and abs(dy) <= epsilon_px)
                move_ok = (abs(ndx) < MOVE_EPS and abs(ndy) < MOVE_EPS)

                # Drive servos if not stable
                if not (err_ok and move_ok):
                    send_move(ndx, ndy)

                if conf_ok and err_ok and move_ok:
                    good_frames += 1
                    if good_frames >= REQUIRED:
                        success = True

                # HUD with live numbers
                hud = (f"{label}  quota {good_frames}/{REQUIRED}  "
                       f"conf {best_conf:.2f}  err {err_px:.0f}px  move {move_norm:.3f}")
            else:
                hud = f"{label}  quota {good_frames}/{REQUIRED}"

            if debug:
                max_conf_seen = max(max_conf_seen, best_conf if best is not None else 0.0)
                if best is not None:
                    min_err_px_seen = min(min_err_px_seen, err_px)
                    min_move_norm_seen = min(min_move_norm_seen, move_norm)

            max_w = int(annotated.shape[1] * 0.92)
            draw_wrapped_text(annotated, hud, 10, 24, max_w)
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * display_scale), int(h * display_scale)))
            cv2.imshow("YOLO Detection", resized)
            cv2.waitKey(1)

    clear_class_filter(model)

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
