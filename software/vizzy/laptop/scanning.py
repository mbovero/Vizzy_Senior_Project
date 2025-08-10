# vizzy/laptop/scanning.py
from __future__ import annotations
import time, statistics, cv2
from typing import Dict, List, Tuple, Iterable, Optional, Set
from .hud import draw_wrapped_text
from .yolo_runner import infer_all

def run_scan_window(cap, model, duration_s: float, class_filter: int,
                    exclude_ids: Optional[Iterable[int]],
                    display_scale: float,
                    get_name, min_frames_for_class: int = 4) -> dict:
    """
    Run YOLO for ~duration_s. Aggregate per-class avg_conf and median center (largest box per frame).
    Returns {"frames":int, "objects":[...sorted by avg_conf desc...]}.
    """
    exclude: Set[int] = set(int(x) for x in (exclude_ids or []))
    t0 = time.time()
    frames = 0
    per_class_conf: Dict[int, List[float]] = {}
    per_class_cx:   Dict[int, List[int]]   = {}
    per_class_cy:   Dict[int, List[int]]   = {}

    hud_text = f"SCANNING ~{int(duration_s*1000)} ms"

    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok:
            break

        results = infer_all(model, frame, None if class_filter == -1 else [class_filter])
        for result in results:
            frames += 1

            # Accumulate largest box per frame per class (respecting exclude)
            if len(result.boxes) > 0:
                largest = None
                max_area = 0
                best_cls = None
                best_conf = None
                for box in result.boxes:
                    cid = int(box.cls)
                    if class_filter != -1 and cid != class_filter:
                        continue
                    if cid in exclude:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest = ((x1 + x2) // 2, (y1 + y2) // 2)
                        best_cls = cid
                        best_conf = float(box.conf[0])

                if largest is not None and best_cls is not None:
                    per_class_conf.setdefault(best_cls, []).append(best_conf)
                    per_class_cx.setdefault(best_cls, []).append(largest[0])
                    per_class_cy.setdefault(best_cls, []).append(largest[1])

            annotated = result.plot()
            draw_wrapped_text(annotated, hud_text, 10, 24, int(annotated.shape[1] * 0.92))
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * display_scale), int(h * display_scale)))
            cv2.imshow("YOLO Detection", resized)
            cv2.waitKey(1)

    objects = []
    for cls_id, confs in per_class_conf.items():
        if len(confs) < min_frames_for_class:
            continue
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
    objects.sort(key=lambda r: r["avg_conf"], reverse=True)
    return {"frames": int(frames), "objects": objects}
