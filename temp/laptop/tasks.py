# vizzy/laptop/tasks.py
from __future__ import annotations

from typing import Any

import cv2
from ultralytics import YOLO

from ..shared import config as C

from .gpt_client import Plan, HighLevelCmd
from .motion import Motion
from .memory import ObjectMemory
from .scanning import run_scan_window
from .centering import center_on_class


def execute_plan(
    *,
    plan: Plan,
    motion: Motion,
    memory: ObjectMemory,
    model: YOLO,
    camera: cv2.VideoCapture,
    config: Any,
    frame_sink,  # FrameBus.publish (main-thread renderer)
) -> None:
    if not plan.actions:
        print("[Tasks] (noop) No actions in plan.")
        return

    for i, action in enumerate(plan.actions, start=1):
        kind = action.kind
        p = action.params
        print(f"[Tasks] Action {i}/{len(plan.actions)}: {kind} {p}")

        if kind == "move_to_pose":
            _do_move_to_pose(p, motion)

        elif kind == "center_on":
            _do_center_on(p, motion, model, camera, frame_sink)

        elif kind == "look_for":
            _do_look_for(p, model, camera, frame_sink)

        elif kind in ("track", "pick", "place"):
            print(f"[Tasks] '{kind}' is not implemented yet.")
        else:
            print(f"[Tasks] Unknown action kind: {kind}")


# --------------------------------------------------------------------------- #
# Action handlers
# --------------------------------------------------------------------------- #

def _do_move_to_pose(p: dict, motion: Motion) -> None:
    try:
        pwm_btm = int(p["pwm_btm"])
        pwm_top  = int(p["pwm_top"])
        slew_ms  = int(p.get("slew_ms", 600))
    except Exception as e:
        print(f"[Tasks] move_to_pose: invalid params: {e}")
        return

    motion.goto_pose_pwm(pwm_btm, pwm_top, slew_ms=slew_ms)
    print(f"[Tasks] Moved to PWM pose btm={pwm_btm}, top={pwm_top}, slew={slew_ms}ms")


def _do_center_on(p: dict, motion: Motion, model: YOLO, camera: cv2.VideoCapture, frame_sink) -> None:
    """Focused centering pass without initiating a full SEARCH sweep."""
    cls_name = str(p.get("class_name", "")).strip()
    if not cls_name:
        print("[Tasks] center_on: missing class_name")
        return

    # Resolve class name -> id (fall back to numeric)
    names = model.names
    target_cls = None
    try:
        if isinstance(names, dict):
            for cid, nm in names.items():
                if str(nm).lower() == cls_name.lower():
                    target_cls = int(cid); break
        else:
            for cid, nm in enumerate(names):
                if str(nm).lower() == cls_name.lower():
                    target_cls = int(cid); break
    except Exception:
        pass
    if target_cls is None:
        try:
            target_cls = int(cls_name)
        except Exception:
            print(f"[Tasks] center_on: could not resolve class '{cls_name}'")
            return

    ok, frame0 = camera.read()
    if not ok:
        print("[Tasks] center_on: camera read failed")
        return
    h0, w0 = frame0.shape[:2]
    cx, cy = w0 // 2, h0 // 2

    print(f"[Tasks] Centering on class id={target_cls} ({cls_name})")
    success = center_on_class(
        cap=camera,
        model=model,
        target_cls=int(target_cls),
        center_x=cx,
        center_y=cy,
        send_move=motion.nudge_scan,
        display_scale=getattr(C, "DISPLAY_SCALE", 1.0),
        label=f"CENTER (task) {cls_name}",
        frame_sink=frame_sink,
    )
    print(f"[Tasks] center_on result: {'SUCCESS' if success else 'FAIL'}")


def _do_look_for(p: dict, model: YOLO, camera: cv2.VideoCapture, frame_sink) -> None:
    """Lightweight scan window without moving the arm; shows annotated frames via frame_sink."""
    cls_name = str(p.get("class_name", "")).strip()
    names = model.names

    summary = run_scan_window(
        cap=camera,
        model=lambda img, *a, **kw: model(img, verbose=C.YOLO_VERBOSE), 
        exclude_ids=[],
        get_name=lambda cid: names[cid] if isinstance(names, (list, tuple)) else names.get(cid, str(cid)),
        min_frames_for_class=int(getattr(C, "SCAN_MIN_FRAMES", 5)),
        frame_sink=frame_sink,
        display_scale=getattr(C, "DISPLAY_SCALE", 1.0),
    )

    objs = summary.get("objects", [])
    if not objs:
        print("[Tasks] look_for: nothing detected in this window.")
        return

    if cls_name:
        matches = [o for o in objs if str(o.get("cls_name", "")).lower() == cls_name.lower()]
        if matches:
            best = max(matches, key=lambda o: float(o.get("avg_conf", 0.0)))
            print(f"[Tasks] look_for: saw '{cls_name}' conf={best.get('avg_conf'):.2f} frames={best.get('frames')}")
        else:
            print(f"[Tasks] look_for: '{cls_name}' not seen in this window.")
    else:
        best = max(objs, key=lambda o: float(o.get("avg_conf", 0.0)))
        print(f"[Tasks] look_for: best={best.get('cls_name')} conf={best.get('avg_conf'):.2f} frames={best.get('frames')}")
