# vizzy/rpi/dispatch.py
from __future__ import annotations
from typing import Tuple, Optional
from ..shared import protocol as P
from . import state
from .servo import move_servos, goto_pwms

def process_messages(pi, msgs, debug: bool
                     ) -> Tuple[bool, Optional[int], Optional[bool], Optional[dict], Optional[dict]]:
    """
    Unified message handler.
    Returns: (stop_requested, center_done_cls, center_success, center_diag, yolo_results)
    """
    stop = False
    center_done_cls: Optional[int] = None
    center_success: Optional[bool] = None
    center_diag: Optional[dict] = None
    yolo_results: Optional[dict] = None

    for msg in msgs:
        mtype = msg.get("type")
        mcmd  = msg.get("cmd")

        if mtype == P.TYPE_MOVE:
            if state.centering_active.is_set() or not state.search_active.is_set():
                move_servos(pi, msg.get("horizontal", 0.0), msg.get("vertical", 0.0))

        elif mtype == P.TYPE_SEARCH:
            if msg.get("active", False):
                if not state.search_active.is_set():
                    print("[RPi] Search ON")
                state.search_active.set()
            else:
                if state.search_active.is_set():
                    print("[RPi] Search OFF")
                state.search_active.clear()

        elif mtype == P.TYPE_STOP:
            state.search_active.clear()
            stop = True

        elif mtype == P.TYPE_CENTER_DONE:
            try:
                center_done_cls = int(msg.get("target_cls", -1))
            except Exception:
                center_done_cls = -1
            center_success = bool(msg.get("success", False))
            center_diag = msg.get("diag", None)

        elif mtype == P.TYPE_YOLO_RESULTS:
            yolo_results = msg

        elif mcmd == P.CMD_GOTO_PWMS:
            if state.search_active.is_set() or state.centering_active.is_set():
                if debug:
                    print("[RPi] Ignoring GOTO_PWMS during search/centering")
                continue
            tb = int(msg.get("pwm_btm", 1500))
            tt = int(msg.get("pwm_top", 1500))
            slew = int(msg.get("slew_ms", 600))
            if debug:
                print(f"[RPi] GOTO_PWMS btm={tb} top={tt} slew_ms={slew}")
            goto_pwms(pi, tb, tt, duration_ms=slew, steps=24)

    return stop, center_done_cls, center_success, center_diag, yolo_results
