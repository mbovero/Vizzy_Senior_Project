# vizzy/rpi/dispatch.py
# -----------------------------------------------------------------------------
# Central message dispatcher for Raspberry Pi side (new laptop-centric protocol)
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Iterable, Optional
import threading

from ..shared import protocol as P
from ..shared import config as C
from ..shared.jsonl import send_json

from . import state
from .servo import move_servos, goto_pwms


# Internal latches used by the search FSM to know when a pose is done.
_last_pose_done_id: Optional[int] = None
_last_pose_done_status: Optional[str] = None
_pose_done_lock = threading.Lock()


def _set_pose_done(pid: int, status: str) -> None:
    global _last_pose_done_id, _last_pose_done_status
    with _pose_done_lock:
        _last_pose_done_id = int(pid)
        _last_pose_done_status = str(status).upper()


def take_pose_done() -> tuple[Optional[int], Optional[str]]:
    """Called by search loop to fetch (and clear) the most recent POSE_DONE."""
    global _last_pose_done_id, _last_pose_done_status
    with _pose_done_lock:
        pid, st = _last_pose_done_id, _last_pose_done_status
        _last_pose_done_id = None
        _last_pose_done_status = None
        return pid, st


def process_messages(pi, conn, messages: Iterable[dict], *, debug: bool = False) -> bool:
    """
    Dispatch a batch of messages from the laptop.

    Returns:
        stop (bool): True if the server should stop and exit.
    """
    stop = False

    for msg in messages:
        mtype = msg.get("type")
        mcmd  = msg.get("cmd")

        # ---------------------- TYPE-* messages ----------------------
        if mtype:
            if mtype == P.TYPE_SCAN_MOVE:
                # Only accept during per-pose centering window
                if state.centering_active.is_set():
                    dx = float(msg.get("horizontal", 0.0))
                    dy = float(msg.get("vertical", 0.0))
                    # Clamp to [-1, 1]
                    dx = -1.0 if dx < -1.0 else (1.0 if dx > 1.0 else dx)
                    dy = -1.0 if dy < -1.0 else (1.0 if dy > 1.0 else dy)
                    move_servos(pi, dx, dy, scale_us=C.MOVE_SCALE_US)
                    if debug:
                        print(f"[RPi] SCAN_MOVE dx={dx:.3f} dy={dy:.3f}")
                else:
                    if debug:
                        print("[RPi] SCAN_MOVE ignored (not in centering window)")

            elif mtype == P.TYPE_SEARCH:
                active = bool(msg.get("active", False))
                if active:
                    state.search_active.set()
                    if debug:
                        print("[RPi] SEARCH: ON")
                else:
                    # Immediate stop: clear search + centering; sweep worker will exit
                    state.search_active.clear()
                    state.centering_active.clear()
                    if debug:
                        print("[RPi] SEARCH: OFF (interrupt)")

            elif mtype == P.TYPE_STOP:
                if debug:
                    print("[RPi] STOP received")
                stop = True

            elif mtype == P.TYPE_POSE_DONE:
                pid = int(msg.get("pose_id", -1))
                status = str(msg.get("status", "SKIP")).upper()
                _set_pose_done(pid, status)
                state.centering_active.clear()  # leave centering window; search loop advances
                if debug:
                    print(f"[RPi] POSE_DONE pose_id={pid} status={status}")

            # TYPE_PWMS is only sent by RPi; ignore if received
            else:
                if debug:
                    print(f"[RPi] Ignored message type: {mtype}")

        # ---------------------- CMD-* messages -----------------------
        elif mcmd:
            if mcmd == P.CMD_GOTO_PWMS:
                # Ignore if busy with search or centering
                if state.search_active.is_set() or state.centering_active.is_set():
                    if debug:
                        print("[RPi] Ignoring GOTO_PWMS during search/centering")
                    continue
                tb = int(msg.get("pwm_btm", C.SERVO_CENTER))
                tt = int(msg.get("pwm_top", C.SERVO_CENTER))
                slew = int(msg.get("slew_ms", 600))
                if debug:
                    print(f"[RPi] GOTO_PWMS btm={tb} top={tt} slew_ms={slew}")
                goto_pwms(pi, tb, tt, duration_ms=slew, steps=24)

            elif mcmd == P.CMD_GET_PWMS:
                payload = {
                    "type": P.TYPE_PWMS,
                    "pwm_btm": int(state.current_horizontal),
                    "pwm_top": int(state.current_vertical),
                }
                try:
                    send_json(conn, payload)
                    if debug:
                        print(f"[RPi] -> PWMS {payload}")
                except Exception as e:
                    if debug:
                        print(f"[RPi] Failed to send PWMS: {e}")

            else:
                if debug:
                    print(f"[RPi] Ignored command: {mcmd}")

        else:
            if debug:
                print(f"[RPi] Unknown message payload (no type/cmd): {msg}")

    return stop
