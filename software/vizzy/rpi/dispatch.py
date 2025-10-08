# vizzy/rpi/dispatch.py
# -----------------------------------------------------------------------------
# Central message dispatcher for Raspberry Pi side (laptop-driven sweep)
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Iterable
import time

from ..shared import protocol as P
from ..shared import config as C
from ..shared.jsonl import send_json

from . import state
from .servo import move_servos, goto_pwms


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
                # Accept nudges only when we're not slewing an absolute GOTO.
                if state.centering_active.is_set():
                    dx = float(msg.get("horizontal", 0.0))
                    dy = float(msg.get("vertical", 0.0))
                    # Clamp to [-1, 1]
                    dx = -1.0 if dx < -1.0 else (1.0 if dx > 1.0 else dx)
                    dy = -1.0 if dy < -1.0 else (1.0 if dy > 1.0 else dy)
                    move_servos(pi, dx, dy, scale_us=C.MOVE_SCALE_US)
                    if debug:
                        print(f"[RPi] SCAN_MOVE dx={dx:.3f} dy={dy:.3f}  "
                              f"btm={state.current_horizontal} top={state.current_vertical}")
                else:
                    if debug:
                        print("[RPi] SCAN_MOVE ignored (slewing/settling)")

            elif mtype == P.TYPE_STOP:
                if debug:
                    print("[RPi] STOP received")
                stop = True

            else:
                if debug:
                    print(f"[RPi] Ignored message type: {mtype}")

        # ---------------------- CMD-* messages -----------------------
        elif mcmd:
            if mcmd == P.CMD_GOTO_PWMS:
                tb   = int(msg.get("pwm_btm", C.SERVO_CENTER))
                tt   = int(msg.get("pwm_top", C.SERVO_CENTER))
                slew = int(msg.get("slew_ms", C.GOTO_POSE_SLEW_MS))
                pose_id = msg.get("pose_id", None)
                if pose_id is not None:
                    try:
                        pose_id = int(pose_id)
                    except Exception:
                        pose_id = None

                if debug:
                    extra = f" pose_id={pose_id}" if pose_id is not None else ""
                    print(f"[RPi] GOTO_PWMS btm={tb} top={tt} slew_ms={slew}{extra}")

                # Block nudges during absolute motion
                state.centering_active.clear()

                # Synchronous goto; steps from config
                goto_pwms(pi, tb, tt, duration_ms=slew, steps=C.GOTO_STEPS)

                # Small settle to reduce oscillation, then allow nudges again
                time.sleep(C.POSE_SETTLE_S)
                state.centering_active.set()

                # Ack move completion (echo pose_id if provided)
                payload = {"type": P.TYPE_POSE_READY}
                if pose_id is not None:
                    payload["pose_id"] = pose_id
                try:
                    send_json(conn, payload)
                    if debug:
                        print(f"[RPi] -> POSE_READY {payload}")
                except Exception as e:
                    if debug:
                        print(f"[RPi] Failed to send POSE_READY: {e}")

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
