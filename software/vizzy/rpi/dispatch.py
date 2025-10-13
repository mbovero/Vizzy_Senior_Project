# vizzy/rpi/dispatch.py
from __future__ import annotations

import time
from typing import Iterable

from ..shared import protocol as P
from ..shared import config as C
from ..shared.jsonl import send_json

from . import state
from .servo import move_servos, goto_pwms


# ============================================================================
# Message Handler Functions
# ============================================================================

def handle_scan_move(pi, msg: dict, debug: bool) -> None:
    """Handle TYPE_SCAN_MOVE: Accept nudges only when centering is active."""
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


def handle_stop(debug: bool) -> bool:
    """Handle TYPE_STOP: Signal server to stop."""
    if debug:
        print("[RPi] STOP received")
    return True


def handle_goto_pwms(pi, conn, msg: dict, debug: bool) -> None:
    """Handle CMD_GOTO_PWMS: Move to absolute PWM position."""
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


def handle_get_pwms(conn, debug: bool) -> None:
    """Handle CMD_GET_PWMS: Return current PWM positions."""
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


def handle_move_to(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_MOVE_TO: STUB - Print and confirm (no actual movement)."""
    x = msg.get("x", 0)
    y = msg.get("y", 0)
    z = msg.get("z", 0)
    print(f"[RPi] RECEIVED: MOVE_TO command")
    print(f"[RPi]   Target position: x={x}, y={y}, z={z} mm")
    print(f"[RPi]   (Stub: Not executing - would calculate IK and move servos)")
    
    try:
        send_json(conn, {
            "type": P.TYPE_CMD_COMPLETE,
            "cmd": P.CMD_MOVE_TO,
            "status": "success"
        })
        print(f"[RPi] CONFIRMED: MOVE_TO completed")
    except Exception as e:
        print(f"[RPi] Failed to send confirmation: {e}")


def handle_grab(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_GRAB: STUB - Print and confirm (no actual gripper control)."""
    print(f"[RPi] RECEIVED: GRAB command")
    print(f"[RPi]   (Stub: Not executing - would close gripper)")
    
    try:
        send_json(conn, {
            "type": P.TYPE_CMD_COMPLETE,
            "cmd": P.CMD_GRAB,
            "status": "success"
        })
        print(f"[RPi] CONFIRMED: GRAB completed")
    except Exception as e:
        print(f"[RPi] Failed to send confirmation: {e}")


def handle_release(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_RELEASE: STUB - Print and confirm (no actual gripper control)."""
    print(f"[RPi] RECEIVED: RELEASE command")
    print(f"[RPi]   (Stub: Not executing - would open gripper)")
    
    try:
        send_json(conn, {
            "type": P.TYPE_CMD_COMPLETE,
            "cmd": P.CMD_RELEASE,
            "status": "success"
        })
        print(f"[RPi] CONFIRMED: RELEASE completed")
    except Exception as e:
        print(f"[RPi] Failed to send confirmation: {e}")


def handle_rot_yaw(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_ROT_YAW: STUB - Print and confirm (no actual servo control)."""
    angle = msg.get("angle", 0.0)
    print(f"[RPi] RECEIVED: ROT_YAW command")
    print(f"[RPi]   Target angle: {angle}°")
    print(f"[RPi]   (Stub: Not executing - would rotate yaw servo)")
    
    try:
        send_json(conn, {
            "type": P.TYPE_CMD_COMPLETE,
            "cmd": P.CMD_ROT_YAW,
            "status": "success"
        })
        print(f"[RPi] CONFIRMED: ROT_YAW completed")
    except Exception as e:
        print(f"[RPi] Failed to send confirmation: {e}")


def handle_rot_pitch(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_ROT_PITCH: STUB - Print and confirm (no actual servo control)."""
    angle = msg.get("angle", 0.0)
    print(f"[RPi] RECEIVED: ROT_PITCH command")
    print(f"[RPi]   Target angle: {angle}°")
    print(f"[RPi]   (Stub: Not executing - would rotate pitch servo)")
    
    try:
        send_json(conn, {
            "type": P.TYPE_CMD_COMPLETE,
            "cmd": P.CMD_ROT_PITCH,
            "status": "success"
        })
        print(f"[RPi] CONFIRMED: ROT_PITCH completed")
    except Exception as e:
        print(f"[RPi] Failed to send confirmation: {e}")


# ============================================================================
# Main Message Dispatcher
# ============================================================================

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
                handle_scan_move(pi, msg, debug)
            elif mtype == P.TYPE_STOP:
                stop = handle_stop(debug)
            else:
                if debug:
                    print(f"[RPi] Ignored message type: {mtype}")

        # ---------------------- CMD-* messages -----------------------
        elif mcmd:
            if mcmd == P.CMD_GOTO_PWMS:
                handle_goto_pwms(pi, conn, msg, debug)
            elif mcmd == P.CMD_GET_PWMS:
                handle_get_pwms(conn, debug)
            elif mcmd == P.CMD_MOVE_TO:
                handle_move_to(conn, msg, debug)
            elif mcmd == P.CMD_GRAB:
                handle_grab(conn, msg, debug)
            elif mcmd == P.CMD_RELEASE:
                handle_release(conn, msg, debug)
            elif mcmd == P.CMD_ROT_YAW:
                handle_rot_yaw(conn, msg, debug)
            elif mcmd == P.CMD_ROT_PITCH:
                handle_rot_pitch(conn, msg, debug)
            else:
                if debug:
                    print(f"[RPi] Ignored command: {mcmd}")

        else:
            if debug:
                print(f"[RPi] Unknown message payload (no type/cmd): {msg}")

    return stop
