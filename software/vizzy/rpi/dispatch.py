# vizzy/rpi/dispatch.py
# -----------------------------------------------------------------------------
# Raspberry Pi message dispatcher for the Vizzy arm.
# Handles laptop-issued protocol commands, maintains the current Cartesian
# target (x, y, z, pitch), and emits acknowledgements. Physical motion will be
# handled by a future IK-backed implementation; for now we log intent.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Iterable

from ..shared import config as C
from ..shared import protocol as P
from ..shared.jsonl import send_json

from . import state


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _sorted_bounds(a: float, b: float) -> tuple[float, float]:
    return (a, b) if a <= b else (b, a)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp_workspace(x: float, y: float, z: float) -> tuple[float, float, float]:
    xmin, xmax = _sorted_bounds(C.SEARCH_X_MIN_MM, C.SEARCH_X_MAX_MM)
    ymin, ymax = _sorted_bounds(C.SEARCH_Y_MIN_MM, C.SEARCH_Y_MAX_MM)
    zmin, zmax = _sorted_bounds(C.SEARCH_Z_MIN_MM, C.SEARCH_Z_MAX_MM)
    return (
        _clamp(x, xmin, xmax),
        _clamp(y, ymin, ymax),
        _clamp(z, zmin, zmax),
    )


def _set_target(x: float, y: float, z: float, pitch: float) -> None:
    with state.target_lock:
        state.current_target.update({
            "x": x,
            "y": y,
            "z": z,
            "pitch": pitch,
        })


def _get_target() -> dict:
    with state.target_lock:
        return dict(state.current_target)


# -----------------------------------------------------------------------------
# Message Handler Functions
# -----------------------------------------------------------------------------

def handle_scan_move(_pi, msg: dict, debug: bool) -> None:
    """Handle TYPE_SCAN_MOVE: adjust stored target by small Cartesian offsets."""
    if not state.centering_active.is_set():
        if debug:
            print("[RPi] SCAN_MOVE ignored (movement in progress)")
        return

    dx = float(msg.get("horizontal", 0.0))
    dy = float(msg.get("vertical", 0.0))
    dx = -1.0 if dx < -1.0 else (1.0 if dx > 1.0 else dx)
    dy = -1.0 if dy < -1.0 else (1.0 if dy > 1.0 else dy)

    step = float(getattr(C, "SCAN_NUDGE_STEP_MM", 5.0))

    with state.target_lock:
        current = dict(state.current_target)
        new_x = current["x"] + dx * step
        new_y = current["y"] + dy * step
        new_z = current["z"]
        new_x, new_y, new_z = _clamp_workspace(new_x, new_y, new_z)
        state.current_target.update({
            "x": new_x,
            "y": new_y,
            "z": new_z,
        })

    print(f"[RPi] SCAN_MOVE Δx={dx:.3f} Δy={dy:.3f} -> target x={new_x:.1f} y={new_y:.1f} z={new_z:.1f}")


def handle_stop(debug: bool) -> bool:
    """Handle TYPE_STOP: Signal server to stop."""
    if debug:
        print("[RPi] STOP received")
    return True


def handle_move_to(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_MOVE_TO: update stored target and confirm completion."""
    target = _get_target()
    x = float(msg.get("x", target["x"]))
    y = float(msg.get("y", target["y"]))
    z = float(msg.get("z", target["z"]))
    pitch = float(msg.get("pitch", target["pitch"]))

    x, y, z = _clamp_workspace(x, y, z)
    _set_target(x, y, z, pitch)
    state.centering_active.set()

    print("[RPi] MOVE_TO received:")
    print(f"        target position: x={x:.1f} mm, y={y:.1f} mm, z={z:.1f} mm")
    print(f"        target pitch   : {pitch:.1f}°")
    print("        (Stub: IK execution pending)")

    payload = {
        "type": P.TYPE_CMD_COMPLETE,
        "cmd": P.CMD_MOVE_TO,
        "status": "success",
    }
    try:
        send_json(conn, payload)
        if debug:
            print(f"[RPi] -> CMD_COMPLETE {payload}")
    except Exception as exc:
        if debug:
            print(f"[RPi] Failed to send CMD_COMPLETE: {exc}")


def handle_get_obj_loc(conn, debug: bool) -> None:
    """Handle CMD_GET_OBJ_LOC: send current target location."""
    target = _get_target()
    payload = {
        "type": P.TYPE_OBJ_LOC,
        "x": float(target["x"]),
        "y": float(target["y"]),
        "z": float(target["z"]),
    }
    try:
        send_json(conn, payload)
        if debug:
            print(f"[RPi] -> OBJ_LOC {payload}")
    except Exception as exc:
        if debug:
            print(f"[RPi] Failed to send OBJ_LOC: {exc}")


def handle_grab(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_GRAB: placeholder that confirms receipt."""
    print("[RPi] GRAB received (stub)")
    payload = {"type": P.TYPE_CMD_COMPLETE, "cmd": P.CMD_GRAB, "status": "success"}
    try:
        send_json(conn, payload)
        if debug:
            print(f"[RPi] -> CMD_COMPLETE {payload}")
    except Exception as exc:
        if debug:
            print(f"[RPi] Failed to send CMD_COMPLETE for GRAB: {exc}")


def handle_release(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_RELEASE: placeholder that confirms receipt."""
    print("[RPi] RELEASE received (stub)")
    payload = {"type": P.TYPE_CMD_COMPLETE, "cmd": P.CMD_RELEASE, "status": "success"}
    try:
        send_json(conn, payload)
        if debug:
            print(f"[RPi] -> CMD_COMPLETE {payload}")
    except Exception as exc:
        if debug:
            print(f"[RPi] Failed to send CMD_COMPLETE for RELEASE: {exc}")


def handle_rot_yaw(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_ROT_YAW: placeholder that confirms receipt."""
    angle = float(msg.get("angle", 0.0))
    print(f"[RPi] ROT_YAW received (stub) -> target angle {angle:.1f}°")
    payload = {"type": P.TYPE_CMD_COMPLETE, "cmd": P.CMD_ROT_YAW, "status": "success"}
    try:
        send_json(conn, payload)
        if debug:
            print(f"[RPi] -> CMD_COMPLETE {payload}")
    except Exception as exc:
        if debug:
            print(f"[RPi] Failed to send CMD_COMPLETE for ROT_YAW: {exc}")


def handle_rot_pitch(conn, msg: dict, debug: bool) -> None:
    """Handle CMD_ROT_PITCH: placeholder that confirms receipt."""
    angle = float(msg.get("angle", 0.0))
    print(f"[RPi] ROT_PITCH received (stub) -> target angle {angle:.1f}°")
    payload = {"type": P.TYPE_CMD_COMPLETE, "cmd": P.CMD_ROT_PITCH, "status": "success"}
    try:
        send_json(conn, payload)
        if debug:
            print(f"[RPi] -> CMD_COMPLETE {payload}")
    except Exception as exc:
        if debug:
            print(f"[RPi] Failed to send CMD_COMPLETE for ROT_PITCH: {exc}")


# -----------------------------------------------------------------------------
# Main Message Dispatcher
# -----------------------------------------------------------------------------

def process_messages(pi, conn, messages: Iterable[dict], *, debug: bool = False) -> bool:
    """
    Dispatch a batch of messages from the laptop.
    Returns:
        stop (bool): True if the server should stop and exit.
    """
    _ = pi  # Reserved for future IK integration (pigpio instance)
    stop = False

    for msg in messages:
        mtype = msg.get("type")
        mcmd = msg.get("cmd")

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
            if mcmd == P.CMD_MOVE_TO:
                handle_move_to(conn, msg, debug)
            elif mcmd == P.CMD_GET_OBJ_LOC:
                handle_get_obj_loc(conn, debug)
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
