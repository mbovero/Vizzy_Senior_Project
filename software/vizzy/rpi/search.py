# vizzy/rpi/search.py
# -----------------------------------------------------------------------------
# Purpose
#   This module contains the **search and centering routines** for the
#   Raspberry Pi side of the Vizzy robotic arm project.
#
# Why this exists
#   - Implements a "search mode" where the arm sweeps its camera across a grid
#     of positions to find and center on detected objects.
#   - Handles requesting YOLO scans from the laptop, deciding which object to
#     target, and issuing commands to precisely center the arm on that object.
#
# How it fits into the project
#   - When the laptop sends a "start search" command, the Pi enters this loop.
#   - The Pi moves the arm to discrete positions, asking the laptop for object
#     detection results at each position.
#   - If a suitable object is found (confidence above a threshold and not yet
#     centered on in this session), the Pi commands the laptop to perform
#     centering.
#   - Once centering is verified, the Pi saves the PWM servo positions so the
#     object can be revisited later without re-searching.
#
# Key points for understanding:
#   - "Scan" = ask the laptop to run YOLO for a short period and return a list
#     of detected objects and their average confidences.
#   - "Center" = fine-tune servo angles until the chosen object is centered in
#     the camera frame, based on live feedback from the laptop.
#   - The Pi uses global state flags (from state.py) to know if search or
#     centering is currently active.
#   - This module directly drives the servos via `pigpio` calls and by using
#     helper functions from servo.py.
# -----------------------------------------------------------------------------

from __future__ import annotations
import time, select
from typing import Optional, Tuple, List, Dict

from .dispatch import process_messages
from ..shared.jsonl import send_json, recv_lines
from ..shared import protocol as P
from . import state
from .servo import goto_pwms
from .config import (
    SERVO_BTM, SERVO_TOP,
    SERVO_MIN, SERVO_MAX,
    SEARCH_MIN_OFFSET, SEARCH_MAX_OFFSET,
    SEARCH_H_STEP, SEARCH_V_STEP,
    POSE_SETTLE_S, SCAN_DURATION_MS, CENTER_DURATION_MS, CENTER_EPSILON_PX,
    MAX_CENTERS_PER_POSE, CONF_THRESHOLD
)

# Define absolute servo limits for scanning (slightly inside mechanical extremes)
SEARCH_MIN = SERVO_MIN + SEARCH_MIN_OFFSET
SEARCH_MAX = SERVO_MAX - SEARCH_MAX_OFFSET

# -----------------------------------------------------------------------------
# Object selection logic
# -----------------------------------------------------------------------------
def process_scan_results(objects: List[dict], excluded: set, conf_threshold: float
                         ) -> Optional[Tuple[int, str]]:
    """
    Given a list of detected objects from the laptop's YOLO scan, pick the best one.

    Args:
        objects: List of dictionaries, each describing an object (cls_id, cls_name, avg_conf, etc.)
        excluded: Set of class IDs already centered on during this search session.
        conf_threshold: Minimum confidence required to consider an object.

    Returns:
        (class_id, class_name) tuple for the chosen target, or None if nothing qualifies.
    """
    # Sort detections from highest to lowest average confidence
    sorted_objs = sorted(objects, key=lambda o: float(o.get("avg_conf", 0.0)), reverse=True)

    for o in sorted_objs:
        cid = int(o.get("cls_id", -1))
        if cid < 0:
            continue  # skip invalid IDs
        if cid in excluded:
            continue  # skip if already centered on this class in this session
        if float(o.get("avg_conf", 0.0)) < conf_threshold:
            continue  # skip if confidence too low
        name = str(o.get("cls_name", str(cid)))
        return cid, name

    return None  # no valid targets

# -----------------------------------------------------------------------------
# Scan request / wait for results
# -----------------------------------------------------------------------------
def request_scan_and_wait(pi, conn, centered_this_session: set, debug: bool) -> Optional[dict]:
    """
    Ask the laptop to perform a YOLO scan, then wait for the results.

    Sends CMD_YOLO_SCAN with:
      - duration_ms: how long to run YOLO inference
      - exclude_cls: list of class IDs to ignore

    Waits until:
      - A YOLO_RESULTS message arrives, or
      - search_active is cleared, or
      - a timeout is reached.
    """
    # Request scan from laptop
    send_json(conn, {
        "cmd": P.CMD_YOLO_SCAN,
        "duration_ms": SCAN_DURATION_MS,
        "exclude_cls": sorted(centered_this_session),
    })

    buf = b""
    t0 = time.time()
    while True:
        # Wait up to 20 ms for a message from laptop
        readable, _, _ = select.select([conn], [], [], 0.02)
        if readable:
            msgs, buf, closed = recv_lines(conn, buf)
            if closed:
                print("[RPi] Connection closed while waiting for scan")
                return None
            # Pass messages to dispatch, capture YOLO results if present
            stop, _, _, _, yolo = process_messages(pi, msgs, debug=debug)
            if stop:
                return None
            if yolo is not None:
                return yolo

        # Bail if search mode ended externally
        if not state.search_active.is_set():
            return None
        # Timeout after ~8 seconds
        if time.time() - t0 > 8.0:
            print("[RPi] Scan wait timeout; continuing")
            return None

# -----------------------------------------------------------------------------
# Centering routine
# -----------------------------------------------------------------------------
def center_on_class_and_return(pi, conn, target_cls: int, target_name: str,
                               saved_h: int, saved_v: int, debug: bool):
    """
    Command the laptop to center on the given object, then return to the saved pose.

    Args:
        target_cls: Class ID of target object.
        target_name: Human-readable name.
        saved_h, saved_v: Servo PWM positions to return to after centering.
        debug: Whether to print extra diagnostic info.

    Returns:
        (success, diag) where:
          - success: True if centering verification passed.
          - diag: Optional diagnostic dictionary from the laptop.
    """
    print(f"[RPi] Centering on {target_name} (id {target_cls})...")
    state.centering_active.set()

    # Send center-on command to laptop
    send_json(conn, {
        "cmd": P.CMD_CENTER_ON,
        "target_cls": int(target_cls),
        "target_name": target_name,
        "duration_ms": CENTER_DURATION_MS,
        "epsilon_px": CENTER_EPSILON_PX
    })

    buf = b""
    t0 = time.time()
    success = False
    diag = None

    while True:
        readable, _, _ = select.select([conn], [], [], 0.02)
        if readable:
            msgs, buf, closed = recv_lines(conn, buf)
            if closed:
                print("[RPi] Connection closed during centering")
                break

            # Process all messages so servo move commands get executed
            stop, done_cls, done_ok, done_diag, _ = process_messages(pi, msgs, debug=debug)
            if stop:
                state.search_active.clear()
                break
            if done_cls is not None and int(done_cls) == int(target_cls):
                # Got final verification result for our target
                success = bool(done_ok)
                diag = done_diag
                break

        if not state.search_active.is_set():
            break
        if time.time() - t0 > 12.0:
            print("[RPi] Centering timeout; continuing")
            break

    state.centering_active.clear()

    # If successful, tell laptop to save this pose to memory
    if success:
        try:
            send_json(conn, {
                "type": P.TYPE_CENTER_SNAPSHOT,
                "cls_id": int(target_cls),
                "cls_name": target_name,
                "pwm_btm": int(state.current_horizontal),
                "pwm_top": int(state.current_vertical),
                "ts": time.time(),
                "diag": diag
            })
        except Exception:
            pass

    # Return to original scanning position
    pi.set_servo_pulsewidth(SERVO_BTM, saved_h)
    pi.set_servo_pulsewidth(SERVO_TOP, saved_v)
    state.current_horizontal, state.current_vertical = saved_h, saved_v
    time.sleep(POSE_SETTLE_S * 0.8)

    return success, diag

# -----------------------------------------------------------------------------
# Main search loop
# -----------------------------------------------------------------------------
def run_search_cycle(pi, conn, debug: bool) -> None:
    """
    Perform a full search cycle:
      - Sweep the arm over a grid of servo positions.
      - At each position, request a YOLO scan from the laptop.
      - If a valid target is found, attempt to center on it.
      - Save successful centerings to session memory so they aren't repeated.
    """
    print("[RPi] Search started")
    conn.setblocking(False)  # non-blocking socket reads

    centered_this_session: set[int] = set()  # track objects we've already centered on

    # Helper functions for scan direction
    def horiz_range(fwd: bool):
        return range(SEARCH_MIN, SEARCH_MAX + 1, SEARCH_H_STEP) if fwd else range(SEARCH_MAX, SEARCH_MIN - 1, -SEARCH_H_STEP)
    def vert_range(fwd: bool):
        return range(SEARCH_MIN, SEARCH_MAX + 1, SEARCH_V_STEP) if fwd else range(SEARCH_MAX, SEARCH_MIN - 1, -SEARCH_V_STEP)

    fwd = True  # sweep direction flag
    while state.search_active.is_set():
        for h in horiz_range(fwd):
            if not state.search_active.is_set():
                break
            pi.set_servo_pulsewidth(SERVO_BTM, h)
            state.current_horizontal = h

            for v in vert_range(fwd):
                if not state.search_active.is_set():
                    break
                pi.set_servo_pulsewidth(SERVO_TOP, v)
                state.current_vertical = v
                time.sleep(POSE_SETTLE_S)  # allow arm to stabilize

                centers_done_at_this_pose = 0
                while state.search_active.is_set() and centers_done_at_this_pose < MAX_CENTERS_PER_POSE:
                    scan = request_scan_and_wait(pi, conn, centered_this_session, debug)
                    if scan is None:
                        break

                    objects = list(scan.get("objects", []))
                    target = process_scan_results(objects, centered_this_session, CONF_THRESHOLD)
                    if target is None:
                        break

                    saved_h, saved_v = state.current_horizontal, state.current_vertical
                    target_cls, target_name = target
                    ok, diag = center_on_class_and_return(pi, conn, target_cls, target_name, saved_h, saved_v, debug)
                    if ok:
                        centered_this_session.add(target_cls)
                        centers_done_at_this_pose += 1
                        print(f"[RPi] ✅ Verified centered on {target_name} (id {target_cls}).")
                    else:
                        if debug:
                            print(f"[RPi] ❌ Centering verification FAILED for {target_name} (id {target_cls}).")
                            if isinstance(diag, dict):
                                thr = diag.get("thresholds", {})
                                obs = diag.get("observed", {})
                                if "required_good_frames" in thr:
                                    print(f"      Thresholds: conf>={thr.get('conf_per_frame')}, "
                                          f"pixel<= {thr.get('pixel_epsilon')} px, "
                                          f"move_norm< {thr.get('move_norm_eps')}, "
                                          f"required_good_frames={thr.get('required_good_frames')}")
                                print(f"      Observed: total_frames={obs.get('total_frames')}, "
                                      f"good_frames={obs.get('good_frames')}, "
                                      f"max_conf_seen={obs.get('max_conf_seen')}, "
                                      f"min_err_px_seen={obs.get('min_err_px_seen')}, "
                                      f"min_move_norm_seen={obs.get('min_move_norm_seen')}")
                # Let control messages (like stop) be processed between poses
                _poll_commands_nonblocking(pi, conn)

            if not state.search_active.is_set():
                break
        fwd = not fwd  # reverse sweep direction

    print("[RPi] Search stopped")

# -----------------------------------------------------------------------------
# Non-blocking control message poll
# -----------------------------------------------------------------------------
def _poll_commands_nonblocking(pi, conn) -> None:
    """
    Quickly check for any control messages from laptop without blocking.
    This ensures the Pi can respond to commands (like stop) even in between
    scanning poses.
    """
    buf = b""
    readable, _, _ = select.select([conn], [], [], 0.0)
    if not readable:
        return
    msgs, buf, closed = recv_lines(conn, buf)
    if closed:
        print("[RPi] Connection closed by laptop")
        state.search_active.clear()
        return
    from .server import process_messages
    process_messages(pi, msgs, debug=False)  # we ignore return values here
