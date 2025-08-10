# vizzy/rpi/search.py
from __future__ import annotations
import time, select
from .server import process_messages
from typing import Optional, Tuple, List, Dict
from ..shared.jsonl import send_json, recv_lines
from ..shared import protocol as P
from . import state
from .servo import goto_pwms
from .config import (
    SERVO_BTM, SERVO_TOP,
    SERVO_MIN, SERVO_MAX, SERVO_CENTER,
    SEARCH_MIN_OFFSET, SEARCH_MAX_OFFSET,
    SEARCH_H_STEP, SEARCH_V_STEP,
    POSE_SETTLE_S, SCAN_DURATION_MS, CENTER_DURATION_MS, CENTER_EPSILON_PX,
    MAX_CENTERS_PER_POSE, CONF_THRESHOLD
)

# Computed search bounds
SEARCH_MIN = SERVO_MIN + SEARCH_MIN_OFFSET
SEARCH_MAX = SERVO_MAX - SEARCH_MAX_OFFSET

def process_scan_results(objects: List[dict], excluded: set, conf_threshold: float
                         ) -> Optional[Tuple[int, str]]:
    """
    Choose the best target: highest avg_conf not excluded and above threshold.
    Returns (class_id, class_name) or None.
    """
    # Sort high to low confidence
    sorted_objs = sorted(objects, key=lambda o: float(o.get("avg_conf", 0.0)), reverse=True)
    for o in sorted_objs:
        cid = int(o.get("cls_id", -1))
        if cid < 0:
            continue
        if cid in excluded:
            continue
        if float(o.get("avg_conf", 0.0)) < conf_threshold:
            continue
        name = str(o.get("cls_name", str(cid)))
        return cid, name
    return None

def request_scan_and_wait(pi, conn, centered_this_session: set, debug: bool) -> Optional[dict]:
    send_json(conn, {
        "cmd": P.CMD_YOLO_SCAN,
        "duration_ms": SCAN_DURATION_MS,
        "exclude_cls": sorted(centered_this_session),
    })
    buf = b""
    t0 = time.time()
    while True:
        readable, _, _ = select.select([conn], [], [], 0.02)
        if readable:
            msgs, buf, closed = recv_lines(conn, buf)
            if closed:
                print("[RPi] Connection closed while waiting for scan")
                return None
            stop, _, _, _, yolo = process_messages(pi, msgs, debug=debug)
            if stop:
                return None
            if yolo is not None:
                return yolo

        if not state.search_active.is_set():
            return None
        if time.time() - t0 > 8.0:
            print("[RPi] Scan wait timeout; continuing")
            return None


def center_on_class_and_return(pi, conn, target_cls: int, target_name: str,
                               saved_h: int, saved_v: int, debug: bool):
    print(f"[RPi] Centering on {target_name} (id {target_cls})...")
    state.centering_active.set()

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

            # IMPORTANT: process all messages, so TYPE_MOVE actually drives servos
            stop, done_cls, done_ok, done_diag, _ = process_messages(pi, msgs, debug=debug)
            if stop:
                state.search_active.clear()
                break
            if done_cls is not None and int(done_cls) == int(target_cls):
                success = bool(done_ok)
                diag = done_diag
                break

        if not state.search_active.is_set():
            break
        if time.time() - t0 > 12.0:
            print("[RPi] Centering timeout; continuing")
            break

    state.centering_active.clear()

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

    # Return to saved scan pose
    pi.set_servo_pulsewidth(SERVO_BTM, saved_h)
    pi.set_servo_pulsewidth(SERVO_TOP, saved_v)
    state.current_horizontal, state.current_vertical = saved_h, saved_v
    time.sleep(POSE_SETTLE_S * 0.8)

    return success, diag

def run_search_cycle(pi, conn, debug: bool) -> None:
    """Top-level search loop. Sweeps grid, scans, centers, and resumes."""
    print("[RPi] Search started")
    conn.setblocking(False)

    centered_this_session: set[int] = set()

    def horiz_range(fwd: bool):
        return range(SEARCH_MIN, SEARCH_MAX + 1, SEARCH_H_STEP) if fwd else range(SEARCH_MAX, SEARCH_MIN - 1, -SEARCH_H_STEP)
    def vert_range(fwd: bool):
        return range(SEARCH_MIN, SEARCH_MAX + 1, SEARCH_V_STEP) if fwd else range(SEARCH_MAX, SEARCH_MIN - 1, -SEARCH_V_STEP)

    fwd = True
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
                time.sleep(POSE_SETTLE_S)

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
                        print(f"[RPi] ? Verified centered on {target_name} (id {target_cls}).")
                    else:
                        if debug:
                            print(f"[RPi] ? Centering verification FAILED for {target_name} (id {target_cls}).")
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
                # allow control messages between poses
                _poll_commands_nonblocking(pi, conn)

            if not state.search_active.is_set():
                break
        fwd = not fwd

    print("[RPi] Search stopped")

def _poll_commands_nonblocking(pi, conn) -> None:
    """Process any control packets without blocking."""
    buf = b""
    readable, _, _ = select.select([conn], [], [], 0.0)
    if not readable:
        return
    msgs, buf, closed = recv_lines(conn, buf)
    if closed:
        print("[RPi] Connection closed by laptop")
        state.search_active.clear()
        return
    # We reuse server.process_messages to keep semantics unified
    from .server import process_messages
    process_messages(pi, msgs, debug=False)  # noncritical path, ignore returns
