#!/usr/bin/env python3
import pigpio
import socket
import json
import time
import select
import argparse
from threading import Event

# ==============================
# CLI (for debug toggle)
# ==============================
parser = argparse.ArgumentParser(description="RPi Arm Server")
parser.add_argument('--debug', action='store_true',
                    help='Print detailed diagnostics when centering completes')
args = parser.parse_args()
DEBUG_DIAG = bool(args.debug)

# ==============================
# Flags
# ==============================
search_active = Event()
centering_active = Event()  # allow laptop to drive servos during centering

# ==============================
# Servo config
# ==============================
SERVO_BTM = 22
SERVO_MID = 27
SERVO_TOP = 17

SERVO_MIN = 1000  # us
SERVO_MAX = 2000  # us
SERVO_CENTER = 1500  # us

current_horizontal = SERVO_CENTER
current_vertical = SERVO_CENTER

def setup_servos(pi):
    pi.set_servo_pulsewidth(SERVO_MID, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_BTM, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_TOP, SERVO_CENTER)
    time.sleep(1)

def move_servos(pi, horizontal, vertical):
    global current_horizontal, current_vertical
    horizontal_change = int(horizontal * 200)
    vertical_change   = int(vertical * 200)
    new_horizontal = max(SERVO_MIN, min(SERVO_MAX, current_horizontal + horizontal_change))
    new_vertical   = max(SERVO_MIN, min(SERVO_MAX, current_vertical   - vertical_change))
    pi.set_servo_pulsewidth(SERVO_BTM, new_horizontal)
    pi.set_servo_pulsewidth(SERVO_TOP, new_vertical)
    current_horizontal = new_horizontal
    current_vertical   = new_vertical

# ==============================
# Search & centering params
# ==============================
SEARCH_MIN = SERVO_MIN + 200
SEARCH_MAX = SERVO_MAX - 200
SEARCH_H_STEP = 250
SEARCH_V_STEP = 100

POSE_SETTLE_S        = 0.35
SCAN_DURATION_MS     = 1500
CENTER_DURATION_MS   = 3000
CENTER_EPSILON_PX    = 25
MAX_CENTERS_PER_POSE = 1

CONF_THRESHOLD = 0.65  # avg_conf must be >= this to be considered

# ==============================
# JSON helpers
# ==============================
def send_json(conn, obj):
    msg = json.dumps(obj) + "\n"
    conn.sendall(msg.encode("utf-8"))

def recv_lines(conn, buf):
    try:
        data = conn.recv(4096)
        if not data:
            return [], buf, True
        buf += data
    except BlockingIOError:
        return [], buf, False
    lines = []
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        if line.strip():
            lines.append(line.decode("utf-8"))
    return lines, buf, False

def parse_json_lines(lines):
    msgs = []
    for line in lines:
        try:
            msgs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return msgs

# -------- Unified message processor --------
def process_messages(pi, msgs):
    """
    Returns: (stop_requested, center_done_cls, center_success, center_diag, yolo_results)
    """
    stop = False
    center_done_cls = None
    center_success = None
    center_diag = None
    yolo_results = None

    for msg in msgs:
        mtype = msg.get("type")

        if mtype == "move":
            if centering_active.is_set() or not search_active.is_set():
                move_servos(pi, msg.get("horizontal", 0.0), msg.get("vertical", 0.0))

        elif mtype == "search":
            if msg.get("active", False):
                if not search_active.is_set():
                    print("[RPi] Search ON")
                search_active.set()
            else:
                if search_active.is_set():
                    print("[RPi] Search OFF")
                search_active.clear()

        elif mtype == "stop":
            search_active.clear()
            stop = True

        elif mtype == "CENTER_DONE":
            try:
                center_done_cls = int(msg.get("target_cls", -1))
            except Exception:
                center_done_cls = -1
            center_success = bool(msg.get("success", False))
            center_diag = msg.get("diag", None)

        elif mtype == "YOLO_RESULTS":
            yolo_results = msg

    return stop, center_done_cls, center_success, center_diag, yolo_results

# ==============================
# Search driver
# ==============================
def run_search_cycle(pi, conn):
    global current_horizontal, current_vertical

    print("[RPi] Search started")
    conn.setblocking(False)
    buf = b""

    centered_this_session = set()

    def request_scan_and_wait():
        nonlocal buf
        send_json(conn, {
            "cmd": "YOLO_SCAN",
            "duration_ms": SCAN_DURATION_MS,
            "exclude_cls": sorted(centered_this_session)
        })
        t_start = time.time()
        while True:
            readable, _, _ = select.select([conn], [], [], 0.02)
            if readable:
                lines, buf, closed = recv_lines(conn, buf)
                if closed:
                    print("[RPi] Connection closed while waiting for scan")
                    return None
                stop, _, _, _, yolo = process_messages(pi, parse_json_lines(lines))
                if stop:
                    return None
                if yolo is not None:
                    return yolo
            if not search_active.is_set():
                return None
            if time.time() - t_start > 8.0:
                print("[RPi] Scan wait timeout")
                return None

    def center_on_class_and_return(target_cls: int, target_name: str, saved_h: int, saved_v: int):
        global current_horizontal, current_vertical
        nonlocal buf
        print(f"[RPi] Centering on {target_name} (id {target_cls})...")
        centering_active.set()
        send_json(conn, {
            "cmd": "CENTER_ON",
            "target_cls": int(target_cls),
            "target_name": target_name,
            "duration_ms": CENTER_DURATION_MS,
            "epsilon_px": CENTER_EPSILON_PX
        })

        t0 = time.time()
        success = False
        diag = None
        while True:
            readable, _, _ = select.select([conn], [], [], 0.02)
            if readable:
                lines, buf, closed = recv_lines(conn, buf)
                if closed:
                    print("[RPi] Connection closed during centering")
                    break
                stop, center_done_cls, center_success, center_diag, _ = process_messages(pi, parse_json_lines(lines))
                if stop:
                    break
                if center_done_cls is not None and center_done_cls == int(target_cls):
                    success = bool(center_success)
                    diag = center_diag
                    break
            if not search_active.is_set():
                break
            if time.time() - t0 > 12.0:
                print("[RPi] Centering timeout; continuing")
                break

        centering_active.clear()

        # === NEW: if verified, snapshot current PWM pose for laptop memory ===
        if success:
            try:
                send_json(conn, {
                    "type": "CENTER_SNAPSHOT",
                    "cls_id": int(target_cls),
                    "cls_name": target_name,
                    "pwm_btm": int(current_horizontal),
                    "pwm_top": int(current_vertical),
                    "ts": time.time(),
                    "diag": diag  # may be None if laptop not in --debug
                })
            except Exception:
                pass

        # Return to saved pose
        pi.set_servo_pulsewidth(SERVO_BTM, saved_h)
        pi.set_servo_pulsewidth(SERVO_TOP, saved_v)
        current_horizontal, current_vertical = saved_h, saved_v
        time.sleep(POSE_SETTLE_S * 0.8)

        return success, diag

    def poll_commands_nonblocking():
        nonlocal buf
        readable, _, _ = select.select([conn], [], [], 0.0)
        if not readable:
            return
        lines, buf, closed = recv_lines(conn, buf)
        if closed:
            print("[RPi] Connection closed by laptop")
            search_active.clear()
            return
        process_messages(pi, parse_json_lines(lines))

    def horiz_range(fwd=True):
        return range(SEARCH_MIN, SEARCH_MAX + 1, SEARCH_H_STEP) if fwd else range(SEARCH_MAX, SEARCH_MIN - 1, -SEARCH_H_STEP)

    def vert_range(fwd=True):
        return range(SEARCH_MIN, SEARCH_MAX + 1, SEARCH_V_STEP) if fwd else range(SEARCH_MAX, SEARCH_MIN - 1, -SEARCH_V_STEP)

    fwd = True
    while search_active.is_set():
        for h in horiz_range(fwd):
            if not search_active.is_set():
                break
            pi.set_servo_pulsewidth(SERVO_BTM, h)
            current_horizontal = h

            for v in vert_range(fwd):
                if not search_active.is_set():
                    break
                pi.set_servo_pulsewidth(SERVO_TOP, v)
                current_vertical = v
                time.sleep(POSE_SETTLE_S)

                centers_done_at_this_pose = 0
                while search_active.is_set() and centers_done_at_this_pose < MAX_CENTERS_PER_POSE:
                    scan = request_scan_and_wait()
                    if scan is None:
                        break

                    objects = scan.get("objects", [])
                    objects.sort(key=lambda o: float(o.get("avg_conf", 0.0)), reverse=True)

                    target = None
                    for o in objects:
                        cls_id = int(o.get("cls_id", -1))
                        if cls_id == -1:
                            continue
                        if cls_id in centered_this_session:
                            continue
                        if float(o.get("avg_conf", 0.0)) < CONF_THRESHOLD:
                            continue
                        target = (cls_id, str(o.get("cls_name", str(cls_id))))
                        break

                    if target is None:
                        break

                    saved_h, saved_v = current_horizontal, current_vertical
                    target_cls, target_name = target
                    ok, diag = center_on_class_and_return(target_cls, target_name, saved_h, saved_v)
                    if ok:
                        centered_this_session.add(target_cls)
                        centers_done_at_this_pose += 1
                        print(f"[RPi] ? Verified centered on {target_name} (id {target_cls}). "
                              f"Session memory now: {sorted(centered_this_session)}")
                    else:
                        if DEBUG_DIAG:
                            print(f"[RPi] ? Centering verification FAILED for {target_name} (id {target_cls}).")
                            if isinstance(diag, dict):
                                thr = diag.get("thresholds", {})
                                obs = diag.get("observed", {})
                                if "required_good_frames" in thr:
                                    print(f"      Thresholds: conf>={thr.get('conf_per_frame')}, "
                                          f"pixel<= {thr.get('pixel_epsilon')} px, "
                                          f"move_norm< {thr.get('move_norm_eps')}, "
                                          f"required_good_frames={thr.get('required_good_frames')}")
                                else:
                                    print(f"      Thresholds: conf>={thr.get('conf_per_frame')}, "
                                          f"pixel<= {thr.get('pixel_epsilon')} px, "
                                          f"move_norm< {thr.get('move_norm_eps')}")
                                print(f"      Observed: total_frames={obs.get('total_frames')}, "
                                      f"good_frames={obs.get('good_frames')}, "
                                      f"max_conf_seen={obs.get('max_conf_seen')}, "
                                      f"min_err_px_seen={obs.get('min_err_px_seen')}, "
                                      f"min_move_norm_seen={obs.get('min_move_norm_seen')}, "
                                      f"last_conf={obs.get('last_conf')}, "
                                      f"last_err_px={obs.get('last_err_px')}, "
                                      f"last_move_norm={obs.get('last_move_norm')}, "
                                      f"frames_conf_ok={obs.get('frames_conf_ok')}, "
                                      f"frames_err_ok={obs.get('frames_err_ok')}, "
                                      f"frames_move_small={obs.get('frames_move_small')}")
                            else:
                                print("      (No diagnostics provided)")

                    poll_commands_nonblocking()

                poll_commands_nonblocking()

            if not search_active.is_set():
                break

        fwd = not fwd  # zig-zag

    print("[RPi] Search stopped")

# ==============================
# Client handler
# ==============================
def handle_client(conn, pi):
    conn.setblocking(False)
    buf = b""
    try:
        while True:
            readable, _, _ = select.select([conn], [], [], 0.2)
            if not readable:
                continue

            lines, buf, closed = recv_lines(conn, buf)
            if closed:
                break

            msgs = parse_json_lines(lines)
            stop, _, _, _, _ = process_messages(pi, msgs)
            if stop:
                return

            if search_active.is_set():
                run_search_cycle(pi, conn)
                setup_servos(pi)

    finally:
        conn.close()
        search_active.clear()
        centering_active.clear()

# ==============================
# Main
# ==============================
def main():
    pi = pigpio.pi()
    if not pi.connected:
        print("Failed to connect to pigpio daemon")
        return

    setup_servos(pi)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 65432))
        s.listen()
        print(f"RPi server started (debug={'on' if DEBUG_DIAG else 'off'}), waiting for connections...")

        try:
            while True:
                conn, addr = s.accept()
                print(f"Connected by {addr}")
                try:
                    handle_client(conn, pi)
                except Exception as e:
                    print(f"[RPi] Client handler error: {e}")
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            pi.set_servo_pulsewidth(SERVO_BTM, 0)
            pi.set_servo_pulsewidth(SERVO_TOP, 0)
            pi.stop()

if __name__ == "__main__":
    main()
