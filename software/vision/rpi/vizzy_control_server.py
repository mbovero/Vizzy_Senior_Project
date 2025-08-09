#!/usr/bin/env python3
import pigpio
import socket
import json
import time
import select
from threading import Event

# ==============================
# Flags
# ==============================
search_active = Event()
centering_active = Event()  # allow laptop to drive servos during centering

# ==============================
# Servo config
# ==============================
SERVO_BTM = 22  # Bottom servo - horizontal/pan movements
SERVO_MID = 27  # Middle servo (not actively used but centered)
SERVO_TOP = 17  # Top servo - vertical/tilt movements

SERVO_MIN = 1000  # us
SERVO_MAX = 2000  # us
SERVO_CENTER = 1500  # us

current_horizontal = SERVO_CENTER
current_vertical = SERVO_CENTER

def setup_servos(pi):
    """Initialize servos to center position"""
    pi.set_servo_pulsewidth(SERVO_MID, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_BTM, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_TOP, SERVO_CENTER)
    time.sleep(1)

def move_servos(pi, horizontal, vertical):
    """
    Move servos based on normalized inputs (-1 to 1).
    Positive horizontal -> increase bottom pulse (pan right)
    Positive vertical   -> decrease top pulse (tilt up).
    """
    global current_horizontal, current_vertical

    horizontal_change = int(horizontal * 200)  # tune sensitivity to match laptop
    vertical_change   = int(vertical * 200)

    new_horizontal = current_horizontal + horizontal_change
    new_vertical   = current_vertical - vertical_change

    new_horizontal = max(SERVO_MIN, min(SERVO_MAX, new_horizontal))
    new_vertical   = max(SERVO_MIN, min(SERVO_MAX, new_vertical))

    pi.set_servo_pulsewidth(SERVO_BTM, new_horizontal)
    pi.set_servo_pulsewidth(SERVO_TOP, new_vertical)

    current_horizontal = new_horizontal
    current_vertical   = new_vertical

# ==============================
# Search grid params
# ==============================
SEARCH_MIN = SERVO_MIN + 200
SEARCH_MAX = SERVO_MAX - 200
SEARCH_H_STEP = 250
SEARCH_V_STEP = 100

POSE_SETTLE_S = 0.35
SCAN_DURATION_MS = 1200
CENTER_DURATION_MS = 1500
CENTER_EPSILON_PX = 25
MAX_CENTERS_PER_POSE = 1

# Confidence gating (works with laptop's MIN_FRAMES filter)
CONF_THRESHOLD = 0.65

# ==============================
# JSON helpers (line-delimited)
# ==============================
def send_json(conn, obj):
    msg = json.dumps(obj) + "\n"
    conn.sendall(msg.encode("utf-8"))

def recv_lines(conn, buf):
    """
    Read available bytes and split into complete JSON lines.
    Returns (lines, remainder_buf, closed)
    """
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

# -------- NEW: unified message processor we can call anywhere --------
def process_messages(pi, msgs):
    """
    Handle incoming messages uniformly. Must be called frequently from
    scan/centering loops so 'move' is applied even during search mode.
    Returns: (stop_requested, center_done_cls, yolo_results)
    """
    stop = False
    center_done_cls = None
    yolo_results = None

    for msg in msgs:
        mtype = msg.get("type")

        if mtype == "move":
            if centering_active.is_set() or not search_active.is_set():
                move_servos(pi, msg.get("horizontal", 0.0), msg.get("vertical", 0.0))

        elif mtype == "search":
            # ? FIX: honor both ON and OFF
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

        elif mtype == "YOLO_RESULTS":
            yolo_results = msg

    return stop, center_done_cls, yolo_results

# ==============================
# Search driver
# ==============================
def run_search_cycle(pi, conn):
    """
    For each scan pose:
      - move + settle
      - repeat up to MAX_CENTERS_PER_POSE:
          * YOLO_SCAN (ask laptop, excluding session-seen classes)
          * pick highest-confidence object whose class hasn't been centered in this session and passes CONF_THRESHOLD
          * CENTER_ON that class (laptop drives servos; we apply 'move' here)
          * return to saved pose
      - continue grid
    """
    global current_horizontal, current_vertical

    print("[RPi] Search started")
    conn.setblocking(False)
    buf = b""

    # Session-wide memory of classes we've already centered
    centered_this_session = set()

    def request_scan_and_wait():
        """Send YOLO_SCAN and wait for YOLO_RESULTS. Returns dict or None."""
        nonlocal buf
        send_json(conn, {
            "cmd": "YOLO_SCAN",
            "duration_ms": SCAN_DURATION_MS,
            "exclude_cls": sorted(centered_this_session)  # tell laptop which to ignore in summary
        })
        t_start = time.time()
        while True:
            readable, _, _ = select.select([conn], [], [], 0.02)
            if readable:
                lines, buf, closed = recv_lines(conn, buf)
                if closed:
                    print("[RPi] Connection closed while waiting for scan")
                    return None
                stop, _, yolo = process_messages(pi, parse_json_lines(lines))
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
        """Tell laptop to center; APPLY 'move' commands here; then return to saved pose."""
        nonlocal buf
        print(f"[RPi] Centering on {target_name} (id {target_cls})...")
        centering_active.set()
        send_json(conn, {
            "cmd": "CENTER_ON",
            "target_cls": int(target_cls),
            "target_name": target_name,          # debug/log only; laptop ignores
            "duration_ms": CENTER_DURATION_MS,
            "epsilon_px": CENTER_EPSILON_PX
        })

        t0 = time.time()
        while True:
            readable, _, _ = select.select([conn], [], [], 0.02)
            if readable:
                lines, buf, closed = recv_lines(conn, buf)
                if closed:
                    print("[RPi] Connection closed during centering")
                    break
                stop, center_done_cls, _ = process_messages(pi, parse_json_lines(lines))
                if stop:
                    break
                if center_done_cls is not None and center_done_cls == int(target_cls):
                    # laptop finished centering on this target
                    break
            if not search_active.is_set():
                break
            # safety timeout
            if time.time() - t0 > 10.0:
                print("[RPi] Centering timeout; continuing")
                break

        centering_active.clear()

        # Return to saved pose
        pi.set_servo_pulsewidth(SERVO_BTM, saved_h)
        pi.set_servo_pulsewidth(SERVO_TOP, saved_v)
        global current_horizontal, current_vertical
        current_horizontal, current_vertical = saved_h, saved_v
        time.sleep(POSE_SETTLE_S * 0.8)

    def poll_commands_nonblocking():
        """Process any control or move packets between poses."""
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

                # settle before scan/center cycle
                time.sleep(POSE_SETTLE_S)

                centers_done_at_this_pose = 0

                # Optionally center multiple distinct (new) classes at this pose
                while search_active.is_set() and centers_done_at_this_pose < MAX_CENTERS_PER_POSE:
                    # Scan at this pose
                    scan = request_scan_and_wait()
                    if scan is None:
                        break

                    # Ensure sorted by confidence (defensive; laptop already sorts)
                    objects = scan.get("objects", [])
                    objects.sort(key=lambda o: float(o.get("avg_conf", 0.0)), reverse=True)

                    # Pick highest-confidence class not yet centered in this SESSION and above threshold
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
                        # Nothing new/credible for this session at this pose
                        break

                    # Save pose, center, mark, return
                    saved_h, saved_v = current_horizontal, current_vertical
                    target_cls, target_name = target
                    center_on_class_and_return(target_cls, target_name, saved_h, saved_v)
                    centered_this_session.add(target_cls)
                    centers_done_at_this_pose += 1

                    # Allow interrupts (and moves) between centerings
                    poll_commands_nonblocking()

                # Allow interrupts between poses
                poll_commands_nonblocking()

            if not search_active.is_set():
                break

        fwd = not fwd  # zig-zag direction

    # Session ended; optional summary
    if centered_this_session:
        print(f"[RPi] Session complete. Centered classes (ids): {sorted(centered_this_session)}")
    else:
        print("[RPi] Session complete. No classes centered.")
    print("[RPi] Search stopped")

# ==============================
# Client handler
# ==============================
def handle_client(conn, pi):
    """
    Handles incoming control packets when NOT actively scanning.
    When a search starts, hands control to run_search_cycle().
    """
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
            # Apply moves/search/stop here too (for manual mode)
            stop, _, _ = process_messages(pi, msgs)
            if stop:
                return

            # Start/stop search is already handled by process_messages via 'search' type.
            if search_active.is_set():
                # Hand control to the search driver; it will do all socket reads and moves.
                run_search_cycle(pi, conn)
                setup_servos(pi)  # re-center after search

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
        print("RPi server started, waiting for connections...")

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
