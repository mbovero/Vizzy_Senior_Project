import pigpio
import socket
import json
import time
import select

from threading import Event

search_active = Event()

# GPIO pin specifications
SERVO_BTM = 22  # Bottom servo - horizontal/pan movements
SERVO_MID = 27
SERVO_TOP = 17  # Top servo - vertical/tilt movements

# Servo configuration
SERVO_MIN = 1000  # us
SERVO_MAX = 2000  # us
SERVO_CENTER = 1500  # us

# Current positions
current_horizontal = SERVO_CENTER
current_vertical = SERVO_CENTER

def setup_servos(pi):
    """Initialize servos to center position"""
    pi.set_servo_pulsewidth(SERVO_MID, SERVO_CENTER)  # Middle servo
    pi.set_servo_pulsewidth(SERVO_BTM, SERVO_CENTER)  # Bottom (horizontal)
    pi.set_servo_pulsewidth(SERVO_TOP, SERVO_CENTER)  # Top (vertical)
    time.sleep(1)

def move_servos(pi, horizontal, vertical):
    """Move servos based on normalized inputs (-1 to 1)"""
    global current_horizontal, current_vertical

    horizontal_change = int(horizontal * 200)  # sensitivity
    vertical_change = int(vertical * 200)

    new_horizontal = current_horizontal + horizontal_change
    new_vertical = current_vertical - vertical_change  # invert for typical cam frame

    new_horizontal = max(SERVO_MIN, min(SERVO_MAX, new_horizontal))
    new_vertical = max(SERVO_MIN, min(SERVO_MAX, new_vertical))

    pi.set_servo_pulsewidth(SERVO_BTM, new_horizontal)
    pi.set_servo_pulsewidth(SERVO_TOP, new_vertical)

    current_horizontal = new_horizontal
    current_vertical = new_vertical

# Search grid params
SEARCH_MIN = SERVO_MIN + 200
SEARCH_MAX = SERVO_MAX - 200
SEARCH_H_STEP = 250
SEARCH_V_STEP = 100
POSE_SETTLE_S = 0.35     # small settle before scan
SCAN_DURATION_MS = 2000   # RPi asks laptop to scan for this long

# ----- JSON helpers (line-delimited) -----
def send_json(conn, obj):
    msg = json.dumps(obj) + "\n"
    conn.sendall(msg.encode("utf-8"))

def recv_lines(conn, buf):
    """Read available bytes and split into complete JSON lines; return (lines, remainder_buf)."""
    try:
        data = conn.recv(4096)
        if not data:
            return [], buf, True  # closed
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
            # ignore bad lines; could log if desired
            pass
    return msgs

# ----- Search driver -----
def run_search_cycle(pi, conn):
    """
    Walk the scan grid. At each pose:
    - Move servos and settle
    - Ask the laptop to YOLO_SCAN for SCAN_DURATION_MS
    - Wait for YOLO_RESULTS (or a stop/disable command)
    """
    global current_horizontal, current_vertical

    print("[RPi] Search started")
    conn.setblocking(False)  # non-blocking for command polling
    buf = b""

    def request_scan_and_wait():
        nonlocal buf
        send_json(conn, {"cmd": "YOLO_SCAN", "duration_ms": SCAN_DURATION_MS})
        # Wait for YOLO_RESULTS or a stop/disable
        t_start = time.time()
        while True:
            # If connection closed, bail
            readable, _, _ = select.select([conn], [], [], 0.2)
            if readable:
                lines, buf, closed = recv_lines(conn, buf)
                if closed:
                    print("[RPi] Connection closed while waiting for scan result")
                    return False
                for msg in parse_json_lines(lines):
                    mtype = msg.get("type")
                    if mtype == "YOLO_RESULTS":
                        # We don't use the contents yet; just acknowledge and continue
                        # You could log: frames = msg.get("frames", 0)
                        return True
                    # Handle stop/disable during scan
                    if msg.get("type") == "search" and not msg.get("active", True):
                        print("[RPi] Search deactivated by laptop during scan")
                        search_active.clear()
                        return False
                    if msg.get("type") == "stop":
                        print("[RPi] Stop received during scan")
                        search_active.clear()
                        return False

            # Safety: if search got cleared elsewhere, stop waiting
            if not search_active.is_set():
                return False

            # Optional: timeout guard in case laptop never responds
            if time.time() - t_start > 5.0:
                print("[RPi] Scan wait timeout; continuing")
                return True

    # Helper to process any incoming control packets between poses
    def poll_commands_nonblocking():
        nonlocal buf
        readable, _, _ = select.select([conn], [], [], 0.0)
        if readable:
            lines, buf, closed = recv_lines(conn, buf)
            if closed:
                print("[RPi] Connection closed by laptop")
                search_active.clear()
                return
            for msg in parse_json_lines(lines):
                if msg.get("type") == "search":
                    if not msg.get("active", True):
                        search_active.clear()
                elif msg.get("type") == "stop":
                    search_active.clear()

    # Zig-zag across grid
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

            v_iter = vert_range(fwd)
            for v in v_iter:
                if not search_active.is_set():
                    break
                pi.set_servo_pulsewidth(SERVO_TOP, v)
                current_vertical = v

                # Small settle to avoid motion blur
                time.sleep(POSE_SETTLE_S)

                # Ask laptop to scan this pose
                if not request_scan_and_wait():
                    break

                # Let laptop/user disable search in-between poses
                poll_commands_nonblocking()

            if not search_active.is_set():
                break

        fwd = not fwd  # zig-zag direction

    print("[RPi] Search stopped")

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
            if readable:
                lines, buf, closed = recv_lines(conn, buf)
                if closed:
                    break
                for msg in parse_json_lines(lines):
                    mtype = msg.get("type")
                    if mtype == "move":
                        if not search_active.is_set():
                            move_servos(pi, msg.get("horizontal", 0.0), msg.get("vertical", 0.0))

                    elif mtype == "search":
                        active = bool(msg.get("active", False))
                        if active and not search_active.is_set():
                            search_active.set()
                            run_search_cycle(pi, conn)  # returns when search stops
                            setup_servos(pi)             # re-center after search
                        elif not active:
                            search_active.clear()

                    elif mtype == "stop":
                        search_active.clear()
                        return
            else:
                # idle loop
                pass
    finally:
        conn.close()
        search_active.clear()

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
