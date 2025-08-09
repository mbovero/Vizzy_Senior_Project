#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
import torch
import socket
import json
import time
import argparse
import threading
import statistics

# ---- HUD drawing with word-wrap ----
import math

HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.7            # ↓ smaller than before
HUD_THICK = 2
HUD_LINE_GAP = 6           # px between lines

def draw_wrapped_text(img, text, x, y, max_width, scale=HUD_SCALE, color=(0,0,255), thick=HUD_THICK):
    """
    Draw 'text' wrapped to fit max_width (pixels), starting at (x,y).
    Returns bottom y coordinate after drawing.
    """
    words = text.split()
    if not words:
        return y

    line = ""
    ascent = cv2.getTextSize("Ag", HUD_FONT, scale, thick)[0][1]
    line_h = int(ascent + HUD_LINE_GAP)

    for w in words:
        test = (w if not line else (line + " " + w))
        (tw, _), _ = cv2.getTextSize(test, HUD_FONT, scale, thick)
        if tw <= max_width:
            line = test
        else:
            # draw current line and wrap
            cv2.putText(img, line, (x, y), HUD_FONT, scale, color, thick, cv2.LINE_AA)
            y += line_h
            line = w

    if line:
        cv2.putText(img, line, (x, y), HUD_FONT, scale, color, thick, cv2.LINE_AA)
        y += line_h

    return y

# ==============================
# Config / CLI
# ==============================
DISPLAY_SCALE = 1.5
SERVO_SPEED = 0.2
DEADZONE = 30

# Scan summarizer robustness
MIN_FRAMES_FOR_CLASS = 4  # ignore classes that appear in too few frames in a scan window

# --- Centering success criteria (tune here) ---
CENTER_CONF_THRESHOLD   = 0.60   # min per-frame conf during centering
CENTER_MOVE_NORM_EPS    = 0.035  # |ndx| and |ndy| must both be below this to be "stable"
REQUIRED_GOOD_FRAMES    = 12     # total (not necessarily consecutive) frames meeting all criteria

parser = argparse.ArgumentParser(description='Laptop client for YOLO-driven robotic arm')
parser.add_argument('--class-id', type=int, default=-1,
                    help='Numeric class ID to track (-1 for all classes). Interpreted using model.names.')
parser.add_argument('--class-name', type=str, default=None,
                    help='Class name to track (e.g., "laptop"). Overrides --class-id if provided.')
parser.add_argument('--ip', type=str, default='192.168.1.30',
                    help='Raspberry Pi IP address')
parser.add_argument('--port', type=int, default=65432,
                    help='Port number for socket connection')
parser.add_argument('--engine', type=str, default='yolo11m-seg.engine',
                    help='Path to YOLO engine or model')
parser.add_argument('--camera-index', type=int, default=4,
                    help='OpenCV camera index')
parser.add_argument('--debug', action='store_true',
                    help='Enable verbose diagnostics during centering verification')
args = parser.parse_args()

DEBUG_DIAG = bool(args.debug)

PI_IP = args.ip
PI_PORT = args.port

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==============================
# Model + Camera
# ==============================
model = YOLO(args.engine)

# Names mapping from the loaded model (authoritative)
NAMES = model.names  # dict[int,str] or list[str]

def get_name(cid: int) -> str:
    try:
        if isinstance(NAMES, dict):
            return str(NAMES[int(cid)])
        else:
            return str(NAMES[int(cid)])
    except Exception:
        return str(cid)

def name_to_id(name: str):
    if name is None:
        return None
    lname = name.strip().lower()
    try:
        if isinstance(NAMES, dict):
            for k, v in NAMES.items():
                if str(v).lower() == lname:
                    return int(k)
        else:
            for i, v in enumerate(NAMES):
                if str(v).lower() == lname:
                    return int(i)
    except Exception:
        pass
    return None

# Resolve target class
resolved_class_id = -1
if args.class_name:
    cid = name_to_id(args.class_name)
    if cid is None:
        print(f"[WARN] Class name '{args.class_name}' not found in model.names. Falling back to ALL CLASSES.")
        resolved_class_id = -1
    else:
        resolved_class_id = cid
else:
    if args.class_id >= 0:
        resolved_class_id = int(args.class_id)

print(f"Tracking: {'ALL CLASSES' if resolved_class_id == -1 else f'{get_name(resolved_class_id)} (id {resolved_class_id})'}")

cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

ok, frame0 = cap.read()
if not ok:
    print("Error: Could not read frame from camera")
    exit(1)

frame_h, frame_w = frame0.shape[:2]
center_x, center_y = frame_w // 2, frame_h // 2

# ==============================
# Socket helpers (line-delimited JSON)
# ==============================
def connect_to_pi():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((PI_IP, PI_PORT))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("Connected to Raspberry Pi")
            return sock
        except (ConnectionRefusedError, OSError) as e:
            print(f"Connection failed, retrying... Error: {e}")
            time.sleep(2)

def send_json(sock, obj):
    msg = json.dumps(obj) + "\n"
    sock.sendall(msg.encode("utf-8"))

pi_socket = connect_to_pi()

# ==============================
# Shared state
# ==============================
search_mode = False

scan_request_lock = threading.Lock()
scan_request = None  # {"duration_ms": 900, "exclude_cls": [...]}

center_request_lock = threading.Lock()
center_request = None  # {"target_cls": 63, "duration_ms": 1500, "epsilon_px": 25}

# ==============================
# Receiver: handle RPi -> Laptop commands
# ==============================
def receiver_loop():
    global pi_socket, scan_request, center_request
    buf = b""
    while True:
        try:
            data = pi_socket.recv(4096)
            if not data:
                print("[Laptop] Connection closed by RPi; reconnecting...")
                pi_socket = connect_to_pi()
                buf = b""
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                # YOLO scan request
                if msg.get("cmd") == "YOLO_SCAN":
                    dur = int(msg.get("duration_ms", 900))
                    excl = msg.get("exclude_cls", [])
                    with scan_request_lock:
                        scan_request = {"duration_ms": dur, "exclude_cls": excl}

                # Centering request
                elif msg.get("cmd") == "CENTER_ON":
                    with center_request_lock:
                        center_request = {
                            "target_cls": int(msg.get("target_cls", -1)),
                            "duration_ms": int(msg.get("duration_ms", 1200)),
                            "epsilon_px": int(msg.get("epsilon_px", 25)),
                        }

        except (ConnectionResetError, BrokenPipeError, OSError):
            print("[Laptop] Socket error; reconnecting...")
            pi_socket = connect_to_pi()
            buf = b""

recv_thread = threading.Thread(target=receiver_loop, daemon=True)
recv_thread.start()

# ==============================
# Utility: movement + search toggle + model filter reset
# ==============================
def calculate_servo_movement(obj_center, frame_center):
    dx = frame_center[0] - obj_center[0]
    dy = obj_center[1] - frame_center[1]
    ndx = dx / frame_center[0] if abs(dx) > DEADZONE else 0.0
    ndy = dy / frame_center[1] if abs(dy) > DEADZONE else 0.0
    return ndx, ndy

def send_servo_command(dx, dy):
    global pi_socket
    cmd = {'type': 'move', 'horizontal': dx * SERVO_SPEED, 'vertical': dy * SERVO_SPEED}
    try:
        send_json(pi_socket, cmd)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pi_socket = connect_to_pi()
        send_json(pi_socket, cmd)

def send_search_command(active: bool):
    global pi_socket, search_mode
    search_mode = bool(active)
    try:
        send_json(pi_socket, {'type': 'search', 'active': search_mode})
    except (BrokenPipeError, ConnectionResetError, OSError):
        pi_socket = connect_to_pi()
        send_json(pi_socket, {'type': 'search', 'active': search_mode})

def clear_model_class_filter():
    """Ensure subsequent calls are NOT class-filtered."""
    try:
        if hasattr(model, "predictor") and hasattr(model.predictor, "args"):
            model.predictor.args.classes = None
    except Exception:
        pass

# ==============================
# SCAN WINDOW: run YOLO ~duration and keep UI alive
# ==============================
def run_scan_window(duration_s: float, class_filter: int, exclude_ids=None):
    """
    Collect detections for ~duration_s and return per-class aggregates.
    exclude_ids: set/list of class IDs to ignore when summarizing (still drawn).
    """
    if exclude_ids is None:
        exclude_ids = set()
    else:
        exclude_ids = set(int(x) for x in exclude_ids)

    t0 = time.time()
    frames = 0

    per_class_conf = {}
    per_class_cx = {}
    per_class_cy = {}

    hud_text = f"SCANNING ~{int(duration_s*1000)} ms"

    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok:
            break

        # ALL classes => classes=None (clears sticky filter)
        if class_filter == -1:
            results = model(frame, classes=None, stream=True)
        else:
            results = model(frame, classes=[class_filter], stream=True)

        for result in results:
            frames += 1

            # accumulate stats using largest box per frame
            if len(result.boxes) > 0:
                largest = None
                max_area = 0
                best_cls = None
                best_conf = None

                for box in result.boxes:
                    cid = int(box.cls)
                    # Respect filter and exclude set for summary
                    if class_filter != -1 and cid != class_filter:
                        continue
                    if cid in exclude_ids:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest = ((x1 + x2) // 2, (y1 + y2) // 2)
                        best_cls = cid
                        best_conf = float(box.conf[0])

                if largest is not None and best_cls is not None:
                    per_class_conf.setdefault(best_cls, []).append(best_conf)
                    per_class_cx.setdefault(best_cls, []).append(largest[0])
                    per_class_cy.setdefault(best_cls, []).append(largest[1])

            # live UI update (we still draw everything, even excluded classes)
            annotated = result.plot()  # uses model.names internally
            cv2.putText(annotated, hud_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
            cv2.imshow("YOLO Detection", resized)
            cv2.waitKey(1)

    # Build summary (apply min-frames gate)
    objects = []
    for cls_id, confs in per_class_conf.items():
        if len(confs) < MIN_FRAMES_FOR_CLASS:
            continue
        avg_conf = sum(confs) / len(confs)
        med_cx = statistics.median(per_class_cx[cls_id])
        med_cy = statistics.median(per_class_cy[cls_id])
        name = get_name(int(cls_id))
        objects.append({
            "cls_id": int(cls_id),
            "cls_name": name,
            "avg_conf": float(avg_conf),
            "median_center": [float(med_cx), float(med_cy)],
            "frames": int(len(confs))
        })
    objects.sort(key=lambda r: r["avg_conf"], reverse=True)

    return {"frames": int(frames), "objects": objects}

# ==============================
# CENTERING LOOP: quota-based verification + optional diagnostics
# ==============================
def center_on_class(duration_s: float, target_cls: int, epsilon_px: int):
    """
    Try to center on target_cls for ~duration_s.

    Success requires a QUOTA of frames (REQUIRED_GOOD_FRAMES) where ALL are true:
      - conf >= CENTER_CONF_THRESHOLD
      - pixel error <= epsilon_px
      - |ndx| and |ndy| < CENTER_MOVE_NORM_EPS

    Sends CENTER_DONE with:
      - 'success': bool
      - 'diag': {...} only if --debug is enabled
    """
    t0 = time.time()
    label = f"CENTERING {get_name(target_cls)} (id {target_cls})"

    good_frames = 0
    success = False

    # Diagnostics (populated only if DEBUG_DIAG)
    total_frames = 0
    frames_conf_ok = 0
    frames_err_ok = 0
    frames_move_small = 0

    max_conf_seen = 0.0
    min_err_px_seen = float('inf')
    min_move_norm_seen = float('inf')

    last_conf = 0.0
    last_err_px = None
    last_move_norm = None

    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, classes=[target_cls], stream=True)

        for result in results:
            total_frames += 1
            annotated = result.plot()

            best = None
            best_conf = 0.0
            max_area = 0

            if len(result.boxes) > 0:
                for box in result.boxes:
                    if int(box.cls) != int(target_cls):
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best = ((x1 + x2) // 2, (y1 + y2) // 2)
                        best_conf = float(box.conf[0])

            conf_ok = False
            err_ok = False
            move_small = False

            if best is not None:
                dx = (center_x - best[0])
                dy = (best[1] - center_y)
                err_px = max(abs(dx), abs(dy))
                ndx = dx / center_x
                ndy = dy / center_y
                move_norm = max(abs(ndx), abs(ndy))

                conf_ok = (best_conf >= CENTER_CONF_THRESHOLD)
                err_ok = (abs(dx) <= epsilon_px and abs(dy) <= epsilon_px)
                move_small = (abs(ndx) < CENTER_MOVE_NORM_EPS and abs(ndy) < CENTER_MOVE_NORM_EPS)

                # Drive servos if not yet inside the stability box
                if not (err_ok and move_small):
                    try:
                        send_json(pi_socket, {'type': 'move',
                                              'horizontal': ndx * SERVO_SPEED,
                                              'vertical':   ndy * SERVO_SPEED})
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        new_sock = connect_to_pi()
                        globals()['pi_socket'] = new_sock
                        send_json(pi_socket, {'type': 'move',
                                              'horizontal': ndx * SERVO_SPEED,
                                              'vertical':   ndy * SERVO_SPEED})

                # Update counters
                if conf_ok and err_ok and move_small:
                    good_frames += 1
                    if good_frames >= REQUIRED_GOOD_FRAMES:
                        success = True  # latch; we still run until duration (or break early if you want)
                if DEBUG_DIAG:
                    max_conf_seen = max(max_conf_seen, best_conf)
                    min_err_px_seen = min(min_err_px_seen, err_px)
                    min_move_norm_seen = min(min_move_norm_seen, move_norm)
                    last_conf = best_conf
                    last_err_px = err_px
                    last_move_norm = move_norm
                    if conf_ok: frames_conf_ok += 1
                    if err_ok: frames_err_ok += 1
                    if move_small: frames_move_small += 1

            if DEBUG_DIAG:
                # Live numbers (fall back to last seen when target is missing)
                disp_conf = best_conf if best is not None else (last_conf if last_conf else 0.0)
                disp_err  = err_px    if best is not None else (last_err_px if last_err_px is not None else 0)
                disp_move = move_norm if best is not None else (last_move_norm if last_move_norm is not None else 0.0)

                # Per-criterion indicators
                conf_tag  = "OK" if disp_conf >= CENTER_CONF_THRESHOLD else "low"
                err_tag   = "OK" if disp_err <= epsilon_px else "high"
                move_tag  = "OK" if disp_move < CENTER_MOVE_NORM_EPS else "moving"

                if best is None:
                    hud = (
                        f"{label}  "
                        f"quota {good_frames}/{REQUIRED_GOOD_FRAMES}  "
                        f"[NO TARGET IN FRAME]  "
                        f"conf {disp_conf:.2f} (>={CENTER_CONF_THRESHOLD:.2f}) {conf_tag}  "
                        f"err {disp_err:.0f}px (<={epsilon_px}px) {err_tag}  "
                        f"move {disp_move:.3f} (<{CENTER_MOVE_NORM_EPS:.3f}) {move_tag}"
                    )
                else:
                    hud = (
                        f"{label}  "
                        f"quota {good_frames}/{REQUIRED_GOOD_FRAMES}  "
                        f"conf {disp_conf:.2f} (>={CENTER_CONF_THRESHOLD:.2f}) {conf_tag}  "
                        f"err {disp_err:.0f}px (<={epsilon_px}px) {err_tag}  "
                        f"move {disp_move:.3f} (<{CENTER_MOVE_NORM_EPS:.3f}) {move_tag}"
                    )
            else:
                hud = (
                    f"{label}  "
                    f"quota {good_frames}/{REQUIRED_GOOD_FRAMES}  "
                )

            max_w = int(annotated.shape[1] * 0.92)
            draw_wrapped_text(annotated, hud, 10, 24, max_w)
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
            cv2.imshow("YOLO Detection", resized)
            cv2.waitKey(1)

    # Make sure future scans use ALL classes again (clear sticky class filter)
    clear_model_class_filter()

    # Build diagnostics (only if debugging)
    diag = None
    if DEBUG_DIAG:
        diag = {
            "thresholds": {
                "conf_per_frame": CENTER_CONF_THRESHOLD,
                "pixel_epsilon": epsilon_px,
                "move_norm_eps": CENTER_MOVE_NORM_EPS,
                "required_good_frames": REQUIRED_GOOD_FRAMES
            },
            "observed": {
                "total_frames": int(total_frames),
                "good_frames": int(good_frames),
                "max_conf_seen": round(max_conf_seen, 4),
                "min_err_px_seen": None if min_err_px_seen == float('inf') else int(min_err_px_seen),
                "min_move_norm_seen": None if min_move_norm_seen == float('inf') else round(min_move_norm_seen, 4),
                "last_conf": round(last_conf, 4),
                "last_err_px": None if last_err_px is None else int(last_err_px),
                "last_move_norm": None if last_move_norm is None else round(last_move_norm, 4),
                "frames_conf_ok": int(frames_conf_ok),
                "frames_err_ok": int(frames_err_ok),
                "frames_move_small": int(frames_move_small),
            }
        }

    # Notify RPi we’re done centering, include success + diagnostics (if any)
    payload = {"type": "CENTER_DONE", "target_cls": int(target_cls), "success": bool(success)}
    if diag is not None:
        payload["diag"] = diag
    try:
        send_json(pi_socket, payload)
    except (BrokenPipeError, ConnectionResetError, OSError):
        new_sock = connect_to_pi()
        globals()['pi_socket'] = new_sock
        send_json(pi_socket, payload)

# ==============================
# Main loop
# ==============================
try:
    while True:
        # Keyboard: toggle Search Mode from laptop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            search_mode = not search_mode
            send_search_command(search_mode)
            print(f"{'Entering' if search_mode else 'Exiting'} search mode")
        elif key == ord('q'):
            break

        # Service pending scan request (from RPi)
        with scan_request_lock:
            pending_scan = scan_request
            scan_request = None

        if pending_scan is not None:
            duration_ms = int(pending_scan.get("duration_ms", 900))
            exclude_ids = pending_scan.get("exclude_cls", [])
            summary = run_scan_window(duration_s=duration_ms / 1000.0,
                                      class_filter=resolved_class_id,
                                      exclude_ids=exclude_ids)
            # Reply with results to let RPi proceed
            try:
                send_json(pi_socket, {"type": "YOLO_RESULTS", **summary})
            except (BrokenPipeError, ConnectionResetError, OSError):
                pi_socket = connect_to_pi()
                send_json(pi_socket, {"type": "YOLO_RESULTS", **summary})
            continue

        # Service pending centering request (from RPi)
        with center_request_lock:
            pending_center = center_request
            center_request = None

        if pending_center is not None:
            target = int(pending_center["target_cls"])
            dur_s = float(pending_center["duration_ms"]) / 1000.0
            eps = int(pending_center["epsilon_px"])
            print(f"[Laptop] Centering on {get_name(target)} (id {target}) for {dur_s:.2f}s "
                  f"{'(debug on)' if DEBUG_DIAG else '(debug off)'}")
            center_on_class(dur_s, target, eps)
            continue

        # Normal live view + optional manual tracking when NOT in search mode
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame")
            break

        # Run YOLO (filtered or not). Be explicit to avoid sticky class filters.
        if resolved_class_id == -1:
            results = model(frame, classes=None, stream=True)  # ALL classes
        else:
            results = model(frame, classes=[resolved_class_id], stream=True)

        for result in results:
            largest_obj = None
            max_area = 0
            best_class = None

            if len(result.boxes) > 0:
                for box in result.boxes:
                    if resolved_class_id != -1 and int(box.cls) != resolved_class_id:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest_obj = ((x1 + x2) // 2, (y1 + y2) // 2)
                        best_class = int(box.cls) if resolved_class_id == -1 else resolved_class_id

                if largest_obj:
                    dx, dy = calculate_servo_movement(largest_obj, (center_x, center_y))
                    if not search_mode:
                        send_servo_command(dx, dy)

                    # Simple overlay
                    cv2.circle(frame, largest_obj, 5, (0, 255, 0), -1)
                    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                    cv2.line(frame, largest_obj, (center_x, center_y), (0, 0, 255), 2)
                    if best_class is not None:
                        label = f"Tracking: {get_name(best_class)} (id {best_class})"
                        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            annotated = result.plot()  # uses model.names internally
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
            cv2.imshow("YOLO Detection", resized)

finally:
    try:
        send_json(pi_socket, {'type': 'stop'})
    except Exception:
        pass
    try:
        pi_socket.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
