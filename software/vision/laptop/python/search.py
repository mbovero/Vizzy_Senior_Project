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
import os
from typing import Dict, Any

# ---- HUD drawing with word-wrap ----
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.7
HUD_THICK = 2
HUD_LINE_GAP = 6

def draw_wrapped_text(img, text, x, y, max_width, scale=HUD_SCALE, color=(0,0,255), thick=HUD_THICK):
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

MIN_FRAMES_FOR_CLASS = 4
CENTER_CONF_THRESHOLD   = 0.60
CENTER_MOVE_NORM_EPS    = 0.035
REQUIRED_GOOD_FRAMES    = 12

parser = argparse.ArgumentParser(description='Laptop client for YOLO-driven robotic arm')
parser.add_argument('--class-id', type=int, default=-1)
parser.add_argument('--class-name', type=str, default=None)
parser.add_argument('--ip', type=str, default='192.168.1.30')
parser.add_argument('--port', type=int, default=65432)
parser.add_argument('--engine', type=str, default='yolo11m-seg.engine')
parser.add_argument('--camera-index', type=int, default=4)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--mem-file', type=str, default='object_memory.json')
args = parser.parse_args()

DEBUG_DIAG = bool(args.debug)
MEM_FILE = args.mem_file

PI_IP = args.ip
PI_PORT = args.port

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==============================
# Object Memory (persistent)
# ==============================
class ObjectMemory:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            else:
                self.data = {}
        except Exception as e:
            print(f"[Memory] Failed to load {self.path}: {e}")
            self.data = {}

    def save(self):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[Memory] Failed to save {self.path}: {e}")

    def reset_session_flags(self):
        for k, v in self.data.items():
            v["updated_this_session"] = 0
        self.save()

    def prune_not_updated(self):
        before = len(self.data)
        self.data = {k: v for k, v in self.data.items() if int(v.get("updated_this_session", 0)) == 1}
        after = len(self.data)
        if before != after:
            print(f"[Memory] Pruned {before - after} stale entries")
        self.save()

    def update_entry(self, cls_id: int, cls_name: str, pwm_btm: int, pwm_top: int, avg_conf: float = None):
        k = str(int(cls_id))
        now = time.time()
        entry = self.data.get(k, {})
        entry.update({
            "cls_id": int(cls_id),
            "cls_name": cls_name,
            "pwm_btm": int(pwm_btm),
            "pwm_top": int(pwm_top),
            "last_seen_ts": now,
            "updated_this_session": 1
        })
        if avg_conf is not None:
            entry["avg_conf"] = float(avg_conf)
        self.data[k] = entry
        self.save()

    def entries_sorted(self):
        # deterministic order; you can swap to last_seen_ts desc if you want
        return [self.data[k] for k in sorted(self.data.keys(), key=lambda x: int(x))]

    def pretty_print(self):
        if not self.data:
            print("[Memory] (empty)")
            return
        print("\n[Memory]")
        for k in sorted(self.data.keys(), key=lambda x: int(x)):
            v = self.data[k]
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v.get("last_seen_ts", 0)))
            print(f"  id {v.get('cls_id'):>3}  {v.get('cls_name','?'):<15} "
                  f"btm={v.get('pwm_btm')}  top={v.get('pwm_top')}  "
                  f"avg_conf={v.get('avg_conf','-')}  updated={v.get('updated_this_session')}  "
                  f"seen={ts}")
        print()

object_memory = ObjectMemory(MEM_FILE)

# ==============================
# Model + Camera
# ==============================
model = YOLO(args.engine)
NAMES = model.names

if not DEBUG_DIAG:
    model.overrides['verbose'] = False

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

PI_IP = args.ip
PI_PORT = args.port
pi_socket = connect_to_pi()

# ==============================
# Shared state
# ==============================
search_mode = False

# NEW: memory recall mode state
recall_mode = False
recall_index = 0  # index into sorted memory entries

scan_request_lock = threading.Lock()
scan_request = None

center_request_lock = threading.Lock()
center_request = None

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

                if msg.get("cmd") == "YOLO_SCAN":
                    dur = int(msg.get("duration_ms", 900))
                    excl = msg.get("exclude_cls", [])
                    with scan_request_lock:
                        scan_request = {"duration_ms": dur, "exclude_cls": excl}

                elif msg.get("cmd") == "CENTER_ON":
                    with center_request_lock:
                        center_request = {
                            "target_cls": int(msg.get("target_cls", -1)),
                            "duration_ms": int(msg.get("duration_ms", 1200)),
                            "epsilon_px": int(msg.get("epsilon_px", 25)),
                        }

                elif msg.get("type") == "CENTER_SNAPSHOT":
                    cls_id = int(msg.get("cls_id"))
                    cls_name = str(msg.get("cls_name", ""))
                    pwm_btm = int(msg.get("pwm_btm"))
                    pwm_top = int(msg.get("pwm_top"))
                    diag = msg.get("diag")
                    avg_conf = None
                    if isinstance(diag, dict):
                        obs = diag.get("observed") or {}
                        if "max_conf_seen" in obs:
                            avg_conf = float(obs["max_conf_seen"])
                    object_memory.update_entry(cls_id, cls_name or str(cls_id), pwm_btm, pwm_top, avg_conf=avg_conf)
                    print(f"[Laptop] Memory updated: {cls_name or cls_id} -> btm={pwm_btm}, top={pwm_top}")

        except (ConnectionResetError, BrokenPipeError, OSError):
            print("[Laptop] Socket error; reconnecting...")
            pi_socket = connect_to_pi()
            buf = b""

recv_thread = threading.Thread(target=receiver_loop, daemon=True)
recv_thread.start()

# ==============================
# Utilities
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

def send_goto_pwms(pwm_btm: int, pwm_top: int, slew_ms: int = 600):
    """NEW: tell the RPi to move to stored PWMs."""
    global pi_socket
    payload = {"cmd": "GOTO_PWMS", "pwm_btm": int(pwm_btm), "pwm_top": int(pwm_top), "slew_ms": int(slew_ms)}
    try:
        send_json(pi_socket, payload)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pi_socket = connect_to_pi()
        send_json(pi_socket, payload)

def clear_model_class_filter():
    try:
        if hasattr(model, "predictor") and hasattr(model.predictor, "args"):
            model.predictor.args.classes = None
    except Exception:
        pass

# ==============================
# SCAN WINDOW (unchanged core)
# ==============================
def run_scan_window(duration_s: float, class_filter: int, exclude_ids=None):
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
        results = model(frame, classes=None if class_filter == -1 else [class_filter], stream=True)
        for result in results:
            frames += 1
            if len(result.boxes) > 0:
                largest = None
                max_area = 0
                best_cls = None
                best_conf = None
                for box in result.boxes:
                    cid = int(box.cls)
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

            annotated = result.plot()
            draw_wrapped_text(annotated, hud_text, 10, 24, int(annotated.shape[1] * 0.92))
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
            cv2.imshow("YOLO Detection", resized)
            cv2.waitKey(1)

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
# CENTERING LOOP (quota-based) - unchanged logic
# ==============================
def center_on_class(duration_s: float, target_cls: int, epsilon_px: int):
    t0 = time.time()
    label = f"CENTERING {get_name(target_cls)} (id {target_cls})"
    good_frames = 0
    success = False

    total_frames = 0
    frames_conf_ok = frames_err_ok = frames_move_small = 0
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
                if conf_ok and err_ok and move_small:
                    good_frames += 1
                    if good_frames >= REQUIRED_GOOD_FRAMES:
                        success = True

                if DEBUG_DIAG:
                    max_conf_seen = max(max_conf_seen, best_conf)
                    min_err_px_seen = min(min_err_px_seen, err_px)
                    min_move_norm_seen = min(min_move_norm_seen, move_norm)
                    last_conf = best_conf
                    last_err_px = err_px
                    last_move_norm = move_norm

                # HUD with live numbers
                disp_conf = best_conf
                disp_err  = err_px
                disp_move = move_norm
                hud = (f"{label}  quota {good_frames}/{REQUIRED_GOOD_FRAMES}  "
                       f"conf {disp_conf:.2f}  err {disp_err:.0f}px  move {disp_move:.3f}")
            else:
                hud = f"{label}  quota {good_frames}/{REQUIRED_GOOD_FRAMES}"

            max_w = int(annotated.shape[1] * 0.92)
            draw_wrapped_text(annotated, hud, 10, 24, max_w)
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
            cv2.imshow("YOLO Detection", resized)
            cv2.waitKey(1)

    clear_model_class_filter()

    diag = None
    if DEBUG_DIAG:
        diag = {
            "thresholds": {
                "conf_per_frame": CENTER_CONF_THRESHOLD,
                "pixel_epsilon": 25,  # provided by caller; value not critical here
                "move_norm_eps": CENTER_MOVE_NORM_EPS,
                "required_good_frames": REQUIRED_GOOD_FRAMES
            },
            "observed": {
                "total_frames": int(total_frames),
                "good_frames": int(good_frames),
                "max_conf_seen": round(max_conf_seen, 4),
                "min_err_px_seen": None if min_err_px_seen == float('inf') else int(min_err_px_seen),
                "min_move_norm_seen": None if min_move_norm_seen == float('inf') else round(min_move_norm_seen, 4),
            }
        }

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
        key = cv2.waitKey(1) & 0xFF

        # Toggle Search Mode
        if key == ord('s'):
            search_mode = not search_mode
            send_search_command(search_mode)
            if search_mode:
                print("Entering search mode: resetting memory session flags...")
                object_memory.reset_session_flags()
                # turning off recall mode if it was on
                if recall_mode:
                    recall_mode = False
                    print("[Recall] OFF (search started)")
            else:
                print("Exiting search mode: pruning memory not updated this session...")
                object_memory.prune_not_updated()
            print(f"{'Entering' if search_mode else 'Exiting'} search mode")

        # Toggle memory recall mode (only when NOT in search mode)
        elif key == ord('m') and not search_mode:
            recall_mode = not recall_mode
            entries = object_memory.entries_sorted()

            if recall_mode:
                if not entries:
                    print("[Recall] No objects in memory; turning OFF.")
                    recall_mode = False
                else:
                    # Print a nicely formatted list of all memory entries
                    print("\n[Recall] ON — stored objects:")
                    print("  idx  name            id   pwm_btm  pwm_top   avg_conf   last_seen")
                    for idx, e in enumerate(entries):
                        ts = e.get("last_seen_ts")
                        ts_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "-"
                        avg_conf = e.get("avg_conf")
                        avg_conf_s = f"{avg_conf:.2f}" if isinstance(avg_conf, (int, float)) else "-"
                        print(f"  {idx:>3}  {e.get('cls_name','?'):<14} {e.get('cls_id'):>3}   "
                            f"{e.get('pwm_btm'):>7}  {e.get('pwm_top'):>7}   {avg_conf_s:>8}   {ts_s}")

                    # Start at the first item and move the arm there
                    recall_index = 0
                    e = entries[recall_index]
                    print(f"[Recall] Selected -> {e['cls_name']} (id {e['cls_id']}), moving to stored pose...")
                    send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)
            else:
                print("[Recall] OFF")

        # While in recall mode: cycle left/right
        elif recall_mode and key == ord('a') and not search_mode:
            entries = object_memory.entries_sorted()
            if entries:
                recall_index = (recall_index - 1) % len(entries)
                e = entries[recall_index]
                print(f"[Recall] {e['cls_name']} (id {e['cls_id']}) <- moving")
                send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)

        elif recall_mode and key == ord('d') and not search_mode:
            entries = object_memory.entries_sorted()
            if entries:
                recall_index = (recall_index + 1) % len(entries)
                e = entries[recall_index]
                print(f"[Recall] {e['cls_name']} (id {e['cls_id']}) -> moving")
                send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)

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

        # Normal live view / manual tracking when NOT in search mode
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame")
            break

        results = model(frame, classes=None if resolved_class_id == -1 else [resolved_class_id], stream=True)
        for result in results:
            annotated = result.plot()
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
            # Small on-screen hint for recall mode
            if recall_mode and not search_mode:
                draw_wrapped_text(resized, "[RECALL MODE] m=exit, a=prev, d=next", 10, 24, int(resized.shape[1]*0.9))
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
