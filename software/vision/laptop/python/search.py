from ultralytics import YOLO
import cv2
import torch
import socket
import json
import time
import argparse
import threading
import statistics

DISPLAY_SCALE = 1.5

COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

parser = argparse.ArgumentParser(description='Object Tracking with YOLO (Laptop client)')
parser.add_argument('--class-id', type=int, default=-1,
                    help='COCO class ID to track (-1 for all classes)')
parser.add_argument('--ip', type=str, default='192.168.1.30',
                    help='Raspberry Pi IP address')
parser.add_argument('--port', type=int, default=65432,
                    help='Port number for socket connection')
args = parser.parse_args()

PI_IP = args.ip
PI_PORT = args.port
TARGET_CLASS = args.class_id
SERVO_SPEED = 0.2
DEADZONE = 30

if TARGET_CLASS != -1 and TARGET_CLASS not in COCO_CLASSES:
    print(f"Error: Class ID {TARGET_CLASS} not found.")
    exit(1)

print(f"Tracking: {'ALL CLASSES' if TARGET_CLASS == -1 else COCO_CLASSES[TARGET_CLASS]} (class ID: {TARGET_CLASS})")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO engine (adjust path as needed)
model = YOLO("yolo11m-seg.engine")

# ----- Camera -----
cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame")
    exit(1)
frame_height, frame_width = frame.shape[:2]
center_x, center_y = frame_width // 2, frame_height // 2

# ----- Socket helpers (line-delimited JSON) -----
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
    try:
        sock.sendall(msg.encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError):
        raise

pi_socket = connect_to_pi()

# ----- Shared flags for search & scan -----
search_mode = False
scan_request_lock = threading.Lock()
scan_request = None  # dict like {"duration_ms": 2000}

# ----- Receiving thread: reacts to YOLO_SCAN -----
def receiver_loop():
    global search_mode, scan_request, pi_socket
    buf = b""
    while True:
        try:
            data = pi_socket.recv(4096)
            if not data:
                print("[Laptop] Connection closed by RPi")
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                # Handle commands from RPi
                if msg.get("cmd") == "YOLO_SCAN":
                    dur = int(msg.get("duration_ms", 2000))
                    with scan_request_lock:
                        scan_request = {"duration_ms": dur}
                # (We can add more RPi->Laptop commands later as needed)
        except (ConnectionResetError, OSError):
            print("[Laptop] Socket error; attempting reconnect...")
            pi_socket = connect_to_pi()
            buf = b""

recv_thread = threading.Thread(target=receiver_loop, daemon=True)
recv_thread.start()

# ----- Servo control (manual tracking only when not in search) -----
def calculate_servo_movement(obj_center, frame_center):
    dx = frame_center[0] - obj_center[0]
    dy = obj_center[1] - frame_center[1]
    norm_dx = dx / frame_center[0] if abs(dx) > DEADZONE else 0.0
    norm_dy = dy / frame_center[1] if abs(dy) > DEADZONE else 0.0
    return norm_dx, norm_dy

def send_servo_command(dx, dy):
    global pi_socket
    cmd = {'type': 'move', 'horizontal': dx * SERVO_SPEED, 'vertical': dy * SERVO_SPEED}
    try:
        send_json(pi_socket, cmd)
    except (BrokenPipeError, ConnectionResetError):
        pi_socket = connect_to_pi()
        send_json(pi_socket, cmd)

def send_search_command(active):
    global pi_socket, search_mode
    cmd = {'type': 'search', 'active': bool(active)}
    search_mode = bool(active)
    try:
        send_json(pi_socket, cmd)
    except (BrokenPipeError, ConnectionResetError):
        pi_socket = connect_to_pi()
        send_json(pi_socket, cmd)

def run_scan_window(duration_s: float, class_filter: int):
    """
    Run YOLO for ~duration_s and return a lightweight summary.
    During the scan window we also update the OpenCV window so the UI stays live.
    """
    import statistics

    t0 = time.time()
    frames = 0

    per_class_conf = {}
    per_class_cx = {}
    per_class_cy = {}

    # Optional: simple HUD text
    hud_text = f"SCANNING ~{int(duration_s*1000)} ms"

    while (time.time() - t0) < duration_s:
        ok, frame = cap.read()
        if not ok:
            break

        # Run model (with or without class filter)
        if class_filter == -1:
            results = model(frame, stream=True)
        else:
            results = model(frame, classes=[class_filter], stream=True)

        for result in results:
            frames += 1

            # ---- accumulate simple stats (largest box per frame) ----
            if len(result.boxes) > 0:
                largest = None
                max_area = 0
                best_cls = None
                best_conf = None

                for box in result.boxes:
                    if class_filter != -1 and int(box.cls) != class_filter:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest = ((x1 + x2) // 2, (y1 + y2) // 2)
                        best_cls = int(box.cls) if class_filter == -1 else class_filter
                        best_conf = float(box.conf[0])

                if largest is not None and best_cls is not None:
                    per_class_conf.setdefault(best_cls, []).append(best_conf)
                    per_class_cx.setdefault(best_cls, []).append(largest[0])
                    per_class_cy.setdefault(best_cls, []).append(largest[1])

            # ---- live UI update during scan window ----
            annotated = result.plot()
            # add a small "SCANNING" HUD
            cv2.putText(annotated, hud_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated,
                                 (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
            cv2.imshow("YOLO Detection", resized)
            cv2.waitKey(1)  # critical: lets OpenCV repaint

    # Build summary to send back to RPi
    objects = []
    for cls_id, confs in per_class_conf.items():
        avg_conf = sum(confs) / len(confs)
        med_cx = statistics.median(per_class_cx[cls_id])
        med_cy = statistics.median(per_class_cy[cls_id])
        objects.append({
            "cls_id": int(cls_id),
            "cls": COCO_CLASSES.get(int(cls_id), str(cls_id)),
            "avg_conf": float(avg_conf),
            "median_center": [float(med_cx), float(med_cy)],
            "frames": int(len(confs))
        })
    objects.sort(key=lambda r: r["avg_conf"], reverse=True)

    return {"frames": int(frames), "objects": objects}

# ----- Main UI / loop -----
try:
    while True:
        # Keyboard: toggle search locally (and tell RPi)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            search_mode = not search_mode
            send_search_command(search_mode)
            print(f"{'Entering' if search_mode else 'Exiting'} search mode")

        # If an RPi scan request is pending, service it immediately
        with scan_request_lock:
            pending = scan_request
            scan_request = None

        if pending is not None:
            duration_ms = int(pending.get("duration_ms", 200))
            summary = run_scan_window(duration_s=duration_ms / 1000.0, class_filter=TARGET_CLASS)
            # Reply with results (RPi currently just waits for this before continuing)
            try:
                send_json(pi_socket, {"type": "YOLO_RESULTS", **summary})
            except (BrokenPipeError, ConnectionResetError):
                pi_socket = connect_to_pi()
                send_json(pi_socket, {"type": "YOLO_RESULTS", **summary})
            # Continue loop (no persistence yet)
            continue

        # Normal live view + (optional) manual centering when NOT in search mode
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame")
            break

        # Run YOLO for visualization/tracking
        if TARGET_CLASS == -1:
            results = model(frame, stream=True)
        else:
            results = model(frame, classes=[TARGET_CLASS], stream=True)

        for result in results:
            largest_obj = None
            max_area = 0
            best_class = None

            if len(result.boxes) > 0:
                for box in result.boxes:
                    if TARGET_CLASS != -1 and int(box.cls) != TARGET_CLASS:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest_obj = ((x1 + x2) // 2, (y1 + y2) // 2)
                        best_class = int(box.cls) if TARGET_CLASS == -1 else TARGET_CLASS

                if largest_obj:
                    dx, dy = calculate_servo_movement(largest_obj, (center_x, center_y))
                    if not search_mode:
                        # Only drive servos in manual/track mode
                        send_servo_command(dx, dy)

                    # Visualization
                    cv2.circle(frame, largest_obj, 5, (0, 255, 0), -1)
                    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                    cv2.line(frame, largest_obj, (center_x, center_y), (0, 0, 255), 2)
                    label = f"Tracking: {COCO_CLASSES[best_class]}" if best_class is not None else "Tracking"
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                annotated = result.plot()
                h, w = annotated.shape[:2]
                resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                cv2.imshow("YOLO Detection", resized)
                cv2.waitKey(1)  # crucial: lets OpenCV repaint

            annotated = result.plot()
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
            cv2.imshow("YOLO Detection", resized)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

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
