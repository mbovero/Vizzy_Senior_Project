# vizzy/laptop/client.py
from __future__ import annotations
import socket, time, threading, cv2
from typing import Optional, Iterable
from .config import make_arg_parser, resolve_class_id_from_name
from .yolo_runner import init_model, infer_all
from .hud import draw_wrapped_text
from .memory import ObjectMemory
from .scanning import run_scan_window
from .centering import center_on_class
from ..shared.jsonl import send_json, recv_lines
from ..shared import protocol as P

def connect_to_pi(ip: str, port: int):
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("Connected to Raspberry Pi")
            return sock
        except (ConnectionRefusedError, OSError) as e:
            print(f"Connection failed, retrying... Error: {e}")
            time.sleep(2)

def main():
    # ----- CLI & model init -----
    args = make_arg_parser().parse_args()
    DEBUG = bool(args.debug)
    DISPLAY_SCALE = float(args.display_scale)

    model = init_model(args.engine, DEBUG)
    NAMES = model.names

    def get_name(cid: int) -> str:
        try:
            if isinstance(NAMES, dict):
                return str(NAMES[int(cid)])
            else:
                return str(NAMES[int(cid)])
        except Exception:
            return str(cid)

    resolved_class_id = resolve_class_id_from_name(NAMES, args.class_id, args.class_name)
    print(f"Tracking: {'ALL CLASSES' if resolved_class_id == -1 else f'{get_name(resolved_class_id)} (id {resolved_class_id})'}")

    # ----- Camera -----
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    ok, frame0 = cap.read()
    if not ok:
        print("Error: Could not read frame from camera")
        return
    frame_h, frame_w = frame0.shape[:2]
    center_x, center_y = frame_w // 2, frame_h // 2

    # ----- Networking -----
    PI_IP, PI_PORT = args.ip, args.port
    pi_socket = connect_to_pi(PI_IP, PI_PORT)

    # ----- Shared state -----
    SERVO_SPEED = 0.2
    DEADZONE = 30

    search_mode = False
    recall_mode = False
    recall_index = 0

    object_memory = ObjectMemory(args.mem_file)

    scan_request_lock = threading.Lock()
    scan_request: Optional[dict] = None

    center_request_lock = threading.Lock()
    center_request: Optional[dict] = None

    # ---------- Receiver thread ----------
    def receiver_loop():
        nonlocal pi_socket, scan_request, center_request
        buf = b""
        while True:
            try:
                msgs, buf, closed = recv_lines(pi_socket, buf)
                if closed:
                    print("[Laptop] Connection closed by RPi; reconnecting...")
                    pi_socket = connect_to_pi(PI_IP, PI_PORT)
                    buf = b""
                    continue

                for msg in msgs:
                    # RPi -> laptop commands
                    if msg.get("cmd") == P.CMD_YOLO_SCAN:
                        with scan_request_lock:
                            scan_request = {
                                "duration_ms": int(msg.get("duration_ms", 900)),
                                "exclude_cls": msg.get("exclude_cls", []),
                            }
                    elif msg.get("cmd") == P.CMD_CENTER_ON:
                        with center_request_lock:
                            center_request = {
                                "target_cls": int(msg.get("target_cls", -1)),
                                "duration_ms": int(msg.get("duration_ms", 1200)),
                                "epsilon_px": int(msg.get("epsilon_px", 25)),
                                "target_name": msg.get("target_name", ""),
                            }
                    elif msg.get("type") == P.TYPE_CENTER_SNAPSHOT:
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
                pi_socket = connect_to_pi(PI_IP, PI_PORT)

    threading.Thread(target=receiver_loop, daemon=True).start()

    # ---------- Helpers ----------
    def calculate_servo_movement(obj_center, frame_center):
        dx = frame_center[0] - obj_center[0]
        dy = obj_center[1] - frame_center[1]
        ndx = dx / frame_center[0] if abs(dx) > DEADZONE else 0.0
        ndy = dy / frame_center[1] if abs(dy) > DEADZONE else 0.0
        return ndx, ndy

    def send_servo_command(dx, dy):
        nonlocal pi_socket  # declare first
        cmd = {"type": P.TYPE_MOVE, "horizontal": dx * SERVO_SPEED, "vertical": dy * SERVO_SPEED}
        try:
            send_json(pi_socket, cmd)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # reconnect and retry once
            pi_socket = connect_to_pi(PI_IP, PI_PORT)
            send_json(pi_socket, cmd)

    def send_search_command(active: bool):
        nonlocal search_mode, pi_socket
        search_mode = bool(active)
        try:
            send_json(pi_socket, {'type': P.TYPE_SEARCH, 'active': search_mode})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(PI_IP, PI_PORT)
            send_json(pi_socket, {'type': P.TYPE_SEARCH, 'active': search_mode})

    def send_goto_pwms(pwm_btm: int, pwm_top: int, slew_ms: int = 600):
        nonlocal pi_socket
        payload = {"cmd": P.CMD_GOTO_PWMS, "pwm_btm": int(pwm_btm), "pwm_top": int(pwm_top), "slew_ms": int(slew_ms)}
        try:
            send_json(pi_socket, payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(PI_IP, PI_PORT)
            send_json(pi_socket, payload)

    # ---------- Main UI loop ----------
    try:
        while True:
            key = cv2.waitKey(1) & 0xFF

            # Toggle search
            if key == ord('s'):
                send_search_command(not search_mode)
                if search_mode:
                    print("Entering search mode: resetting memory session flags...")
                    object_memory.reset_session_flags()
                    if recall_mode:
                        recall_mode = False
                        print("[Recall] OFF (search started)")
                else:
                    print("Exiting search mode: pruning memory not updated this session...")
                    object_memory.prune_not_updated()
                print(f"{'Entering' if search_mode else 'Exiting'} search mode")

            # Toggle recall
            elif key == ord('m') and not search_mode:
                recall_mode = not recall_mode
                entries = object_memory.entries_sorted()
                if recall_mode:
                    if not entries:
                        print("[Recall] No objects in memory; turning OFF.")
                        recall_mode = False
                    else:
                        object_memory.print_table()
                        recall_index = 0
                        e = entries[recall_index]
                        print(f"[Recall] Selected -> {e['cls_name']} (id {e['cls_id']}), moving to stored pose...")
                        send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)
                else:
                    print("[Recall] OFF")

            # Cycle recall
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

            # Service pending scan request
            with scan_request_lock:
                pending_scan = scan_request
                scan_request = None

            if pending_scan is not None:
                duration_ms = int(pending_scan.get("duration_ms", 900))
                exclude_ids = pending_scan.get("exclude_cls", [])
                summary = run_scan_window(
                    cap, model, duration_s=duration_ms / 1000.0,
                    class_filter=resolved_class_id,
                    exclude_ids=exclude_ids,
                    display_scale=DISPLAY_SCALE,
                    get_name=get_name,
                )
                try:
                    send_json(pi_socket, { "type": P.TYPE_YOLO_RESULTS, **summary })
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pi_socket = connect_to_pi(PI_IP, PI_PORT)
                    send_json(pi_socket, { "type": P.TYPE_YOLO_RESULTS, **summary })
                continue

            # Service pending centering request
            with center_request_lock:
                pending_center = center_request
                center_request = None

            if pending_center is not None:
                target = int(pending_center["target_cls"])
                dur_s = float(pending_center["duration_ms"]) / 1000.0
                eps   = int(pending_center["epsilon_px"])
                label = f"CENTERING {get_name(target)} (id {target})"
                if DEBUG:
                    print(f"[Laptop] Centering on {get_name(target)} (id {target}) for {dur_s:.2f}s (debug on)")

                thresholds = {
                    "conf": 0.60,              # per-frame minimum
                    "move_norm_eps": 0.035,    # normalized motion stability
                    "required_frames": 12      # quota (not necessarily consecutive)
                }

                def move_cb(ndx, ndy):
                    send_servo_command(ndx, ndy)

                success, diag = center_on_class(
                    cap, model, dur_s, target, eps, center_x, center_y,
                    thresholds, move_cb, DISPLAY_SCALE, label, DEBUG
                )
                payload = {"type": P.TYPE_CENTER_DONE, "target_cls": int(target), "success": bool(success)}
                if DEBUG and diag is not None:
                    payload["diag"] = diag
                try:
                    send_json(pi_socket, payload)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pi_socket = connect_to_pi(PI_IP, PI_PORT)
                    send_json(pi_socket, payload)
                continue

            # Normal live view (and optional manual tracking outside search)
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame")
                break

            results = infer_all(model, frame, None if resolved_class_id == -1 else [resolved_class_id])
            for result in results:
                annotated = result.plot()
                h, w = annotated.shape[:2]
                resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                if recall_mode and not search_mode:
                    draw_wrapped_text(resized, "[RECALL MODE] m=exit, a=prev, d=next", 10, 24, int(resized.shape[1]*0.9))
                cv2.imshow("YOLO Detection", resized)

    finally:
        try:
            send_json(pi_socket, {'type': P.TYPE_STOP})
        except Exception:
            pass
        try:
            pi_socket.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
