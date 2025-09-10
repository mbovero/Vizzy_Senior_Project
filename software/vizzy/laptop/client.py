# -----------------------------------------------------------------------------
# vizzy/laptop/client.py
#
# Purpose:
#   Main laptop-side controller for Vizzy. This process:
#     • Loads YOLO and reads frames from a local camera.
#     • Connects to the Raspberry Pi (RPi) over TCP (JSON line protocol).
#     • Responds to RPi commands:
#         - YOLO scan window at a stationary search pose.
#         - Center on a requested class (driving servos via RPi).
#         - Update local “object memory” when centering is verified.
#     • Provides a small keyboard UI for:
#         - Toggling Search Mode (s).
#         - Toggling Memory Recall Mode (m) and cycling stored objects (a/d).
#         - Quit (q).
#
# High-level flow:
#   1) Boot: parse CLI → init YOLO → open camera → connect to RPi.
#   2) Start a receiver thread that continuously reads messages from RPi and
#      fills small “mailboxes” (scan_request, center_request).
#   3) Main loop:
#        - Handle keyboard input.
#        - If a scan request arrives, run a stationary scan and send results.
#        - If a center request arrives, run centering loop and send outcome.
#        - Otherwise, keep the live annotated YOLO view responsive.
#
# Notes:
#   - Networking uses newline-delimited JSON (see vizzy/shared/jsonl.py).
#   - Protocol constants live in vizzy/shared/protocol.py.
#   - Memory of verified objects (PWM pose per class) is persisted to disk.
# -----------------------------------------------------------------------------

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
    """
    Persistent connect helper:
      - Tries to connect to the RPi TCP server.
      - If it fails, waits 2s and retries forever.
      - Disables Nagles algorithm (TCP_NODELAY) for low-latency small packets.
    Returns:
      A connected socket object.
    """
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
    # ----------------------------- CLI & model init --------------------------
    args = make_arg_parser().parse_args()
    DEBUG = bool(args.debug)
    DISPLAY_SCALE = float(args.display_scale)

    # Load YOLO engine and quiet logs unless --debug set.
    model = init_model(args.engine, DEBUG)
    NAMES = model.names  # authoritative class ID → name mapping from the model

    def get_name(cid: int) -> str:
        """Return human-readable label for a class ID using model.names."""
        try:
            if isinstance(NAMES, dict):
                return str(NAMES[int(cid)])
            else:
                return str(NAMES[int(cid)])
        except Exception:
            return str(cid)

    # Resolve user’s target class (by name or id). -1 means detect ALL classes.
    resolved_class_id = resolve_class_id_from_name(NAMES, args.class_id, args.class_name)
    print(f"Tracking: {'ALL CLASSES' if resolved_class_id == -1 else f'{get_name(resolved_class_id)} (id {resolved_class_id})'}")

    # ------------------------------- Camera ----------------------------------
    # Open the video device (uses V4L2 backend on Linux). Immediately test
    # a frame so we can report failures early.
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    # Request MJPG (smaller USB payload than raw YUYV)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(800))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(600))
    cap.set(cv2.CAP_PROP_FPS,          float(30.0))

    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    ok, frame0 = cap.read()
    if not ok:
        print("Error: Could not read frame from camera")
        return
    frame_h, frame_w = frame0.shape[:2]
    print(f"video width: {frame_w}, video height: {frame_h}")
    center_x, center_y = frame_w // 2, frame_h // 2  # frame center in pixels

    # ------------------------------ Networking -------------------------------
    # Connect to the RPi server and keep the socket around for the duration.
    PI_IP, PI_PORT = args.ip, args.port
    pi_socket = connect_to_pi(PI_IP, PI_PORT)

    # ------------------------------ Shared state -----------------------------
    # Servo motion scaling and deadzone for small errors.
    SERVO_SPEED = 0.2
    DEADZONE = 30

    # UI / mode flags:
    search_mode = False         # true when the RPi is sweeping the grid
    recall_mode = False         # true when user is navigating stored objects
    recall_index = 0            # index into the memory table when recalling

    # On-disk memory of verified objects (PWM pose per class).
    object_memory = ObjectMemory(args.mem_file)

    # Small, thread-safe "mailboxes" for requests received from the RPi:
    scan_request_lock = threading.Lock()
    scan_request: Optional[dict] = None

    center_request_lock = threading.Lock()
    center_request: Optional[dict] = None

    # ----------------------------- Receiver thread ---------------------------
    def receiver_loop():
        """
        Background loop that:
          - Reads newline-delimited JSON from the RPi socket.
          - For command messages, writes into the “mailboxes” protected by locks:
              • scan_request: asks us to run a scan window now.
              • center_request: asks us to center on a class now.
          - For snapshot messages, updates local object memory.
        If the socket closes/errors, it reconnects and continues.
        """
        nonlocal pi_socket, scan_request, center_request
        buf = b""
        while True:
            try:
                # recv_lines() returns any full JSON messages parsed from buf
                msgs, buf, closed = recv_lines(pi_socket, buf)
                if closed:
                    print("[Laptop] Connection closed by RPi; reconnecting...")
                    pi_socket = connect_to_pi(PI_IP, PI_PORT)
                    buf = b""
                    continue

                # Process each discrete JSON message
                for msg in msgs:
                    # RPi → Laptop: request to run a YOLO scan window
                    if msg.get("cmd") == P.CMD_YOLO_SCAN:
                        with scan_request_lock:
                            scan_request = {
                                "duration_ms": int(msg.get("duration_ms", 900)),
                                "exclude_cls": msg.get("exclude_cls", []),
                            }

                    # RPi → Laptop: request to center on a given class
                    elif msg.get("cmd") == P.CMD_CENTER_ON:
                        with center_request_lock:
                            center_request = {
                                "target_cls": int(msg.get("target_cls", -1)),
                                "duration_ms": int(msg.get("duration_ms", 1200)),
                                "epsilon_px": int(msg.get("epsilon_px", 25)),
                                "target_name": msg.get("target_name", ""),
                            }

                    # RPi → Laptop: snapshot of verified centering (store in memory)
                    elif msg.get("type") == P.TYPE_CENTER_SNAPSHOT:
                        cls_id  = int(msg.get("cls_id"))
                        cls_name= str(msg.get("cls_name", ""))
                        pwm_btm = int(msg.get("pwm_btm"))
                        pwm_top = int(msg.get("pwm_top"))
                        diag    = msg.get("diag")

                        # Try to capture a representative confidence to save with the entry
                        avg_conf = None
                        if isinstance(diag, dict):
                            obs = diag.get("observed") or {}
                            if "max_conf_seen" in obs:
                                avg_conf = float(obs["max_conf_seen"])

                        object_memory.update_entry(
                            cls_id, cls_name or str(cls_id), pwm_btm, pwm_top, avg_conf=avg_conf
                        )
                        print(f"[Laptop] Memory updated: {cls_name or cls_id} -> btm={pwm_btm}, top={pwm_top}")

            except (ConnectionResetError, BrokenPipeError, OSError):
                print("[Laptop] Socket error; reconnecting...")
                pi_socket = connect_to_pi(PI_IP, PI_PORT)

    # Start the receiver thread (daemon so it won’t block exit)
    threading.Thread(target=receiver_loop, daemon=True).start()

    # ------------------------------ Helpers ----------------------------------
    def calculate_servo_movement(obj_center, frame_center):
        """
        Convert pixel error (object center vs frame center) to normalized
        motion commands for the servos. Small errors inside DEADZONE → 0.
        Returns:
          (ndx, ndy) normalized deltas in range roughly [-1..1].
        """
        dx = frame_center[0] - obj_center[0]  # >0 means object left of center
        dy = obj_center[1] - frame_center[1]  # >0 means object below center
        ndx = dx / frame_center[0] if abs(dx) > DEADZONE else 0.0
        ndy = dy / frame_center[1] if abs(dy) > DEADZONE else 0.0
        return ndx, ndy

    def send_servo_command(dx, dy):
        """
        Send a relative move command to the RPi (which actually drives servos).
        If the link drops, reconnect and retry once.
        """
        nonlocal pi_socket  # declare before use so Python knows it’s outer-scope
        cmd = {"type": P.TYPE_MOVE, "horizontal": dx * SERVO_SPEED, "vertical": dy * SERVO_SPEED}
        try:
            send_json(pi_socket, cmd)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(PI_IP, PI_PORT)
            send_json(pi_socket, cmd)

    def send_search_command(active: bool):
        """
        Toggle Search Mode on the RPi. We mirror the state locally for UI.
        """
        nonlocal search_mode, pi_socket
        search_mode = bool(active)
        try:
            send_json(pi_socket, {'type': P.TYPE_SEARCH, 'active': search_mode})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(PI_IP, PI_PORT)
            send_json(pi_socket, {'type': P.TYPE_SEARCH, 'active': search_mode})

    def send_goto_pwms(pwm_btm: int, pwm_top: int, slew_ms: int = 600):
        """
        Ask the RPi to slew both servos to absolute PWM values smoothly.
        Used by Recall Mode to move to stored object poses.
        """
        nonlocal pi_socket
        payload = {"cmd": P.CMD_GOTO_PWMS, "pwm_btm": int(pwm_btm), "pwm_top": int(pwm_top), "slew_ms": int(slew_ms)}
        try:
            send_json(pi_socket, payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(PI_IP, PI_PORT)
            send_json(pi_socket, payload)

    # ------------------------------ Main UI loop ------------------------------
    try:
        while True:
            # Poll for keyboard input at ~1kHz via OpenCV’s event loop.
            key = cv2.waitKey(1) & 0xFF

            # ---- Toggle Search Mode (s) ----
            if key == ord('s'):
                # Flip search flag and notify RPi
                send_search_command(not search_mode)

                if search_mode:
                    # We just turned ON search mode on the RPi.
                    # Reset session flags so freshly centered objects are marked updated.
                    print("Entering search mode: resetting memory session flags...")
                    object_memory.reset_session_flags()

                    # If user was in Recall Mode, switch it off (search takes over).
                    if recall_mode:
                        recall_mode = False
                        print("[Recall] OFF (search started)")
                else:
                    # We just turned OFF search mode; prune memory entries that
                    # were never updated during the session.
                    print("Exiting search mode: pruning memory not updated this session...")
                    object_memory.prune_not_updated()

                print(f"{'Entering' if search_mode else 'Exiting'} search mode")

            # ---- Toggle Recall Mode (m) – only when NOT in search mode ----
            elif key == ord('m') and not search_mode:
                recall_mode = not recall_mode
                entries = object_memory.entries_sorted()

                if recall_mode:
                    if not entries:
                        # No saved objects → nothing to recall; turn mode back off.
                        print("[Recall] No objects in memory; turning OFF.")
                        recall_mode = False
                    else:
                        # Print a table of stored objects for context
                        object_memory.print_table()

                        # Select first entry and ask RPi to move there
                        recall_index = 0
                        e = entries[recall_index]
                        print(f"[Recall] Selected -> {e['cls_name']} (id {e['cls_id']}), moving to stored pose...")
                        send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)
                else:
                    print("[Recall] OFF")

            # ---- Recall Mode: cycle left (a) ----
            elif recall_mode and key == ord('a') and not search_mode:
                entries = object_memory.entries_sorted()
                if entries:
                    recall_index = (recall_index - 1) % len(entries)
                    e = entries[recall_index]
                    print(f"[Recall] {e['cls_name']} (id {e['cls_id']}) <- moving")
                    send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)

            # ---- Recall Mode: cycle right (d) ----
            elif recall_mode and key == ord('d') and not search_mode:
                entries = object_memory.entries_sorted()
                if entries:
                    recall_index = (recall_index + 1) % len(entries)
                    e = entries[recall_index]
                    print(f"[Recall] {e['cls_name']} (id {e['cls_id']}) -> moving")
                    send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)

            # ---- Quit (q) ----
            elif key == ord('q'):
                break

            # ------------------- Service pending scan request -------------------
            # If the RPi asked us (via receiver thread) to scan at the current
            # stationary pose, process it now and send back summarized results.
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
                continue  # Go back to top of loop after handling request

            # ----------------- Service pending centering request ----------------
            # If the RPi asked us to center on a class, run the centering loop
            # (which drives servos via move_cb → TYPE_MOVE messages) and then
            # report whether verification succeeded.
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

                # Thresholds used by centering verification (quota-based).
                thresholds = {
                    "conf": 0.60,              # per-frame minimum confidence
                    "move_norm_eps": 0.035,    # normalized motion stability
                    "required_frames": 12      # number of “good” frames (not necessarily consecutive)
                }

                # Callback that the centering loop uses to request relative servo moves.
                def move_cb(ndx, ndy):
                    send_servo_command(ndx, ndy)

                # Run the centering attempt; this keeps the UI responsive and
                # sends TYPE_MOVE messages to the RPi while tracking.
                success, diag = center_on_class(
                    cap, model, dur_s, target, eps, center_x, center_y,
                    thresholds, move_cb, DISPLAY_SCALE, label, DEBUG
                )

                # Report completion to the RPi; include diagnostics only if DEBUG.
                payload = {"type": P.TYPE_CENTER_DONE, "target_cls": int(target), "success": bool(success)}
                if DEBUG and diag is not None:
                    payload["diag"] = diag
                try:
                    send_json(pi_socket, payload)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pi_socket = connect_to_pi(PI_IP, PI_PORT)
                    send_json(pi_socket, payload)
                continue  # Done servicing centering; return to event loop

            # ----------------------- Idle: live annotated view -------------------
            # No pending requests → keep the UI responsive by running YOLO on
            # live frames (optionally filtered to a single class).
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame")
                break

            results = infer_all(model, frame, None if resolved_class_id == -1 else [resolved_class_id])
            for result in results:
                annotated = result.plot()
                h, w = annotated.shape[:2]
                resized = cv2.resize(annotated, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                # Show a small hint when recall mode is active
                if recall_mode and not search_mode:
                    draw_wrapped_text(resized, "[RECALL MODE] m=exit, a=prev, d=next", 10, 24, int(resized.shape[1]*0.9))
                cv2.imshow("YOLO Detection", resized)

    finally:
        # On exit, try to tell the RPi we’re stopping; then clean up resources.
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
