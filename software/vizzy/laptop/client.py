from __future__ import annotations

import socket
import time
import cv2
import threading
from queue import Queue, Empty
from typing import Optional, Dict, Any

import torch
from ultralytics import YOLO

from ..shared import config as C
from ..shared import protocol as P
from ..shared.jsonl import recv_lines, send_json

from .memory import ObjectMemory
from .scanning import run_scan_window
from .centering import center_on_class
from .hud import draw_wrapped_text


# --------------------------------------------------------------------------------------
# Networking helpers
# --------------------------------------------------------------------------------------
def connect_to_pi(ip: str, port: int):
    """Connect to the RPi and return a blocking TCP socket."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.connect((ip, port))
    print(f"[Laptop] Connected to RPi at {ip}:{port}")
    return s


# --------------------------------------------------------------------------------------
# Main client
# --------------------------------------------------------------------------------------
def main():
    # ----------------------------- YOLO model initialization -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(C.YOLO_MODEL)

    NAMES = model.names  # class ID --> name mapping from the model

    def get_name(cid: int) -> str:
        try:
            if isinstance(NAMES, dict):
                return str(NAMES[int(cid)])
            else:
                # Some exports store names as a list
                return str(NAMES[int(cid)])
        except Exception:
            return str(cid)

    # ----------------------------- Camera ----------------------------------------------
    cap = cv2.VideoCapture(C.CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {C.CAM_INDEX}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(800))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(600))
    cap.set(cv2.CAP_PROP_FPS,          float(30.0))

    # Grab one frame to compute frame center and set a nice window size
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read a frame from the camera on startup.")
    h0, w0 = frame0.shape[:2]
    center_x, center_y = w0 // 2, h0 // 2

    # ----------------------------- State / Memory --------------------------------------
    object_memory = ObjectMemory(C.MEM_FILE)

    # UI state
    search_mode = False
    recall_mode = False
    recall_index = 0

    # ----------------------------- Networking ------------------------------------------
    pi_socket = connect_to_pi(C.PI_IP, C.PI_PORT)
    buf = b""

    # Minimal “mailboxes”
    pose_ready_q: "Queue[int]" = Queue(maxsize=8)  # pose_id queue
    pwms_event = threading.Event()
    pwms_payload: Dict[str, int] = {}  # {"pwm_btm": int, "pwm_top": int}
    sweep_completed_flag = threading.Event()

    # ----------------------------- Sender helpers --------------------------------------
    def send_scan_move(ndx: float, ndy: float):
        """Laptop → RPi: SCAN_MOVE with normalized deltas in [-1, 1]."""
        nonlocal pi_socket
        # Clamp to [-1, 1]
        dx = max(-1.0, min(1.0, float(ndx)))
        dy = max(-1.0, min(1.0, float(ndy)))
        payload = {"type": P.TYPE_SCAN_MOVE, "horizontal": dx, "vertical": dy}
        try:
            send_json(pi_socket, payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(C.PI_IP, C.PI_PORT)
            send_json(pi_socket, payload)

    def send_search(active: bool):
        """Laptop → RPi: toggle search mode."""
        nonlocal pi_socket, search_mode
        search_mode = bool(active)
        payload = {"type": P.TYPE_SEARCH, "active": search_mode}
        try:
            send_json(pi_socket, payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(C.PI_IP, C.PI_PORT)
            send_json(pi_socket, payload)

    def send_goto_pwms(pwm_btm: int, pwm_top: int, slew_ms: int = 600):
        """Laptop → RPi: absolute move for recall mode."""
        nonlocal pi_socket
        payload = {"cmd": P.CMD_GOTO_PWMS, "pwm_btm": int(pwm_btm), "pwm_top": int(pwm_top), "slew_ms": int(slew_ms)}
        try:
            send_json(pi_socket, payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(C.PI_IP, C.PI_PORT)
            send_json(pi_socket, payload)

    def request_pwms(timeout_s: float = 0.3) -> Optional[Dict[str, int]]:
        """Laptop → RPi: ask for current PWMs and wait briefly for TYPE_PWMS."""
        nonlocal pi_socket, pwms_payload
        pwms_event.clear()
        try:
            send_json(pi_socket, {"cmd": P.CMD_GET_PWMS})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(C.PI_IP, C.PI_PORT)
            send_json(pi_socket, {"cmd": P.CMD_GET_PWMS})
        ok = pwms_event.wait(timeout_s)
        return dict(pwms_payload) if ok else None

    # ----------------------------- Receiver thread -------------------------------------
    def receiver_loop():
        nonlocal pi_socket, buf, pwms_payload, search_mode
        while True:
            try:
                msgs, buf, closed = recv_lines(pi_socket, buf)
                if closed:
                    print("[Laptop] RPi closed connection; reconnecting...")
                    pi_socket = connect_to_pi(C.PI_IP, C.PI_PORT)
                    buf = b""
                    continue

                for msg in msgs:
                    mtype = msg.get("type")
                    mcmd = msg.get("cmd")

                    # RPi → Laptop: sweep has reached a new pose and is settled
                    if mtype == P.TYPE_POSE_READY:
                        pid = int(msg.get("pose_id", 0))
                        try:
                            pose_ready_q.put_nowait(pid)
                        except Exception:
                            # If queue is full, drop oldest to keep things moving
                            _ = None
                            try:
                                pose_ready_q.get_nowait()
                            except Empty:
                                pass
                            pose_ready_q.put_nowait(pid)

                    # RPi → Laptop: sweep completed by default stop
                    elif mtype == P.TYPE_SEARCH and (msg.get("active") is False):
                        # RPi finished the sweep (default stop)
                        search_mode = False           # <-- add this line
                        sweep_completed_flag.set()


                    # RPi → Laptop: current PWMs response
                    elif mtype == P.TYPE_PWMS:
                        try:
                            pwms_payload.clear()
                            pwms_payload.update({"pwm_btm": int(msg["pwm_btm"]), "pwm_top": int(msg["pwm_top"])})
                            pwms_event.set()
                        except Exception:
                            # Ignore malformed
                            pass

            except (BrokenPipeError, ConnectionResetError, OSError):
                print("[Laptop] Socket error; reconnecting...")
                pi_socket = connect_to_pi(C.PI_IP, C.PI_PORT)
                buf = b""

    threading.Thread(target=receiver_loop, daemon=True).start()

    # ----------------------------- Pose processing helper ------------------------------
    def process_pose(pose_id: int):
        """Run scan → choose → center → (if success) GET_PWMS → update memory → POSE_DONE."""
        # Exclude classes already updated this session
        exclude_ids = [int(e["cls_id"]) for e in object_memory.entries_sorted() if e.get("updated_this_session") == 1]

        summary = run_scan_window(
            cap=cap,
            model=model,
            exclude_ids=exclude_ids,
            get_name=get_name,
            min_frames_for_class=C.SCAN_MIN_FRAMES,
        )

        # Pick best candidate by avg_conf, then frames; enforce scan gates
        objs = summary.get("objects", [])
        target = None
        for o in objs:
            if o.get("avg_conf", 0.0) >= C.SCAN_MIN_CONF and int(o.get("frames", 0)) >= int(C.SCAN_MIN_FRAMES):
                target = o
                break

        if target is None:
            # Nothing worth centering at this pose
            try:
                send_json(pi_socket, {"type": P.TYPE_POSE_DONE, "pose_id": int(pose_id), "status": "SKIP"})
            except (BrokenPipeError, ConnectionResetError, OSError):
                # try reconnect & resend once
                s = connect_to_pi(C.PI_IP, C.PI_PORT)
                send_json(s, {"type": P.TYPE_POSE_DONE, "pose_id": int(pose_id), "status": "SKIP"})
            return

        cls_id = int(target["cls_id"])
        label = f"CENTER {target.get('cls_name', get_name(cls_id))} (id {cls_id})"

        # Centering loop (drives SCAN_MOVE bursts)
        success_tuple = center_on_class(
            cap=cap,
            model=model,
            target_cls=cls_id,
            center_x=center_x,
            center_y=center_y,
            send_move=send_scan_move,
            display_scale=C.DISPLAY_SCALE,
            label=label,
        )
        # center_on_class may return (bool) or (bool, diag)
        success = success_tuple if isinstance(success_tuple, bool) else bool(success_tuple[0])

        if success:
            # While still centered, query current PWMs from RPi, then update memory locally
            pwms = request_pwms(timeout_s=0.3)
            if pwms is not None and "pwm_btm" in pwms and "pwm_top" in pwms:
                object_memory.update_entry(
                    cls_id=cls_id,
                    cls_name=target.get("cls_name", get_name(cls_id)),
                    pwm_btm=int(pwms["pwm_btm"]),
                    pwm_top=int(pwms["pwm_top"]),
                )
                print(f"[Memory] Updated: {get_name(cls_id)} -> btm={pwms['pwm_btm']}, top={pwms['pwm_top']}")
            else:
                print("[Memory] Warning: did not receive PWMS in time; not updating pose for this object.")

            # Tell RPi we’re done with this pose
            try:
                send_json(pi_socket, {"type": P.TYPE_POSE_DONE, "pose_id": int(pose_id), "status": "SUCCESS"})
            except (BrokenPipeError, ConnectionResetError, OSError):
                s = connect_to_pi(C.PI_IP, C.PI_PORT)
                send_json(s, {"type": P.TYPE_POSE_DONE, "pose_id": int(pose_id), "status": "SUCCESS"})
        else:
            try:
                send_json(pi_socket, {"type": P.TYPE_POSE_DONE, "pose_id": int(pose_id), "status": "FAIL"})
            except (BrokenPipeError, ConnectionResetError, OSError):
                s = connect_to_pi(C.PI_IP, C.PI_PORT)
                send_json(s, {"type": P.TYPE_POSE_DONE, "pose_id": int(pose_id), "status": "FAIL"})

    # ----------------------------- UI / Main loop --------------------------------------
    try:
        while True:
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            # Toggle search mode (s)
            if key == ord("s"):
                search_mode = not search_mode
                if search_mode:
                    print("[Search] ON")
                    object_memory.reset_session_flags()
                else:
                    print("[Search] OFF (interrupt)")
                    object_memory.prune_not_updated()
                send_search(search_mode)

            # Toggle recall mode (m)
            elif key == ord("m") and not search_mode:
                recall_mode = not recall_mode
                if recall_mode:
                    entries = object_memory.entries_sorted()
                    if entries:
                        recall_index = 0
                        e = entries[recall_index]
                        print(f"[Recall] Selected -> {e['cls_name']} (id {e['cls_id']}), moving to stored pose...")
                        send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)
                else:
                    print("[Recall] OFF")

            # Recall left (a)
            elif recall_mode and key == ord("a") and not search_mode:
                entries = object_memory.entries_sorted()
                if entries:
                    recall_index = (recall_index - 1) % len(entries)
                    e = entries[recall_index]
                    print(f"[Recall] {e['cls_name']} (id {e['cls_id']}) <- moving")
                    send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)

            # Recall right (d)
            elif recall_mode and key == ord("d") and not search_mode:
                entries = object_memory.entries_sorted()
                if entries:
                    recall_index = (recall_index + 1) % len(entries)
                    e = entries[recall_index]
                    print(f"[Recall] {e['cls_name']} (id {e['cls_id']}) -> moving")
                    send_goto_pwms(e["pwm_btm"], e["pwm_top"], slew_ms=600)

            # Quit (q)
            elif key == ord("q"):
                break

            # If RPi reports sweep completion, prune memory and clear flag
            if sweep_completed_flag.is_set():
                object_memory.prune_not_updated()
                sweep_completed_flag.clear()
                print("[Search] Completed (pruned memory entries not updated this session)")

            # Process any ready poses
            try:
                pose_id = pose_ready_q.get_nowait()
            except Empty:
                pose_id = None

            if pose_id is not None:
                process_pose(int(pose_id))
                continue  # Next UI tick

            # Live preview when idle
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame")
                break
            results = model(frame)
            for result in results:
                annotated = result.plot()
                h, w = annotated.shape[:2]
                resized = cv2.resize(annotated, (int(w * C.DISPLAY_SCALE), int(h * C.DISPLAY_SCALE)))
                if recall_mode and not search_mode:
                    draw_wrapped_text(resized, "RECALL MODE", 12, 12, int(resized.shape[1] * 0.6))
                cv2.imshow("YOLO Detection", resized)

    finally:
        # Try to stop gracefully
        try:
            send_json(pi_socket, {"type": P.TYPE_STOP})
        except Exception:
            pass
        try:
            pi_socket.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
