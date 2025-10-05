from . import config
import socket, time, cv2, threading
from ultralytics import YOLO
import torch
from .memory import ObjectMemory
from typing import Optional
from ..shared.jsonl import recv_lines, send_json
from ..shared import protocol as P


# TODO move this?
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
    # ----------------------------- YOLO model initialization -----------------------------
    # Choose GPU (cuda) if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the specified YOLO model
    model = YOLO(config.YOLO_MODEL)

    NAMES = model.names  # class ID --> name mapping from the model
    def get_name(cid: int) -> str:
        """Return human-readable label for a class ID using model.names."""
        try:
            if isinstance(NAMES, dict):
                return str(NAMES[int(cid)])
            else:
                return str(NAMES[int(cid)])
        except Exception:
            return str(cid)


    # ------------------------------- Camera setup ----------------------------------
    # Open video device using V4L2 backend on Linux
    cap = cv2.VideoCapture(config.CAM_INDEX, cv2.CAP_V4L2)

    # Configure image type, resolution, and FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(800))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(600))
    cap.set(cv2.CAP_PROP_FPS,          float(30.0))

    # Ensure camera connection and interfacing is successful
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    ok, frame0 = cap.read()
    if not ok:
        print("Error: Could not read frame from camera")
        return
    
    # DEBUG -------------------------------
    frame_h, frame_w = frame0.shape[:2]
    print(f"video width: {frame_w}, video height: {frame_h}")
    # -------------------------------

    # Store image frame center for object centering later on
    center_x, center_y = frame_w // 2, frame_h // 2     # pixels


    # ------------------------------ Networking -------------------------------
    # Connect to the RPi server and store the socket
    pi_socket = connect_to_pi(config.PI_IP, config.PI_PORT)


    # ------------------------------ Shared state -----------------------------
    # TODO: we need to abstract the whole search mode to its own thread/function; 
    # other modes will include idle (waiting for user input), and task execution
    # UI / mode flags:
    search_mode = False         # true when the RPi is sweeping the grid
    recall_mode = False         # true when user is navigating stored objects
    recall_index = 0            # index into the memory table when recalling

    # Initialize memory of objects in the arm's workspace
    object_memory = ObjectMemory(config.MEM_FILE)

    # Small, thread-safe one-slot "inboxes" for requests received from the RPi:
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
                # Track any full JSON messages parsed from buffer
                msgs, buf, closed = recv_lines(pi_socket, buf)
                # If disconnected, attempt reconnection
                if closed:
                    print("[Laptop] Connection closed by RPi; reconnecting...")
                    pi_socket = connect_to_pi(config.PI_IP, config.PI_PORT)
                    buf = b""
                    continue

                # Process each discrete JSON message
                # TODO: this command should stay, but might need modification; it should trigger a full 
                # scan of the current search path position - including object centering
                for msg in msgs:
                    # RPi --> Laptop: request to run a YOLO scan window
                    if msg.get("cmd") == P.CMD_YOLO_SCAN:
                        with scan_request_lock:
                            scan_request = {
                                "duration_ms": int(msg.get("duration_ms", 900)),
                                "exclude_cls": msg.get("exclude_cls", []),
                            }

                    # RPi --> Laptop: request to center on a given class
                    # TODO: not sure why RPi has control over this; laptop should have all tools necessary 
                    # to do this on its own. Can hopefully remove this entirely
                    elif msg.get("cmd") == P.CMD_CENTER_ON:
                        with center_request_lock:
                            center_request = {
                                "target_cls": int(msg.get("target_cls", -1)),
                                "duration_ms": int(msg.get("duration_ms", 1200)),
                                "epsilon_px": int(msg.get("epsilon_px", 25)),
                                "target_name": msg.get("target_name", ""),
                            }

                    # RPi --> Laptop: snapshot of verified centering (store in memory)
                    # TODO: this should be changed to store XYZ position of the identified object; should 
                    # probably also trigger a calculation for grasping angle
                    elif msg.get("type") == P.TYPE_CENTER_SNAPSHOT:
                        cls_id  = int(msg.get("cls_id"))
                        cls_name= str(msg.get("cls_name", ""))
                        pwm_btm = int(msg.get("pwm_btm"))
                        pwm_top = int(msg.get("pwm_top"))
                        diag    = msg.get("diag")   #TODO remove

                        # Try to capture a representative confidence to save with the entry
                        # TODO don't really need these diagnostics
                        avg_conf = None
                        if isinstance(diag, dict):
                            obs = diag.get("observed") or {}
                            if "max_conf_seen" in obs:
                                avg_conf = float(obs["max_conf_seen"])
                        #TODO keep this, but remove confidence; eventually change pwm to pos + orientation
                        object_memory.update_entry(
                            cls_id, cls_name or str(cls_id), pwm_btm, pwm_top, avg_conf=avg_conf
                        )
                        print(f"[Laptop] Memory updated: {cls_name or cls_id} -> btm={pwm_btm}, top={pwm_top}")

            # Attempt reconnecting when errors occur
            except (ConnectionResetError, BrokenPipeError, OSError):
                print("[Laptop] Socket error; reconnecting...")
                pi_socket = connect_to_pi(config.PI_IP, config.PI_PORT)

    # Start the receiver thread (as daemon so it won’t block exit)
    threading.Thread(target=receiver_loop, daemon=True).start()


    # ------------------------------ Command Helpers ----------------------------------
    # TODO: this should be swapped out for a generic relative move command that requires the 
    # arm to keep the camera parallel to the ground while moving left/right/forward/back
    def send_servo_command(dx, dy):
        """
        Send a relative move command to the RPi.
        """
        nonlocal pi_socket  # declare before use so Python knows it’s outer-scope
        cmd = {"type": P.TYPE_MOVE, "horizontal": dx * config.SERVO_SPEED, "vertical": dy * config.SERVO_SPEED} # TODO move SERVO+SPEED to RPi side of things
        try:
            send_json(pi_socket, cmd)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pi_socket = connect_to_pi(config.PI_IP, config.PI_PORT)
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
            pi_socket = connect_to_pi(config.PI_IP, config.PI_PORT)
            send_json(pi_socket, {'type': P.TYPE_SEARCH, 'active': search_mode})

    # TODO: This isn't needed for searching; something similar will be used by the LLM later tho
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
            pi_socket = connect_to_pi(config.PI_IP, config.PI_PORT)
            send_json(pi_socket, payload)
