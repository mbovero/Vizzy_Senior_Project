# vizzy/laptop/app.py
# -----------------------------------------------------------------------------
# StateManager: laptop-side orchestrator (IDLE preview, SEARCH cycles, TaskAgent).
# Threads used:
#   - ReceiverThread (existing in client.py; we’ll reuse its mailbox logic)
#   - ScanWorker (new; started on start_search())
#   - TaskAgent (new; waits for user request -> plan -> execute -> back to IDLE)
#
# Display policy:
#   - IDLE: plain YOLO preview (no HUD)
#   - SEARCH: ScanWorker generates annotated frames; main thread displays them
# -----------------------------------------------------------------------------

from __future__ import annotations

import threading
import time
from queue import Empty
from typing import Optional, Literal

import cv2
import torch
from ultralytics import YOLO

from ..shared import config as C
from ..shared.jsonl import recv_lines, send_json
from ..shared import protocol as P

# NEW: frame bus for cross-thread rendering
from .display_bus import FrameBus
from .yolo_runner import build_allowed_class_ids

# NEW: LLM worker manager for semantic enrichment
from .llm_worker import WorkerManager
from .memory import ObjectMemory

StateName = Literal["IDLE", "SEARCH", "PROCESS_QUERY", "EXECUTE_TASK", "INTERRUPT"]


class Events:
    """Coordination primitives shared across threads."""
    def __init__(self):
        self.search_requested = threading.Event()
        self.scan_active = threading.Event()
        self.scan_abort = threading.Event()       # reserved for future interrupt
        self.scan_finished = threading.Event()
        self.query_ready = threading.Event()

# Used by scan cycle and execution
class Mailboxes:
    """Network mailboxes the ReceiverThread fills."""
    def __init__(self):
        from queue import Queue
        self.cmd_complete_q: "Queue[dict]" = Queue(maxsize=8)  # Receives CMD_COMPLETE messages
        self.obj_loc_event = threading.Event()
        self.obj_loc_payload: dict = {}


class StateManager:
    def __init__(self):         
        # State
        self.state: StateName = "IDLE"
        self._prev_state: Optional[StateName] = None
        self.events = Events()
        self.mail = Mailboxes()
        self.idle_deadline: float = time.time() + C.IDLE_TIMEOUT_S
        self.search_mode: bool = False 

        # IO / vision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(C.YOLO_MODEL)
        self.cap = cv2.VideoCapture(C.CAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {C.CAM_INDEX}")

        # TODO add in USB cam config

        ok, frame0 = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError("Failed to read an initial frame from the camera.")
        self.h0, self.w0 = frame0.shape[:2]
        self.allowed_class_ids = build_allowed_class_ids(
            self.model.names,
            C.OBJ_BLACKLIST,
        )

        # Networking
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            print(f"[StateManager] Connecting to RPi server at {C.PI_IP}:{C.PI_PORT}...")
            self.sock.connect((C.PI_IP, C.PI_PORT))
            print(f"[StateManager] Connected to RPi server successfully")
        except Exception as e:
            print(f"[StateManager] ERROR: Failed to connect to RPi server: {e}")
            raise

        # Thread handles (filled later)
        self.receiver_thread: Optional[threading.Thread] = None     # Networking
        self.scan_worker: Optional[threading.Thread] = None         # Scanning thread
        self.task_agent: Optional[threading.Thread] = None          # User query -> task scheduler -> execution

        # Motion facade & camera manager (motion is created lazily)
        self.motion = None

        # NEW: Frame bus for worker->main display handoff
        # Larger buffer for smoother GUI (reduces frame drops)
        self.frame_bus = FrameBus(maxsize=8)

        # NEW: Shared memory and LLM worker manager
        print("[StateManager] Initializing object memory...")
        self.memory = ObjectMemory(C.MEM_FILE)
        print("[StateManager] Initializing LLM worker manager...")
        self.llm_worker = WorkerManager(
            memory=self.memory,
            max_workers=C.LLM_WORKERS,
            model=C.IMAGE_PROCESS_MODEL,
        )
        print("[StateManager] Memory and LLM components initialized")

    # ------------------------------- Triggers ---------------------------------

    def request_search(self, reason: str = "idle") -> None:
        """Public trigger: ask to start a search cycle (debounced by start_search())."""
        self.events.search_requested.set()

    def start_search(self) -> None:
        """
        Spawn a ScanWorker and set scan_active if not already scanning.
        (Spawns the worker, wires in Motion, and flips scan_active.)
        """
        # Don't start scan worker if we're skipping scan cycle
        if C.SKIP_TO_TASK_SCHEDULER or C.SKIP_SCAN_CYCLE:
            print("[StateManager] Scan cycle skipped, scan worker will not run")
            return
            
        if self.events.scan_active.is_set():
            print("[StateManager] Search already active, skipping")
            return
        
        print("[StateManager] Starting search cycle...")
        self.events.search_requested.clear()
        self.events.scan_finished.clear()
        self.events.scan_abort.clear()

        # Lazy imports to avoid circular refs until files exist
        from .scan_worker import ScanWorker
        from .motion import Motion

        # Ensure we have a Motion façade
        if self.motion is None:
            print("[StateManager] Creating Motion facade...")
            self.motion = Motion(
                self.sock,
                self.mail.cmd_complete_q,
                self.mail.obj_loc_event,
                self.mail.obj_loc_payload,
                abort_event=self.events.scan_abort,
            )

        # Hand off the camera to the worker directly; worker pushes frames to frame_bus
        print("[StateManager] Creating ScanWorker thread...")
        worker = ScanWorker(
            sock=self.sock,
            cap=self.cap,
            model=self.model,
            mail=self.mail,
            events=self.events,
            motion=self.motion,
            display_scale=C.DISPLAY_SCALE,
            frame_sink=self.frame_bus.publish,   # NEW: push frames to main thread
            llm_worker=self.llm_worker,          # NEW: LLM worker pool for enrichment
            memory=self.memory,                  # NEW: Share same memory instance!
            allowed_class_ids=self.allowed_class_ids,
        )
        worker.daemon = True
        print("[StateManager] Starting ScanWorker thread...")
        worker.start()
        self.scan_worker = worker
        self.events.scan_active.set()
        self._switch_state("SEARCH")
        print("[StateManager] Entered SEARCH state, ScanWorker is running")

    def finish_search(self) -> None:
        """Handle end-of-grid or manual stop; return to IDLE and reset idle timer."""
        self.events.scan_active.clear()
        self.events.scan_finished.clear()
        self._switch_state("IDLE")
        self.idle_deadline = time.time() + C.IDLE_TIMEOUT_S

    # --------------------------- Receiver thread ------------------------------

    def _receiver_loop(self) -> None:
        """
        Continuously read text responses from rpi_control_server.
        Server sends text lines like "ACK ik\n" or "ERR ...\n"
        Also handles JSON protocol for backward compatibility.
        """
        import socket
        buf = b""
        from queue import Empty
        print("[Receiver] Receiver thread started, waiting for server responses...")

        while True:
            try:
                # Try to read data (text protocol - newline-delimited)
                # This will block until data is available, which is fine
                try:
                    data = self.sock.recv(4096)
                    if not data:
                        # Connection closed
                        print("[Receiver] Connection closed by server, attempting reconnect...")
                        raise ConnectionError("Connection closed by server")
                except (ConnectionError, OSError, socket.error) as e:
                    # Attempt a blocking reconnect
                    print(f"[Receiver] Connection error: {e}, attempting reconnect...")
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                    try:
                        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        self.sock.connect((C.PI_IP, C.PI_PORT))
                        print(f"[Receiver] Reconnected to server at {C.PI_IP}:{C.PI_PORT}")
                        buf = b""
                        continue
                    except Exception as reconnect_err:
                        print(f"[Receiver] Reconnect failed: {reconnect_err}, retrying in 1s...")
                        time.sleep(1.0)
                        continue

                # Accumulate data in buffer
                buf += data
                
                # Process complete lines
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if not line_str:
                        continue
                    
                    # Parse text protocol responses from rpi_control_server
                    if line_str.startswith("ACK"):
                        # Format: "ACK ik" or "ACK rest"
                        print(f"[Receiver] Received: {line_str}")
                        # Put string message in queue for motion.py to handle
                        try:
                            self.mail.cmd_complete_q.put_nowait(line_str)
                        except Exception:
                            # Queue full, drop oldest and try again
                            try:
                                self.mail.cmd_complete_q.get_nowait()
                            except Empty:
                                pass
                            try:
                                self.mail.cmd_complete_q.put_nowait(line_str)
                            except Exception:
                                pass
                    elif line_str.startswith("ERR"):
                        # Format: "ERR ..."
                        print(f"[Receiver] Received ERROR: {line_str}")
                        # Put error message in queue
                        try:
                            self.mail.cmd_complete_q.put_nowait(line_str)
                        except Exception:
                            try:
                                self.mail.cmd_complete_q.get_nowait()
                            except Empty:
                                pass
                            try:
                                self.mail.cmd_complete_q.put_nowait(line_str)
                            except Exception:
                                pass
                    else:
                        # Try to parse as JSON (backward compatibility)
                        try:
                            import json
                            msg = json.loads(line_str)
                            mtype = msg.get("type")
                            mcmd = msg.get("cmd")

                            if mtype == P.TYPE_CMD_COMPLETE:
                                try:
                                    self.mail.cmd_complete_q.put_nowait(msg)
                                except Exception:
                                    try:
                                        self.mail.cmd_complete_q.get_nowait()
                                    except Empty:
                                        pass
                                    self.mail.cmd_complete_q.put_nowait(msg)

                            elif mtype == P.TYPE_OBJ_LOC:
                                try:
                                    self.mail.obj_loc_payload.clear()
                                    self.mail.obj_loc_payload.update({
                                        "x": float(msg["x"]),
                                        "y": float(msg["y"]),
                                        "z": float(msg["z"]),
                                    })
                                    self.mail.obj_loc_event.set()
                                except Exception:
                                    pass
                        except (json.JSONDecodeError, ValueError):
                            # Not JSON, ignore
                            pass

            except Exception as e:
                # Reconnect path (simple)
                print(f"[Receiver] Unexpected error in receiver loop: {e}")
                import traceback
                traceback.print_exc()
                try:
                    self.sock.close()
                except Exception:
                    pass
                try:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.sock.connect((C.PI_IP, C.PI_PORT))
                    print(f"[Receiver] Reconnected after error")
                    buf = b""
                except Exception as reconnect_err:
                    print(f"[Receiver] Reconnect failed: {reconnect_err}")
                    time.sleep(1.0)  # Wait before retrying

    # ----------------------------- Idle preview -------------------------------
    @staticmethod
    def _draw_idle_hud(frame, text: str, *, display_scale: float) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(28 * display_scale)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        from .hud import draw_wrapped_text
        draw_wrapped_text(frame, text, 8, 8, int(w * 0.8))

    def _run_idle_preview_once(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            print("[StateManager] WARNING: Failed to read camera frame in IDLE")
            return

        #print("[StateManager] Running YOLO inference on IDLE frame...") TODO
        verbose = C.YOLO_VERBOSE
        try:
            results = self.model(frame, classes=self.allowed_class_ids, verbose=verbose)
        except TypeError:
            results = self.model(frame, classes=self.allowed_class_ids)
        except Exception as e:
            print(f"[StateManager] WARNING: YOLO inference failed: {e}")
            results = []

        annotated = frame
        try:
            for r in results:
                annotated = r.plot()
        except Exception as e:
            print(f"[StateManager] WARNING: YOLO plot failed: {e}")
            pass

        secs_left = max(0.0, self.idle_deadline - time.time())
        status = f"IDLE | auto-search in {secs_left:0.1f}s"

        self._draw_idle_hud(annotated, status, display_scale=C.DISPLAY_SCALE)

        h, w = annotated.shape[:2]
        resized = cv2.resize(annotated, (int(w * C.DISPLAY_SCALE), int(h * C.DISPLAY_SCALE)))
        # Route all GUI frames through the display bus
        self.frame_bus.publish(resized)

    # ------------------------------ Main loop --------------------------------

    def start(self) -> None:
        """Start background threads and enter the FSM main loop."""
        print("[StateManager] Starting background threads...")
        
        # Start LLM worker manager
        print("[StateManager] Starting LLM worker...")
        self.llm_worker.start()
        print("[StateManager] LLM worker started")
        
        # Initialize command dispatcher for task execution
        print("[StateManager] Initializing command dispatcher...")
        from .dispatch import CommandDispatcher
        self.dispatcher = CommandDispatcher(
            sock=self.sock,
            cmd_complete_queue=self.mail.cmd_complete_q
        )
        print("[StateManager] Command dispatcher initialized")
        
        # Start receiver (network)
        print("[StateManager] Starting receiver thread...")
        self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self.receiver_thread.start()
        print("[StateManager] Receiver thread started")

        # Start TaskAgent (user input -> plan -> execute)
        print("[StateManager] Starting TaskAgent...")
        from .task_agent import TaskAgent
        agent = TaskAgent(state_mgr=self, events=self.events)
        agent.daemon = True
        agent.start()
        self.task_agent = agent
        print("[StateManager] TaskAgent started")

        print(f"[Laptop] StateManager started on device={self.device}")
        print("[StateManager] Entering main loop...")

        self.run()

    def run(self) -> None:
        """Small FSM: IDLE preview + timers; kicks off SEARCH when requested."""
        print("[StateManager] Main loop started, entering IDLE state")
        
        # Check if SKIP_TO_TASK_SCHEDULER or SKIP_SCAN_CYCLE is enabled
        if C.SKIP_TO_TASK_SCHEDULER or C.SKIP_SCAN_CYCLE:
            skip_mode = "TASK_SCHEDULER_ONLY" if C.SKIP_TO_TASK_SCHEDULER else "SKIP_SCAN"
            print("="*70)
            print(f"[StateManager] {skip_mode} MODE")
            print("[StateManager] Bypassing scan cycle - using existing object memory")
            print("[StateManager] Memory will NOT be cleared")
            print("[StateManager] Ready for user queries...")
            print("="*70)
            # Skip scan cycle, just wait for user input - memory persists
            while True:
                time.sleep(0.1)
                # Display any frames from task execution
                while True:
                    try:
                        frame = self.frame_bus.get_nowait()
                        cv2.imshow("Vizzy (Task Scheduler)", frame)
                        cv2.waitKey(1)
                    except Exception:
                        break
            return
        
        try:
            while True:
                now = time.time()

                # Idle timeout -> request search
                if self.state == "IDLE" and now >= self.idle_deadline:
                    self.request_search("idle-timeout")

                # Kick off search if requested and not active
                if self.events.search_requested.is_set() and not self.events.scan_active.is_set():
                    self.start_search()

                # If a scan ended (via interrupt) TODO
                if self.events.scan_finished.is_set() and self.events.scan_active.is_set():
                    self.finish_search()

                # --- Display handling (main-thread only) ---
                if self.state == "IDLE":
                    # Close active-view window when returning to IDLE
                    if self._prev_state in ("SEARCH", "EXECUTE_TASK"):
                        try:
                            cv2.destroyWindow("Vizzy (SEARCH)")
                        except Exception:
                            pass
                    # Publish an IDLE frame via the display bus
                    self._run_idle_preview_once()
                    # Drain and display the latest IDLE frame on main thread
                    last = None
                    for frame in self.frame_bus.drain():
                        last = frame
                    if last is not None:
                        cv2.imshow("Vizzy (IDLE Preview)", last)

                elif self.state in ("SEARCH", "EXECUTE_TASK"):
                    # Close IDLE window when entering an active view
                    if self._prev_state == "IDLE":
                        try:
                            cv2.destroyWindow("Vizzy (IDLE Preview)")
                        except Exception:
                            pass
                    # Drain frames from the bus and render the latest one (faster GUI)
                    last = None
                    frame_count = 0
                    for frame in self.frame_bus.drain():
                        last = frame
                        frame_count += 1
                    if last is not None:
                        cv2.imshow("Vizzy (SEARCH)", last)  # reuse one window for all active modes
                        # Use waitKey(1) for faster GUI updates (non-blocking)
                    else:
                        # No frame available, use minimal delay
                        pass

                # Minimal UI handling (only 'q' to quit for now)
                # waitKey(1) is non-blocking and allows fast GUI updates
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # remember previous state for next tick
                self._prev_state = self.state

                # Reduced sleep for faster GUI responsiveness
                time.sleep(0.005)  # Faster main loop for responsive GUI

        finally:
            # Stop LLM worker gracefully
            try:
                print("[Laptop] Stopping LLM worker...")
                self.llm_worker.stop(wait=True)
            except Exception as e:
                print(f"[Laptop] Error stopping LLM worker: {e}")
            
            try:
                send_json(self.sock, {"type": P.TYPE_STOP})
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.cap.release()
            cv2.destroyAllWindows()

    # ------------------------------ utilities --------------------------------

    def _switch_state(self, new_state: StateName) -> None:
        """Set state and allow transition-specific cleanup in the loop."""
        self.state = new_state
