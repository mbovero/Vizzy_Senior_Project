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
from typing import Optional, Literal

import cv2
import torch
from ultralytics import YOLO

from ..shared import config as C
from ..shared.jsonl import recv_lines, send_json
from ..shared import protocol as P

# NEW: frame bus for cross-thread rendering
from .display_bus import FrameBus

StateName = Literal["IDLE", "SEARCH", "PROCESS_QUERY", "EXECUTE_TASK", "INTERRUPT"]


class Events:
    """Coordination primitives shared across threads."""
    def __init__(self):
        self.search_requested = threading.Event()
        self.scan_active = threading.Event()
        self.scan_abort = threading.Event()       # reserved for future interrupt
        self.scan_finished = threading.Event()
        self.search_completed_from_rpi = threading.Event()  # ReceiverThread sets on SEARCH {active:false}
        self.query_ready = threading.Event()


class Mailboxes:
    """Network mailboxes the ReceiverThread fills (same semantics as current client.py)."""
    def __init__(self):
        from queue import Queue
        self.pose_ready_q: "Queue[int]" = Queue(maxsize=8)
        self.pwms_event = threading.Event()
        self.pwms_payload: dict = {}


class StateManager:
    def __init__(self):
        # State
        self.state: StateName = "IDLE"
        self._prev_state: Optional[StateName] = None  # NEW: to manage window teardown
        self.events = Events()
        self.mail = Mailboxes()
        self.idle_deadline: float = time.time() + getattr(C, "IDLE_TIMEOUT_S", 45.0)
        self.search_mode: bool = False  # mirror of wire SEARCH state

        # IO / vision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(C.YOLO_MODEL)
        self.cap = cv2.VideoCapture(C.CAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {C.CAM_INDEX}")

        ok, frame0 = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError("Failed to read an initial frame from the camera.")
        self.h0, self.w0 = frame0.shape[:2]

        # Networking
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((C.PI_IP, C.PI_PORT))

        # Thread handles (filled later)
        self.receiver_thread: Optional[threading.Thread] = None
        self.scan_worker: Optional[threading.Thread] = None
        self.task_agent: Optional[threading.Thread] = None

        # Motion facade & camera manager (motion is created lazily)
        self.motion = None

        # NEW: Frame bus for worker->main display handoff
        self.frame_bus = FrameBus(maxsize=4)

    # ------------------------------- Triggers ---------------------------------

    def request_search(self, reason: str = "idle") -> None:
        """Public trigger: ask to start a search cycle (debounced by start_search())."""
        self.events.search_requested.set()

    def start_search(self) -> None:
        """
        Spawn a ScanWorker and set scan_active if not already scanning.
        (Spawns the worker, wires in Motion, and flips scan_active.)
        """
        if self.events.scan_active.is_set():
            return
        self.events.search_requested.clear()
        self.events.scan_finished.clear()
        self.events.scan_abort.clear()

        # Lazy imports to avoid circular refs until files exist
        from .scan_worker import ScanWorker
        from .motion import Motion

        # Ensure we have a Motion façade
        if self.motion is None:
            self.motion = Motion(self.sock, self.mail.pwms_event, self.mail.pwms_payload)

        # Hand off the camera to the worker directly; worker pushes frames to frame_bus
        worker = ScanWorker(
            sock=self.sock,
            cap=self.cap,
            model=self.model,
            mail=self.mail,
            events=self.events,
            motion=self.motion,
            display_scale=C.DISPLAY_SCALE,
            frame_sink=self.frame_bus.publish,   # NEW: push frames to main thread
        )
        worker.daemon = True
        worker.start()
        self.scan_worker = worker
        self.events.scan_active.set()
        self._switch_state("SEARCH")

    def finish_search(self) -> None:
        """Handle end-of-grid or manual stop; return to IDLE and reset idle timer."""
        self.events.scan_active.clear()
        self.events.scan_finished.clear()
        self._switch_state("IDLE")
        self.idle_deadline = time.time() + getattr(C, "IDLE_TIMEOUT_S", 45.0)

    # --------------------------- Receiver thread ------------------------------

    def _receiver_loop(self) -> None:
        """Continuously read JSONL from RPi and fill mailboxes / events."""
        buf = b""
        from queue import Empty

        while True:
            try:
                msgs, buf, closed = recv_lines(self.sock, buf)
                if closed:
                    # Attempt a blocking reconnect; simple strategy for now
                    import socket
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.sock.connect((C.PI_IP, C.PI_PORT))
                    buf = b""
                    continue

                for msg in msgs:
                    mtype = msg.get("type")
                    mcmd = msg.get("cmd")

                    if mtype == P.TYPE_POSE_READY:
                        try:
                            self.mail.pose_ready_q.put_nowait(int(msg.get("pose_id", 0)))
                        except Exception:
                            try:
                                self.mail.pose_ready_q.get_nowait()
                            except Empty:
                                pass
                            self.mail.pose_ready_q.put_nowait(int(msg.get("pose_id", 0)))

                    elif mtype == P.TYPE_SEARCH:
                        # RPi default completion
                        if msg.get("active") is False:
                            self.search_mode = False
                            self.events.search_completed_from_rpi.set()

                    elif mtype == P.TYPE_PWMS:
                        try:
                            self.mail.pwms_payload.clear()
                            self.mail.pwms_payload.update({
                                "pwm_btm": int(msg["pwm_btm"]),
                                "pwm_top": int(msg["pwm_top"]),
                            })
                            self.mail.pwms_event.set()
                        except Exception:
                            pass

            except Exception:
                # Reconnect path (simple)
                import socket
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.connect((C.PI_IP, C.PI_PORT))
                buf = b""

    # ----------------------------- Idle preview -------------------------------

    def _run_idle_preview_once(self) -> None:
        """Render one IDLE preview frame."""
        ok, frame = self.cap.read()
        if not ok:
            return
        results = self.model(frame, verbose = C.YOLO_VERBOSE)
        for result in results:
            annotated = result.plot()
            h, w = annotated.shape[:2]
            resized = cv2.resize(annotated, (int(w * C.DISPLAY_SCALE), int(h * C.DISPLAY_SCALE)))
            cv2.imshow("Vizzy (IDLE Preview)", resized)

    # ------------------------------ Main loop --------------------------------

    def start(self) -> None:
        """Start background threads and enter the FSM main loop."""
        # Start receiver (network)
        self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self.receiver_thread.start()

        # Start TaskAgent (user input -> plan -> execute)
        from .task_agent import TaskAgent
        agent = TaskAgent(state_mgr=self, events=self.events)
        agent.daemon = True
        agent.start()
        self.task_agent = agent

        print(f"[Laptop] StateManager started on device={self.device}")

        self.run()

    def run(self) -> None:
        """Small FSM: IDLE preview + timers; kicks off SEARCH when requested."""
        try:
            while True:
                now = time.time()

                # Idle timeout -> request search
                if self.state == "IDLE" and now >= self.idle_deadline:
                    self.request_search("idle-timeout")

                # Kick off search if requested and not active
                if self.events.search_requested.is_set() and not self.events.scan_active.is_set():
                    self.start_search()

                # If RPi completed the sweep, ScanWorker should end; finish when we see it
                if self.events.search_completed_from_rpi.is_set() and self.events.scan_active.is_set():
                    if self.events.scan_finished.is_set():
                        self.finish_search()
                        self.events.search_completed_from_rpi.clear()

                # If a scan ended (e.g., normal path), finish
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
                    self._run_idle_preview_once()

                elif self.state in ("SEARCH", "EXECUTE_TASK"):
                    # Close IDLE window when entering an active view
                    if self._prev_state == "IDLE":
                        try:
                            cv2.destroyWindow("Vizzy (IDLE Preview)")
                        except Exception:
                            pass
                    # Drain frames from the bus and render the latest one
                    last = None
                    for frame in self.frame_bus.drain():
                        last = frame
                    if last is not None:
                        cv2.imshow("Vizzy (SEARCH)", last)  # reuse one window for all active modes

                # Minimal UI handling (only 'q' to quit for now)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # remember previous state for next tick
                self._prev_state = self.state

                time.sleep(0.01)

        finally:
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
