# vizzy/laptop/scan_worker.py
# -----------------------------------------------------------------------------
# ScanWorker: runs a full SEARCH cycle.
# Per pose:
#   - Repeat:
#       scan window -> pick next unseen viable class -> center -> GET_PWMS -> memory update
#     until no new viable objects remain at that pose.
#   - Then send a single POSE_DONE with status: SUCCESS (>=1 success), FAIL (>0 tries, 0 success), or SKIP (0 tries).
# Exits when:
#   - RPi completes the grid (SEARCH {active:false}), or
#   - scan_abort is set (future), or
#   - socket problems (safe exit).
#
# Notes:
# - Memory flags are reset at the start of the search.
# - Memory is pruned on completion (default stop or early exit).
# - All GUI rendering is forwarded via frame_sink to the main thread.
# -----------------------------------------------------------------------------

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Any, Callable

import cv2
from ultralytics import YOLO

from ..shared import config as C
from ..shared import protocol as P
from ..shared.jsonl import send_json

from .memory import ObjectMemory
from .scanning import run_scan_window
from .centering import center_on_class
from .motion import Motion


FrameSink = Optional[Callable[[Any], None]]


class ScanWorker(threading.Thread):
    def __init__(
        self,
        *,
        sock,
        cap: cv2.VideoCapture,
        model: YOLO,
        mail,
        events,
        motion: Motion,
        display_scale: float = 1.0,
        frame_sink: FrameSink = None,   # <--- new: push frames to main thread
    ):
        super().__init__(name="ScanWorker")
        self.sock = sock
        self.cap = cap
        self.model = model
        self.mail = mail
        self.events = events
        self.motion = motion
        self.display_scale = float(display_scale)
        self.frame_sink = frame_sink

        # names map for labels
        self._names = model.names

        # memory
        self.memory = ObjectMemory(C.MEM_FILE)

        # frame center (set from a fresh read; fallback to last known if needed)
        ok, frame0 = self.cap.read()
        if ok:
            h0, w0 = frame0.shape[:2]
            self.center_x, self.center_y = w0 // 2, h0 // 2
        else:
            # Reasonable defaults; centering will still function visually
            self.center_x, self.center_y = 640 // 2, 480 // 2

    # ------------------------------- helpers ---------------------------------

    def _get_name(self, cid: int) -> str:
        try:
            names = self._names
            if isinstance(names, dict):
                return str(names[int(cid)])
            return str(names[int(cid)])
        except Exception:
            return str(cid)

    def _send(self, payload: Dict[str, Any]) -> None:
        try:
            send_json(self.sock, payload)
        except Exception:
            # Best-effort; upstream orchestrator handles reconnects
            pass

    # ------------------------------- main run --------------------------------

    def run(self) -> None:
        # Mark entries as not updated for this session, announce SEARCH start
        self.memory.reset_session_flags()
        self._send({"type": P.TYPE_SEARCH, "active": True})

        # Clear completion flags for a fresh run
        self.events.scan_finished.clear()

        try:
            while not self.events.scan_abort.is_set():
                # If RPi has already finished (edge case), stop
                if self.events.search_completed_from_rpi.is_set():
                    break

                # Wait for next pose_id from receiver/mailbox
                try:
                    pose_id = self.mail.pose_ready_q.get(timeout=0.05)
                except Exception:
                    # No pose yet; small idle tick
                    time.sleep(0.01)
                    continue

                # Per-pose repeat: try to find and center multiple distinct classes
                successes = 0
                attempts = 0
                fails_at_pose: Dict[int, int] = {}

                while not self.events.scan_abort.is_set():
                    # Build exclusion list: already updated this session, or too many failures at this pose
                    exclude_ids = [int(e["cls_id"]) for e in self.memory.entries_sorted() if e.get("updated_this_session") == 1]
                    for cid, fcount in fails_at_pose.items():
                        if fcount >= int(C.MAX_FAILS_PER_POSE):
                            exclude_ids.append(int(cid))

                    # Scan the window at this pose (emit frames via frame_sink if provided)
                    summary = run_scan_window(
                        cap=self.cap,
                        model=self.model,
                        exclude_ids=exclude_ids,
                        get_name=self._get_name,
                        min_frames_for_class=int(C.SCAN_MIN_FRAMES),
                        frame_sink=self.frame_sink,               # <--- forward frames
                        display_scale=self.display_scale,          # keep consistent sizing
                    )

                    objs = summary.get("objects", [])
                    candidate = None
                    for o in objs:
                        if (
                            float(o.get("avg_conf", 0.0)) >= float(C.SCAN_MIN_CONF)
                            and int(o.get("frames", 0)) >= int(C.SCAN_MIN_FRAMES)
                            and int(o.get("cls_id")) not in exclude_ids
                        ):
                            candidate = o
                            break

                    if candidate is None:
                        # Nothing new to try at this pose -> report POSE_DONE and move on
                        status = "SUCCESS" if successes > 0 else ("FAIL" if attempts > 0 else "SKIP")
                        self._send({"type": P.TYPE_POSE_DONE, "pose_id": int(pose_id), "status": status})
                        break  # advance to next pose

                    attempts += 1
                    cls_id = int(candidate["cls_id"])
                    label = f"CENTER {candidate.get('cls_name', self._get_name(cls_id))} (id {cls_id})"

                    # Centering loop; this owns the HUD during SEARCH
                    result = center_on_class(
                        cap=self.cap,
                        model=self.model,
                        target_cls=cls_id,
                        center_x=self.center_x,
                        center_y=self.center_y,
                        send_move=self.motion.nudge_scan,          # normalized deltas
                        display_scale=self.display_scale,
                        label=label,
                        frame_sink=self.frame_sink,                # <--- forward frames
                    )
                    success = result if isinstance(result, bool) else bool(result[0])

                    if success:
                        successes += 1
                        # While still centered, ask for PWMs and update memory
                        pwms = self.motion.get_pwms(timeout_s=0.3)
                        if pwms:
                            self.memory.update_entry(
                                cls_id=cls_id,
                                cls_name=candidate.get("cls_name", self._get_name(cls_id)),
                                pwm_btm=int(pwms["pwm_btm"]),
                                pwm_top=int(pwms["pwm_top"]),
                            )
                        # Loop again at the SAME pose to look for another unseen class
                        continue
                    else:
                        # Track failures to avoid infinite loops on hard cases
                        fails_at_pose[cls_id] = fails_at_pose.get(cls_id, 0) + 1
                        # If we’ve hit the per-pose failure cap for this class, it’ll be excluded next iteration
                        continue

                # end while per-pose

                # Check for end-of-grid signal
                if self.events.search_completed_from_rpi.is_set():
                    break

            # Normal exit path: RPi signals end-of-grid or abort was requested
        finally:
            # Prune entries not updated this session (end-of-search behavior)
            try:
                self.memory.prune_not_updated()
            except Exception:
                pass

            # Tell orchestrator we’re done
            self.events.scan_finished.set()
