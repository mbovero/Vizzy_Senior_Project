# vizzy/laptop/scan_worker.py
# -----------------------------------------------------------------------------
# ScanWorker: laptop-driven SEARCH cycle.
# For each pose from build_search_path():
#   - GOTO baseline -> wait for POSE_READY
#   - repeat at SAME pose:
#       scan window -> pick unseen viable class -> center -> GET_PWMS -> memory
#       -> return to baseline (GOTO) -> wait for POSE_READY -> dwell
#     until no new viable objects remain at that pose.
# Exits when path is exhausted or scan_abort is set.
# -----------------------------------------------------------------------------

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Any, Callable

import cv2
from ultralytics import YOLO

from ..shared import config as C

from .memory import ObjectMemory
from .scanning import run_scan_window, build_search_path
from .centering import center_on_class
from .motion import Motion


FrameSink = Optional[Callable[[Any], None]]


class ScanWorker(threading.Thread):
    def __init__(
        self,
        *,
        sock,  # kept for parity with callers, not directly used here
        cap: cv2.VideoCapture,
        model: YOLO,
        mail,
        events,
        motion: Motion,
        display_scale: float = 1.0,
        frame_sink: FrameSink = None,
        llm_worker=None,  # NEW: LLM WorkerManager for semantic enrichment
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
        self.llm_worker = llm_worker  # NEW: LLM worker pool for semantic enrichment

        # names map for labels
        self._names = model.names

        # memory
        self.memory = ObjectMemory(C.MEM_FILE)
        
        # Image capture directory for LLM processing
        self.image_dir = getattr(C, "IMAGE_DIR", "captured_images")
        import os
        os.makedirs(self.image_dir, exist_ok=True)

        # frame center (set from a fresh read; fallback to last known if needed)
        ok, frame0 = self.cap.read()
        if ok:
            h0, w0 = frame0.shape[:2]
            self.center_x, self.center_y = w0 // 2, h0 // 2
        else:
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

    def _drain_pose_ready(self) -> None:
        """Clear any stale POSE_READY ids in the mailbox queue."""
        try:
            while True:
                _ = self.mail.pose_ready_q.get_nowait()
        except Exception:
            pass

    def _wait_pose_ready(self, expected_id: Optional[int], slew_ms: int) -> bool:
        """
        Wait for a TYPE_POSE_READY ack. If expected_id is not None, only return
        True when that id is seen; otherwise accept the first ack.
        Timeout is derived from the motion duration plus settle.
        """
        t_deadline = time.time() + (slew_ms / 1000.0) + C.POSE_SETTLE_S + 1.5
        while time.time() < t_deadline and not self.events.scan_abort.is_set():
            try:
                pid = self.mail.pose_ready_q.get(timeout=0.05)
                if expected_id is None or int(pid) == int(expected_id):
                    return True
                # else: ignore mismatched acks (could be late from a prior goto)
            except Exception:
                pass
        return False

    def _capture_and_enrich(self, object_id: str) -> Optional[str]:
        """
        Capture the current centered frame, save it, upload to OpenAI, and submit to LLM worker.
        This is non-blocking - the LLM processing happens asynchronously in the worker pool.
        
        Args:
            object_id: The unique object ID to enrich
        
        Returns:
            Path to the saved image, or None on failure
        """
        if self.llm_worker is None:
            return None
        
        import os
        from .llm_semantics import upload_image
        
        try:
            # Capture current centered frame
            ok, frame = self.cap.read()
            if not ok:
                print(f"[ScanWorker] Failed to capture image for {object_id}")
                return None
            
            # Save image to disk with timestamp
            timestamp = int(time.time() * 1000)
            image_filename = f"{object_id}_{timestamp}.jpg"
            image_path = os.path.join(self.image_dir, image_filename)
            cv2.imwrite(image_path, frame)
            print(f"[ScanWorker] Captured image: {image_path}")
            
            # Upload to OpenAI
            file_id = upload_image(image_path)
            print(f"[ScanWorker] Uploaded image {image_path} -> file_id={file_id}")
            
            # Submit to LLM worker pool (non-blocking - returns immediately)
            uid, fut = self.llm_worker.submit(object_id=object_id, file_id=file_id)
            print(f"[ScanWorker] Submitted enrichment task {uid[:8]} for {object_id}")
            
            return image_path
        
        except Exception as e:
            print(f"[ScanWorker] Error capturing/enriching image for {object_id}: {e}")
            return None

    # ------------------------------- main run --------------------------------

    def run(self) -> None:
        # Mark entries as not updated for this session
        self.memory.reset_session_flags()
        self.events.scan_finished.clear()

        path = build_search_path()  # [{"pose_id","pwm_btm","pwm_top","slew_ms"}, ...]

        try:
            # Iterate poses locally; RPi just executes gotos & nudges.
            for pose in path:
                if self.events.scan_abort.is_set():
                    break

                pid = int(pose["pose_id"])
                btm = int(pose["pwm_btm"])
                top = int(pose["pwm_top"])
                slew = int(pose["slew_ms"])

                # Go to baseline pose for this grid point
                self._drain_pose_ready()
                self.motion.goto_pose_pwm(btm, top, slew_ms=slew, pose_id=pid)
                if not self._wait_pose_ready(pid, slew):
                    # If ack missing, proceed cautiously after an extra dwell
                    time.sleep(C.POSE_SETTLE_S)

                # Per-pose repeat: try to find and center multiple distinct classes
                successes = 0
                attempts = 0
                fails_at_pose: Dict[int, int] = {}

                while not self.events.scan_abort.is_set():
                    # Exclusions: already updated this session, or too many fails here
                    exclude_ids = [int(e["cls_id"]) for e in self.memory.entries_sorted()
                                   if e.get("updated_this_session") == 1]
                    for cid, fcount in fails_at_pose.items():
                        if fcount >= C.MAX_FAILS_PER_POSE:
                            exclude_ids.append(int(cid))

                    # Scan the window at this pose (publish frames via frame_sink)
                    summary = run_scan_window(
                        cap=self.cap,
                        model=self.model,
                        exclude_ids=exclude_ids,
                        get_name=self._get_name,
                        min_frames_for_class=C.SCAN_MIN_FRAMES,
                        frame_sink=self.frame_sink,
                        display_scale=self.display_scale,
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
                        # Nothing new to try at this pose -> advance to next pose
                        break

                    attempts += 1
                    cls_id = int(candidate["cls_id"])
                    label = f"CENTER {candidate.get('cls_name', self._get_name(cls_id))} (id {cls_id})"

                    # Centering loop; this owns the HUD during SEARCH
                    success = bool(
                        center_on_class(
                            cap=self.cap,
                            model=self.model,
                            target_cls=cls_id,
                            center_x=self.center_x,
                            center_y=self.center_y,
                            send_move=self.motion.nudge_scan,      # normalized deltas
                            display_scale=self.display_scale,
                            label=label,
                            frame_sink=self.frame_sink,
                        )
                    )

                    if success:
                        successes += 1
                        # While still centered, ask for PWMs and create object in memory
                        pwms = self.motion.get_pwms(timeout_s=0.3)
                        if pwms:
                            # Create object entry with unique ID (returns the object_id)
                            object_id = self.memory.update_entry(
                                cls_id=cls_id,
                                cls_name=candidate.get("cls_name", self._get_name(cls_id)),
                                pwm_btm=int(pwms["pwm_btm"]),
                                pwm_top=int(pwms["pwm_top"]),
                                x=0.0,  # TODO: Calculate from IK
                                y=0.0,  # TODO: Calculate from IK
                                z=0.0,  # TODO: Read from laser sensor
                            )
                            
                            # Capture image and submit for LLM enrichment (non-blocking)
                            # The LLM worker will process this asynchronously and update semantics
                            image_path = self._capture_and_enrich(object_id)
                            
                            # Update object with image path if capture succeeded
                            if image_path and object_id:
                                obj = self.memory.get_object(object_id)
                                if obj:
                                    obj["image_path"] = image_path
                                    self.memory.data["objects"][object_id] = obj
                                    self.memory.save()
                    else:
                        # Track failures to avoid infinite loops on hard cases
                        fails_at_pose[cls_id] = fails_at_pose.get(cls_id, 0) + 1

                    # --- Always return to the baseline pose before next attempt ---
                    self._drain_pose_ready()
                    self.motion.goto_pose_pwm(
                        btm, top,
                        slew_ms=C.GOTO_POSE_SLEW_MS,
                        pose_id=pid
                    )
                    _ = self._wait_pose_ready(pid, C.GOTO_POSE_SLEW_MS)
                    time.sleep(C.RETURN_TO_POSE_DWELL_S)

                # end while per-pose

                if self.events.scan_abort.is_set():
                    break

            # end for pose in path

        finally:
            # Prune entries not updated this session (end-of-search behavior)
            try:
                self.memory.prune_not_updated()
            except Exception:
                pass

            # Tell orchestrator we’re done
            self.events.scan_finished.set()
