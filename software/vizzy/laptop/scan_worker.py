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
import numpy as np
from ultralytics import YOLO

from ..shared import config as C

from .memory import ObjectMemory
from .scanning import run_scan_window, build_search_path
from .centering import center_on_class
from .motion import Motion
from .orientation import calculate_grasp_angle


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
        memory=None,  # NEW: Shared ObjectMemory instance
        allowed_class_ids=None,
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

        # Use shared memory instance or create new one (backward compatible)
        self.memory = memory if memory is not None else ObjectMemory(C.MEM_FILE)
        self.allowed_class_ids = list(allowed_class_ids) if allowed_class_ids is not None else None
        
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
        self._centering_active = False

    # ------------------------------- helpers ---------------------------------

    def _get_name(self, cid: int) -> str:
        try:
            names = self._names
            if isinstance(names, dict):
                return str(names[int(cid)])
            return str(names[int(cid)])
        except Exception:
            return str(cid)

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

    def _flush_capture(self, frames: int = 5) -> None:
        """Drop buffered frames so fresh scans reflect the baseline pose."""
        try:
            for _ in range(max(0, frames)):
                self.cap.grab()
        except Exception:
            pass
    
    def _calculate_orientation_from_collected_frames(self, frames_list: list) -> dict:
        """
        Calculate averaged grasp orientation from frames collected during centering.
        
        Args:
            frames_list: List of dicts with keys "frame" (ndarray) and "mask" (tensor)
        
        Returns:
            Dictionary with orientation data: {"grasp_yaw": float}
        """
        if not frames_list:
            print("[ScanWorker] WARNING: No frames collected for orientation calculation")
            return {"grasp_yaw": 0.0}
        
        print(f"[ScanWorker] Calculating orientation from {len(frames_list)} collected frames...")
        
        angles = []
        
        for i, item in enumerate(frames_list):
            try:
                # Extract mask tensor (already from YOLO during centering)
                mask_tensor = item["mask"]
                
                # Convert mask tensor to uint8 mask for PCA
                mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
                h, w = item["frame"].shape[:2]
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_u8 = (mask_resized * 255).astype(np.uint8)
                
                # Calculate orientation using PCA
                orientation = calculate_grasp_angle(mask_u8)
                
                if orientation.get("success", False):
                    angle = orientation["yaw_angle"]
                    angles.append(angle)
                    print(f"[ScanWorker] Frame {i+1}/{len(frames_list)}: angle={angle:.1f}°")
                else:
                    print(f"[ScanWorker] Frame {i+1}/{len(frames_list)}: Orientation calc failed")
                
            except Exception as e:
                print(f"[ScanWorker] Frame {i+1}/{len(frames_list)}: Error: {e}")
                continue
        
        # Calculate averaged results
        if len(angles) == 0:
            print("[ScanWorker] WARNING: No successful orientation measurements")
            return {"grasp_yaw": 0.0}
        
        # Use circular mean for angles (handles wraparound at -90/+90)
        angles_rad = np.deg2rad(angles)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        avg_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
        
        print(f"[ScanWorker] Orientation result: {len(angles)} samples, angle={avg_angle:.1f}°")
        
        return {"grasp_yaw": round(float(avg_angle), 2)}

    # ------------------------------- main run --------------------------------

    def run(self) -> None:
        print("[ScanWorker] Thread started, initializing scan...")
        # Mark entries as not updated for this session
        self.memory.reset_session_flags()
        self.events.scan_finished.clear()

        print("[ScanWorker] Building search path...")
        path = build_search_path()  # [{"pose_id","pwm_btm","pwm_top","slew_ms"}, ...]
        print(f"[ScanWorker] Search path has {len(path)} poses")

        try:
            # Iterate poses locally; RPi just executes gotos & nudges.
            print("[ScanWorker] Starting pose iteration...")
            for pose in path:
                if self.events.scan_abort.is_set():
                    break

                pid = int(pose["pose_id"])
                btm = int(pose["pwm_btm"])
                top = int(pose["pwm_top"])
                print(f"[ScanWorker] Pose {pid}: BTM={btm}, TOP={top}, SLEW={pose.get('slew_ms')}")

                # Go to baseline pose for this grid point
                print(f"[ScanWorker] Sending GOTO_POSE to RPi...")
                ok = self.motion.goto_pose_pwm(btm, top, pose_id=pid)
                if ok:
                    print(f"[ScanWorker] POSE_READY received")
                else:
                    # If ack missing, proceed cautiously after an extra dwell
                    print(f"[ScanWorker] No POSE_READY received, settling...")
                    time.sleep(C.POSE_SETTLE_S)

                # Per-pose repeat: try to find and center multiple distinct classes
                attempts = 0
                fails_at_pose: Dict[int, int] = {}
                print(f"[ScanWorker] Starting scan loop at pose {pid}...")

                # Ensure we start each scan loop with fresh frames from the baseline pose
                self._flush_capture()

                while not self.events.scan_abort.is_set():
                    if self._centering_active:
                        # Defensive: should never run when centering flag is set.
                        time.sleep(0.01)
                        continue

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
                        allowed_class_ids=self.allowed_class_ids,
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
                    self._centering_active = True
                    center_success, collected_frames = center_on_class(
                        cap=self.cap,
                        model=self.model,
                        target_cls=cls_id,
                        center_x=self.center_x,
                        center_y=self.center_y,
                        send_move=self.motion.nudge_scan,      # normalized deltas
                        display_scale=self.display_scale,
                        label=label,
                        frame_sink=self.frame_sink,
                        collect_frames=True,  # Collect frames for orientation calculation
                    )
                    self._centering_active = False

                    # When success centering, get grasp orientation and create object
                    if center_success:
                        # Request position data 
                        pwms = self.motion.get_pwms(timeout_s=0.3)
                        
                        # Calculate grasp orientation from collected frames (no new YOLO inference needed!)
                        orientation_data = self._calculate_orientation_from_collected_frames(collected_frames)
                        
                        # Create object entry with unique ID (returns the object_id)
                        object_id = self.memory.update_entry(
                            cls_id=cls_id,
                            cls_name=candidate.get("cls_name", self._get_name(cls_id)),
                            pwm_btm=int(pwms["pwm_btm"]),
                            pwm_top=int(pwms["pwm_top"]),
                            x=0.0,  # TODO: Calculate from IK
                            y=0.0,  # TODO: Calculate from IK
                            z=0.0,  # TODO: Read from laser sensor
                            orientation=orientation_data,  # Add orientation data
                        )
                        
                        # Capture image and submit for LLM enrichment (non-blocking)
                        # Skip if SKIP_SEMANTIC_ENRICHMENT is enabled
                        if not getattr(C, "SKIP_SEMANTIC_ENRICHMENT", False):
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
                            print(f"[ScanWorker] Skipping semantic enrichment (SKIP_SEMANTIC_ENRICHMENT enabled)")
                    else:
                        # Track failures to avoid infinite loops on hard cases
                        fails_at_pose[cls_id] = fails_at_pose.get(cls_id, 0) + 1

                    # --- Always return to the baseline pose before next attempt ---
                    returned = self.motion.goto_pose_pwm(btm, top, pose_id=pid)
                    if not returned:
                        print("[ScanWorker] RETURN pose ack missing or aborted; continuing after settle...")
                    time.sleep(C.RETURN_TO_POSE_DWELL_S)
                    self._flush_capture()

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
