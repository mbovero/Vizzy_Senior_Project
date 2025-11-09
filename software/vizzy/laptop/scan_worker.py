# vizzy/laptop/scan_worker.py
# -----------------------------------------------------------------------------
# ScanWorker: laptop-driven SEARCH cycle.
# For each pose from build_search_path():
#   - MOVE_TO baseline -> wait for CMD_COMPLETE
#   - repeat at SAME pose:
#       scan window -> pick unseen viable class -> center -> GET_OBJ_LOC -> memory
#       -> return to baseline (MOVE_TO) -> wait for CMD_COMPLETE -> dwell
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
        
        # Build list of valid class IDs for search (only fork and cup)
        valid_class_names = getattr(C, 'SEARCH_VALID_CLASS_NAMES', ["fork", "cup"])
        self.search_valid_class_ids = []
        if isinstance(self._names, dict):
            for cls_id, cls_name in self._names.items():
                if isinstance(cls_name, str) and cls_name.lower() in [n.lower() for n in valid_class_names]:
                    self.search_valid_class_ids.append(int(cls_id))
        elif hasattr(self._names, '__iter__'):
            # Handle case where names is a list or sequence
            for idx, cls_name in enumerate(self._names):
                if isinstance(cls_name, str) and cls_name.lower() in [n.lower() for n in valid_class_names]:
                    self.search_valid_class_ids.append(int(idx))
        
        print(f"[ScanWorker] Valid search objects: {valid_class_names}, class IDs: {self.search_valid_class_ids}")
        
        # Image capture directory for LLM processing
        self.image_dir = C.IMAGE_DIR
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
    
    def _center_and_register_object(self, cls_name: str, cls_id: int, x: float, y: float, z: float, pitch: float) -> None:
        """
        Center on object and register it. Scan cycle will resume after centering completes.
        Centering continues until movement is < 5mm in each direction (measured when arm is stopped).
        After centering and registration, immediately resume scan cycle.
        """
        print(f"[ScanWorker] Starting centering on {cls_name} (class ID: {cls_id})")
        print(f"[ScanWorker] Centering will continue until movement < {C.CENTER_MIN_MOVEMENT_MM}mm")
        print(f"[ScanWorker] Measurements only occur when arm is stopped (after {C.CENTER_MEASURE_WAIT_TIME_S}s wait)")
        
        # Set centering flag
        self._centering_active = True
        
        # Centering loop - continues until movement < 5mm and object is centered
        center_success, collected_frames, final_x_mm, final_y_mm = center_on_class(
            cap=self.cap,
            model=self.model,
            target_cls=cls_id,
            center_x=self.center_x,
            center_y=self.center_y,
            send_move=self.motion.nudge_scan,      # legacy callback (not used if motion provided)
            display_scale=self.display_scale,
            label=f"CENTER {cls_name}",
            frame_sink=self.frame_sink,
            collect_frames=True,  # Collect frames for orientation calculation
            current_x_mm=x,       # Baseline x position in mm (where scan was at)
            current_y_mm=y,       # Baseline y position in mm (where scan was at)
            current_z_mm=z,       # Baseline z position in mm
            current_pitch_deg=pitch,  # Baseline pitch in degrees
            motion=self.motion,   # Motion object for absolute moves
        )
        
        # Clear centering flag
        self._centering_active = False
        
        # Only register object if centering was successful (movement < 5mm and object centered)
        if center_success:
            print(f"[ScanWorker] Centering SUCCESS - movement < {C.CENTER_MIN_MOVEMENT_MM}mm")
            print(f"[ScanWorker] Final centered position: x={final_x_mm:.2f}mm, y={final_y_mm:.2f}mm, z={z:.2f}mm")
            
            # Use the final centered position for object location
            loc = {
                "x": final_x_mm,  # Final x after centering
                "y": final_y_mm,  # Final y after centering
                "z": z,           # Z unchanged
            }
            
            # Calculate grasp orientation from collected frames
            orientation_data = self._calculate_orientation_from_collected_frames(collected_frames)
            
            # Create object entry with unique ID
            object_id = self.memory.update_entry(
                cls_id=cls_id,
                cls_name=cls_name,
                x=float(loc["x"]),
                y=float(loc["y"]),
                z=float(loc["z"]),
                orientation=orientation_data,
            )
            print(f"[ScanWorker] Object registered with ID: {object_id}")
            
            # Capture image and submit for LLM enrichment (non-blocking)
            if not C.SKIP_SEMANTIC_ENRICHMENT:
                image_path = self._capture_and_enrich(object_id)
                if image_path and object_id:
                    obj = self.memory.get_object(object_id)
                    if obj:
                        obj["image_path"] = image_path
                        self.memory.data["objects"][object_id] = obj
                        self.memory.save()
            else:
                print(f"[ScanWorker] Skipping semantic enrichment (SKIP_SEMANTIC_ENRICHMENT enabled)")
            
            print(f"[ScanWorker] Object centered and registered successfully")
        else:
            print(f"[ScanWorker] Centering FAILED - movement not < {C.CENTER_MIN_MOVEMENT_MM}mm or object lost")
            print(f"[ScanWorker] Object NOT registered - resuming scan cycle")
        
        # Return to baseline pose before resuming scan
        print(f"[ScanWorker] Returning to baseline pose: x={x:.2f}mm, y={y:.2f}mm, z={z:.2f}mm")
        returned = self.motion.move_to_target(x, y, z, pitch)
        if not returned:
            print("[ScanWorker] Warning: Baseline MOVE_TO did not confirm; continuing after dwell")
        time.sleep(C.RETURN_TO_POSE_DWELL_S)
        self._flush_capture()
        print(f"[ScanWorker] Returned to baseline pose - ready to resume scan")
    
    def _display_registered_object_info(self, cls_name: str, x: float, y: float, z: float, object_id: int) -> None:
        """Display registered object information and stop scanning."""
        import cv2
        from .hud import draw_wrapped_text
        
        # Keep displaying until scan is aborted
        while not self.events.scan_abort.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            
            # Annotate frame with registration info
            annotated = frame.copy()
            h, w = annotated.shape[:2]
            
            # Draw info overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            
            # Display text
            info_lines = [
                f"===== OBJECT REGISTERED =====",
                f"Object: {cls_name}",
                f"Object ID: {object_id}",
                f"",
                f"Current Position:",
                f"  X: {x:.2f} mm",
                f"  Y: {y:.2f} mm",
                f"  Z: {z:.2f} mm",
                f"",
                f"Scanning STOPPED",
                f"Press 'q' to quit"
            ]
            
            y_pos = 30
            for line in info_lines:
                draw_wrapped_text(annotated, line, 10, y_pos, int(w * 0.9))
                y_pos += 25
            
            # Resize and publish
            resized = cv2.resize(annotated, (int(w * self.display_scale), int(h * self.display_scale)))
            self.frame_sink(resized)
            
            time.sleep(0.1)  # Update at 10 FPS
    
    def _display_object_in_frame_info(self, frame, cls_name: str, cls_id: int, conf: float, x: float, y: float, z: float) -> None:
        """Display object detection info when object is in frame (stop moving)."""
        import cv2
        from .hud import draw_wrapped_text
        
        # Run YOLO to get annotated frame
        results = self.model(frame, classes=[cls_id], verbose=False)
        annotated = frame.copy()
        for r in results:
            try:
                annotated = r.plot()
            except Exception:
                pass
        
        h, w = annotated.shape[:2]
        
        # Draw info overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(120 * self.display_scale)), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        
        # Display text
        info_lines = [
            f"OBJECT IN FRAME: {cls_name} (conf: {conf:.2f})",
            f"Current Position: X={x:.1f}mm Y={y:.1f}mm Z={z:.1f}mm",
            f"Movement STOPPED - Object detected"
        ]
        
        y_pos = 25
        for line in info_lines:
            draw_wrapped_text(annotated, line, 10, y_pos, int(w * 0.9))
            y_pos += 25
        
        # Resize and publish
        resized = cv2.resize(annotated, (int(w * self.display_scale), int(h * self.display_scale)))
        self.frame_sink(resized)
    
    def _display_object_detected_loop(self, cls_name: str, cls_id: int, x: float, y: float, z: float) -> None:
        """
        Continuously display object detection info while object is in frame.
        When object is removed from frame, resume scan cycle.
        GUI stays live - frames are continuously published to frame_sink at high FPS.
        Arm stays in current position while object is detected.
        """
        import cv2
        from .hud import draw_wrapped_text
        
        print(f"[ScanWorker] ========== OBJECT DETECTED - SCAN PAUSED ==========")
        print(f"[ScanWorker] Object: {cls_name}")
        print(f"[ScanWorker] Arm LOCKED at: x={x:.1f}mm, y={y:.1f}mm, z={z:.1f}mm")
        print(f"[ScanWorker] Scan will RESUME when object is removed from frame")
        print(f"[ScanWorker] ==================================================")
        
        frame_count = 0
        last_log_time = time.time()
        frames_without_object = 0
        frames_needed_to_resume = 10  # Need 10 consecutive frames without object to resume
        
        # Keep displaying while object is detected
        # Check continuously if object is still in frame
        # If object is removed, exit loop and resume scan cycle
        while not self.events.scan_abort.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.016)  # ~60 FPS if read fails
                continue
            
            frame_count += 1
            
            # Run YOLO to check if object is still in frame
            results = self.model(frame, classes=[cls_id], verbose=False)
            annotated = frame.copy()
            best_conf = 0.0
            object_still_detected = False
            
            for r in results:
                try:
                    annotated = r.plot()
                    # Check if object is still detected
                    if len(r.boxes) > 0:
                        for i in range(len(r.boxes)):
                            det_cls_id = int(r.boxes.cls[i].item())
                            if det_cls_id == cls_id:
                                conf = float(r.boxes.conf[i].item())
                                if conf >= C.SCAN_MIN_CONF:
                                    object_still_detected = True
                                    if conf > best_conf:
                                        best_conf = conf
                except Exception:
                    pass
            
            # Check if object is still in frame
            if object_still_detected:
                # Object still detected - reset counter and keep displaying
                frames_without_object = 0
            else:
                # Object not detected - increment counter
                frames_without_object += 1
                
                # If object has been missing for enough frames, resume scan cycle
                if frames_without_object >= frames_needed_to_resume:
                    print(f"[ScanWorker] =========================================")
                    print(f"[ScanWorker] Object no longer detected in frame")
                    print(f"[ScanWorker] RESUMING scan cycle...")
                    print(f"[ScanWorker] =========================================")
                    # Exit loop to resume scan cycle
                    return
            
            h, w = annotated.shape[:2]
            
            # Draw info overlay (minimal overlay to keep camera feed visible)
            overlay = annotated.copy()
            # Only overlay on top portion - smaller overlay for faster rendering
            overlay_height = min(int(140 * self.display_scale), h // 4)
            cv2.rectangle(overlay, (0, 0), (w, overlay_height), (0, 0, 0), -1)
            alpha = 0.6  # Less opaque for better visibility
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            
            # Display text (minimal text for faster rendering)
            if object_still_detected:
                status_text = f"OBJECT DETECTED - Scan paused (conf: {best_conf:.2f})"
            else:
                status_text = f"Object removed - Resuming in {frames_needed_to_resume - frames_without_object} frames..."
            
            info_lines = [
                status_text,
                f"Position: X={x:.0f}mm Y={y:.0f}mm Z={z:.0f}mm",
                f"Arm locked - Waiting for object removal"
            ]
            
            y_pos = 25
            for line in info_lines:
                draw_wrapped_text(annotated, line, 10, y_pos, int(w * 0.9))
                y_pos += 22
            
            # Resize and publish frame to keep GUI live (fast path)
            resized = cv2.resize(annotated, (int(w * self.display_scale), int(h * self.display_scale)))
            self.frame_sink(resized)
            
            # Log periodically (every 10 seconds) to show loop is running
            current_time = time.time()
            if current_time - last_log_time >= 10.0:
                fps = frame_count / (current_time - (last_log_time - 10.0))
                status = "detected" if object_still_detected else "removed"
                print(f"[ScanWorker] Display loop: {frame_count} frames, ~{fps:.1f} FPS, object: {status}")
                last_log_time = current_time
                frame_count = 0
            
            # Very short sleep for high FPS GUI (~60 FPS target)
            time.sleep(0.016)  # ~60 FPS update rate for responsive GUI
    
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
        path = build_search_path()  # [{"pose_id","x","y","z","pitch"}, ...]
        print(f"[ScanWorker] Search path has {len(path)} poses")

        try:
            # Iterate poses locally; RPi just executes gotos & nudges.
            print("[ScanWorker] Starting pose iteration...")
            for pose in path:
                if self.events.scan_abort.is_set():
                    break

                pid = int(pose["pose_id"])
                x = float(pose["x"])
                y = float(pose["y"])
                z = float(pose["z"])
                pitch = float(pose.get("pitch", C.SEARCH_PITCH_DEG))
                print(f"[ScanWorker] Pose {pid}: x={x:.1f} y={y:.1f} z={z:.1f} pitch={pitch:.1f}")

                # Go to baseline pose for this grid point
                print(f"[ScanWorker] Moving to pose {pid}: x={x:.1f}mm, y={y:.1f}mm, z={z:.1f}mm")
                ok = self.motion.move_to_target(x, y, z, pitch)
                if not ok:
                    print("[ScanWorker] Warning: MOVE_TO baseline timed out or failed; continuing after settle")

                time.sleep(C.MOVE_SETTLE_S)
                
                # CRITICAL: Check if object of interest is in frame BEFORE starting scan loop
                # If detected, STOP scan cycle immediately - do NOT proceed with scanning or centering
                ok, check_frame = self.cap.read()
                if ok:
                    # Quick detection check for objects we care about
                    check_results = self.model(check_frame, classes=self.search_valid_class_ids, verbose=False)
                    object_detected = False
                    detected_cls_name = None
                    detected_cls_id = None
                    detected_conf = 0.0
                    
                    for r in check_results:
                        if len(r.boxes) > 0:
                            for i in range(len(r.boxes)):
                                cls_id = int(r.boxes.cls[i].item())
                                conf = float(r.boxes.conf[i].item())
                                if cls_id in self.search_valid_class_ids and conf >= C.SCAN_MIN_CONF:
                                    object_detected = True
                                    detected_cls_name = self._get_name(cls_id)
                                    detected_cls_id = cls_id
                                    detected_conf = conf
                                    break
                            if object_detected:
                                break
                    
                    if object_detected:
                        print(f"[ScanWorker] =========================================")
                        print(f"[ScanWorker] OBJECT DETECTED: {detected_cls_name} (conf={detected_conf:.2f})")
                        print(f"[ScanWorker] SCAN CYCLE PAUSED - Starting centering...")
                        print(f"[ScanWorker] Current position: x={x:.1f}mm, y={y:.1f}mm, z={z:.1f}mm")
                        print(f"[ScanWorker] =========================================")
                        
                        # CENTER on object - centering will continue until movement < 5mm
                        # After centering succeeds and object is registered, scan will resume
                        self._center_and_register_object(detected_cls_name, detected_cls_id, x, y, z, pitch)
                        
                        # Centering complete and object registered - resume scan cycle
                        print(f"[ScanWorker] Object centered and registered - RESUMING scan cycle")
                        print(f"[ScanWorker] Continuing to next pose in scan path...")
                        # Continue with the scan cycle (next pose)
                        continue  # Skip to next pose in the path
                
                # Only proceed with scan loop if NO object detected
                # Per-pose repeat: try to find and center multiple distinct classes
                attempts = 0
                fails_at_pose: Dict[int, int] = {}
                print(f"[ScanWorker] No object detected - starting scan loop at pose {pid}...")

                # Ensure we start each scan loop with fresh frames from the baseline pose
                self._flush_capture()

                while not self.events.scan_abort.is_set():
                    if self._centering_active:
                        # Defensive: should never run when centering flag is set.
                        time.sleep(0.01)
                        continue
                    
                    # BEFORE scanning, check again if object is in frame
                    # If object appears during scan loop, pause and wait for removal
                    ok, pre_scan_frame = self.cap.read()
                    if ok:
                        pre_scan_results = self.model(pre_scan_frame, classes=self.search_valid_class_ids, verbose=False)
                        object_detected_pre = False
                        detected_cls_name_pre = None
                        detected_cls_id_pre = None
                        detected_conf_pre = 0.0
                        
                        for r in pre_scan_results:
                            if len(r.boxes) > 0:
                                for i in range(len(r.boxes)):
                                    cls_id = int(r.boxes.cls[i].item())
                                    conf = float(r.boxes.conf[i].item())
                                    if cls_id in self.search_valid_class_ids and conf >= C.SCAN_MIN_CONF:
                                        object_detected_pre = True
                                        detected_cls_name_pre = self._get_name(cls_id)
                                        detected_cls_id_pre = cls_id
                                        detected_conf_pre = conf
                                        break
                                if object_detected_pre:
                                    break
                        
                        if object_detected_pre:
                            print(f"[ScanWorker] Object detected during scan: {detected_cls_name_pre} (conf={detected_conf_pre:.2f})")
                            print(f"[ScanWorker] PAUSING scan - starting centering...")
                            # Center on object - centering will continue until movement < 5mm
                            self._center_and_register_object(detected_cls_name_pre, detected_cls_id_pre, x, y, z, pitch)
                            # Object centered and registered - resume scan at this pose
                            print(f"[ScanWorker] Object centered and registered - resuming scan at current pose")
                            break  # Exit inner scan loop, continue to next pose

                    # Exclusions: already updated this session, or too many fails here
                    exclude_ids = [int(e["cls_id"]) for e in self.memory.entries_sorted()
                                   if e.get("updated_this_session") == 1]
                    for cid, fcount in fails_at_pose.items():
                        if fcount >= C.MAX_FAILS_PER_POSE:
                            exclude_ids.append(int(cid))

                    # Scan the window at this pose (publish frames via frame_sink)
                    # Only scan for fork and cup (use search_valid_class_ids instead of allowed_class_ids)
                    summary = run_scan_window(
                        cap=self.cap,
                        model=self.model,
                        exclude_ids=exclude_ids,
                        get_name=self._get_name,
                        min_frames_for_class=C.SCAN_MIN_FRAMES,
                        frame_sink=self.frame_sink,
                        display_scale=self.display_scale,
                        allowed_class_ids=self.search_valid_class_ids,  # Only fork and cup
                    )

                    objs = summary.get("objects", [])
                    candidate = None
                    
                    # Find candidate object that meets criteria (already filtered to fork/cup by scan_window)
                    for o in objs:
                        cls_id = int(o.get("cls_id"))
                        
                        # Double-check it's a valid search object (should already be filtered)
                        if cls_id not in self.search_valid_class_ids:
                            continue
                        
                        if (
                            float(o.get("avg_conf", 0.0)) >= float(C.SCAN_MIN_CONF)
                            and int(o.get("frames", 0)) >= int(C.SCAN_MIN_FRAMES)
                            and cls_id not in exclude_ids
                        ):
                            candidate = o
                            break

                    if candidate is None:
                        # Nothing new to try at this pose -> advance to next pose
                        break
                    
                    # If we found a candidate object during scan window, this means object is in frame
                    # According to user: "scan can only happen if there is not an object of interest in frame"
                    # So if object is detected during scan, we should CENTER on it
                    cls_id = int(candidate["cls_id"])
                    detected_cls_name = candidate.get("cls_name", self._get_name(cls_id))
                    detected_conf = float(candidate.get("avg_conf", 0.0))
                    
                    print(f"[ScanWorker] =========================================")
                    print(f"[ScanWorker] Object detected during scan: {detected_cls_name} (conf={detected_conf:.2f})")
                    print(f"[ScanWorker] SCAN CYCLE PAUSED - Starting centering...")
                    print(f"[ScanWorker] Current position: x={x:.1f}mm, y={y:.1f}mm, z={z:.1f}mm")
                    print(f"[ScanWorker] =========================================")
                    
                    # CENTER on object - centering will continue until movement < 5mm
                    # After centering succeeds and object is registered, scan will resume
                    self._center_and_register_object(detected_cls_name, cls_id, x, y, z, pitch)
                    
                    # Object centered and registered - resume scan at this pose
                    print(f"[ScanWorker] Object centered and registered - RESUMING scan at current pose")
                    # Break out of inner scan loop, but continue with next iteration of pose loop
                    break  # Exit inner scan loop, continue to next pose

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
