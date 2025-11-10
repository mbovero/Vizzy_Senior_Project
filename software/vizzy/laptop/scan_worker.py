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
        # Arm movement state tracking
        self._arm_moving = threading.Event()  # Set when arm is moving, cleared when stopped
        self._last_move_time = 0.0  # Timestamp of last movement command
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
        print(f"[ScanWorker] DEBUG: _centering_active set to True")
        
        # Centering loop - continues until movement < 5mm and object is centered
        # Will NOT exit until item is successfully centered (no timeout)
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
            abort_event=self.events.scan_abort,  # Allow abort via scan_abort event
        )
        
        # Clear centering flag
        self._centering_active = False
        print(f"[ScanWorker] DEBUG: _centering_active set to False")
        
        # Only register object if centering was successful (movement < 5mm and object centered)
        if center_success:
            print(f"[ScanWorker] Centering SUCCESS - movement < {C.CENTER_MIN_MOVEMENT_MM}mm")
            print(f"[ScanWorker] Final centered position: x={final_x_mm:.2f}mm, y={final_y_mm:.2f}mm, z=0.0mm (stored as 0)")
            
            # Use the final centered position for object location
            # Z coordinate is always stored as 0 (objects are on the table/work surface)
            loc = {
                "x": final_x_mm,  # Final x after centering
                "y": final_y_mm,  # Final y after centering
                "z": 0.0,         # Z always stored as 0 (object is on table/work surface)
            }
            
            # Calculate grasp orientation from collected frames
            orientation_data = self._calculate_orientation_from_collected_frames(collected_frames)
            
            # Apply yaw transformation: positive yaw - 90°, negative yaw + 90°
            # Then convert to radians for storage
            import math
            yaw_deg = orientation_data.get("grasp_yaw", 0.0)
            
            if yaw_deg > 0:
                corrected_yaw_deg = yaw_deg - 90.0
            elif yaw_deg < 0:
                corrected_yaw_deg = yaw_deg + 90.0
            else:
                corrected_yaw_deg = yaw_deg  # yaw is exactly 0, no change needed
            
            # Convert to radians for storage
            corrected_yaw_rad = math.radians(corrected_yaw_deg)
            
            print(f"[ScanWorker] Yaw transformation: {yaw_deg:.2f}° -> {corrected_yaw_deg:.2f}° -> {corrected_yaw_rad:.4f} rad")
            
            # Update orientation data with corrected yaw in radians
            orientation_data["grasp_yaw"] = corrected_yaw_rad
            
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
            
            # Return to baseline pose after successful centering and registration
            # Continuously publish frames while returning to baseline
            print(f"[ScanWorker] Returning to baseline pose: x={x:.2f}mm, y={y:.2f}mm, z={z:.2f}mm")
            # Mark arm as moving before returning to baseline
            self._arm_moving.set()
            returned = self.motion.move_to_target(x, y, z, pitch)
            if not returned:
                print("[ScanWorker] Warning: Baseline MOVE_TO did not confirm; continuing after dwell")
            
            # Wait for arm to settle after returning to baseline
            # Use same settle time as when moving to a pose
            settle_time = C.MOVE_SETTLE_S + getattr(C, 'POSE_CV_DELAY_S', 0.3)
            print(f"[ScanWorker] Waiting {settle_time:.1f}s for arm to settle at baseline after centering...")
            settle_start = time.time()
            while (time.time() - settle_start) < settle_time:
                grabbed = self.cap.grab()
                if grabbed:
                    ok, frame = self.cap.retrieve()
                else:
                    ok, frame = self.cap.read()
                if ok and self.frame_sink:
                    h, w = frame.shape[:2]
                    from .hud import draw_wrapped_text
                    elapsed = time.time() - settle_start
                    status = f"Returning to baseline... {settle_time - elapsed:.1f}s"
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, int(28 * self.display_scale)), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                    draw_wrapped_text(frame, status, 8, 8, int(w * 0.8))
                    resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)))
                    self.frame_sink(resized)
                time.sleep(0.033)  # ~30 FPS while waiting
            
            # Mark arm as stopped after settling
            self._arm_moving.clear()
            print(f"[ScanWorker] Arm stopped at baseline after centering")
            
            # Wait for dwell time while continuously publishing frames
            dwell_start = time.time()
            while (time.time() - dwell_start) < C.RETURN_TO_POSE_DWELL_S:
                grabbed = self.cap.grab()
                if grabbed:
                    ok, frame = self.cap.retrieve()
                else:
                    ok, frame = self.cap.read()
                if ok and self.frame_sink:
                    h, w = frame.shape[:2]
                    from .hud import draw_wrapped_text
                    elapsed = time.time() - dwell_start
                    status = f"Baseline dwell... {C.RETURN_TO_POSE_DWELL_S - elapsed:.1f}s"
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, int(28 * self.display_scale)), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                    draw_wrapped_text(frame, status, 8, 8, int(w * 0.8))
                    resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)))
                    self.frame_sink(resized)
                time.sleep(0.033)  # ~30 FPS while waiting
            
            self._flush_capture()
            print(f"[ScanWorker] Returned to baseline pose - ready to resume scan")
        else:
            # Centering was aborted or camera failed - should not happen in normal operation
            # since centering now continues until success
            if self.events.scan_abort.is_set():
                print(f"[ScanWorker] Centering ABORTED - scan cycle cancelled")
            else:
                print(f"[ScanWorker] Centering EXITED unexpectedly (camera failure?)")
            print(f"[ScanWorker] Object NOT registered")
            
            # Still return to baseline pose even if centering was aborted
            # Continuously publish frames while returning to baseline
            print(f"[ScanWorker] Returning to baseline pose: x={x:.2f}mm, y={y:.2f}mm, z={z:.2f}mm")
            # Mark arm as moving before returning to baseline
            self._arm_moving.set()
            returned = self.motion.move_to_target(x, y, z, pitch)
            if not returned:
                print("[ScanWorker] Warning: Baseline MOVE_TO did not confirm; continuing after dwell")
            
            # Wait for arm to settle after returning to baseline
            settle_time = C.MOVE_SETTLE_S + getattr(C, 'POSE_CV_DELAY_S', 0.3)
            print(f"[ScanWorker] Waiting {settle_time:.1f}s for arm to settle at baseline after centering abort...")
            settle_start = time.time()
            while (time.time() - settle_start) < settle_time:
                grabbed = self.cap.grab()
                if grabbed:
                    ok, frame = self.cap.retrieve()
                else:
                    ok, frame = self.cap.read()
                if ok and self.frame_sink:
                    h, w = frame.shape[:2]
                    from .hud import draw_wrapped_text
                    elapsed = time.time() - settle_start
                    status = f"Returning to baseline... {settle_time - elapsed:.1f}s"
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, int(28 * self.display_scale)), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                    draw_wrapped_text(frame, status, 8, 8, int(w * 0.8))
                    resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)))
                    self.frame_sink(resized)
                time.sleep(0.033)  # ~30 FPS while waiting
            
            # Mark arm as stopped after settling
            self._arm_moving.clear()
            print(f"[ScanWorker] Arm stopped at baseline after centering abort")
            
            # Wait for dwell time while continuously publishing frames
            dwell_start = time.time()
            while (time.time() - dwell_start) < C.RETURN_TO_POSE_DWELL_S:
                grabbed = self.cap.grab()
                if grabbed:
                    ok, frame = self.cap.retrieve()
                else:
                    ok, frame = self.cap.read()
                if ok and self.frame_sink:
                    h, w = frame.shape[:2]
                    from .hud import draw_wrapped_text
                    elapsed = time.time() - dwell_start
                    status = f"Baseline dwell... {C.RETURN_TO_POSE_DWELL_S - elapsed:.1f}s"
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, int(28 * self.display_scale)), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                    draw_wrapped_text(frame, status, 8, 8, int(w * 0.8))
                    resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)))
                    self.frame_sink(resized)
                time.sleep(0.033)  # ~30 FPS while waiting
            
            self._flush_capture()
            print(f"[ScanWorker] Returned to baseline pose")
    
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
                time.sleep(0.033)  # ~30 FPS if read fails (matches display rate)
                continue
            
            frame_count += 1
            
            # Run YOLO to check if object is still in frame
            # Use half precision for faster inference
            results = self.model(frame, classes=[cls_id], verbose=False, half=True)
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
            
            # Balanced sleep for smooth GUI without excessive CPU usage (~30 FPS target)
            time.sleep(0.033)  # ~30 FPS update rate for smooth, responsive GUI
    
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
                print(f"[ScanWorker] Moving to pose {pid}: x={x:.1f}mm, y={y:.1f}mm, z={z:.1f}mm, pitch={pitch:.3f}°")
                
                # Mark arm as moving
                self._arm_moving.set()
                self._last_move_time = time.time()
                
                ok = self.motion.move_to_target(x, y, z, pitch)
                if not ok:
                    print("[ScanWorker] Warning: MOVE_TO baseline timed out or failed; continuing after settle")

                # Wait for arm to settle while continuously publishing frames to keep GUI live
                settle_time = C.MOVE_SETTLE_S + getattr(C, 'POSE_CV_DELAY_S', 0.3)
                print(f"[ScanWorker] Waiting {settle_time:.1f}s for arm to settle at pose {pid}...")
                settle_start = time.time()
                while (time.time() - settle_start) < settle_time:
                    # Continuously read and publish frames to prevent GUI freezing
                    grabbed = self.cap.grab()
                    if grabbed:
                        ok, frame = self.cap.retrieve()
                    else:
                        ok, frame = self.cap.read()
                    if ok:
                        # Publish frame with status (no YOLO during movement)
                        h, w = frame.shape[:2]
                        from .hud import draw_wrapped_text
                        elapsed = time.time() - settle_start
                        status = f"Moving to pose {pid}... waiting {settle_time - elapsed:.1f}s"
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, int(28 * self.display_scale)), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                        draw_wrapped_text(frame, status, 8, 8, int(w * 0.8))
                        resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)))
                        if self.frame_sink:
                            self.frame_sink(resized)
                    time.sleep(0.033)  # ~30 FPS while waiting
                
                # Add 5 second delay after first command (pose 0) - but keep publishing frames
                if pid == 0:
                    print("[ScanWorker] First pose reached - waiting 5 seconds...")
                    wait_start = time.time()
                    while (time.time() - wait_start) < 5.0:
                        # Continuously read and publish frames
                        grabbed = self.cap.grab()
                        if grabbed:
                            ok, frame = self.cap.retrieve()
                        else:
                            ok, frame = self.cap.read()
                        if ok:
                            h, w = frame.shape[:2]
                            from .hud import draw_wrapped_text
                            elapsed = time.time() - wait_start
                            status = f"First pose - waiting {5.0 - elapsed:.1f}s"
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (w, int(28 * self.display_scale)), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                            draw_wrapped_text(frame, status, 8, 8, int(w * 0.8))
                            resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)))
                            if self.frame_sink:
                                self.frame_sink(resized)
                        time.sleep(0.033)  # ~30 FPS while waiting
                
                # Mark arm as stopped (ready for YOLO detection)
                self._arm_moving.clear()
                print(f"[ScanWorker] Arm stopped at pose {pid} - YOLO detection enabled")
                
                # Wait a bit more to ensure arm is fully settled before detection - but keep publishing frames
                wait_start = time.time()
                while (time.time() - wait_start) < 0.2:
                    grabbed = self.cap.grab()
                    if grabbed:
                        ok, frame = self.cap.retrieve()
                    else:
                        ok, frame = self.cap.read()
                    if ok:
                        # Publish frame (no YOLO yet, just status)
                        h, w = frame.shape[:2]
                        from .hud import draw_wrapped_text
                        status = f"Arm settling at pose {pid}..."
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, int(28 * self.display_scale)), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                        draw_wrapped_text(frame, status, 8, 8, int(w * 0.8))
                        resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)))
                        if self.frame_sink:
                            self.frame_sink(resized)
                    time.sleep(0.033)  # ~30 FPS while waiting
                
                # CRITICAL: Check if object of interest is in frame BEFORE starting scan loop
                # If detected, STOP scan cycle immediately - do NOT proceed with scanning or centering
                # Use grab/retrieve for non-blocking read
                grabbed = self.cap.grab()
                if grabbed:
                    ok, check_frame = self.cap.retrieve()
                else:
                    ok, check_frame = self.cap.read()
                
                if ok:
                    # Quick detection check for objects we care about
                    # Use half precision for faster inference
                    check_results = self.model(check_frame, classes=self.search_valid_class_ids, verbose=False, half=True)
                    object_detected = False
                    detected_cls_name = None
                    detected_cls_id = None
                    detected_conf = 0.0
                    existing_objects = []  # Initialize to avoid NameError
                    
                    # Annotate frame with YOLO results and publish it so user can see what YOLO sees
                    annotated_frame = check_frame.copy()
                    for r in check_results:
                        try:
                            annotated_frame = r.plot()  # YOLO-annotated frame
                        except Exception:
                            pass
                        
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
                    
                    # Always publish the annotated frame so user can see YOLO detections
                    h, w = annotated_frame.shape[:2]
                    from .hud import draw_wrapped_text
                    if object_detected:
                        status = f"Object detected: {detected_cls_name} (conf: {detected_conf:.2f})"
                    else:
                        status = f"Scanning at pose {pid}..."
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, int(28 * self.display_scale)), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
                    draw_wrapped_text(annotated_frame, status, 8, 8, int(w * 0.8))
                    resized = cv2.resize(annotated_frame, (int(w * self.display_scale), int(h * self.display_scale)))
                    if self.frame_sink:
                        self.frame_sink(resized)
                    
                    if object_detected:
                        # Check if objects of this class have already been registered
                        existing_objects = self.memory.get_objects_by_class(detected_cls_id)
                        if len(existing_objects) > 0:
                            print(f"[ScanWorker] =========================================")
                            print(f"[ScanWorker] OBJECT DETECTED: {detected_cls_name} (conf={detected_conf:.2f})")
                            print(f"[ScanWorker] BUT: {len(existing_objects)} object(s) of class '{detected_cls_name}' already registered")
                            print(f"[ScanWorker] SKIPPING centering, but will still scan for other objects")
                            print(f"[ScanWorker] Continuing to scan window...")
                            print(f"[ScanWorker] =========================================")
                            # Object already registered - skip centering but STILL run scan window
                            # to look for OTHER objects (e.g., cups, knives, spoons)
                            # Fall through to scan loop below
                        else:
                            # Object detected and NOT registered - center on it immediately
                            print(f"[ScanWorker] =========================================")
                            print(f"[ScanWorker] OBJECT DETECTED: {detected_cls_name} (conf={detected_conf:.2f})")
                            print(f"[ScanWorker] SCAN CYCLE PAUSED - Starting centering...")
                            print(f"[ScanWorker] Current position: x={x:.1f}mm, y={y:.1f}mm, z={z:.1f}mm")
                            print(f"[ScanWorker] =========================================")
                            
                            # CENTER on object - centering will continue until movement < 5mm
                            # After centering succeeds and object is registered, scan will resume
                            self._center_and_register_object(detected_cls_name, detected_cls_id, x, y, z, pitch)
                            
                            # Centering complete and object registered - continue scan cycle
                            print(f"[ScanWorker] Object centered and registered - CONTINUING scan cycle to next pose")
                            print(f"[ScanWorker] Skipping to next pose in scan path...")
                            # Continue with the scan cycle (next pose)
                            continue  # Skip to next pose in the path
                
                # Run scan loop to look for objects (even if one was detected but already registered)
                # This ensures we scan for ALL object types (fork, cup, knife, spoon)
                # Per-pose repeat: try to find and center multiple distinct classes
                attempts = 0
                fails_at_pose: Dict[int, int] = {}
                scan_window_count = 0  # Track number of scan windows run at this pose
                if object_detected and len(existing_objects) > 0:
                    print(f"[ScanWorker] Starting scan loop at pose {pid} to look for other objects...")
                else:
                    print(f"[ScanWorker] Starting scan loop at pose {pid}...")

                # Ensure we start each scan loop with fresh frames from the baseline pose
                self._flush_capture()

                while not self.events.scan_abort.is_set():
                    if self._centering_active:
                        # Defensive: should never run when centering flag is set.
                        # But still publish frames to keep GUI live
                        print(f"[ScanWorker] DEBUG: Scan loop waiting - centering is active (pose {pid})")
                        grabbed = self.cap.grab()
                        if grabbed:
                            ok, frame = self.cap.retrieve()
                        else:
                            ok, frame = self.cap.read()
                        if ok and self.frame_sink:
                            h, w = frame.shape[:2]
                            resized = cv2.resize(frame, (int(w * self.display_scale), int(h * self.display_scale)))
                            self.frame_sink(resized)
                        time.sleep(0.033)  # ~30 FPS while centering is active
                        continue
                    
                    # BEFORE scanning, check again if object is in frame
                    # Only check if arm is stopped (no detection during movement)
                    if not self._arm_moving.is_set():
                        # Non-blocking camera read
                        grabbed = self.cap.grab()
                        if grabbed:
                            ok, pre_scan_frame = self.cap.retrieve()
                        else:
                            ok, pre_scan_frame = self.cap.read()
                        
                        if ok:
                            # Only run YOLO when arm is stopped
                            # Use half precision for faster inference
                            pre_scan_results = self.model(pre_scan_frame, classes=self.search_valid_class_ids, verbose=False, half=True)
                            
                            object_detected_pre = False
                            detected_cls_name_pre = None
                            detected_cls_id_pre = None
                            detected_conf_pre = 0.0
                            
                            # Annotate frame with YOLO results and publish it
                            annotated_pre_frame = pre_scan_frame.copy()
                            for r in pre_scan_results:
                                try:
                                    annotated_pre_frame = r.plot()  # YOLO-annotated frame
                                except Exception:
                                    pass
                                
                                if len(r.boxes) > 0:
                                    for i in range(len(r.boxes)):
                                        cls_id = int(r.boxes.cls[i].item())
                                        conf = float(r.boxes.conf[i].item())
                                        if cls_id in self.search_valid_class_ids and conf >= C.SCAN_MIN_CONF:
                                            # Check if objects of this class have already been registered
                                            existing_objects = self.memory.get_objects_by_class(cls_id)
                                            if len(existing_objects) > 0:
                                                # Object already registered - skip it
                                                print(f"[ScanWorker] Object '{self._get_name(cls_id)}' detected but already registered - ignoring")
                                                continue
                                            
                                            object_detected_pre = True
                                            detected_cls_name_pre = self._get_name(cls_id)
                                            detected_cls_id_pre = cls_id
                                            detected_conf_pre = conf
                                            break
                                    if object_detected_pre:
                                        break
                            
                            # Always publish the annotated frame so user can see YOLO detections
                            h, w = annotated_pre_frame.shape[:2]
                            from .hud import draw_wrapped_text
                            if object_detected_pre:
                                status = f"Object detected: {detected_cls_name_pre} (conf: {detected_conf_pre:.2f})"
                            else:
                                status = f"Scanning at pose {pid}..."
                            overlay = annotated_pre_frame.copy()
                            cv2.rectangle(overlay, (0, 0), (w, int(28 * self.display_scale)), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.35, annotated_pre_frame, 0.65, 0, annotated_pre_frame)
                            draw_wrapped_text(annotated_pre_frame, status, 8, 8, int(w * 0.8))
                            resized = cv2.resize(annotated_pre_frame, (int(w * self.display_scale), int(h * self.display_scale)))
                            if self.frame_sink:
                                self.frame_sink(resized)
                            
                            if object_detected_pre:
                                print(f"[ScanWorker] Object detected during scan: {detected_cls_name_pre} (conf={detected_conf_pre:.2f})")
                                print(f"[ScanWorker] PAUSING scan - starting centering...")
                                print(f"[ScanWorker] DEBUG: Before centering - scan_window_count={scan_window_count} at pose {pid}")
                                # Center on object - centering will continue until movement < 5mm
                                self._center_and_register_object(detected_cls_name_pre, detected_cls_id_pre, x, y, z, pitch)
                                # Object centered and registered - continue to next pose
                                print(f"[ScanWorker] Object centered and registered - CONTINUING to next pose in scan cycle")
                                print(f"[ScanWorker] DEBUG: After centering - breaking out of scan loop, scan_window_count={scan_window_count} at pose {pid}")
                                break  # Exit inner scan loop, continue to next pose

                    # Exclusions: already updated this session, or too many fails here
                    exclude_ids = [int(e["cls_id"]) for e in self.memory.entries_sorted()
                                   if e.get("updated_this_session") == 1]
                    for cid, fcount in fails_at_pose.items():
                        if fcount >= C.MAX_FAILS_PER_POSE:
                            exclude_ids.append(int(cid))

                    # CRITICAL: Check if centering just completed - if so, don't run scan window
                    # This prevents scan windows from running immediately after centering when arm might still be moving
                    if self._centering_active:
                        print(f"[ScanWorker] DEBUG: Skipping scan window - centering is still active (should not happen here)")
                        continue
                    
                    # Also check if arm is moving - wait for it to stop before running scan window
                    # This ensures scan windows only run when arm is fully settled
                    if self._arm_moving.is_set():
                        print(f"[ScanWorker] DEBUG: Arm is moving - waiting for arm to stop before scan window")
                        # Wait for arm to stop (with timeout to prevent infinite wait)
                        wait_start = time.time()
                        while self._arm_moving.is_set() and (time.time() - wait_start) < 5.0:
                            time.sleep(0.1)
                            if self._centering_active:
                                print(f"[ScanWorker] DEBUG: Centering started while waiting for arm - breaking")
                                break
                        if self._centering_active:
                            continue
                    
                    # Scan the window at this pose (publish frames via frame_sink)
                    # Only scan for fork and cup (use search_valid_class_ids instead of allowed_class_ids)
                    # Pass arm movement state checker - only run YOLO when arm is stopped
                    scan_window_count += 1
                    print(f"[ScanWorker] DEBUG: Starting scan window #{scan_window_count} at pose {pid} (centering_active: {self._centering_active}, arm_moving: {self._arm_moving.is_set()})")
                    scan_start_time = time.time()
                    summary = run_scan_window(
                        cap=self.cap,
                        model=self.model,
                        exclude_ids=exclude_ids,
                        get_name=self._get_name,
                        min_frames_for_class=C.SCAN_MIN_FRAMES,
                        frame_sink=self.frame_sink,
                        display_scale=self.display_scale,
                        allowed_class_ids=self.search_valid_class_ids,  # Only fork and cup
                        arm_is_stopped=lambda: not self._arm_moving.is_set(),  # Only run YOLO when arm is stopped
                    )
                    scan_duration = time.time() - scan_start_time
                    print(f"[ScanWorker] DEBUG: Scan window #{scan_window_count} completed in {scan_duration:.3f}s at pose {pid} (expected: {C.SCAN_DURATION_MS/1000.0:.3f}s)")
                    
                    # Check if centering became active during scan window - if so, break immediately
                    if self._centering_active:
                        print(f"[ScanWorker] DEBUG: Centering became active during scan window - breaking out of scan loop")
                        break

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
                    print(f"[ScanWorker] DEBUG: Before centering (from scan window results) - scan_window_count={scan_window_count} at pose {pid}")
                    self._center_and_register_object(detected_cls_name, cls_id, x, y, z, pitch)
                    
                    # Object centered and registered - continue to next pose
                    print(f"[ScanWorker] Object centered and registered - CONTINUING to next pose in scan cycle")
                    print(f"[ScanWorker] DEBUG: After centering (from scan window results) - breaking out of scan loop, scan_window_count={scan_window_count} at pose {pid}")
                    # Break out of inner scan loop, continue with next iteration of pose loop
                    break  # Exit inner scan loop, continue to next pose

                # end while per-pose
                print(f"[ScanWorker] DEBUG: Exiting scan loop at pose {pid} - total scan windows run: {scan_window_count}")
                print(f"[ScanWorker] Continuing to next pose in scan path...")

                if self.events.scan_abort.is_set():
                    print(f"[ScanWorker] Scan abort requested - stopping scan cycle")
                    break

            # end for pose in path
            print(f"[ScanWorker] Completed scan cycle - all poses visited")

        finally:
            # Prune entries not updated this session (end-of-search behavior)
            try:
                self.memory.prune_not_updated()
            except Exception:
                pass

            # Tell orchestrator we’re done
            self.events.scan_finished.set()
