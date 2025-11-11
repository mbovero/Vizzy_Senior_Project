from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .hud import draw_wrapped_text
from .yolo_runner import clear_class_filter
from ..shared import config as C


# ------------------------------ helpers ------------------------------------- #

def _contour_center(mask_u8: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (cx, cy) using contour moments of the largest blob in a binary mask."""
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def instance_center(box_xyxy, mask_tensor, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """
    Return (cx, cy) for a single detection:
      - If a mask is provided, compute segmentation center (contour moments).
      - Else, return box center.
    """
    if mask_tensor is not None:
        # mask tensor is (Hmask, Wmask) float/bool; resize to frame dims, make binary u8
        mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
        mask_np = cv2.resize(mask_np, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        mask_u8 = (mask_np * 255).astype(np.uint8)
        c = _contour_center(mask_u8)
        if c is not None:
            return c

    # fallback: box center
    x1, y1, x2, y2 = map(int, box_xyxy)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def calculate_movement_needed(
    obj_offset_x_camera_mm: float,
    obj_offset_y_camera_mm: float,
    working_distance_mm: float,
    scale_factor: float = 1.0
) -> Tuple[float, float]:
    """
    Calculate the movement needed in x and y to center the object.
    This matches object_centering.py exactly.
    
    Camera orientation:
    - Camera y-axis (top/bottom in image) maps to global x-axis movement
    - Camera x-axis (left/right in image) maps to global y-axis movement (negated)
    
    Movement calculation (matching object_centering.py):
    - Camera y offset (top/bottom) → x movement
    - Camera x offset (left/right) → y movement (negated)
    
    Args:
        obj_offset_x_camera_mm: Object offset in camera x direction (left/right) in mm
        obj_offset_y_camera_mm: Object offset in camera y direction (top/bottom) in mm
        working_distance_mm: Working distance from camera to objects (mm) - kept for compatibility
        scale_factor: Scale factor to multiply movement values (default 1.0)
    
    Returns:
        Tuple of (movement_x_mm, movement_y_mm)
        - movement_x_mm: Movement needed in global x direction (scaled)
        - movement_y_mm: Movement needed in global y direction (scaled)
    """
    # Camera y offset (top/bottom) → x movement
    movement_x_mm = obj_offset_y_camera_mm
    
    # Camera x offset → y movement (negated)
    movement_y_mm = -obj_offset_x_camera_mm
    
    # Apply scale factor to both movements
    movement_x_mm *= scale_factor
    movement_y_mm *= scale_factor
    
    return movement_x_mm, movement_y_mm


# ----------------------------- main routine --------------------------------- #

def center_on_class(
    cap,
    model,
    target_cls: int,
    center_x: int,
    center_y: int,
    send_move: Callable[[float, float], None],
    display_scale: float,
    label: str,
    frame_sink: Callable[[np.ndarray], None],
    collect_frames: bool = False,
    current_x_mm: float = 0.0,
    current_y_mm: float = 0.0,
    current_z_mm: float = 275.0,
    current_pitch_deg: float = 0.0,
    motion=None,
    abort_event=None,
) -> Tuple[bool, list, float, float]:
    """
    Closed-loop centering on a class until item is successfully centered.
    Will continue indefinitely until success is achieved (unless aborted).

    Args:
        cap: OpenCV VideoCapture.
        model: YOLO model instance (Ultralytics).
        target_cls: Class ID to center on.
        center_x, center_y: Pixel coordinates of frame center.
        send_move: Callback taking normalized (dx, dy) in [-1,1] (legacy, not used if motion is provided).
        display_scale: Scale factor for displayed frames.
        label: HUD label prefix.
        frame_sink: Callback to publish frames to main/UI thread.
        collect_frames: If True, collect stable frames with masks for orientation calculation.
        current_x_mm: Current x position in mm (for absolute moves).
        current_y_mm: Current y position in mm (for absolute moves).
        current_z_mm: Current z position in mm (for absolute moves).
        current_pitch_deg: Current pitch angle in degrees (for absolute moves).
        motion: Motion object for absolute moves (if None, uses send_move callback).
        abort_event: Optional threading.Event to check for abort signal. If set, centering will exit.

    Returns:
        (success, collected_frames, final_x_mm, final_y_mm): 
        - success: bool
        - collected_frames: list of dicts with "frame" and "mask"
        - final_x_mm: final x position after centering (mm)
        - final_y_mm: final y position after centering (mm)
    """
    t0 = time.time()
    good_frames = 0
    success = False
    captured_frames = []
    # Initialize last_move_time to a time in the past so first measurement can happen immediately
    last_move_time = time.time() - (C.CENTER_MEASURE_WAIT_TIME_S + 1.0)  # Start with arm already "settled"
    arm_is_moving = False  # Track if arm is currently moving
    # Track when current movement cycle started - timeout resets after each movement
    last_movement_cycle_start = time.time()  # Start of current movement cycle (resets after each move)
    
    # Track current position for absolute moves (relative to baseline pose, not origin)
    # Start from the baseline pose passed in - this is where the arm currently is
    x_mm = current_x_mm  # Current x position relative to origin (mm) - baseline pose
    y_mm = current_y_mm  # Current y position relative to origin (mm) - baseline pose
    z_mm = current_z_mm  # Current z position relative to origin (mm) - baseline pose
    pitch_deg = current_pitch_deg  # Current pitch (degrees) - baseline pose
    
    print(f"[Centering] Starting centering at baseline position: x={x_mm:.2f}mm, y={y_mm:.2f}mm, z={z_mm:.2f}mm")
    print(f"[Centering] Measurements will only occur when arm is stopped")
    print(f"[Centering] Wait time after movement command: {C.CENTER_MEASURE_WAIT_TIME_S}s before next measurement")
    print(f"[Centering] Timeout: {C.CENTER_TIMEOUT_S}s per movement cycle - resets after each movement toward object")

    # Loop until success achieved or timeout
    # Exit on: success, timeout, abort event, or camera failure
    while not success:
        # Check for abort signal
        if abort_event is not None and abort_event.is_set():
            print(f"[Centering] Abort signal received - exiting centering mode")
            break
        
        current_time = time.time()
        
        # Check for timeout - based on time since last movement cycle start (resets after each move)
        time_since_last_cycle = current_time - last_movement_cycle_start
        if time_since_last_cycle >= C.CENTER_TIMEOUT_S:
            print(f"[Centering] Timeout exceeded ({time_since_last_cycle:.2f}s >= {C.CENTER_TIMEOUT_S}s) since last movement cycle - canceling centering")
            print(f"[Centering] Returning to scan pose")
            success = False
            break
        
        # Check if arm has settled after last movement
        # Only measure movement when 2 seconds have passed since the movement command
        time_since_last_move = current_time - last_move_time
        
        # If enough time has passed since last movement, arm is no longer moving
        if arm_is_moving and time_since_last_move >= C.CENTER_MEASURE_WAIT_TIME_S:
            arm_is_moving = False
            print(f"[Centering] Arm has stopped moving ({time_since_last_move:.2f}s elapsed) - ready for measurement")
        
        arm_is_settled = (time_since_last_move >= C.CENTER_MEASURE_WAIT_TIME_S) and not arm_is_moving
        
        ok, frame = cap.read()
        if not ok:
            break

        # Skip YOLO inference and movement measurement if arm is still moving or hasn't settled
        # Just show status and publish frame for live feed
        if not arm_is_settled:
            # Skip YOLO inference to save computation - just show status
            # Use a lightweight frame copy without heavy processing
            annotated = frame.copy()
            h, w = annotated.shape[:2]
            cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
            remaining_time = C.CENTER_MEASURE_WAIT_TIME_S - time_since_last_move
            if remaining_time > 0:
                status = f"Waiting for arm to stop ({time_since_last_move:.1f}s/{C.CENTER_MEASURE_WAIT_TIME_S}s)..."
            else:
                status = "Arm ready for measurement"
            hud = f"{label}  {status}"
            max_w = int(annotated.shape[1] * 0.92)
            draw_wrapped_text(annotated, hud, 10, 24, max_w)
            resized = cv2.resize(annotated, (int(w * display_scale), int(h * display_scale)))
            frame_sink(resized)
            time.sleep(0.033)  # ~30 FPS when waiting (reduces CPU usage)
            continue
        
        # Model inference filtered to the target class (only when arm is settled)
        # Use half precision and disable verbose for faster inference
        results = model(frame, classes=[int(target_cls)], verbose=False, half=True)  # half=True for faster inference
        annotated = frame  # fallback
        best_conf = 0.0
        best_area = 0
        best_xy = None
        best_mask_tensor = None
        
        for result in results:
            annotated = result.plot()

            # Choose highest-confidence instance (break ties by larger bbox area)
            if len(result.boxes) > 0:
                masks = getattr(result, "masks", None)
                mask_list = list(masks.data) if (masks is not None and masks.data is not None) else None

                n = len(result.boxes)
                for i in range(n):
                    cid = int(result.boxes.cls[i].item())
                    if cid != int(target_cls):
                        continue

                    box_xyxy = result.boxes.xyxy[i].detach().cpu().numpy()
                    conf = float(result.boxes.conf[i].item())
                    x1, y1, x2, y2 = map(int, box_xyxy)
                    area = max(0, (x2 - x1)) * max(0, (y2 - y1))

                    mask_tensor = mask_list[i] if mask_list is not None and i < len(mask_list) else None
                    cx, cy = instance_center(box_xyxy, mask_tensor, annotated.shape[1], annotated.shape[0])

                    if (conf > best_conf) or (conf == best_conf and area > best_area):
                        best_conf = conf
                        best_area = area
                        best_xy = (int(cx), int(cy))
                        best_mask_tensor = mask_tensor
        
        # Process best detection (outside the result loop)
        # Only process measurements when arm is settled (stopped)
        movement_x_mm = 0.0
        movement_y_mm = 0.0
        best_xy_valid = (best_xy is not None)
        movement_small_enough = False
        conf_ok = False
        err_ok = False
        move_ok = False
        dx_px = 0
        dy_px = 0
        err_px = 0
        
        if best_xy_valid:
            bx, by = best_xy
            # Note: bx is column (image x), by is row (image y)
            # center_x is column (image x), center_y is row (image y)

            # Pixel error relative to frame center (for display/stability checks)
            dx_px = center_x - bx     # >0 → object left of center (in image)
            dy_px = by - center_y     # >0 → object below center (in image)
            err_px = max(abs(dx_px), abs(dy_px))

            # Calculate camera coordinate offsets (matching object_centering.py exactly)
            # Note: In object_centering.py:
            #   - center_x = frame_h // 2 (row, y-coordinate in image)
            #   - center_y = frame_w // 2 (column, x-coordinate in image)
            #   - cx is column (x-coordinate), cy is row (y-coordinate)
            #   - cam_to_obj_x_camera = cx - center_y (column - center_column)
            #   - cam_to_obj_y_camera = cy - center_x (row - center_row)
            # In our code (from scan_worker):
            #   - center_x = w0 // 2 (column, x-coordinate in image)
            #   - center_y = h0 // 2 (row, y-coordinate in image)
            #   - bx is column (x-coordinate), by is row (y-coordinate)
            # To match object_centering.py calculation:
            #   - cam_to_obj_x_camera = bx - center_x (column - center_column)
            #   - cam_to_obj_y_camera = by - center_y (row - center_row)
            cam_to_obj_x_camera = bx - center_x  # column offset (camera x direction, left/right)
            cam_to_obj_y_camera = by - center_y  # row offset (camera y direction, top/bottom)
            
            # Convert to millimeters
            obj_offset_x_camera_mm = cam_to_obj_x_camera * C.PIXEL_TO_MM
            obj_offset_y_camera_mm = cam_to_obj_y_camera * C.PIXEL_TO_MM
            
            # Calculate movement needed in mm (using same logic as object_centering.py)
            movement_x_mm, movement_y_mm = calculate_movement_needed(
                obj_offset_x_camera_mm=obj_offset_x_camera_mm,
                obj_offset_y_camera_mm=obj_offset_y_camera_mm,
                working_distance_mm=C.WORKING_DISTANCE_MM,
                scale_factor=C.MOVEMENT_SCALE_FACTOR
            )
            
            # Normalized error for display/stability checks
            ndx = dx_px / max(1, center_x)
            ndy = dy_px / max(1, center_y)
            move_norm = max(abs(ndx), abs(ndy))

            # Threshold checks
            conf_ok = (best_conf >= C.CENTER_CONF)
            err_ok = (abs(dx_px) <= C.CENTER_EPSILON_PX and abs(dy_px) <= C.CENTER_EPSILON_PX)
            move_ok = (abs(ndx) < C.CENTER_MOVE_NORM and abs(ndy) < C.CENTER_MOVE_NORM)
            
            # NEW: Check if movement is less than 5mm in each direction
            # Movement must be < 5mm in both x and y to be considered "centered"
            movement_small_enough = (abs(movement_x_mm) < C.CENTER_MIN_MOVEMENT_MM and 
                                     abs(movement_y_mm) < C.CENTER_MIN_MOVEMENT_MM)

            # Command movement ONLY if movement is too large (>= 5mm in either direction)
            # If movement is small enough (< 5mm), we consider it stable regardless of pixel/stability thresholds
            if not movement_small_enough:
                if motion is not None:
                    # Movement is >= 5mm in at least one direction - need to move
                    # Calculate new absolute position by adding movement to CURRENT position
                    # This is relative to the current arm position, not the origin
                    # We accumulate movements from the baseline pose
                    new_x_mm = x_mm + movement_x_mm  # Add movement to current x
                    new_y_mm = y_mm + movement_y_mm  # Add movement to current y
                    
                    # Send absolute move command (new position relative to origin)
                    print(f"[Centering] Current pos: ({x_mm:.2f}, {y_mm:.2f})mm")
                    print(f"[Centering] Movement: dx={movement_x_mm:.2f}mm dy={movement_y_mm:.2f}mm")
                    print(f"[Centering] Movement too large (>= {C.CENTER_MIN_MOVEMENT_MM}mm) - moving arm")
                    print(f"[Centering] New pos: ({new_x_mm:.2f}, {new_y_mm:.2f})mm")
                    
                    ok = motion.move_to_target(new_x_mm, new_y_mm, z_mm, pitch_deg)
                    if ok:
                        # Update tracked position after successful move (accumulate from baseline)
                        x_mm = new_x_mm
                        y_mm = new_y_mm
                        print(f"[Centering] Move command sent and acknowledged")
                        print(f"[Centering] Updated target position to ({x_mm:.2f}, {y_mm:.2f})mm")
                        # Mark that arm is moving and record move time
                        # The ACK means the server received the command, but arm is still physically moving
                        arm_is_moving = True
                        last_move_time = time.time()
                        # Reset timeout timer - new movement cycle starts now
                        last_movement_cycle_start = time.time()
                        print(f"[Centering] Arm movement started - will wait {C.CENTER_MEASURE_WAIT_TIME_S}s before next measurement")
                        print(f"[Centering] Timeout timer reset - {C.CENTER_TIMEOUT_S}s per movement cycle")
                    else:
                        print(f"[Centering] Warning: Move command failed or timed out, keeping old position ({x_mm:.2f}, {y_mm:.2f})mm")
                        # If command failed, don't mark as moving
                        # But still wait a bit in case there's residual motion
                        last_move_time = time.time()
                        arm_is_moving = True
                    
                    # Don't set arm_is_moving = False here!
                    # The arm is physically moving even after ACK is received
                    # We'll check the time in the next iteration to see if enough time has passed
                    # Publish current frame with movement info before continuing to keep GUI live
                    if best_xy_valid:
                        bx, by = best_xy
                        cv2.circle(annotated, (bx, by), 6, (0, 0, 255), -1)
                        cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
                        cv2.line(annotated, (bx, by), (center_x, center_y), (0, 255, 0), 2)
                    else:
                        cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
                    hud = f"{label}  Moving arm... waiting {C.CENTER_MEASURE_WAIT_TIME_S}s"
                    max_w = int(annotated.shape[1] * 0.92)
                    draw_wrapped_text(annotated, hud, 10, 24, max_w)
                    h, w = annotated.shape[:2]
                    resized = cv2.resize(annotated, (int(w * display_scale), int(h * display_scale)))
                    frame_sink(resized)
                    # Sleep to prevent excessive CPU usage while waiting for arm movement
                    time.sleep(0.033)  # ~30 FPS update rate
                    # Continue to next iteration to wait for movement to complete
                    continue
                else:
                    # Legacy: use normalized moves (if motion not provided)
                    send_move(ndx, ndy)
            else:
                # Movement is small enough (< 5mm) - consider it stable
                if best_xy_valid:
                    print(f"[Centering] Movement small enough: dx={movement_x_mm:.2f}mm dy={movement_y_mm:.2f}mm (< {C.CENTER_MIN_MOVEMENT_MM}mm) - STABLE")

            # Count stable frames toward quota
            # PRIMARY REQUIREMENT: Movement must be small enough (< 5mm)
            # When movement is < 5mm, we consider it stable and count it as good
            # even if pixel/stability thresholds aren't perfect (movement threshold takes priority)
            # Only count frames where:
            # 1. Movement is small enough (< 5mm) - PRIMARY REQUIREMENT (must be met)
            # 2. Object is detected (best_xy_valid)
            # 3. Confidence is high enough (conf_ok)
            # Pixel error and motion stability (err_ok, move_ok) are NOT required when movement is < 5mm
            if movement_small_enough and best_xy_valid and conf_ok:
                good_frames += 1
                
                # Collect frames with masks if requested (only up to CENTER_FRAMES)
                # Only collect frames where item is successfully centered (still in frame)
                if collect_frames and len(captured_frames) < C.CENTER_FRAMES:
                    # Use the best mask tensor we found (item is in frame)
                    if best_mask_tensor is not None:
                        captured_frames.append({
                            "frame": frame.copy(),
                            "mask": best_mask_tensor
                        })
                
                # Success criteria: enough stable frames where item is centered, in frame, and movement < 5mm
                if good_frames >= C.CENTER_FRAMES:
                    success = True
                    print(f"[Centering] Success! Item centered (movement < {C.CENTER_MIN_MOVEMENT_MM}mm) for {good_frames} frames")
                    break  # Exit immediately when quota reached

            # ---------------- HUD ----------------
            if best_xy_valid:
                bx, by = best_xy
                # Object center (red)
                cv2.circle(annotated, (bx, by), 6, (0, 0, 255), -1)
                # Frame center (blue)
                cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
                # Line between them (green)
                cv2.line(annotated, (bx, by), (center_x, center_y), (0, 255, 0), 2)

                # Show movement status in HUD
                if movement_small_enough:
                    movement_status = f"OK (<{C.CENTER_MIN_MOVEMENT_MM}mm)"
                else:
                    movement_status = f"LARGE (>= {C.CENTER_MIN_MOVEMENT_MM}mm)"
                
                hud = (
                    f"{label}  quota {good_frames}/{C.CENTER_FRAMES}  "
                    f"conf {best_conf:.2f}  err {err_px:.0f}px  "
                    f"move_x {movement_x_mm:.1f}mm move_y {movement_y_mm:.1f}mm [{movement_status}]"
                )
            else:
                # No detection — item not in frame
                # Don't count this toward success (item must be in frame to be registered)
                cv2.circle(annotated, (center_x, center_y), 6, (255, 0, 0), -1)
                hud = f"{label}  quota {good_frames}/{C.CENTER_FRAMES}  (NO DETECTION - item not in frame)"

        # Draw HUD and publish frame (always, regardless of detection)
        max_w = int(annotated.shape[1] * 0.92)
        draw_wrapped_text(annotated, hud, 10, 24, max_w)
        h, w = annotated.shape[:2]
        resized = cv2.resize(annotated, (int(w * display_scale), int(h * display_scale)))
        frame_sink(resized)
        # Frame rate limiting for smooth GUI (~30 FPS)
        time.sleep(0.033)

    # Clear any class filter set during this centering attempt
    clear_class_filter(model)
    
    # Return final position after centering (so caller can return to baseline if needed)
    elapsed_total = time.time() - t0
    time_since_last_cycle_final = time.time() - last_movement_cycle_start
    if success:
        print(f"[Centering] Centering SUCCESS - item was in frame and centered")
        print(f"[Centering] Final centered position: x={x_mm:.2f}mm, y={y_mm:.2f}mm")
        print(f"[Centering] Object location will be registered at this position")
        print(f"[Centering] Total centering time: {elapsed_total:.2f}s")
    else:
        if abort_event is not None and abort_event.is_set():
            print(f"[Centering] Centering ABORTED by user/system")
        elif time_since_last_cycle_final >= C.CENTER_TIMEOUT_S:
            print(f"[Centering] Centering TIMED OUT - no movement within {time_since_last_cycle_final:.2f}s (limit: {C.CENTER_TIMEOUT_S}s per cycle)")
            print(f"[Centering] Returning to scan pose without registering object")
        else:
            print(f"[Centering] Centering EXITED - camera failure or other error")
        print(f"[Centering] Final position: x={x_mm:.2f}mm, y={y_mm:.2f}mm")
    print(f"[Centering] Exiting centering mode")

    return success, captured_frames, x_mm, y_mm
