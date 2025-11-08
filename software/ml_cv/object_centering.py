#!/usr/bin/env python3
"""
Object Centering Script - Live Feed
Shows live video feed with real-time x and y movement needed to center object.
Displays how far the arm needs to move in x and y to make the camera y-axis perpendicular to the object.
"""

from __future__ import annotations

import cv2
import numpy as np
import math
import time
from pathlib import Path
from typing import Optional, Tuple
from ultralytics import YOLO


# ============================================================================
# HARD-CODED PARAMETERS
# ============================================================================

# Camera configuration
CAMERA_INDEX = 4

# Pixel to mm conversion factor (adjustable parameter)
PIXEL_TO_MM = 1.0 / 2.90  # mm per pixel

# Working distance for object detection (mm)
# This is the approximate distance from camera to objects in the workspace
# Used to calculate movement needed - adjust based on your setup
WORKING_DISTANCE_MM = 600.0  # mm (typical working distance for arm operations)

# Movement scale factor (adjustable parameter)
# Multiplies the calculated movement to account for camera distance/perspective effects
# Increase this value if movements are too small, decrease if movements are too large
MOVEMENT_SCALE_FACTOR = 1.7 # Start with 1.0 and adjust as needed

# YOLO model path (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YOLO_MODEL_PATH = str(PROJECT_ROOT / "software" / "vizzy" / "laptop" / "yolo11m-seg.engine")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    
    Camera orientation:
    - Camera y-axis (top) aligns with global x-axis (radial direction)
    - Camera x-axis (right) points in negative global y direction (tangential)
    - To center object: camera x offset should be zero (object centered horizontally)
    
    Movement calculation (axes flipped):
    - Camera x offset (left/right) → x movement
    - Camera y offset (top/bottom) → y movement (with sign flip)
    
    Args:
        obj_offset_x_camera_mm: Object offset in camera x direction (left/right) in mm
        obj_offset_y_camera_mm: Object offset in camera y direction (top/bottom) in mm
        working_distance_mm: Working distance from camera to objects (mm) - not directly used but kept for future use
        scale_factor: Scale factor to multiply movement values (default 1.0)
    
    Returns:
        Tuple of (movement_x_mm, movement_y_mm)
        - movement_x_mm: Movement needed in global x direction (scaled)
        - movement_y_mm: Movement needed in global y direction (scaled)
    """
    # Swapped: Camera y offset maps to x movement, Camera x offset maps to y movement
    # Camera y offset (top/bottom) → x movement
    # Camera x offset (left/right) → y movement
    movement_x_mm = obj_offset_y_camera_mm
    
    # Camera x offset → y movement
    movement_y_mm = -obj_offset_x_camera_mm
    
    # Apply scale factor to both movements
    movement_x_mm *= scale_factor
    movement_y_mm *= scale_factor
    
    return movement_x_mm, movement_y_mm


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("=" * 80)
    print("Object Centering - Live Feed")
    print("=" * 80)
    print("Press 'q' to quit")
    print("=" * 80)
    
    # Load YOLO model
    print(f"\n[1] Loading YOLO model from: {YOLO_MODEL_PATH}")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"    Model loaded successfully")
    except Exception as e:
        print(f"    ERROR: Failed to load YOLO model: {e}")
        return
    
    # Open camera
    print(f"\n[2] Opening camera (index {CAMERA_INDEX})")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"    ERROR: Could not open camera index {CAMERA_INDEX}")
        return
    
    # Configure camera settings
    print(f"    Configuring camera settings...")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(800))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(600))
    cap.set(cv2.CAP_PROP_FPS, float(30.0))
    
    # Wait for camera to boot up
    print(f"    Waiting for camera to boot up (2 seconds)...")
    time.sleep(2.0)
    
    # Flush a few frames to ensure camera is ready
    print(f"    Flushing initial frames...")
    for _ in range(5):
        cap.grab()
    
    print(f"    Starting live feed...")
    
    # Get frame dimensions
    ok, test_frame = cap.read()
    if not ok or test_frame is None:
        print(f"    ERROR: Failed to read frame from camera")
        cap.release()
        return
    
    frame_h, frame_w = test_frame.shape[:2]
    center_x = frame_h // 2  # half of image height
    center_y = frame_w // 2  # half of image width
    
    print(f"    Frame size: {frame_w}x{frame_h} pixels")
    print(f"    Frame center: ({center_x}, {center_y})")
    
    # Main loop
    frame_count = 0
    fps_start_time = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("    WARNING: Failed to read frame")
            break
        
        frame_count += 1
        
        # Run YOLO inference
        results = model(frame, verbose=False)
        
        # Process detections
        best_detection = None
        best_conf = 0.0
        best_area = 0
        
        for result in results:
            if len(result.boxes) == 0:
                continue
            
            masks = getattr(result, "masks", None)
            mask_list = list(masks.data) if (masks is not None and masks.data is not None) else None
            
            n = len(result.boxes)
            for i in range(n):
                box_xyxy = result.boxes.xyxy[i].detach().cpu().numpy()
                conf = float(result.boxes.conf[i].item())
                cls_id = int(result.boxes.cls[i].item())
                x1, y1, x2, y2 = map(int, box_xyxy)
                area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                
                mask_tensor = mask_list[i] if mask_list is not None and i < len(mask_list) else None
                cx, cy = instance_center(box_xyxy, mask_tensor, frame_w, frame_h)
                
                # Object priority: highest confidence, break ties by larger area
                if (conf > best_conf) or (conf == best_conf and area > best_area):
                    best_conf = conf
                    best_area = area
                    best_detection = {
                        "cls_id": cls_id,
                        "cls_name": model.names[cls_id] if hasattr(model, "names") else str(cls_id),
                        "conf": conf,
                        "area": area,
                        "obj_x": cx,
                        "obj_y": cy,
                        "box_xyxy": box_xyxy,
                        "mask_tensor": mask_tensor,
                    }
        
        # Create annotated frame
        annotated = frame.copy()
        
        # Draw YOLO annotations
        for result in results:
            try:
                annotated = result.plot()
            except Exception:
                pass
        
        # Draw frame center (blue circle)
        center_img_x = center_y  # column (x-coordinate in image)
        center_img_y = center_x  # row (y-coordinate in image)
        cv2.circle(annotated, (center_img_x, center_img_y), 8, (255, 0, 0), -1)
        cv2.circle(annotated, (center_img_x, center_img_y), 12, (255, 0, 0), 2)
        
        # Calculate and display movement needed
        if best_detection is not None:
            cx = best_detection["obj_x"]  # column
            cy = best_detection["obj_y"]  # row
            
            # Calculate offsets in camera coordinate system
            cam_to_obj_x_camera = cx - center_y  # column offset (camera x direction, left/right)
            cam_to_obj_y_camera = cy - center_x  # row offset (camera y direction, top/bottom)
            
            # Convert to millimeters
            obj_offset_x_camera_mm = cam_to_obj_x_camera * PIXEL_TO_MM
            obj_offset_y_camera_mm = cam_to_obj_y_camera * PIXEL_TO_MM
            
            # Calculate movement needed
            movement_x_mm, movement_y_mm = calculate_movement_needed(
                obj_offset_x_camera_mm=obj_offset_x_camera_mm,
                obj_offset_y_camera_mm=obj_offset_y_camera_mm,
                working_distance_mm=WORKING_DISTANCE_MM,
                scale_factor=MOVEMENT_SCALE_FACTOR
            )
            
            # Draw object center (red circle)
            obj_img_x = cx
            obj_img_y = cy
            cv2.circle(annotated, (obj_img_x, obj_img_y), 8, (0, 0, 255), -1)
            cv2.circle(annotated, (obj_img_x, obj_img_y), 12, (0, 0, 255), 2)
            
            # Draw line connecting centers
            cv2.line(annotated, (obj_img_x, obj_img_y), (center_img_x, center_img_y), (0, 255, 0), 2)
        else:
            movement_x_mm = 0.0
            movement_y_mm = 0.0
            obj_offset_x_camera_mm = 0.0
            obj_offset_y_camera_mm = 0.0
        
        # Overlay text with movement information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        line_height = 30
        y_offset = 40
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        # Build text lines
        if best_detection is None:
            text_lines = [
                "No detections found",
                "",
                "Waiting for object...",
            ]
        else:
            text_lines = [
                f"Object: {best_detection['cls_name']} (conf={best_conf:.2f})",
                "",
                "MOVEMENT NEEDED:",
                f"  X: {movement_x_mm:+.2f} mm",
                f"  Y: {movement_y_mm:+.2f} mm",
                "",
                f"Scale: {MOVEMENT_SCALE_FACTOR:.2f}",
            ]
        
        # Draw text with background
        for i, line in enumerate(text_lines):
            y_pos = y_offset + i * line_height
            # Get text size for background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
            # Draw background rectangle
            cv2.rectangle(annotated, (10, y_pos - text_h - 5), (10 + text_w + 10, y_pos + 5), bg_color, -1)
            # Draw text
            cv2.putText(annotated, line, (10, y_pos), font, font_scale, color, font_thickness)
        
        # Calculate and display FPS
        if frame_count % 30 == 0:
            fps_elapsed = time.time() - fps_start_time
            fps = 30.0 / fps_elapsed if fps_elapsed > 0 else 0
            fps_start_time = time.time()
            fps_text = f"FPS: {fps:.1f}"
            (fps_w, fps_h), _ = cv2.getTextSize(fps_text, font, 0.5, 1)
            cv2.rectangle(annotated, (frame_w - fps_w - 20, 10), (frame_w - 10, 10 + fps_h + 10), bg_color, -1)
            cv2.putText(annotated, fps_text, (frame_w - fps_w - 15, 10 + fps_h + 5), font, 0.5, color, 1)
        
        # Display frame
        cv2.imshow("Object Centering - Live Feed (Press 'q' to quit)", annotated)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[3] Cleanup complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
