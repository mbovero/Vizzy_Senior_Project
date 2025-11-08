#!/usr/bin/env python3
"""
Object Centering Script
Prototype script that captures a frame from camera, detects objects using YOLO segmentation,
calculates object center coordinates, and transforms them from camera frame coordinates to
robotic arm plane coordinates.
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

# Arm claw coordinates in mm
X_A = 370.0  # mm
Y_A = -24.0    # mm

# Add a calibration factor
SCALE_FACTOR_X = 1.0  # Tune this
SCALE_FACTOR_Y = 1.0  # Tune this

# Camera offset: camera is 35mm behind the claw position along the radius line
CAMERA_OFFSET_MM = -34.5  # mm

# Pixel to mm conversion factor (adjustable parameter)
PIXEL_TO_MM = 1.0 / 2.90  # mm per pixel

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


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("=" * 80)
    print("Object Centering Script")
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
    
    # Capture frame
    print(f"    Capturing frame...")
    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"    ERROR: Failed to read frame from camera")
        cap.release()
        return
    
    frame_h, frame_w = frame.shape[:2]
    print(f"    Frame captured: {frame_w}x{frame_h} pixels")
    
    # Calculate frame center (corrected orientation)
    center_x = frame_h // 2  # half of image height
    center_y = frame_w // 2  # half of image width
    print(f"    Frame center: ({center_x}, {center_y})")
    
    # Run YOLO inference
    print(f"\n[3] Running YOLO inference...")
    results = model(frame, verbose=False)
    
    # Process detections
    print(f"    Processing detections...")
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
    
    if best_detection is None:
        print(f"    WARNING: No detections found - will display frame without calculations")
        cx = None
        cy = None
        obj_x = None
        obj_y = None
        cam_to_obj_x = None
        cam_to_obj_y = None
        x_c = None
        y_c = None
        xT = None
        yT = None
        x_cam = None
        y_cam = None
        x_arm = None
        y_arm = None
        theta_rad = None
        theta_deg = None
    else:
        print(f"    Best detection: {best_detection['cls_name']} (conf={best_conf:.3f})")
        
        # Calculate camera-to-object offsets in pixels
        # Due to camera orientation: camera x-axis maps to image rows (height), 
        # camera y-axis maps to image columns (width)
        cx = best_detection["obj_x"]  # column (x-coordinate in image space)
        cy = best_detection["obj_y"]  # row (y-coordinate in image space)
        print(f"    Object center (image pixels): ({cx}, {cy})")
        
        obj_x = cy  # camera x = image row (due to orientation)
        obj_y = cx  # camera y = image column (due to orientation)
        print(f"    Object center (camera coords): ({obj_x}, {obj_y})")
        cam_to_obj_x = obj_x - center_x
        cam_to_obj_y = obj_y - center_y
        
        print(f"\n[4] Camera frame coordinates:")
        print(f"    cam_to_obj_x (pixels): {cam_to_obj_x}")
        print(f"    cam_to_obj_y (pixels): {cam_to_obj_y}")
        
        # Convert to millimeters
        x_c = cam_to_obj_x * PIXEL_TO_MM
        y_c = cam_to_obj_y * PIXEL_TO_MM
        print(f"    x_c (mm): {x_c:.3f}")
        print(f"    y_c (mm): {y_c:.3f}")
        
        # Calculate rotation angle
        print(f"\n[5] Coordinate transformation:")
        print(f"    Arm claw position: ({X_A:.1f}, {Y_A:.1f}) mm")
        
        if abs(Y_A) < 1e-6:
            # When Y_A = 0, planes are aligned (theta = 0)
            theta_rad = 0.0
            print(f"    NOTE: y_a is zero, setting theta = 0 (planes aligned)")
        else:
            theta_rad = math.atan(X_A / Y_A)
        
        theta_deg = math.degrees(theta_rad)
        print(f"    Theta (rotation angle): {theta_rad:.4f} rad ({theta_deg:.2f} deg)")
        
        # Apply rotation matrix transformation
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        xT = (c * x_c - s * y_c) * SCALE_FACTOR_X
        yT = (s * x_c + c * y_c) * SCALE_FACTOR_Y
        
        # Sign correction based on testing:
        # - When Y_A < 0: signs are correct, no flip needed
        # - When Y_A > 0: both xT and yT need to be flipped
        # - When Y_A = 0: only yT needs to be flipped
        if Y_A > 0:
            xT = -xT
            yT = -yT
            print(f"    NOTE: Y_A > 0, flipped signs of xT and yT")
        elif abs(Y_A) < 1e-6:  # Y_A == 0
            yT = -yT
            print(f"    NOTE: Y_A == 0, flipped sign of yT only")
        
        print(f"    xT (transformed offset): {xT:.3f} mm")
        print(f"    yT (transformed offset): {yT:.3f} mm")
        
        # Calculate camera position in arm plane (camera is 35mm behind claw)
        print(f"\n[6] Camera offset calculation:")
        r_a = math.sqrt(X_A**2 + Y_A**2)
        print(f"    Claw radius (r_a): {r_a:.3f} mm")
        
        if r_a < 1e-6:
            print(f"    ERROR: Claw radius is zero, cannot calculate camera position")
            cap.release()
            return
        
        r_cam = r_a - CAMERA_OFFSET_MM
        if r_cam < 0:
            print(f"    WARNING: Camera offset ({CAMERA_OFFSET_MM} mm) is larger than claw radius ({r_a:.3f} mm)")
            print(f"    Setting r_cam to 0")
            r_cam = 0.0
        
        print(f"    Camera radius (r_cam): {r_cam:.3f} mm")
        
        # Calculate camera position (scaled along radius line)
        scale = r_cam / r_a if r_a > 0 else 0.0
        x_cam = X_A * scale
        y_cam = Y_A * scale
        print(f"    Camera position: ({x_cam:.3f}, {y_cam:.3f}) mm")
        
        # Calculate camera offset components along radius line
        # This represents the offset from claw to camera in arm plane coordinates
        if r_a > 1e-6:
            offset_scale = CAMERA_OFFSET_MM / r_a
            camera_offset_x = X_A * offset_scale
            camera_offset_y = Y_A * offset_scale
            print(f"    Camera offset from claw: ({camera_offset_x:.3f}, {camera_offset_y:.3f}) mm")
        else:
            camera_offset_x = 0.0
            camera_offset_y = 0.0
        
        # Final arm plane coordinates relative to claw position
        # Object position = claw position + object offset from camera + camera offset
        # xT and yT are the object offset from camera center in arm plane coordinates
        # We add the camera offset to translate from camera reference frame to claw reference frame
        x_arm = X_A + xT - camera_offset_x
        y_arm = Y_A + yT + camera_offset_y
        
        print(f"\n[7] Final arm plane coordinates (relative to claw):")
        print(f"    x_arm: {x_arm:.3f} mm")
        print(f"    y_arm: {y_arm:.3f} mm")
    
    # Create annotated frame for visualization
    print(f"\n[8] Creating visualization...")
    annotated = frame.copy()
    
    # Draw YOLO annotations
    for result in results:
        try:
            annotated = result.plot()
        except Exception:
            pass
    
    # Draw frame center (blue circle) - center_x is row, center_y is column in image
    center_img_x = center_y  # column (x-coordinate in image)
    center_img_y = center_x  # row (y-coordinate in image)
    cv2.circle(annotated, (center_img_x, center_img_y), 8, (255, 0, 0), -1)
    cv2.circle(annotated, (center_img_x, center_img_y), 12, (255, 0, 0), 2)
    
    # Draw object center and line only if detection exists
    if best_detection is not None and cx is not None and cy is not None:
        # Draw object center (red circle) - use image coordinates (cx, cy)
        obj_img_x = cx  # column (x-coordinate in image)
        obj_img_y = cy  # row (y-coordinate in image)
        cv2.circle(annotated, (obj_img_x, obj_img_y), 8, (0, 0, 255), -1)
        cv2.circle(annotated, (obj_img_x, obj_img_y), 12, (0, 0, 255), 2)
        
        # Draw line connecting centers
        cv2.line(annotated, (obj_img_x, obj_img_y), (center_img_x, center_img_y), (0, 255, 0), 2)
    
    # Overlay text with calculations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 20
    y_offset = 30
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    # Build text lines based on whether detection exists
    if best_detection is None:
        text_lines = [
            f"No detections found",
            f"",
            f"Camera Frame Center:",
            f"  center_x: {center_x}",
            f"  center_y: {center_y}",
            f"",
            f"Arm Claw Position:",
            f"  X_A: {X_A:.1f} mm",
            f"  Y_A: {Y_A:.1f} mm",
        ]
    else:
        text_lines = [
            f"Object: {best_detection['cls_name']} (conf={best_conf:.3f})",
            f"",
            f"Camera Frame (pixels):",
            f"  cam_to_obj_x: {cam_to_obj_x:.1f}",
            f"  cam_to_obj_y: {cam_to_obj_y:.1f}",
            f"",
            f"Camera Frame (mm):",
            f"  x_c: {x_c:.3f} mm",
            f"  y_c: {y_c:.3f} mm",
            f"",
            f"Transformation:",
            f"  theta: {theta_deg:.2f} deg",
            f"  xT: {xT:.3f} mm",
            f"  yT: {yT:.3f} mm",
            f"",
            f"Camera Position:",
            f"  x_cam: {x_cam:.3f} mm",
            f"  y_cam: {y_cam:.3f} mm",
            f"",
            f"Final Arm Coordinates:",
            f"  x_arm: {x_arm:.3f} mm",
            f"  y_arm: {y_arm:.3f} mm",
        ]
    
    # Draw text with background
    for i, line in enumerate(text_lines):
        y_pos = y_offset + i * line_height
        # Get text size for background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
        # Draw background rectangle
        cv2.rectangle(annotated, (10, y_pos - text_h - 5), (10 + text_w + 5, y_pos + 5), bg_color, -1)
        # Draw text
        cv2.putText(annotated, line, (10, y_pos), font, font_scale, color, font_thickness)
    
    # Display frame
    print(f"    Displaying frame (press any key to close)...")
    cv2.imshow("Object Centering", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Cleanup
    cap.release()
    print(f"\n[9] Cleanup complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

