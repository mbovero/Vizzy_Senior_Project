#!/usr/bin/env python3
"""
Test script for orientation detection.

Runs YOLO on camera feed and displays orientation angles for detected objects.
Press 'q' to quit.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from orientation import calculate_grasp_angle, visualize_orientation
from ..shared import config as C


def main():
    print("[Orientation Test] Starting...")
    
    # Load YOLO model
    model = YOLO(C.YOLO_MODEL)
    print(f"[Orientation Test] Loaded YOLO model: {C.YOLO_MODEL}")
    
    # Open camera
    cap = cv2.VideoCapture(C.CAM_INDEX)
    if not cap.isOpened():
        print(f"[Orientation Test] ERROR: Cannot open camera {C.CAM_INDEX}")
        return
    
    print(f"[Orientation Test] Camera opened. Press 'q' to quit.")
    print("-" * 60)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Orientation Test] Failed to read frame")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Run YOLO
        results = model(frame, verbose=False)
        
        # Process each detection
        display_frame = frame.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for idx, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                
                # Get mask if available
                if results[0].masks is not None and idx < len(results[0].masks):
                    mask_tensor = results[0].masks[idx].data[0]
                    
                    # Convert to binary numpy array
                    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
                    mask_resized = cv2.resize(mask_np, (w, h), 
                                             interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_resized * 255).astype(np.uint8)
                    
                    # Calculate orientation
                    orientation = calculate_grasp_angle(mask_binary, method="minrect")
                    
                    if orientation["success"]:
                        # Draw orientation visualization
                        display_frame = visualize_orientation(display_frame, 
                                                             mask_binary, 
                                                             orientation)
                        
                        # Print info (every 30 frames to avoid spam)
                        if frame_count % 30 == 0:
                            yaw = orientation["yaw_angle"]
                            confidence = orientation["confidence"]
                            width = orientation.get("grasp_width_px", 0)
                            aspect = orientation.get("aspect_ratio", 0)
                            
                            print(f"[{cls_name}] "
                                  f"Yaw: {yaw:+6.1f}° | "
                                  f"Conf: {confidence:.2f} | "
                                  f"Width: {width:.0f}px | "
                                  f"Aspect: {aspect:.2f}")
                        
                        # Draw bounding box with class name
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), 
                                    (0, 255, 0), 2)
                        
                        # Label with orientation
                        yaw = orientation["yaw_angle"]
                        conf_ori = orientation["confidence"]
                        label = f"{cls_name} {conf:.2f} | Yaw:{yaw:+.0f}° ({conf_ori:.2f})"
                        
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # No mask, just draw box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), 
                                (255, 0, 0), 2)
                    cv2.putText(display_frame, f"{cls_name} {conf:.2f}", 
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add instructions
        cv2.putText(display_frame, "Press 'q' to quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Orientation Test", display_frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    print("\n[Orientation Test] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

