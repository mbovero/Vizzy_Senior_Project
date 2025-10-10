#!/usr/bin/env python3
"""
Test script for orientation detection.

Runs YOLO on camera feed and displays orientation angles for detected objects.
Press 'q' to quit.

Usage:
    python test_orientation.py              # Use minrect (default)
    python test_orientation.py --method pca # Use PCA
    python test_orientation.py -m moments   # Use moments
"""

import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from orientation import calculate_grasp_angle, visualize_orientation

def main(method="minrect"):
    print("[Orientation Test] Starting...")
    print(f"[Orientation Test] Using method: {method}")
    
    # Load YOLO model
    model = YOLO("yolo11m-seg.engine")
    
    # Open camera
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print(f"[Orientation Test] ERROR: Cannot open camera ")
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
                    orientation = calculate_grasp_angle(mask_binary, method=method)
                    
                    if orientation["success"]:
                        # Draw orientation visualization
                        display_frame = visualize_orientation(display_frame, 
                                                             mask_binary, 
                                                             orientation)
                        
                        # Print info (every 30 frames to avoid spam)
                        if frame_count % 30 == 0:
                            yaw = orientation["yaw_angle"]
                            confidence = orientation["confidence"]
                            width = orientation.get("grasp_width_px")
                            aspect = orientation.get("aspect_ratio")
                            elongation = orientation.get("elongation")
                            
                            # Build output string
                            output = f"[{cls_name}] Yaw: {yaw:+6.1f} | Conf: {confidence:.2f}"
                            
                            if width is not None:
                                output += f" | Width: {width:.0f}px"
                            if aspect is not None:
                                output += f" | Aspect: {aspect:.2f}"
                            if elongation is not None:
                                output += f" | Elongation: {elongation:.2f}"
                            
                            print(output)
                        
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
        cv2.putText(display_frame, f"Method: {method.upper()} | Press 'q' to quit", 
                   (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow(f"Orientation Test - {method.upper()}", display_frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    print("\n[Orientation Test] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test orientation detection on live camera feed"
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        default="minrect",
        choices=["minrect", "pca", "moments"],
        help="Orientation calculation method (default: minrect)"
    )
    
    args = parser.parse_args()
    main(method=args.method)

