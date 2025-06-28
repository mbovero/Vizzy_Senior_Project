from ultralytics import YOLO
import cv2
import numpy as np

# Initialize model
model = YOLO("yolo11x-seg.engine")

def get_contour_center(mask):
    """Calculate object center using contour moments"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    return None

try:
    for result in model(source="4", show=True, stream=True, device=0):
        frame = result.orig_img.copy()
        h, w = frame.shape[:2]
        
        # Process both boxes and masks if available
        if result.boxes is not None and result.masks is not None:
            for i, (box, mask) in enumerate(zip(result.boxes, result.masks.data)):
                # Convert mask to numpy array
                mask_np = mask.cpu().numpy().astype(np.uint8)
                mask_np = cv2.resize(mask_np, (w, h)) * 255
                
                # Get segmentation-based center (red)
                seg_center = get_contour_center(mask_np)
                if seg_center:
                    seg_x, seg_y = seg_center
                    cv2.circle(frame, (seg_x, seg_y), 6, (0, 0, 255), -1)  # Red dot
                    cv2.putText(frame, f"S({seg_x},{seg_y})", (seg_x + 10, seg_y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Get bounding box center (blue)
                box_xyxy = box.xyxy[0].cpu().numpy()
                box_x1, box_y1, box_x2, box_y2 = box_xyxy.astype(int)
                box_cx = int((box_x1 + box_x2) / 2)
                box_cy = int((box_y1 + box_y2) / 2)
                cv2.circle(frame, (box_cx, box_cy), 6, (255, 0, 0), -1)  # Blue dot
                cv2.putText(frame, f"B({box_cx},{box_cy})", (box_cx + 10, box_cy + 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Draw connecting line between centers
                if seg_center:
                    cv2.line(frame, (seg_x, seg_y), (box_cx, box_cy), (0, 255, 0), 1)
                
                # Draw bounding box (optional)
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
        
        # Show frame with both centers
        cv2.imshow("Dual Center Visualization", frame)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProcessing stopped")
finally:
    cv2.destroyAllWindows()