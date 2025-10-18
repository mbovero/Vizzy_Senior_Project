import cv2
from ultralytics import YOLO
import pigpio
import time

# Initialize servos
pi = pigpio.pi()
# GPIO pin specifications for each Proto Vizzy V2 servo
SERVO_TOP = 17    # Typically tilt (vertical)
SERVO_MID = 27    # Not used in this example
SERVO_BTM = 22    # Typically pan (horizontal)

# Servo configuration
SERVO_MIN = 1000   # Minimum pulse width (us)
SERVO_MAX = 2000   # Maximum pulse width (us)
SERVO_CENTER = 1500
current_pan = SERVO_CENTER  # Track current position

try:
    # Initialize servos to center position
    pi.set_servo_pulsewidth(SERVO_TOP, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_MID, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_BTM, SERVO_CENTER)
    time.sleep(1)
    print("Servos initialized to center position!")
except KeyboardInterrupt:
    print("Program interrupted by user.")

# Open the USB camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit()

# Get frame dimensions for center calculation
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center_x = frame_width // 2

# Load the NCNN model
ncnn_model = YOLO("yolo11n_ncnn_model")

# PID controller constants (simple proportional control for now)
KP = 0.2  # Proportional gain
DEADZONE = 25  # Pixels - don't adjust for small errors

def move_servo(target_x):
    """Move servo to center the target in the frame"""
    global current_pan
    
    # Calculate error from center
    error = target_x - frame_center_x
    
    # Only move if error is outside deadzone
    if abs(error) > DEADZONE:
        # Calculate adjustment (proportional control)
        adjustment = KP * error
        
        # Convert pixel error to servo pulse width
        new_pan = current_pan - adjustment
        
        # Constrain to servo limits
        new_pan = max(SERVO_MIN, min(SERVO_MAX, new_pan))
        
        # Move servo
        pi.set_servo_pulsewidth(SERVO_BTM, int(new_pan))
        current_pan = new_pan
        print(f"Moving servo to {new_pan} (error: {error:.1f}px)")

try:
    while True:
        # Capture a single frame
        ret, frame = cap.read()
        
        if ret:
            # Run YOLO11 inference on the frame
            results = ncnn_model(frame)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Get detection information
            boxes = results[0].boxes.xyxy  # Bounding boxes
            classes = results[0].boxes.cls  # Class indices
            confidences = results[0].boxes.conf  # Confidence scores
            
            # Find the best target object detection
            best_obj = None
            max_conf = 0.3  # Minimum confidence threshold
            
            for box, cls, conf in zip(boxes, classes, confidences):
                class_name = ncnn_model.names[int(cls)]
                if class_name == "mouse" and conf > max_conf:
                    max_conf = conf
                    best_obj = box
            
            # If bottle found, track it
            if best_obj is not None:
                x1, y1, x2, y2 = best_obj
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Draw detection info
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"Tracking object", 
                           (center_x - 50, center_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Move servo to center the object in frame
                move_servo(center_x)
            
            # Display the resulting frame
            cv2.imshow("Camera", annotated_frame)
        else:
            print("Failed to capture image")
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord("q"):
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pi.set_servo_pulsewidth(SERVO_BTM, SERVO_CENTER)  # Return to center
    time.sleep(1)
    pi.stop()
    print("Servos centered and resources released")
