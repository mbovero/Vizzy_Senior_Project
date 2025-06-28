from ultralytics import YOLO
import cv2
import torch
import socket
import json
import time
import argparse

DISPLAY_SCALE = 2  # Adjust this value to make the window larger (1.5 = 150%)

# COCO class names and IDs (partial list)
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Object Tracking with YOLO')
parser.add_argument('--class-id', type=int, default=72,
                   help=f'COCO class ID to track (default: 72 for cell phone)\nAvailable classes:\n{COCO_CLASSES}')
parser.add_argument('--ip', type=str, default='192.168.1.30',
                   help='Raspberry Pi IP address')
parser.add_argument('--port', type=int, default=65432,
                   help='Port number for socket connection')
args = parser.parse_args()

# Configuration
PI_IP = args.ip
PI_PORT = args.port
TARGET_CLASS = args.class_id
SERVO_SPEED = 0.2
DEADZONE = 30

# Verify class ID is valid
if TARGET_CLASS not in COCO_CLASSES:
    print(f"Error: Class ID {TARGET_CLASS} not found in COCO classes")
    print("Available classes:")
    for id, name in COCO_CLASSES.items():
        print(f"{id}: {name}")
    exit()

print(f"Tracking: {COCO_CLASSES[TARGET_CLASS]} (class ID: {TARGET_CLASS})")

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model
model = YOLO("yolo11m-seg.engine").to(device)
# if device == 'cuda':
#     model.model.half()
#     _ = model.predict(torch.zeros(1, 3, 640, 640).half().to(device))

# Socket connection
def connect_to_pi():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((PI_IP, PI_PORT))
            print("Connected to Raspberry Pi")
            return sock
        except (ConnectionRefusedError, OSError) as e:
            print(f"Connection failed, retrying... Error: {e}")
            time.sleep(5)

pi_socket = connect_to_pi()

# Camera setup
cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame")
    exit()

frame_height, frame_width = frame.shape[:2]
center_x, center_y = frame_width // 2, frame_height // 2

def calculate_servo_movement(obj_center, frame_center):
    """Calculate movements with corrected directions"""
    dx = frame_center[0] - obj_center[0]  # Horizontal (corrected)
    dy = obj_center[1] - frame_center[1]  # Vertical

    # Normalize and apply deadzone
    norm_dx = dx / frame_center[0] if abs(dx) > DEADZONE else 0
    norm_dy = dy / frame_center[1] if abs(dy) > DEADZONE else 0

    return norm_dx, norm_dy

def send_servo_command(dx, dy):
    """Send commands to Pi"""
    global pi_socket
    command = {
        'type': 'move',
        'horizontal': dx * SERVO_SPEED,
        'vertical': dy * SERVO_SPEED
    }
    try:
        pi_socket.sendall(json.dumps(command).encode('utf-8'))
    except (ConnectionResetError, BrokenPipeError):
        print("Reconnecting...")
        pi_socket = connect_to_pi()
        pi_socket.sendall(json.dumps(command).encode('utf-8'))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame, classes=[TARGET_CLASS], stream=True)  # Use specified class

        if len(results[0]) > 0:
            largest_obj = None
            max_area = 0

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_obj = ((x1 + x2) // 2, (y1 + y2) // 2)

            if largest_obj:
                dx, dy = calculate_servo_movement(largest_obj, (center_x, center_y))
                send_servo_command(dx, dy)

                # Visualization
                cv2.circle(frame, largest_obj, 5, (0, 255, 0), -1)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                cv2.line(frame, largest_obj, (center_x, center_y), (0, 0, 255), 2)
                cv2.putText(frame, f"Tracking: {COCO_CLASSES[TARGET_CLASS]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #cv2.imshow("YOLO Detection", results[0].plot())
        annotated_frame = results[0].plot()
        height, width = annotated_frame.shape[:2]
        resized_frame = cv2.resize(annotated_frame,
                                  (int(width * DISPLAY_SCALE),
                                  int(height * DISPLAY_SCALE)))
        cv2.imshow("YOLO Detection", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_command = {'type': 'stop'}
    try:
        pi_socket.sendall(json.dumps(stop_command).encode('utf-8'))
    except:
        pass
    pi_socket.close()
    cap.release()
    cv2.destroyAllWindows()
