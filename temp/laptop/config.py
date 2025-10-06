# Computer Vision parameters
YOLO_MODEL = 'yolo11m-seg.engine'
CAM_INDEX = 4

# Raspberry Pi connections parameters
PI_IP = '192.168.1.30'
PI_PORT = 65432

# Object centering parameters
CENTER_DEADZONE = 30
SERVO_SPEED = 0.2 # TODO: this should be baked into motor/servo control API; remove later on

MEM_FILE = 'object_memory.json'

DISPLAY_SCALE = 1.3