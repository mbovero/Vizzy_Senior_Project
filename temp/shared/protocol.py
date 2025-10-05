from __future__ import annotations

# TODO this needs a complete redo; laptop should handle all YOLO related tasks instead 
# of the weird back and forth between the laptop and rpi

# -----------------------------
# Message "type" constants
# -----------------------------
# These usually indicate **what kind of data/event** the message represents.

TYPE_MOVE            = "move"              # (Laptop -> RPi) Move servos by offset or absolute target.
TYPE_SEARCH          = "search"            # (Laptop -> RPi) Toggle search mode ON/OFF.
TYPE_STOP            = "stop"              # (Laptop -> RPi) Stop all motion/operations cleanly.
TYPE_YOLO_RESULTS    = "YOLO_RESULTS"      # (Laptop -> RPi) Send YOLO detections and confidences.
TYPE_CENTER_DONE     = "CENTER_DONE"       # (RPi -> Laptop) Report result of centering attempt.
TYPE_CENTER_SNAPSHOT = "CENTER_SNAPSHOT"   # (RPi -> Laptop) Send PWM pose + class info for memory.

# -----------------------------
# Message "cmd" constants
# -----------------------------
# These are **requests** to perform a specific action.

CMD_YOLO_SCAN        = "YOLO_SCAN"         # Ask laptop to run YOLO inference for N milliseconds.
CMD_CENTER_ON        = "CENTER_ON"         # Ask RPi to center on a given object class.
CMD_GOTO_PWMS        = "GOTO_PWMS"         # Ask RPi to move servos to specific PWM positions.
