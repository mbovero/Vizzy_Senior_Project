# vizzy/shared/protocol.py
# -----------------------------------------------------------------------------
# Purpose
#   Centralized definition of **message types** and **commands** for the Vizzy
#   Laptop <-> Raspberry Pi communication protocol.
#
# Why this exists
#   The laptop and RPi exchange messages over TCP using JSONL (newline-delimited
#   JSON) framing. Each message contains fields like:
#       { "type": "move", "horizontal": 0.1, "vertical": -0.05 }
#     or
#       { "cmd": "YOLO_SCAN", "duration_ms": 900 }
#
#   To avoid typos and keep the protocol consistent, we store all valid "type"
#   and "cmd" strings in one place here. Both the laptop and RPi import these
#   constants instead of hardcoding strings all over the codebase.
#
# How it fits into the project
#   - **Types** describe the nature of the message — usually data or events
#     flowing from one side to the other.
#   - **Commands** are requests telling the other side to do something.
#   - The RPi server and Laptop client both use these constants in send/receive
#     logic, dispatch tables, and message processing functions.
#
# Benefits:
#   - If a protocol string ever needs to change, we update it in one file.
#   - Code completion / linting can help catch invalid values.
#   - Clear separation of "type" vs. "cmd" messages.
#
# Example:
#   Laptop sending a command to start a YOLO scan:
#       send_json(sock, { "cmd": CMD_YOLO_SCAN, "duration_ms": 900 })
#
#   RPi sending YOLO detection results back:
#       send_json(sock, { "type": TYPE_YOLO_RESULTS, "objects": [...] })
#
# -----------------------------------------------------------------------------

from __future__ import annotations

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
