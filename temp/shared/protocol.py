from __future__ import annotations

# -----------------------------------------------------------------------------
# Vizzy Protocol (shared/protocol.py)
# Minimal, laptop-centric handshake
# -----------------------------------------------------------------------------
# All TYPE/CMD identifiers are UPPERCASE. Payload keys are lowercase and minimal.
#
# Message "type" constants (events/push)
TYPE_SCAN_MOVE   = "SCAN_MOVE"   # (Laptop -> RPi) Fine corrections during centering; values in [-1, 1].
TYPE_SEARCH      = "SEARCH"      # (Both ways) Toggle search mode ON/OFF. { "active": true|false }
TYPE_STOP        = "STOP"        # (Laptop -> RPi) Stop all motion/operations cleanly.
TYPE_POSE_READY  = "POSE_READY"  # (RPi -> Laptop) Arm is settled at next grid pose. { "pose_id": int }
TYPE_POSE_DONE   = "POSE_DONE"   # (Laptop -> RPi) Pose finished. { "pose_id": int, "status": "SUCCESS"|"SKIP"|"FAIL" }
TYPE_PWMS        = "PWMS"        # (RPi -> Laptop) Current servo positions. { "pwm_btm": int, "pwm_top": int }

# Message "cmd" constants (requests)
CMD_GOTO_PWMS    = "GOTO_PWMS"   # (Laptop -> RPi) Move servos to absolute PWM positions.
CMD_GET_PWMS     = "GET_PWMS"    # (Laptop -> RPi) Request current servo PWMs.
