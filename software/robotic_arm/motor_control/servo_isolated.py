import pigpio


pi = pigpio.pi()
init_servos(pi)



def init_servos(pi) -> None:
    """
    Initialize servo outputs at SERVO_CENTER so they are powered and not limp.
    Safe to call multiple times.
    """
    pi.set_servo_pulsewidth(C.SERVO_BTM, C.SERVO_CENTER)
    pi.set_servo_pulsewidth(C.SERVO_MID, C.SERVO_CENTER)
    pi.set_servo_pulsewidth(C.SERVO_TOP, C.SERVO_CENTER)
    state.current_horizontal = C.SERVO_CENTER
    state.current_vertical = C.SERVO_CENTER



# -----------------------------
# Servos & Sweep (RPi)
# -----------------------------
# Servo GPIO pins (BCM numbering for pigpio)
# Adjust to match your wiring.
SERVO_BTM = 0
SERVO_MID = 5
SERVO_TOP = 6

# Pulse width bounds (µs)
SERVO_MIN    = 1000
SERVO_MAX    = 2000
SERVO_CENTER = 1500