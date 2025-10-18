#!/usr/bin/env python3
# center_servos.py — centers all Vizzy servos and exits

import pigpio
import time

# GPIO pins for Vizzy servos
SERVO_TOP = 17
SERVO_MID = 27
SERVO_BTM = 22

# Pulse width configuration (microseconds)
SERVO_MIN = 1000
SERVO_MAX = 2000
SERVO_CENTER = 1500

def main():
    print("Connecting to pigpio daemon...")
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")

    print("Centering all servos...")
    pi.set_servo_pulsewidth(SERVO_TOP, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_MID, SERVO_CENTER)
    pi.set_servo_pulsewidth(SERVO_BTM, SERVO_CENTER)
    time.sleep(1)

    # Stop PWM signals and clean up
    pi.set_servo_pulsewidth(SERVO_TOP, 0)
    pi.set_servo_pulsewidth(SERVO_MID, 0)
    pi.set_servo_pulsewidth(SERVO_BTM, 0)
    pi.stop()
    print("All servos centered and released.")

if __name__ == "__main__":
    main()
