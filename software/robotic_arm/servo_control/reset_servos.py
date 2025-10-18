#!/usr/bin/env python3
# manual_servo_control.py — center all servos, then allow manual PWM input

import pigpio
import time

# GPIO pins for Vizzy servos
SERVO_TOP = 17
SERVO_MID = 27
SERVO_BTM = 22

# Servo pulse width limits (µs)
SERVO_MIN = 1000
SERVO_MAX = 2000
SERVO_CENTER = 1500

# Map IDs to GPIO pins and names
SERVOS = {
    1: ("Bottom", SERVO_BTM),
    2: ("Middle", SERVO_MID),
    3: ("Top", SERVO_TOP),
}

def clamp_pwm(pwm):
    """Clamp PWM signal to safe servo range."""
    return max(SERVO_MIN, min(SERVO_MAX, pwm))

def main():
    print("Connecting to pigpio daemon...")
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")

    print("\nCentering all servos...")
    for name, pin in SERVOS.values():
        pi.set_servo_pulsewidth(pin, SERVO_CENTER)
    time.sleep(1)
    print("All servos initialized to 1500 µs (center position).\n")

    print("Manual Servo Control:")
    print("  1 = <pwm>  → Bottom servo (GPIO 22)")
    print("  2 = <pwm>  → Middle servo (GPIO 27)")
    print("  3 = <pwm>  → Top servo (GPIO 17)")
    print("  c          → Center all servos")
    print("  q          → Quit\n")

    try:
        while True:
            cmd = input("Enter command: ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "c":
                for _, pin in SERVOS.values():
                    pi.set_servo_pulsewidth(pin, SERVO_CENTER)
                print("Centered all servos.")
                continue

            if "=" not in cmd:
                print("Invalid format. Use: <servo#> = <pwm>")
                continue

            try:
                sid_str, pwm_str = cmd.split("=")
                sid = int(sid_str.strip())
                pwm = int(float(pwm_str.strip()))
            except ValueError:
                print("Invalid input. Example: 1 = 1500")
                continue

            if sid not in SERVOS:
                print("Servo ID must be 1, 2, or 3.")
                continue

            pwm = clamp_pwm(pwm)
            name, pin = SERVOS[sid]
            pi.set_servo_pulsewidth(pin, pwm)
            print(f"→ Set {name} servo (GPIO {pin}) to {pwm} µs")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping all servos...")
        for _, pin in SERVOS.values():
            pi.set_servo_pulsewidth(pin, 0)
        pi.stop()
        print("Servos released and pigpio stopped.")

if __name__ == "__main__":
    main()
