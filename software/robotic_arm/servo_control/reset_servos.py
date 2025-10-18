#!/usr/bin/env python3
# manual_servo_control.py — center all servos, then allow manual PWM input
# Plus: "<servo#> = sweep" toggles continuous sweep for that servo.

import pigpio
import time
import threading

# GPIO pins for Vizzy servos
SERVO_TOP = 4
SERVO_MID = 17
SERVO_BTM = 27

# Servo pulse width limits (µs)
SERVO_MIN = 1250
SERVO_MAX = 2250
SERVO_CENTER = 1750

# Sweep behavior
SWEEP_STEP_US = 10       # step size in microseconds per update
SWEEP_DELAY_S = 0.01     # delay between steps

# Map IDs to GPIO pins and names
SERVOS = {
    1: ("Bottom", SERVO_BTM),
    2: ("Middle", SERVO_MID),
    3: ("Top",    SERVO_TOP),
}

def clamp_pwm(pwm):
    return max(SERVO_MIN, min(SERVO_MAX, pwm))

class ServoSweeper:
    """Manages a per-servo sweep thread that can be started/stopped."""
    def __init__(self, pi, pin):
        self.pi = pi
        self.pin = pin
        self._stop_evt = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        with self._lock:
            if self.is_running():
                return
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            if not self.is_running():
                return
            self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def toggle(self):
        if self.is_running():
            self.stop()
            return False
        else:
            self.start()
            return True

    def _run(self):
        # Begin at min, sweep to max, then back, repeat
        direction = +1
        pwm = SERVO_MIN
        self.pi.set_servo_pulsewidth(self.pin, pwm)
        while not self._stop_evt.is_set():
            pwm += direction * SWEEP_STEP_US
            if pwm >= SERVO_MAX:
                pwm = SERVO_MAX
                direction = -1
            elif pwm <= SERVO_MIN:
                pwm = SERVO_MIN
                direction = +1
            self.pi.set_servo_pulsewidth(self.pin, pwm)
            time.sleep(SWEEP_DELAY_S)

def main():
    print("Connecting to pigpio daemon...")
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")

    # Create sweep managers per servo
    sweepers = {sid: ServoSweeper(pi, pin) for sid, (_, pin) in SERVOS.items()}

    # Center all servos on startup
    print("\nCentering all servos...")
    for _, pin in SERVOS.values():
        pi.set_servo_pulsewidth(pin, SERVO_CENTER)
    time.sleep(1)
    print("All servos initialized to 1500 µs (center position).\n")

    print("Manual Servo Control:")
    print("  1 = <pwm>   → Bottom servo (GPIO 22)")
    print("  2 = <pwm>   → Middle servo (GPIO 27)")
    print("  3 = <pwm>   → Top servo (GPIO 17)")
    print("  <n> = sweep → Toggle sweep mode for that servo")
    print("  c           → Center all servos (stops all sweeps)")
    print("  q           → Quit\n")

    try:
        while True:
            cmd = input("Enter command: ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "c":
                # Stop all sweeps and center everything
                for sw in sweepers.values():
                    sw.stop()
                for _, pin in SERVOS.values():
                    pi.set_servo_pulsewidth(pin, SERVO_CENTER)
                print("Centered all servos and stopped sweeps.")
                continue

            if "=" not in cmd:
                print("Invalid format. Use: <servo#> = <pwm>  or  <servo#> = sweep")
                continue

            left, right = cmd.split("=", 1)
            try:
                sid = int(left.strip())
            except ValueError:
                print("Servo ID must be 1, 2, or 3.")
                continue

            if sid not in SERVOS:
                print("Servo ID must be 1, 2, or 3.")
                continue

            arg = right.strip()
            name, pin = SERVOS[sid]
            sweeper = sweepers[sid]

            if arg == "sweep":
                running = sweeper.toggle()
                state = "started" if running else "stopped"
                print(f"→ Sweep {state} for {name} servo (GPIO {pin})")
                continue

            # Otherwise treat as numeric PWM
            try:
                pwm = int(float(arg))
            except ValueError:
                print("Invalid PWM. Example: 1 = 1500  or  2 = sweep")
                continue

            # Stop sweep if currently running on this servo
            if sweeper.is_running():
                sweeper.stop()
                print(f"Stopped sweep for {name} servo to set PWM.")

            pwm = clamp_pwm(pwm)
            pi.set_servo_pulsewidth(pin, pwm)
            print(f"→ Set {name} servo (GPIO {pin}) to {pwm} µs")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping all sweeps and servos...")
        for sw in sweepers.values():
            sw.stop()
        for _, pin in SERVOS.values():
            pi.set_servo_pulsewidth(pin, 0)
        pi.stop()
        print("Servos released and pigpio stopped.")

if __name__ == "__main__":
    main()
