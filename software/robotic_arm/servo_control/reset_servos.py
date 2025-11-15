#!/usr/bin/env python3
# manual_servo_control.py — center all servos, then allow manual PWM input
# Plus: "<servo#> = sweep" toggles continuous sweep for that servo.

import pigpio
import time
import threading

# GPIO pins for Vizzy servos
SERVO_TOP = 4       # Yaw (1740 mid)
SERVO_MID = 17      # Claw (1300-1650 closed to open)
SERVO_BTM = 27      # Wrist

# Servo pulse width limits (µs)
SERVO_MIN = 1000
SERVO_MAX = 2500
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

# Claw (servo 2) reinforcement behavior
CLAW_SERVO_ID = 2
CLAW_HOLD_INTERVAL_S = 0.2  # seconds between reinforcement writes

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

class ClawHoldManager:
    """Continuously reasserts the claw PWM target to push through resistance."""
    def __init__(self, pi, pin, interval_s, initial_pwm):
        self.pi = pi
        self.pin = pin
        self.interval_s = interval_s
        self._target_pwm = initial_pwm
        self._paused = False
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._update_evt = threading.Event()
        self._update_evt.set()  # ensure immediate first write
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_evt.is_set():
            self._update_evt.wait(self.interval_s)
            self._update_evt.clear()
            with self._lock:
                paused = self._paused
                target = self._target_pwm
            if not paused and target is not None:
                self.pi.set_servo_pulsewidth(self.pin, target)

    def set_pwm(self, pwm):
        with self._lock:
            self._target_pwm = pwm
        self.pi.set_servo_pulsewidth(self.pin, pwm)
        self._update_evt.set()

    def pause(self):
        with self._lock:
            self._paused = True
        self._update_evt.set()

    def resume(self):
        with self._lock:
            self._paused = False
        self._update_evt.set()

    def stop(self):
        self._stop_evt.set()
        self._update_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

def main():
    print("Connecting to pigpio daemon...")
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")

    # Create sweep managers per servo
    sweepers = {sid: ServoSweeper(pi, pin) for sid, (_, pin) in SERVOS.items()}

    # Create claw hold manager (continuous PWM reassertion) if claw servo is defined
    claw_hold = None
    if CLAW_SERVO_ID in SERVOS:
        _, claw_pin = SERVOS[CLAW_SERVO_ID]
        claw_hold = ClawHoldManager(pi, claw_pin, CLAW_HOLD_INTERVAL_S, SERVO_CENTER)

    # Center all servos on startup
    print("\nCentering all servos...")
    for sid, (_, pin) in SERVOS.items():
        if sid == CLAW_SERVO_ID and claw_hold:
            claw_hold.resume()
            claw_hold.set_pwm(SERVO_CENTER)
        else:
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
                if claw_hold:
                    claw_hold.resume()
                for sid, (_, pin) in SERVOS.items():
                    if sid == CLAW_SERVO_ID and claw_hold:
                        claw_hold.set_pwm(SERVO_CENTER)
                    else:
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
                if sid == CLAW_SERVO_ID and claw_hold:
                    if running:
                        claw_hold.pause()
                    else:
                        claw_hold.resume()
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
                if sid == CLAW_SERVO_ID and claw_hold:
                    claw_hold.resume()

            pwm = clamp_pwm(pwm)
            if sid == CLAW_SERVO_ID and claw_hold:
                claw_hold.resume()
                claw_hold.set_pwm(pwm)
                print(f"→ Set {name} servo (GPIO {pin}) to {pwm} µs (continuous hold)")
            else:
                pi.set_servo_pulsewidth(pin, pwm)
                print(f"→ Set {name} servo (GPIO {pin}) to {pwm} µs")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping all sweeps and servos...")
        for sw in sweepers.values():
            sw.stop()
        if claw_hold:
            claw_hold.stop()
        for _, pin in SERVOS.values():
            pi.set_servo_pulsewidth(pin, 0)
        pi.stop()
        print("Servos released and pigpio stopped.")

if __name__ == "__main__":
    main()
