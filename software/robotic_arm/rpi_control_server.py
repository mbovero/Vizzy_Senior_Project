#!/usr/bin/env python3
# rpi_arm_server.py — Pi-side control of 3 moteus motors + 3 GPIO hobby servos (pitch, yaw, claw)
# Protocol (from client):  ik x y z pitch_rad yaw_rad O|C
#   - Pitch: radians go into IK; we use IK's q4 (PWM) directly (preserves original behavior)
#   - Yaw:   radians mapped to your specified PWM endpoints
#   - Claw:  O=1700 (open), C=1200 (closed) - snaps immediately (no interpolation)
# Pitch and yaw servos tween in sync with the motor group move duration. Claw snaps immediately.

from __future__ import annotations
import argparse, asyncio, math, time, sys, contextlib, pprint, re

# ---------- Servo Pins (BCM) ----------
SERVO_YAW   = 4
SERVO_CLAW  = 17
SERVO_PITCH = 27

# ---------- Global PWM bounds ----------
SERVO_MIN    = 500
SERVO_MAX    = 2500

# Pitch center (original server behavior: pitch straight up)
SERVO_PITCH_CENTER = 1770

# Yaw PWM mapping (your exact endpoints)
YAW_PWM_CENTER = 1700
YAW_PWM_MIN    = 1025   # clockwise (negative)
YAW_PWM_MAX    = 2375   # counter-clockwise (positive)
YAW_RAD_MIN    = -math.pi/2   # -90°
YAW_RAD_MAX    =  math.pi/2   # +90°

# Claw PWM mapping (your exact endpoints)
CLAW_PWM_OPEN   = 1650
CLAW_PWM_CLOSED = 1300

# Tween timing for servos
SERVO_TWEEN_MIN_DT = 0.01   # s
SERVO_TWEEN_MAX_DT = 0.025  # s
IDLE_SLEEP          = 0.02   # s
DURATION_FLOOR_S    = 0.1    # s (if motors barely move)

# ---------- Moteus / IK ----------
import moteus
from ik_callable import make_default_arm, ik_cmds_bounded

# ---- IK Tunables ----
IK_RADIUS_M = 0.15
IK_Z_MIN    = 0.0
IK_Z_MAX    = 0.4

# ---- q2 linear scaling (map ±2.25 -> ±2.10, 0 -> 0) ----
Q2_MAX_IN   = 2.25
Q2_MAX_OUT  = 2.15
Q2_SCALE    = (Q2_MAX_OUT / Q2_MAX_IN)  # ~0.955...

# ---- Comm/Heartbeat ----
COMM_FAIL_THRESHOLD = 3
RECOVER_SLEEP_S     = 0.5
HEARTBEAT_S         = 2.0

# pigpio
try:
    import pigpio
except Exception as e:
    print("ERROR: pigpio not available. Install via: sudo apt-get install pigpio", file=sys.stderr)
    raise

# Optional transports
def build_transport(kind: str):
    kind = (kind or "pi3hat").lower()
    if kind == "pi3hat":
        try:
            import moteus_pi3hat
            return moteus_pi3hat.Pi3HatRouter()
        except Exception:
            print("WARN: pi3hat not available, falling back to default moteus transport.", file=sys.stderr)
            return None
    if kind.startswith("socketcan"):
        parts = kind.split(":", 1)
        iface = parts[1] if len(parts) > 1 else "can0"
        try:
            return moteus.SocketcanRouter(channel=iface)
        except Exception:
            print(f"WARN: socketcan {iface} not available, using default.", file=sys.stderr)
            return None
    return None

# ---- Tunables (motors) ----
ID1, ID2, ID3      = 1, 2, 3

REST1               = 0.0
REST2               = 1.0
REST3               = 3.25

STEP_SIZE_RAD       = 0.08
STEP_DELAY_S        = 0.01

ACCEL_LIMIT         = 4
VEL_LIMIT           = 8.0
HOLD_ACCEL_LIMIT    = 4.0
HOLD_VEL_LIMIT      = 4.0
WATCHDOG            = 15.0

GROUP_SPEED_RAD_S   = 1.6
ACCEL_MULT          = 2.4
MIN_SPEED           = 0.02
POS_TOL             = 0.01
SYNC_POLL_S         = 0.05
SYNC_TIMEOUT_EXTRA  = 0.1

TEMP_LIMIT          = 50.0
COOL_RESUME_TEMP    = 40.0

WIGGLE_AMPLITUDE    = 0.5
WIGGLE_REPS         = 2
WIGGLE_DWELL_S      = 0.15

# ---------- Helpers ----------
def ang_err(a: float, b: float) -> float:
    return math.remainder(a - b, 2.0 * math.pi)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def lerp(a, b, t):
    return a + (b - a) * t

def map_yaw_rad_to_pwm(rad: float) -> int:
    """Piecewise map radians to PWM using the exact center/min/max."""
    r = max(YAW_RAD_MIN, min(YAW_RAD_MAX, rad))
    if abs(r) < 1e-12:
        return YAW_PWM_CENTER
    if r < 0.0:
        # [YAW_RAD_MIN .. 0] -> [YAW_PWM_MIN .. YAW_PWM_CENTER]
        t = (r - YAW_RAD_MIN) / (0.0 - YAW_RAD_MIN)  # 0 at min, 1 at 0
        pwm = YAW_PWM_MIN + t * (YAW_PWM_CENTER - YAW_PWM_MIN)
    else:
        # [0 .. YAW_RAD_MAX] -> [YAW_PWM_CENTER .. YAW_PWM_MAX]
        t = r / YAW_RAD_MAX  # 0 at 0, 1 at max
        pwm = YAW_PWM_CENTER + t * (YAW_PWM_MAX - YAW_PWM_CENTER)
    return int(round(clamp(pwm, SERVO_MIN, SERVO_MAX)))

async def hold(ctrl: moteus.Controller, pos: float):
    await ctrl.set_position(
        position=float(pos),
        velocity=float('nan'),
        accel_limit=HOLD_ACCEL_LIMIT,
        velocity_limit=HOLD_VEL_LIMIT,
        watchdog_timeout=WATCHDOG,
        query=False,
    )

async def move_ramped(ctrl: moteus.Controller, current: float, target: float) -> float:
    delta = ang_err(target, current)
    steps = max(1, int(abs(delta) / STEP_SIZE_RAD))
    step  = delta / steps
    pos   = current
    for _ in range(steps):
        pos += step
        await hold(ctrl, pos)
        await asyncio.sleep(STEP_DELAY_S)
    await hold(ctrl, target)
    return target

# on_duration callback receives the chosen group move duration (seconds)
async def move_group_sync_time(items, on_duration=None):
    if not items:
        return []
    deltas = [ang_err(t, c) for (_, c, t) in items]
    max_delta = max(abs(d) for d in deltas) if deltas else 0.0

    if max_delta < 1e-9:
        duration = DURATION_FLOOR_S
        if on_duration is not None:
            asyncio.create_task(on_duration(duration))
        await asyncio.gather(*(hold(ctrl, tgt) for (ctrl, _, tgt) in items))
        return [t for (_, _, t) in items]

    group_speed = max(GROUP_SPEED_RAD_S, MIN_SPEED)
    duration = max(max_delta / group_speed, DURATION_FLOOR_S)

    if on_duration is not None:
        asyncio.create_task(on_duration(duration))

    cmds = []
    for (ctrl, cur, tgt), d in zip(items, deltas):
        vel_lim = max(abs(d) / duration, MIN_SPEED)
        acc_lim = max(vel_lim * ACCEL_MULT, ACCEL_LIMIT)
        cmds.append(ctrl.set_position(
            position=float(tgt),
            velocity=float('nan'),
            accel_limit=acc_lim,
            velocity_limit=vel_lim,
            watchdog_timeout=WATCHDOG,
            query=False,
        ))
    await asyncio.gather(*cmds)

    t_end = time.monotonic() + duration + SYNC_TIMEOUT_EXTRA
    ctrls   = [ctrl for (ctrl, _, _) in items]
    targets = [t for (_, _, t) in items]

    while True:
        if time.monotonic() >= t_end:
            break
        try:
            states = await asyncio.gather(*(ctrl.query() for ctrl in ctrls))
            pos = [float(st.values.get('position', 0.0)) for st in states]
        except Exception:
            pos = [None] * len(ctrls)

        if all(p is not None and abs(ang_err(t, p)) <= POS_TOL
               for p, t in zip(pos, targets)):
            break
        await asyncio.sleep(SYNC_POLL_S)

    await asyncio.gather(*(hold(ctrl, tgt) for ctrl, tgt in zip(ctrls, targets)))
    return targets

def extract_temperature(vals: dict):
    try:
        temp = vals.get(moteus.Register.TEMPERATURE)
        if isinstance(temp, (int, float)):
            return float(temp)
    except Exception:
        pass
    temp = vals.get(0x00e)
    if not isinstance(temp, (int, float)):
        temp = vals.get(14)
    return float(temp) if isinstance(temp, (int, float)) else None

def extract_voltage(vals: dict):
    for k in ("voltage", "motor_voltage", "bus_voltage",
              getattr(moteus, "Register", object()).__dict__.get("VOLTAGE") if hasattr(moteus, "Register") else None,
              0x010, 16):
        if k in vals and isinstance(vals[k], (int, float)):
            return float(vals[k])
    return None

def has_fault(vals: dict) -> bool:
    f = vals.get("fault", 0)
    if isinstance(f, (int, float)) and f != 0:
        return True
    for k, v in vals.items():
        kn = str(k).lower()
        if kn.startswith("fault") or kn.startswith("error"):
            if isinstance(v, (int, float)) and v != 0:
                return True
            if isinstance(v, str) and v.strip():
                return True
    return False

def collect_fault_fields(vals: dict) -> dict:
    out = {}
    for key in ("fault", "fault_state", "fault_code", "fault_information", "error", "last_error"):
        if key in vals:
            out[key] = vals[key]
    for k, v in vals.items():
        kn = str(k).lower()
        if (kn.startswith("fault") or kn.startswith("error")) and (k not in out):
            out[k] = v
    return out

async def gentle_clear_fault(ctrl: moteus.Controller, name: str):
    try:
        await ctrl.set_stop()
        await asyncio.sleep(0.05)
        st = await ctrl.query()
        if not has_fault(st.values):
            print(f"[rpi] FAULT-CLEAR: {name} cleared by stop().")
        else:
            print(f"[rpi] FAULT-PERSIST: {name} still faulted after stop().", file=sys.stderr)
    except Exception as e:
        print(f"[rpi] FAULT-CLEAR-ERR: {name} stop/query raised: {e}", file=sys.stderr)

# ---------- Server ----------
class ArmServer:
    def __init__(self, host: str, port: int, _unused_servo_pin: int, transport_kind: str):
        self.host = host
        self.port = port

        # pigpio (single daemon controls all three servos)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")

        # Servo channels state (startup pose: pitch up, yaw centered, claw open)
        self.servos = {
            "pitch": {"pin": SERVO_PITCH, "current": SERVO_PITCH_CENTER, "target": SERVO_PITCH_CENTER},
            "yaw":   {"pin": SERVO_YAW,   "current": YAW_PWM_CENTER,     "target": YAW_PWM_CENTER},
            "claw":  {"pin": SERVO_CLAW,  "current": CLAW_PWM_OPEN,      "target": CLAW_PWM_OPEN},
        }
        for ch in self.servos.values():
            self.pi.set_servo_pulsewidth(ch["pin"], ch["current"])

        # moteus controllers
        self._transport_kind = transport_kind
        router = build_transport(transport_kind)
        self.m1 = moteus.Controller(id=ID1, transport=router)
        self.m2 = moteus.Controller(id=ID2, transport=router)
        self.m3 = moteus.Controller(id=ID3, transport=router)

        # IK arm instance
        self.ik_arm = make_default_arm()

        # Targets/state (motors)
        self.t1 = self.t2 = self.t3 = 0.0
        self.c1 = self.c2 = self.c3 = 0.0

        # Cooling / comm
        self.cooling2 = False
        self.cooling3 = False
        self.last_temp2 = None
        self.last_temp3 = None
        self._comm_failures = 0

        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._servo_task: asyncio.Task | None = None
        self._pi_reconnect_backoff_s = 0.5

    # ===== pigpio resilience =====
    async def _ensure_pi(self):
        """Ensure pigpio connection is alive; (re)connect if needed."""
        if (self.pi is not None) and getattr(self.pi, "connected", False):
            return
        try:
            if self.pi is not None and getattr(self.pi, "connected", False):
                self.pi.stop()
        except Exception:
            pass

        start = time.monotonic()
        while True:
            self.pi = pigpio.pi()
            if getattr(self.pi, "connected", False):
                # Re-apply current PWM to channels so state is consistent
                for _, ch in self.servos.items():
                    try:
                        self.pi.set_servo_pulsewidth(ch["pin"], int(ch["current"]))
                    except Exception:
                        try: self.pi.stop()
                        except Exception: pass
                        self.pi = None
                        await asyncio.sleep(self._pi_reconnect_backoff_s)
                        continue
                if self._pi_reconnect_backoff_s < 2.0:
                    self._pi_reconnect_backoff_s *= 1.25
                print("[rpi] pigpio reconnected.")
                return
            self.pi = None
            await asyncio.sleep(self._pi_reconnect_backoff_s)
            if (time.monotonic() - start) > 5.0:
                print("[rpi] WARN: pigpio reconnect still pending...")
                start = time.monotonic()

    async def _safe_servo_write(self, name: str, pwm_us: int):
        """Write with auto-reconnect protection."""
        try:
            await self._ensure_pi()
            await self._servo_write_pwm(name, pwm_us)
        except (AttributeError, pigpio.error, OSError) as e:
            print(f"[rpi] pigpio write error ({name}): {e} → reconnecting...", file=sys.stderr)
            try:
                if self.pi is not None:
                    self.pi.stop()
            except Exception:
                pass
            self.pi = None
            await self._ensure_pi()
            await self._servo_write_pwm(name, pwm_us)

    async def _cancel_servo_task(self):
        """Cancel any in-flight tween task cleanly."""
        if self._servo_task and not self._servo_task.done():
            self._servo_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._servo_task
        self._servo_task = None

    async def _start_servo_tween(self, duration: float):
        """Ensure only one tween task runs; start a new one for 'duration'."""
        await self._cancel_servo_task()
        self._servo_task = asyncio.create_task(self.servos_tween_multi(duration))

    # ===== SERVO I/O =====
    async def _servo_write_pwm(self, name: str, pwm_us: int):
        ch = self.servos[name]
        pwm = int(clamp(pwm_us, SERVO_MIN, SERVO_MAX))
        self.pi.set_servo_pulsewidth(ch["pin"], pwm)
        ch["current"] = pwm

    def set_servo_targets(self, *, pitch_pwm: int | None = None, yaw_pwm: int | None = None, claw_pwm: int | None = None):
        if pitch_pwm is not None:
            self.servos["pitch"]["target"] = int(clamp(pitch_pwm, SERVO_MIN, SERVO_MAX))
        if yaw_pwm is not None:
            self.servos["yaw"]["target"]   = int(clamp(yaw_pwm,   SERVO_MIN, SERVO_MAX))
        if claw_pwm is not None:
            # Claw target set (will be continuously sent)
            self.servos["claw"]["target"]  = int(clamp(claw_pwm,  SERVO_MIN, SERVO_MAX))

    async def servos_tween_multi(self, duration: float):
        """Tween pitch and yaw linearly from their current to target PWM over 'duration' seconds.
        Claw PWM is continuously sent at target value (no interpolation)."""
        await self._ensure_pi()

        t_claw = self.servos["claw"]["target"]

        if duration <= 0.0:
            await asyncio.gather(
                self._safe_servo_write("pitch", self.servos["pitch"]["target"]),
                self._safe_servo_write("yaw",   self.servos["yaw"]["target"]),
                self._safe_servo_write("claw",  t_claw),
            )
            return

        s_pitch, t_pitch = self.servos["pitch"]["current"], self.servos["pitch"]["target"]
        s_yaw,   t_yaw   = self.servos["yaw"]["current"],   self.servos["yaw"]["target"]

        t0 = time.monotonic()
        dt = max(SERVO_TWEEN_MIN_DT, min(SERVO_TWEEN_MAX_DT, duration / 50.0))
        while True:
            el = time.monotonic() - t0
            if el >= duration:
                break
            a = el / duration
            # Continuously send claw PWM along with pitch/yaw updates
            await asyncio.gather(
                self._safe_servo_write("pitch", int(round(lerp(s_pitch, t_pitch, a)))),
                self._safe_servo_write("yaw",   int(round(lerp(s_yaw,   t_yaw,   a)))),
                self._safe_servo_write("claw",  t_claw),
            )
            await asyncio.sleep(dt)

        await asyncio.gather(
            self._safe_servo_write("pitch", t_pitch),
            self._safe_servo_write("yaw",   t_yaw),
            self._safe_servo_write("claw",  t_claw),
        )

    async def servos_snap(self, *, pitch_pwm: int, yaw_pwm: int, claw_pwm: int):
        """Set servos *once* without tweening. Best-effort; won't block the motor park."""
        self.set_servo_targets(pitch_pwm=pitch_pwm, yaw_pwm=yaw_pwm, claw_pwm=claw_pwm)
        with contextlib.suppress(Exception):
            await self._cancel_servo_task()
        try:
            await self._ensure_pi()
            await asyncio.gather(
                self._safe_servo_write("pitch", self.servos["pitch"]["target"]),
                self._safe_servo_write("yaw",   self.servos["yaw"]["target"]),
                self._safe_servo_write("claw",  self.servos["claw"]["target"]),
            )
        except Exception as e:
            print(f"[rpi] EMERG: servo snap failed (continuing cooldown): {e}", file=sys.stderr)

    async def park_to_rest_emergency(self):
        """Emergency cooldown: snap servos, then motor-only blocking park to REST."""
        await self.servos_snap(
            pitch_pwm=SERVO_PITCH_CENTER,
            yaw_pwm=YAW_PWM_CENTER,
            claw_pwm=CLAW_PWM_OPEN
        )
        finals = await move_group_sync_time([
            (self.m1, self.c1, REST1),
            (self.m2, self.c2, REST2),
            (self.m3, self.c3, REST3),
        ], on_duration=None)
        self.c1, self.c2, self.c3 = finals
        self.t1, self.t2, self.t3 = REST1, REST2, REST3
        await asyncio.gather(
            hold(self.m1, REST1),
            hold(self.m2, REST2),
            hold(self.m3, REST3),
        )

    # ===== MOTEUS & SYSTEM =====
    async def init_positions(self):
        await asyncio.gather(self.m1.set_stop(), self.m2.set_stop(), self.m3.set_stop())
        try:
            s1, s2, s3 = await asyncio.gather(self.m1.query(), self.m2.query(), self.m3.query())
            self.t1 = self.c1 = float(s1.values.get('position', 0.0))
            self.t2 = self.c2 = float(s2.values.get('position', 0.0))
            self.t3 = self.c3 = float(s3.values.get('position', 0.0))
            await self._report_fault_if_any("m1", s1.values); await self._report_fault_if_any("m2", s2.values); await self._report_fault_if_any("m3", s3.values)
        except Exception as e:
            print(f"[rpi] init: query error: {e}", file=sys.stderr)
            self.t1 = self.t2 = self.t3 = 0.0
            self.c1 = self.c2 = self.c3 = 0.0
        print(f"[rpi] init: ID1={self.c1:.3f}, ID2={self.c2:.3f}, ID3={self.c3:.3f}")

    async def _recover_bus(self):
        print("[rpi] COMM: attempting bus recovery...", file=sys.stderr)
        with contextlib.suppress(Exception):
            await asyncio.gather(self.m1.set_stop(), self.m2.set_stop(), self.m3.set_stop())

        router = build_transport(self._transport_kind)
        self.m1 = moteus.Controller(id=ID1, transport=router)
        self.m2 = moteus.Controller(id=ID2, transport=router)
        self.m3 = moteus.Controller(id=ID3, transport=router)

        # Park servos to your safe defaults (best-effort)
        await self.servos_snap(
            pitch_pwm=SERVO_PITCH_CENTER,
            yaw_pwm=YAW_PWM_CENTER,
            claw_pwm=CLAW_PWM_OPEN
        )
        await asyncio.sleep(RECOVER_SLEEP_S)
        self._comm_failures = 0
        print("[rpi] COMM: recovery complete.")

    async def _report_fault_if_any(self, name: str, vals: dict):
        if has_fault(vals):
            ff = collect_fault_fields(vals)
            print(f"[rpi] FAULT: {name}: {ff}", file=sys.stderr)
            await gentle_clear_fault(getattr(self, name), name)

    async def control_loop(self):
        next_temp = time.monotonic()
        try:
            while not self._stop.is_set():
                async with self._lock:
                    group = [
                        (self.m1, self.c1, self.t1),
                        (self.m2, self.c2, self.t2),
                        (self.m3, self.c3, self.t3),
                    ]

                    try:
                        finals = await move_group_sync_time(group, on_duration=self._start_servo_tween)
                        self.c1, self.c2, self.c3 = finals
                        self._comm_failures = 0
                    except Exception as e:
                        self._comm_failures += 1
                        print(f"[rpi] COMM: move_group exception ({self._comm_failures}/{COMM_FAIL_THRESHOLD}): {e}", file=sys.stderr)
                        if self._comm_failures >= COMM_FAIL_THRESHOLD:
                            await self._recover_bus()

                now = time.monotonic()
                if now >= next_temp:
                    try:
                        s2, s3 = await asyncio.gather(self.m2.query(), self.m3.query())
                        self.last_temp2 = extract_temperature(s2.values)
                        self.last_temp3 = extract_temperature(s3.values)
                        await self._report_fault_if_any("m2", s2.values)
                        await self._report_fault_if_any("m3", s3.values)
                    except Exception as e:
                        self._comm_failures += 1
                        print(f"[rpi] COMM: temp query exception ({self._comm_failures}/{COMM_FAIL_THRESHOLD}): {e}", file=sys.stderr)
                        if self._comm_failures >= COMM_FAIL_THRESHOLD:
                            await self._recover_bus()
                    next_temp = now + 1.0

                # Over-temp handling → EMERGENCY PARK (motor-only, servos snapped best-effort)
                if (self.last_temp2 is not None) and (not self.cooling2) and (self.last_temp2 > TEMP_LIMIT):
                    print(f"[rpi] WARN: m2 > {TEMP_LIMIT:.1f} °C → EMERGENCY PARK (motor-only).")
                    await self.park_to_rest_emergency()
                    self.cooling2 = True

                if (self.last_temp3 is not None) and (not self.cooling3) and (self.last_temp3 > TEMP_LIMIT):
                    print(f"[rpi] WARN: m3 > {TEMP_LIMIT:.1f} °C → EMERGENCY PARK (motor-only).")
                    await self.park_to_rest_emergency()
                    self.cooling3 = True

                if self.cooling2 and (self.last_temp2 is not None) and (self.last_temp2 <= COOL_RESUME_TEMP):
                    print(f"[rpi] COOL: m2 <= {COOL_RESUME_TEMP:.1f} °C → wiggle2 + resume check.")
                    self.c2 = await wiggle(self.m2, REST2, self.c2); self.t2 = REST2; self.cooling2 = False

                if self.cooling3 and (self.last_temp3 is not None) and (self.last_temp3 <= COOL_RESUME_TEMP):
                    print(f"[rpi] COOL: m3 <= {COOL_RESUME_TEMP:.1f} °C → wiggle3 + resume check.")
                    self.c3 = await wiggle(self.m3, REST3, self.c3); self.t3 = REST3; self.cooling3 = False

                await asyncio.sleep(IDLE_SLEEP)
        finally:
            with contextlib.suppress(Exception):
                await asyncio.gather(self.m1.set_stop(), self.m2.set_stop(), self.m3.set_stop())
            try:
                await self._cancel_servo_task()
                if self.pi is not None and getattr(self.pi, "connected", False):
                    for ch in self.servos.values():
                        self.pi.set_servo_pulsewidth(ch["pin"], 0)
                    self.pi.stop()
            except Exception:
                pass

    async def heartbeat_loop(self):
        names = [("m1", self.m1), ("m2", self.m2), ("m3", self.m3)]
        while not self._stop.is_set():
            servo_state = "connected" if (self.pi is not None and getattr(self.pi, "connected", False)) else "DISCONNECTED"
            sp = self.servos["pitch"]; sy = self.servos["yaw"]; sc = self.servos["claw"]
            print(f"[rpi] HB: servo={servo_state} P={sp['current']}/{sp['target']}  "
                  f"Y={sy['current']}/{sy['target']}  C={sc['current']}/{sc['target']}", flush=True)

            try:
                results = await asyncio.gather(*(ctrl.query() for _, ctrl in names), return_exceptions=True)
                for (name, ctrl), res in zip(names, results):
                    if isinstance(res, Exception):
                        print(f"[rpi] HEARTBEAT: {name} query error: {res}", file=sys.stderr)
                        continue

                    vals = res.values
                    print(f"[rpi] HB FULL {name} values:")
                    pprint.pprint(vals, width=100, compact=False, sort_dicts=False)

                    temp  = extract_temperature(vals)
                    volts = extract_voltage(vals)
                    pos   = vals.get('position')
                    ps = f"{pos:.3f}" if isinstance(pos, (int, float)) else "n/a"
                    ts = f"{temp:.1f}°C" if isinstance(temp, (int, float)) else "n/a"
                    vs = f"{volts:.2f}V" if isinstance(volts, (int, float)) else "n/a"
                    print(f"[rpi] HB SUM {name} pos={ps} temp={ts} bus={vs}")

                    if has_fault(vals):
                        ff = collect_fault_fields(vals)
                        print(f"[rpi] FAULT: {name}: {ff}", file=sys.stderr)
                        await gentle_clear_fault(ctrl, name)

            except Exception as e:
                print(f"[rpi] HEARTBEAT: unexpected error: {e}", file=sys.stderr)

            await asyncio.sleep(HEARTBEAT_S)

    # ---- Command parsing helpers ----
    @staticmethod
    def _split_payload(s: str):
        """
        Accept 'ik x y z pitch_rad yaw_rad O/C' or comma-separated equivalent.
        Returns list of 6 strings after 'ik'.
        """
        s = re.sub(r'[,\s]+', ' ', s.strip())
        parts = s.split()
        if len(parts) < 7:
            raise ValueError("expected 6 values after 'ik'")
        return parts[1:7]

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        print(f"[rpi] client connected: {addr}")
        try:
            while not self._stop.is_set():
                line = await reader.readline()
                if not line:
                    break
                cmd_raw = line.decode("utf-8").strip()
                if not cmd_raw:
                    continue

                cmd_lower = cmd_raw.lower()

                if cmd_lower in ("quit", "q"):
                    writer.write(b"ACK bye\n"); await writer.drain()
                    self._stop.set()
                    break

                if cmd_lower == "rest":
                    try:
                        async with self._lock:
                            self.t1, self.t2, self.t3 = REST1, REST2, REST3
                            # pitch straight up, yaw centered, claw open
                            await self.servos_snap(
                                pitch_pwm=SERVO_PITCH_CENTER,
                                yaw_pwm=YAW_PWM_CENTER,
                                claw_pwm=CLAW_PWM_OPEN
                            )
                        writer.write(b"ACK rest\n"); await writer.drain()
                    except Exception as e:
                        print(f"[rpi] REST: error: {e}", file=sys.stderr)
                        writer.write(b"ERR rest failed\n"); await writer.drain()
                    continue

                if cmd_lower.startswith("ik "):
                    try:
                        sx, sy, sz, spitch, syaw, sclaw = self._split_payload(cmd_raw)
                        x = float(sx); y = float(sy); z = float(sz)
                        pitch_rad = float(spitch)
                        yaw_rad   = float(syaw)
                        claw_ch   = sclaw.strip().upper()
                        if claw_ch not in ("O", "C"):
                            raise ValueError("claw must be 'O' or 'C'")
                    except Exception:
                        writer.write(b"ERR ik parse (usage: ik x y z pitch_rad yaw_rad O|C)\n")
                        await writer.drain()
                        continue

                    # IK for motors + pitch PWM via IK (pitch is radians)
                    try:
                        cmds = ik_cmds_bounded(
                            self.ik_arm,
                            (x, y, z),
                            pitch_rad,  # radians
                            radius_m=IK_RADIUS_M,
                            z_min=IK_Z_MIN,
                            z_max=IK_Z_MAX
                        )
                        q1 = float(cmds["q1"])
                        q2 = float(cmds["q2"]) * Q2_SCALE
                        q3 = float(cmds["q3"])
                        q4_pwm = int(round(cmds["q4"]))  # IK-provided PWM for pitch
                    except Exception as e:
                        print(f"[rpi] IK error: {e}", file=sys.stderr)
                        writer.write(b"ERR ik failed\n"); await writer.drain()
                        continue

                    # Yaw/claw mapping
                    yaw_pwm  = map_yaw_rad_to_pwm(yaw_rad)
                    claw_pwm = CLAW_PWM_CLOSED if claw_ch == "C" else CLAW_PWM_OPEN

                    try:
                        async with self._lock:
                            # set servo targets (tween will run during group move, continuously sending claw PWM)
                            self.set_servo_targets(pitch_pwm=q4_pwm, yaw_pwm=yaw_pwm, claw_pwm=claw_pwm)
                            # set motor targets (unless cooling)
                            if not (self.cooling2 or self.cooling3):
                                self.t1, self.t2, self.t3 = q1, q2, q3
                        writer.write(b"ACK ik\n"); await writer.drain()
                    except Exception as e:
                        print(f"[rpi] IK: set targets error: {e}", file=sys.stderr)
                        writer.write(b"ERR ik servo\n"); await writer.drain()
                    continue

                writer.write(b"ERR expected: 'ik x y z pitch_rad yaw_rad O|C' or 'rest' or 'quit'\n")
                await writer.drain()

        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            print(f"[rpi] client disconnected: {addr}")

    async def serve(self):
        await self.init_positions()
        ctrl_task = asyncio.create_task(self.control_loop())
        hb_task   = asyncio.create_task(self.heartbeat_loop())
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        print(f"[rpi] listening on {addrs} | servos: pitch={SERVO_PITCH} yaw={SERVO_YAW} claw={SERVO_CLAW}")
        async with server:
            await self._stop.wait()
            server.close()
            await server.wait_closed()
        ctrl_task.cancel(); hb_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ctrl_task; await hb_task
        with contextlib.suppress(Exception):
            await self._cancel_servo_task()

# ---- wiggle helper ----
async def wiggle(ctrl: moteus.Controller, center: float, start_pos: float) -> float:
    left  = center - WIGGLE_AMPLITUDE
    right = center + WIGGLE_AMPLITUDE
    pos = start_pos
    for _ in range(WIGGLE_REPS):
        pos = await move_ramped(ctrl, pos, left)
        await asyncio.sleep(WIGGLE_DWELL_S)
        pos = await move_ramped(ctrl, pos, right)
        await asyncio.sleep(WIGGLE_DWELL_S)
    pos = await move_ramped(ctrl, pos, center)
    return pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=65432)
    ap.add_argument("--servo-pin", type=int, default=SERVO_PITCH, help="(unused—kept for backward compat)")
    ap.add_argument("--transport", default="pi3hat", help="pi3hat | socketcan:can0 | auto")
    args = ap.parse_args()

    srv = ArmServer(args.host, args.port, args.servo_pin, args.transport)
    try:
        asyncio.run(srv.serve())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
