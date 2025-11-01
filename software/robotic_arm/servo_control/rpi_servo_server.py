#!/usr/bin/env python3
# rpi_arm_server.py — Pi-side control of 3 moteus motors + 1 GPIO hobby servo
# IK protocol: accepts `ik <x> <y> <z> <wrist>` and maps to q1,q2,q3 + servo PWM (q4)

from __future__ import annotations
import argparse, asyncio, math, time, sys, contextlib, pprint

# ---------- Servo -----------
SERVO_MIN    = 500
SERVO_MAX    = 2500
SERVO_CENTER = 1750

try:
    import pigpio
except Exception as e:
    print("ERROR: pigpio not available. Install via: sudo apt-get install pigpio", file=sys.stderr)
    raise

# ---------- Moteus ----------
import moteus

# ---------- IK ----------
from ik_callable import make_default_arm, ik_cmds_bounded

# ---- IK Tunables ----
IK_RADIUS_M = 0.15
IK_Z_MIN    = 0.0
IK_Z_MAX    = 0.4

# ---- q2 linear scaling (map ±2.25 -> ±2.10, 0 -> 0) ----
Q2_MAX_IN   = 2.25
Q2_MAX_OUT  = 2.15
Q2_SCALE    = (Q2_MAX_OUT / Q2_MAX_IN)  # 0.933333...

# ---- Comm fault handling ----
COMM_FAIL_THRESHOLD = 3
RECOVER_SLEEP_S     = 0.5

# ---- Heartbeat ----
HEARTBEAT_S         = 2.0

# ---- Servo reconnect tunables ----
PIGPIO_RETRY_DELAY_S = 0.2
PIGPIO_MAX_RETRIES   = 2

# Optional transports (choose one at startup)
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
STEP_DELAY_S        = 0.02

IDLE_SLEEP          = 0.02
ACCEL_LIMIT         = 2.0
VEL_LIMIT           = 4.0
WATCHDOG            = 15.0

GROUP_SPEED_RAD_S   = 0.8
ACCEL_MULT          = 1.2
MIN_SPEED           = 0.01
POS_TOL             = 0.01
SYNC_POLL_S         = 0.05
SYNC_TIMEOUT_EXTRA  = 1.0

TEMP_LIMIT          = 50.0
COOL_RESUME_TEMP    = 40.0

WIGGLE_AMPLITUDE    = 0.5
WIGGLE_REPS         = 2
WIGGLE_DWELL_S      = 0.15

# ---------- Helpers ----------
def ang_err(a: float, b: float) -> float:
    return math.remainder(a - b, 2.0 * math.pi)

async def hold(ctrl: moteus.Controller, pos: float):
    await ctrl.set_position(
        position=float(pos),
        velocity=float('nan'),
        accel_limit=ACCEL_LIMIT,
        velocity_limit=VEL_LIMIT,
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

async def move_group_sync_time(items):
    if not items:
        return []
    deltas = [ang_err(t, c) for (_, c, t) in items]
    max_delta = max(abs(d) for d in deltas) if deltas else 0.0
    if max_delta < 1e-9:
        await asyncio.gather(*(hold(ctrl, tgt) for (ctrl, _, tgt) in items))
        return [t for (_, _, t) in items]

    group_speed = max(GROUP_SPEED_RAD_S, MIN_SPEED)
    duration = max(max_delta / group_speed, 0.2)

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
    def __init__(self, host: str, port: int, servo_pin: int, transport_kind: str):
        self.host = host
        self.port = port
        self.servo_pin = servo_pin

        # Servo driver (robust)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")
        self.current_pwm = SERVO_CENTER
        self.target_pwm  = SERVO_CENTER
        self.pi.set_servo_pulsewidth(self.servo_pin, SERVO_CENTER)

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

        # Cooling state
        self.cooling2 = False
        self.cooling3 = False
        self.last_temp2 = None
        self.last_temp3 = None

        # Comm fault tracking
        self._comm_failures = 0

        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()

    # ===== SERVO ROBUSTNESS =====
    async def _reconnect_pigpio(self) -> bool:
        try:
            if self.pi is not None:
                with contextlib.suppress(Exception):
                    self.pi.stop()
        except Exception:
            pass

        try:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                print("[rpi] SERVO: pigpio reconnect failed (not connected).", file=sys.stderr)
                return False
            self.pi.set_servo_pulsewidth(self.servo_pin, int(self.target_pwm))
            self.current_pwm = int(self.target_pwm)
            print("[rpi] SERVO: pigpio reconnected.")
            return True
        except Exception as e:
            print(f"[rpi] SERVO: pigpio reconnect exception: {e}", file=sys.stderr)
            return False

    async def servo_set(self, pwm_us: int):
        pwm = int(max(SERVO_MIN, min(SERVO_MAX, pwm_us)))
        self.target_pwm = pwm

        if self.pi is None or not getattr(self.pi, "connected", False):
            print("[rpi] SERVO: pigpio disconnected, attempting reconnect...", file=sys.stderr)
            if not await self._reconnect_pigpio():
                return

        for attempt in range(PIGPIO_MAX_RETRIES + 1):
            try:
                self.pi.set_servo_pulsewidth(self.servo_pin, pwm)
                self.current_pwm = pwm
                return
            except Exception as e:
                print(f"[rpi] SERVO: write error (attempt {attempt+1}): {e}", file=sys.stderr)
                try:
                    with contextlib.suppress(Exception):
                        self.pi.stop()
                except Exception:
                    pass
                self.pi = None
                if attempt < PIGPIO_MAX_RETRIES:
                    await asyncio.sleep(PIGPIO_RETRY_DELAY_S)
                    await self._reconnect_pigpio()
                else:
                    return

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

        await self.servo_set(SERVO_CENTER)
        await asyncio.sleep(RECOVER_SLEEP_S)

        try:
            s1, s2, s3 = await asyncio.gather(self.m1.query(), self.m2.query(), self.m3.query())
            self.t1 = self.c1 = float(s1.values.get('position', 0.0))
            self.t2 = self.c2 = float(s2.values.get('position', 0.0))
            self.t3 = self.c3 = float(s3.values.get('position', 0.0))
            await self._report_fault_if_any("m1", s1.values); await self._report_fault_if_any("m2", s2.values); await self._report_fault_if_any("m3", s3.values)
        except Exception as e:
            print(f"[rpi] COMM: recovery query error: {e}", file=sys.stderr)
            self.t1 = self.t2 = self.t3 = 0.0
            self.c1 = self.c2 = self.c3 = 0.0

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
                    await self.servo_set(self.target_pwm)

                    group = [
                        (self.m1, self.c1, self.t1),
                        (self.m2, self.c2, self.t2),
                        (self.m3, self.c3, self.t3),
                    ]
                    try:
                        finals = await move_group_sync_time(group)
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

                if (self.last_temp2 is not None) and (not self.cooling2) and (self.last_temp2 > TEMP_LIMIT):
                    print(f"[rpi] WARN: m2 > {TEMP_LIMIT:.1f} °C → return to REST; enter cooldown2.")
                    await self.servo_set(SERVO_CENTER)
                    finals = await move_group_sync_time([
                        (self.m1, self.c1, REST1),
                        (self.m2, self.c2, REST2),
                        (self.m3, self.c3, REST3),
                    ])
                    self.c1, self.c2, self.c3 = finals
                    self.t1, self.t2, self.t3 = REST1, REST2, REST3
                    self.cooling2 = True

                if (self.last_temp3 is not None) and (not self.cooling3) and (self.last_temp3 > TEMP_LIMIT):
                    print(f"[rpi] WARN: m3 > {TEMP_LIMIT:.1f} °C → return to REST; enter cooldown3.")
                    await self.servo_set(SERVO_CENTER)
                    finals = await move_group_sync_time([
                        (self.m1, self.c1, REST1),
                        (self.m2, self.c2, REST2),
                        (self.m3, self.c3, REST3),
                    ])
                    self.c1, self.c2, self.c3 = finals
                    self.t1, self.t2, self.t3 = REST1, REST2, REST3
                    self.cooling3 = True

                if self.cooling2 and (self.last_temp2 is not None) and (self.last_temp2 <= COOL_RESUME_TEMP):
                    print(f"[rpi] COOL: m2 <= {COOL_RESUME_TEMP:.1f} °C → wiggle2 + resume check.")
                    self.c2 = await wiggle(self.m2, REST2, self.c2)
                    self.t2 = REST2
                    self.cooling2 = False

                if self.cooling3 and (self.last_temp3 is not None) and (self.last_temp3 <= COOL_RESUME_TEMP):
                    print(f"[rpi] COOL: m3 <= {COOL_RESUME_TEMP:.1f} °C → wiggle3 + resume check.")
                    self.c3 = await wiggle(self.m3, REST3, self.c3)
                    self.t3 = REST3
                    self.cooling3 = False

                await asyncio.sleep(IDLE_SLEEP)
        finally:
            with contextlib.suppress(Exception):
                await asyncio.gather(self.m1.set_stop(), self.m2.set_stop(), self.m3.set_stop())
            try:
                if self.pi is not None and getattr(self.pi, "connected", False):
                    self.pi.set_servo_pulsewidth(self.servo_pin, 0)
                    self.pi.stop()
            except Exception:
                pass

    async def heartbeat_loop(self):
        """Periodic status: prints FULL query dict for each motor."""
        names = [("m1", self.m1), ("m2", self.m2), ("m3", self.m3)]
        while not self._stop.is_set():
            # Servo status
            servo_state = "connected" if (self.pi is not None and getattr(self.pi, "connected", False)) else "DISCONNECTED"
            print(f"[rpi] HB: servo={servo_state} target_pwm={self.target_pwm}", flush=True)

            try:
                results = await asyncio.gather(*(ctrl.query() for _, ctrl in names), return_exceptions=True)
                for (name, ctrl), res in zip(names, results):
                    if isinstance(res, Exception):
                        print(f"[rpi] HEARTBEAT: {name} query error: {res}", file=sys.stderr)
                        continue

                    vals = res.values
                    # ---- FULL DICT PRINT HERE ----
                    print(f"[rpi] HB FULL {name} values:")
                    pprint.pprint(vals, width=100, compact=False, sort_dicts=False)

                    # Optional: still surface summarized fields & faults
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

                cmd = cmd_raw.lower()

                if cmd in ("quit", "q"):
                    writer.write(b"ACK bye\n"); await writer.drain()
                    self._stop.set()
                    break

                if cmd == "rest":
                    try:
                        await self.servo_set(SERVO_CENTER)
                        async with self._lock:
                            self.t1, self.t2, self.t3 = REST1, REST2, REST3
                            self.target_pwm = SERVO_CENTER
                        writer.write(b"ACK rest\n")
                        await writer.drain()
                    except Exception as e:
                        print(f"[rpi] REST: error: {e}", file=sys.stderr)
                        writer.write(b"ERR rest failed\n"); await writer.drain()
                    continue

                if cmd.startswith("ik "):
                    parts = cmd_raw.split()
                    if len(parts) != 5:
                        writer.write(b"ERR ik usage: 'ik <x> <y> <z> <wrist>'\n")
                        await writer.drain()
                        continue
                    try:
                        _, sx, sy, sz, sw = parts
                        x = float(sx); y = float(sy); z = float(sz); wrist = float(sw)
                    except ValueError:
                        writer.write(b"ERR ik parse\n"); await writer.drain()
                        continue

                    try:
                        cmds = ik_cmds_bounded(
                            self.ik_arm,
                            (x, y, z),
                            wrist,
                            radius_m=IK_RADIUS_M,
                            z_min=IK_Z_MIN,
                            z_max=IK_Z_MAX
                        )
                        q1 = float(cmds["q1"])
                        q2 = float(cmds["q2"]) * Q2_SCALE
                        q3 = float(cmds["q3"])
                        q4_pwm = int(round(cmds["q4"]))
                    except Exception as e:
                        print(f"[rpi] IK error: {e}", file=sys.stderr)
                        writer.write(b"ERR ik failed\n"); await writer.drain()
                        continue

                    try:
                        await self.servo_set(q4_pwm)
                        async with self._lock:
                            self.target_pwm = q4_pwm
                            if not (self.cooling2 or self.cooling3):
                                self.t1, self.t2, self.t3 = q1, q2, q3
                        writer.write(b"ACK ik\n"); await writer.drain()
                    except Exception as e:
                        print(f"[rpi] IK: servo error: {e}", file=sys.stderr)
                        writer.write(b"ERR ik servo\n"); await writer.drain()
                    continue

                writer.write(b"ERR expected: 'ik <x> <y> <z> <wrist>' or 'rest' or 'quit'\n")
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
        print(f"[rpi] listening on {addrs}, servo pin BCM {self.servo_pin}")
        async with server:
            await self._stop.wait()
            server.close()
            await server.wait_closed()
        ctrl_task.cancel(); hb_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ctrl_task; await hb_task

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
    ap.add_argument("--servo-pin", type=int, default=27, help="BCM GPIO pin for servo signal")
    ap.add_argument("--transport", default="pi3hat",
                    help="pi3hat | socketcan:can0 | auto")
    args = ap.parse_args()

    srv = ArmServer(args.host, args.port, args.servo_pin, args.transport)
    try:
        asyncio.run(srv.serve())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
