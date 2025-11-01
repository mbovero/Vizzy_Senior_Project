#!/usr/bin/env python3
# rpi_arm_server.py — Pi-side control of 3 moteus motors + 1 GPIO hobby servo
# With hard-fault (red LED) self-recovery via GPIO-controlled power cycle.

from __future__ import annotations
import argparse, asyncio, math, time, sys
import contextlib

# ---------- Servo -----------
SERVO_MIN    = 500
SERVO_MAX    = 2500
SERVO_CENTER = 1750

try:
    import pigpio
except Exception as e:
    print("ERROR: pigpio not available. Install via: sudo apt-get install pigpio", file=sys.stderr)
    raise

# ---------- GPIO for motor power (relay / high-side switch) ----------
# Wire this BCM pin to control the relay or HSD that powers the moteus bus.
MOTOR_PWR_EN          = 22      # BCM pin; active-high enables power to moteus bus
POWER_ACTIVE_HIGH     = True    # Set False if your board is active-low
POWER_CYCLE_OFF_S     = 2.0     # How long to cut power on reset
POWER_RAIL_STABILIZE  = 1.0     # How long to wait after power on before talking CAN

# ---------- Moteus ----------
import moteus

# ---------- IK ----------
# Uses the provided IK interface. q1,q2,q3 are motor angles (rad), q4 is a wrist PWM (us).
from ik_callable import make_default_arm, ik_cmds_bounded

# ---- IK Tunables ----
IK_RADIUS_M = 0.15
IK_Z_MIN    = 0.0
IK_Z_MAX    = 0.4

# Optional transports (choose one at startup)
def build_transport(kind: str):
    kind = (kind or "pi3hat").lower()
    if kind == "pi3hat":
        try:
            import moteus_pi3hat
            return moteus_pi3hat.Pi3HatRouter()  # shared router
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

TEMP_LIMIT          = 50.0       # °C — enter cooldown when >
COOL_RESUME_TEMP    = 40.0       # °C — resume when <=

WIGGLE_AMPLITUDE    = 0.5
WIGGLE_REPS         = 2
WIGGLE_DWELL_S      = 0.15

# ---- Fault / recovery tunables ----
MAX_CONSEC_CMD_ERRORS   = 5      # after this many consecutive command/query errors, begin recovery
SOFT_RESET_RETRIES      = 3      # try soft reset attempts before hard power cycle
SOFT_RESET_BACKOFF_S    = 0.5

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
    """
    items: list[(ctrl, current_pos, target_pos)]
    """
    if not items:
        return []
    try:
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
    except Exception as e:
        # Bubble up to allow higher level to count errors & recover
        raise

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

# ---------- Server ----------
class ArmServer:
    def __init__(self, host: str, port: int, servo_pin: int, transport_kind: str):
        self.host = host
        self.port = port
        self.servo_pin = servo_pin
        self.transport_kind = transport_kind

        # Servo driver
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")
        self.current_pwm = SERVO_CENTER
        self.pi.set_servo_pulsewidth(self.servo_pin, SERVO_CENTER)

        # Motor power GPIO
        self._power_gpio =  MOTOR_PWR_EN
        self.pi.set_mode(self._power_gpio, pigpio.OUTPUT)
        self._power_set(True)   # ensure power is ON at startup

        # moteus controllers
        self.router = build_transport(transport_kind)
        self._make_controllers()

        # IK arm instance
        self.ik_arm = make_default_arm()

        # Targets/state
        self.t1 = self.t2 = self.t3 = 0.0
        self.c1 = self.c2 = self.c3 = 0.0
        self.target_pwm = SERVO_CENTER

        # Cooling state (separate flags for m2 and m3)
        self.cooling2 = False
        self.cooling3 = False
        self.last_temp2 = None
        self.last_temp3 = None

        # Fault / health bookkeeping
        self._consec_errors = 0
        self._recovering = False
        self._last_recovery = None
        self._last_fault_msg = None

        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()

    # ---------- Power control ----------
    def _power_set(self, on: bool):
        level = 1 if on == POWER_ACTIVE_HIGH else 0
        level = level if on else (0 if POWER_ACTIVE_HIGH else 1)
        self.pi.write(self._power_gpio, level)

    async def _power_cycle(self):
        print("[rpi] RESET: power cycling moteus bus...")
        self._power_set(False)
        await asyncio.sleep(POWER_CYCLE_OFF_S)
        self._power_set(True)
        await asyncio.sleep(POWER_RAIL_STABILIZE)

    # ---------- Controller (re)build ----------
    def _make_controllers(self):
        self.m1 = moteus.Controller(id=ID1, transport=self.router)
        self.m2 = moteus.Controller(id=ID2, transport=self.router)
        self.m3 = moteus.Controller(id=ID3, transport=self.router)

    async def _reopen_transport_and_controllers(self):
        # Some transports tolerate power drops without rebuild; others need a fresh router.
        self.router = build_transport(self.transport_kind)
        self._make_controllers()
        # small grace
        await asyncio.sleep(0.1)

    # ---------- Init / health ----------
    async def init_positions(self):
        with contextlib.suppress(Exception):
            await asyncio.gather(self.m1.set_stop(), self.m2.set_stop(), self.m3.set_stop())
        try:
            s1, s2, s3 = await asyncio.gather(self.m1.query(), self.m2.query(), self.m3.query())
            self.t1 = self.c1 = float(s1.values.get('position', 0.0))
            self.t2 = self.c2 = float(s2.values.get('position', 0.0))
            self.t3 = self.c3 = float(s3.values.get('position', 0.0))
        except Exception as e:
            self._last_fault_msg = f"init_positions query fail: {e}"
            self.t1 = self.t2 = self.t3 = 0.0
            self.c1 = self.c2 = self.c3 = 0.0
        print(f"[rpi] init: ID1={self.c1:.3f}, ID2={self.c2:.3f}, ID3={self.c3:.3f}")

    async def servo_set(self, pwm_us: int):
        pwm = int(max(SERVO_MIN, min(SERVO_MAX, pwm_us)))
        if pwm != self.current_pwm:
            self.pi.set_servo_pulsewidth(self.servo_pin, pwm)
            self.current_pwm = pwm

    # ---------- Recovery paths ----------
    async def _soft_reset_attempt(self) -> bool:
        """Try to clear faults if controller is at least somewhat responsive."""
        try:
            await asyncio.gather(self.m1.set_stop(), self.m2.set_stop(), self.m3.set_stop())
            # Probe quickly
            states = await asyncio.gather(self.m1.query(), self.m2.query(), self.m3.query())
            # If we can read positions, consider it alive.
            _ = [st.values.get('position') for st in states]
            return True
        except Exception as e:
            self._last_fault_msg = f"soft_reset_attempt failed: {e}"
            return False

    async def _hard_reset(self):
        """Power-cycle the motor bus and rebuild transport + controllers."""
        # Safe the servo (so wrist doesn't fight during bus down)
        await self.servo_set(SERVO_CENTER)
        # Go to REST targets so on resume we're safe
        self.t1, self.t2, self.t3 = REST1, REST2, REST3
        # Cut power and re-open
        await self._power_cycle()
        await self._reopen_transport_and_controllers()
        # Try to talk; if it works, move to REST and hold
        await self.init_positions()
        with contextlib.suppress(Exception):
            finals = await move_group_sync_time([
                (self.m1, self.c1, REST1),
                (self.m2, self.c2, REST2),
                (self.m3, self.c3, REST3),
            ])
            if finals and len(finals) == 3:
                self.c1, self.c2, self.c3 = finals
        self._last_recovery = time.strftime("%Y-%m-%d %H:%M:%S")

    async def _recover_if_needed(self, reason: str):
        """Called whenever we observe consecutive command errors beyond threshold."""
        if self._recovering:
            return
        self._recovering = True
        print(f"[rpi] WARN: entering recovery: {reason}")
        try:
            # Try soft reset a few times
            for i in range(SOFT_RESET_RETRIES):
                ok = await self._soft_reset_attempt()
                if ok:
                    print("[rpi] RECOVER: soft reset succeeded")
                    self._consec_errors = 0
                    return
                await asyncio.sleep(SOFT_RESET_BACKOFF_S)

            # Hard reset (power cycle)
            print("[rpi] RECOVER: soft reset failed, performing HARD power-cycle")
            await self._hard_reset()
            self._consec_errors = 0
        finally:
            self._recovering = False

    # ---------- Main control loop ----------
    async def control_loop(self):
        """Main control loop: applies targets, enforces thermal policy, ~50 Hz."""
        next_temp = time.monotonic()
        try:
            while not self._stop.is_set():
                # During recovery, just idle a beat
                if self._recovering:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    async with self._lock:
                        # Apply servo first so it never waits on motor motion
                        await self.servo_set(self.target_pwm)

                        # Then schedule/await the synchronized motor move
                        group = [
                            (self.m1, self.c1, self.t1),
                            (self.m2, self.c2, self.t2),
                            (self.m3, self.c3, self.t3),
                        ]
                        finals = await move_group_sync_time(group)
                        self.c1, self.c2, self.c3 = finals

                    # If we got here, consider it a healthy cycle
                    self._consec_errors = 0

                except Exception as e:
                    # Count and maybe recover
                    self._consec_errors += 1
                    self._last_fault_msg = f"control_loop move/query error: {e}"
                    print(f"[rpi] ERROR: {self._last_fault_msg}")
                    if self._consec_errors >= MAX_CONSEC_CMD_ERRORS:
                        await self._recover_if_needed("too many comm/command errors")
                        # After recovery, continue loop
                        continue

                # Temp check ~1 Hz
                now = time.monotonic()
                if now >= next_temp and not self._recovering:
                    try:
                        s2, s3 = await asyncio.gather(self.m2.query(), self.m3.query())
                        self.last_temp2 = extract_temperature(s2.values)
                        self.last_temp3 = extract_temperature(s3.values)
                    except Exception as e:
                        # Comms problem—count it, possibly trigger recovery
                        self._consec_errors += 1
                        self._last_fault_msg = f"temp query fail: {e}"
                        print(f"[rpi] WARN: {self._last_fault_msg}")
                        if self._consec_errors >= MAX_CONSEC_CMD_ERRORS:
                            await self._recover_if_needed("too many errors during temp poll")
                    next_temp = now + 1.0

                # Thermal policy (independent for m2 and m3). If either overheats, hold all at REST.
                if not self._recovering:
                    if (self.last_temp2 is not None) and (not self.cooling2) and (self.last_temp2 > TEMP_LIMIT):
                        print(f"[rpi] WARN: m2 > {TEMP_LIMIT:.1f} °C → return to REST; enter cooldown2.")
                        await self.servo_set(SERVO_CENTER)
                        self.target_pwm = SERVO_CENTER
                        with contextlib.suppress(Exception):
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
                        self.target_pwm = SERVO_CENTER
                        with contextlib.suppress(Exception):
                            finals = await move_group_sync_time([
                                (self.m1, self.c1, REST1),
                                (self.m2, self.c2, REST2),
                                (self.m3, self.c3, REST3),
                            ])
                            self.c1, self.c2, self.c3 = finals
                        self.t1, self.t2, self.t3 = REST1, REST2, REST3
                        self.cooling3 = True

                    # Resume conditions (wiggle the motor that cooled)
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
            self.pi.set_servo_pulsewidth(self.servo_pin, 0)
            # Leave motor power ON so you can inspect, or turn it off:
            # self._power_set(False)
            self.pi.stop()

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

                # Don't lower-case for numeric parsing; use a shadow var
                cmd_lc = cmd_raw.lower()

                if cmd_lc in ("quit", "q"):
                    writer.write(b"ACK bye\n"); await writer.drain()
                    self._stop.set()
                    break

                if cmd_lc == "rest":
                    await self.servo_set(SERVO_CENTER)
                    async with self._lock:
                        self.t1, self.t2, self.t3 = REST1, REST2, REST3
                        self.target_pwm = SERVO_CENTER
                    writer.write(b"ACK rest\n"); await writer.drain()
                    continue

                if cmd_lc == "health":
                    msg = (
                        f"temps: m2={self.last_temp2} C, m3={self.last_temp3} C; "
                        f"cooling2={self.cooling2}, cooling3={self.cooling3}; "
                        f"errors={self._consec_errors}; recovering={self._recovering}; "
                        f"last_recovery={self._last_recovery}; last_fault='{self._last_fault_msg}'\n"
                    )
                    writer.write(msg.encode("utf-8")); await writer.drain()
                    continue

                # New IK command: ik <x> <y> <z> <wrist>
                if cmd_lc.startswith("ik "):
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
                        q2 = float(cmds["q2"]) * 0.933333
                        q3 = float(cmds["q3"])
                        q4_pwm = int(round(cmds["q4"]))
                    except Exception as e:
                        print(f"[rpi] IK error: {e}", file=sys.stderr)
                        writer.write(b"ERR ik failed\n"); await writer.drain()
                        continue

                    await self.servo_set(q4_pwm)
                    async with self._lock:
                        self.target_pwm = q4_pwm
                        # During cooldown/recovery, motors hold at REST
                        if not (self.cooling2 or self.cooling3 or self._recovering):
                            self.t1, self.t2, self.t3 = q1, q2, q3
                    writer.write(b"ACK ik\n"); await writer.drain()
                    continue

                # Unknown command
                writer.write(b"ERR expected: 'ik <x> <y> <z> <wrist>' or 'rest' or 'health' or 'quit'\n")
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
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        print(f"[rpi] listening on {addrs}, servo pin BCM {self.servo_pin}, motor pwr pin BCM {self._power_gpio}")
        async with server:
            await self._stop.wait()
            server.close()
            await server.wait_closed()
        ctrl_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ctrl_task

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
