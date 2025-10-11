#!/usr/bin/python3 -B
# Requires: pip install moteus

import asyncio
import contextlib
import moteus
import math

# ---- Tunables ----
IDLE_SLEEP       = 0.02
ACCEL_LIMIT      = 3
VEL_LIMIT        = 3
WATCHDOG         = 5          # a bit longer to avoid timeout during ramps

MAX_TORQUE_HOLD  = float('nan') # use firmware limit (servo.max_current_A)
KP_SCALE_HOLD    = 3.0
KD_SCALE_HOLD    = 1.0

# Ramp settings (prevents “instant big step” stalls)
STEP_SIZE_RAD    = 0.08         # break big jumps into ~0.08 rad steps
STEP_DELAY_S     = 0.02         # dwell between sub-steps

# Fault polling
RECOVER_POLL_N   = 10

ID_MOTOR_1 = 1
ID_MOTOR_2 = 2
ID_MOTOR_3 = 3

async def hold(ctrl: moteus.Controller, pos: float):
    await ctrl.set_position(
        position=float(pos),
        velocity=float('nan'),
        accel_limit=ACCEL_LIMIT,
        velocity_limit=VEL_LIMIT,
        kp_scale=KP_SCALE_HOLD,
        kd_scale=KD_SCALE_HOLD,
        maximum_torque=MAX_TORQUE_HOLD,
        watchdog_timeout=WATCHDOG,
        query=False,
    )

async def recover_if_faulted(ctrl: moteus.Controller, hold_pos: float):
    try:
        st = await ctrl.query()
        fault = int(st.values.get('fault', 0))
        if fault != 0:
            print(f"[recover] ID{ctrl.id} fault={fault} -> stop & re-hold")
            await ctrl.set_stop()
            await asyncio.sleep(0.02)
            await hold(ctrl, hold_pos)
    except Exception as e:
        print(f"[recover] ID{ctrl.id} query err: {e} -> stop & re-hold")
        with contextlib.suppress(Exception):
            await ctrl.set_stop()
            await asyncio.sleep(0.02)
            await hold(ctrl, hold_pos)

async def move_ramped(ctrl: moteus.Controller, current: float, target: float) -> float:
    """Move from current -> target in small position steps, holding at each."""
    delta = target - current
    steps = max(1, int(abs(delta) / STEP_SIZE_RAD))
    step = delta / steps
    pos = current
    for i in range(steps):
        pos += step
        await hold(ctrl, pos)
        await asyncio.sleep(STEP_DELAY_S)
        # Optional: quick sanity check & light telemetry
        try:
            st = await ctrl.query()
            mode  = int(st.values.get('mode', -1))
            fault = int(st.values.get('fault', 0))
            if fault != 0 or mode == 0:
                print(f"[ramp] ID{ctrl.id} mode={mode} fault={fault} at pos={pos:.3f} -> recover")
                await recover_if_faulted(ctrl, pos)
        except Exception:
            pass
    # final hold at exact target
    await hold(ctrl, target)
    return target

async def read_targets(queue: asyncio.Queue):
    while True:
        text = await asyncio.to_thread(input, "enter '<p1> <p2> <p3>' (rad), or 'q' to quit: ")
        t = text.strip().lower()
        if t in ("q", "quit", "exit"):
            await queue.put(None);  return
        try:
            parts = [float(x) for x in text.split()]
            while len(parts) < 3: parts.append(None)
            await queue.put(tuple(parts[:3]))
        except ValueError:
            print(f"Invalid input: {text!r}")

async def main():
    m1 = moteus.Controller(id=ID_MOTOR_1)
    m2 = moteus.Controller(id=ID_MOTOR_2)
    m3 = moteus.Controller(id=ID_MOTOR_3)

    await asyncio.gather(m1.set_stop(), m2.set_stop(), m3.set_stop())
    try:
        s1, s2, s3 = await asyncio.gather(m1.query(), m2.query(), m3.query())
        t1 = c1 = float(s1.values.get('position', 0.0))
        t2 = c2 = float(s2.values.get('position', 0.0))
        t3 = c3 = float(s3.values.get('position', 0.0))
    except Exception:
        t1 = t2 = t3 = 0.0
        c1 = c2 = c3 = 0.0
    print(f"[init] ID1={c1:.3f}, ID2={c2:.3f}, ID3={c3:.3f} rad")

    q = asyncio.Queue()
    reader_task = asyncio.create_task(read_targets(q))
    poll = 0

    try:
        while True:
            updated1 = updated2 = updated3 = False
            while not q.empty():
                nxt = await q.get()
                if nxt is None: return
                p1, p2, p3 = nxt
                if p1 is not None: t1 = p1; updated1 = True
                if p2 is not None: t2 = p2; updated2 = True
                if p3 is not None: t3 = p3; updated3 = True

            # Ramp only when the target changes; otherwise keep holding
            if updated1: c1 = await move_ramped(m1, c1, t1)
            else:        await hold(m1, t1)

            if updated2: c2 = await move_ramped(m2, c2, t2)
            else:        await hold(m2, t2)

            if updated3: c3 = await move_ramped(m3, c3, t3)
            else:        await hold(m3, t3)

            # periodic fault recovery so it never stays disabled
            poll = (poll + 1) % RECOVER_POLL_N
            if poll == 0:
                await asyncio.gather(
                    recover_if_faulted(m1, t1),
                    recover_if_faulted(m2, t2),
                    recover_if_faulted(m3, t3),
                )

            await asyncio.sleep(IDLE_SLEEP)

    finally:
        reader_task.cancel()
        with contextlib.suppress(Exception):
            await reader_task
        await asyncio.gather(m1.set_stop(), m2.set_stop(), m3.set_stop())

if __name__ == '__main__':
    asyncio.run(main())
