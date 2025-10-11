#!/usr/bin/python3 -B
# Requires: pip install keyboard moteus

import asyncio
import moteus
import keyboard
import contextlib
from math import isnan

# ---- Tunables ----
IDLE_SLEEP      = 0.02        # control-loop period (s)
VEL_MAG         = 20.0        # rad/s for manual drive (ID 2 only)
ACCEL_LIMIT     = 1.0
VEL_LIMIT       = 20.0
BRAKE_ACCEL     = 50.0
BRAKE_VEL_LIM   = 50.0
KP_SCALE_HOLD   = 1.2
KD_SCALE_BRAKE  = 2.0
MAX_TORQUE_CMD  = float('nan')   # NaN -> use servo.max_current_A in firmware
WATCHDOG        = 0.8

# Optional gravity bias (Nm). Tune per joint if needed.
FF_TORQUE_1     = 0.0
FF_TORQUE_2     = 0.0
FF_TORQUE_3     = 0.0

# Controller IDs
ID_MOTOR_1 = 1   # held/brake
ID_MOTOR_2 = 2   # A/D control
ID_MOTOR_3 = 3   # held/brake

# Telemetry printing cadence
PRINT_EVERY = 25   # 25 * 0.02s = ~0.5s


async def command_velocity(ctrl: moteus.Controller, vel: float | None, ff=0.0):
    """Velocity mode when vel is number; hold/brake when vel is None."""
    if vel is None:
        # hold/brake (zero velocity target with damping)
        await ctrl.set_position(
            position=float('nan'),
            velocity=0.0,
            accel_limit=BRAKE_ACCEL,
            velocity_limit=BRAKE_VEL_LIM,
            kp_scale=KP_SCALE_HOLD,
            kd_scale=KD_SCALE_BRAKE,
            maximum_torque=MAX_TORQUE_CMD,
            feedforward_torque=ff,
            watchdog_timeout=WATCHDOG,
            query=False,
        )
    else:
        await ctrl.set_position(
            position=float('nan'),
            velocity=vel,
            accel_limit=ACCEL_LIMIT,
            velocity_limit=VEL_LIMIT,
            maximum_torque=MAX_TORQUE_CMD,
            feedforward_torque=ff,
            watchdog_timeout=WATCHDOG,
            query=False,
        )


async def manual_rearm(ctrl, ff=0.0):
    """Manual re-enable if you hit R: clear & assert a hold."""
    await ctrl.set_stop()
    await asyncio.sleep(0.02)
    await ctrl.set_position(
        position=float('nan'),
        velocity=0.0,
        accel_limit=BRAKE_ACCEL,
        velocity_limit=BRAKE_VEL_LIM,
        kp_scale=KP_SCALE_HOLD,
        kd_scale=KD_SCALE_BRAKE,
        maximum_torque=MAX_TORQUE_CMD,
        feedforward_torque=ff,
        watchdog_timeout=WATCHDOG,
        query=False,
    )


def _v(vals, *names, default=float('nan')):
    """Return first present value from possible field names."""
    for n in names:
        if n in vals and vals[n] is not None:
            return vals[n]
    return default


async def main():
    # Create controllers
    m1 = moteus.Controller(id=ID_MOTOR_1)
    m2 = moteus.Controller(id=ID_MOTOR_2)
    m3 = moteus.Controller(id=ID_MOTOR_3)

    # Clear any faults and start all in a safe hold
    await asyncio.gather(m1.set_stop(), m2.set_stop(), m3.set_stop())
    await asyncio.gather(
        command_velocity(m1, None, FF_TORQUE_1),
        command_velocity(m2, None, FF_TORQUE_2),
        command_velocity(m3, None, FF_TORQUE_3),
    )

    loop_count = 0

    try:
        while True:
            # Telemetry: print ID2 torque/current/temp/voltage/mode/fault
            if PRINT_EVERY and (loop_count % PRINT_EVERY == 0):
                try:
                    st = await m2.query()
                    vals = st.values
                    torque_nm = _v(vals, 'torque', 'torque_Nm', default=0.0)
                    q_current = _v(vals, 'q_current', 'current_A', 'current', default=float('nan'))
                    temp_c    = _v(vals, 'temperature', 'temperature_C', default=float('nan'))
                    volt_v    = _v(vals, 'voltage', 'voltage_V', default=float('nan'))
                    mode      = int(_v(vals, 'mode', default=-1))
                    fault     = int(_v(vals, 'fault', default=0))

                    # Pretty printing with NA fallback
                    def fmt(x, unit=""):
                        if isinstance(x, (int, float)) and not isnan(x):
                            return f"{x:.3f}{unit}" if unit else f"{x:.3f}"
                        return "NA"

                    print(f"[ID2] τ={fmt(torque_nm,' Nm')} | Iq={fmt(q_current,' A')} | "
                          f"T={fmt(temp_c,' °C')} | Vbus={fmt(volt_v,' V')} | mode={mode} fault={fault}")
                except Exception:
                    pass
            loop_count += 1

            # Manual re-arm hotkey (clears any clamped/latched state)
            if keyboard.is_pressed('r'):
                await manual_rearm(m2, FF_TORQUE_2)

            # ID1 & ID3 always hold/brake
            vel1 = None
            vel3 = None

            # ID2: A/D keyboard control
            if keyboard.is_pressed('d'):
                vel2 = +VEL_MAG
            elif keyboard.is_pressed('a'):
                vel2 = -VEL_MAG
            else:
                vel2 = None

            # Send commands (no parallel queries here)
            await asyncio.gather(
                command_velocity(m1, vel1, FF_TORQUE_1),
                command_velocity(m2, vel2, FF_TORQUE_2),
                command_velocity(m3, vel3, FF_TORQUE_3),
            )

            await asyncio.sleep(IDLE_SLEEP)

    finally:
        await asyncio.gather(m1.set_stop(), m2.set_stop(), m3.set_stop())


if __name__ == '__main__':
    asyncio.run(main())
