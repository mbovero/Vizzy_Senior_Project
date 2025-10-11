#!/usr/bin/python3 -B
# Requires: pip install moteus

import asyncio
import moteus

# ---- Tunables ----
IDLE_SLEEP      = 2.5          # toggle period (s) – 3 seconds per move
ACCEL_LIMIT     = 5.0        # fast acceleration
VEL_LIMIT       = 10.0        # high top speed
MAX_TORQUE      = 5.0          # torque limit (Nm)
WATCHDOG        = 3.0          # keep slightly above sleep time

# Two controllers
ID_MOTOR_1 = 1
ID_MOTOR_2 = 2

async def command_position(ctrl: moteus.Controller, pos: float):
    """Send absolute position command."""
    await ctrl.set_position(
        position=float(pos),
        velocity=float('nan'),
        accel_limit=ACCEL_LIMIT,
        velocity_limit=VEL_LIMIT,
        maximum_torque=MAX_TORQUE,
        watchdog_timeout=WATCHDOG,
        query=False,
    )

async def main():
    m1 = moteus.Controller(id=ID_MOTOR_1)
    m2 = moteus.Controller(id=ID_MOTOR_2)

    # Clear any faults
    await asyncio.gather(m1.set_stop(), m2.set_stop())
    print("Toggling both motors between 0 rad and 1 rad every 3 seconds...")

    try:
        while True:
            # Move both to 1 rad
            await asyncio.gather(
                command_position(m1, 0),
                command_position(m2, 0),
            )
            await asyncio.sleep(IDLE_SLEEP)

            # Move both to 0 rad
            await asyncio.gather(
                command_position(m1, 1.5),
                command_position(m2, -1),
            )
            await asyncio.sleep(IDLE_SLEEP)

            await asyncio.gather(
                command_position(m1, -2),
                command_position(m2, -1),
            )
            await asyncio.sleep(IDLE_SLEEP)

    finally:
        await asyncio.gather(m1.set_stop(), m2.set_stop())
        print("Motors stopped.")

if __name__ == '__main__':
    asyncio.run(main())
