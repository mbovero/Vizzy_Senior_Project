#!/usr/bin/python3 -B
# Requires: pip install keyboard moteus

import asyncio
import moteus
import keyboard

# ---- Tunables ----
IDLE_SLEEP      = 0.02         # control-loop period (s)
VEL_MAG         = 5.0          # rad/s magnitude for manual drive
ACCEL_LIMIT     = 5.0
VEL_LIMIT       = 15.0
BRAKE_ACCEL     = 10.0
BRAKE_VEL_LIM   = 10.0
KP_SCALE_HOLD   = 1.0          # soften position spring when holding
KD_SCALE_BRAKE  = 1.0          # damping for braking
MAX_TORQUE_BRK  = 5.0
WATCHDOG        = 0.05

# Set your controller IDs here (must be unique on the daisy chain)
ID_MOTOR_1 = 1   # W/S
ID_MOTOR_2 = 2   # A/D

async def command_velocity(ctrl: moteus.Controller, vel: float):
    """Send velocity/hold command to a single controller."""
    if vel is None:
        # hold/brake
        await ctrl.set_position(
            position=float('nan'),      # velocity mode
            velocity=0.0,               # stop
            accel_limit=BRAKE_ACCEL,
            velocity_limit=BRAKE_VEL_LIM,
            kp_scale=KP_SCALE_HOLD,
            kd_scale=KD_SCALE_BRAKE,
            maximum_torque=MAX_TORQUE_BRK,
            watchdog_timeout=WATCHDOG,
            query=False,
        )
    else:
        await ctrl.set_position(
            position=float('nan'),      # velocity mode
            velocity=vel,
            accel_limit=ACCEL_LIMIT,
            velocity_limit=VEL_LIMIT,
            watchdog_timeout=WATCHDOG,
            query=False,
        )

async def main():
    # Create separate controllers for each target ID
    m1 = moteus.Controller(id=ID_MOTOR_1)
    m2 = moteus.Controller(id=ID_MOTOR_2)

    # Clear any faults
    await asyncio.gather(m1.set_stop(), m2.set_stop())

    try:
        while True:
            # Motor 1 (W/S): W = +VEL, S = -VEL
            if keyboard.is_pressed('w'):
                vel1 = +VEL_MAG
            elif keyboard.is_pressed('s'):
                vel1 = -VEL_MAG
            else:
                vel1 = None  # hold/brake

            # Motor 2 (A/D): D = +VEL, A = -VEL
            if keyboard.is_pressed('a'):
                vel2 = +VEL_MAG
            elif keyboard.is_pressed('d'):
                vel2 = -VEL_MAG
            else:
                vel2 = None  # hold/brake

            # Issue both commands concurrently
            await asyncio.gather(
                command_velocity(m1, vel1),
                command_velocity(m2, vel2),
            )

            await asyncio.sleep(IDLE_SLEEP)

    finally:
        await asyncio.gather(m1.set_stop(), m2.set_stop())

if __name__ == '__main__':
    asyncio.run(main())
