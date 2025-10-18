from __future__ import annotations
from dataclasses import dataclass, field
from math import atan2, cos, sin, sqrt, hypot, pi
from typing import Optional, Tuple, Dict

# ========================= USER SETUP CHECKLIST =========================
# 1) Set ArmParams geometry L1..L4 in meters (center-to-center distances).
# 2) Set JointCalib per joint: theta0 [rad], sign (+1/-1), limits [rad].
#    - If pwm_m/pwm_b are set, rad <-> PWM µs is used.
#    - Else, rad <-> motor turns is used via gear_ratio.
# 3) prefer_elbow_up: True/False branch.
# 4) In example_usage(): set your target_xyz (m) and target_pitch_rad (rad).
# =======================================================================

RAD2DEG = 180.0 / pi
DEG2RAD = pi / 180.0

# --- q4 servo mapping (270°, 500–2500 µs, center 1500 µs) ---
Q4_PWM_CENTER_US   = 1750
Q4_PWM_MIN_US      = 500.0
Q4_PWM_MAX_US      = 2500.0
Q4_PWM_HALFSPAN_US = 1000.0
Q4_MAX_DEG         = 135.0
Q4_US_PER_RAD      = Q4_PWM_HALFSPAN_US / (Q4_MAX_DEG * DEG2RAD)  # µs per rad


@dataclass
class JointCalib:
    """
    Calibration for a single joint.

    theta0: geometric angle (rad) corresponding to hardware logical zero.
            angle_math = sign * (angle_hw) + theta0
    sign: +1 if increasing hardware command increases positive geometric angle; else -1.

    Option A (PWM mapping):
      angle(rad) = pwm_m * PWM(µs) + pwm_b
      rad_to_cmd returns PWM µs; cmd_to_rad expects PWM µs.

    Option B (gear mapping; default):
      Commands are in motor turns (1.0 = one motor revolution), using gear_ratio.
    """
    theta0: float = 0.0
    sign: int = 1
    min_rad: float = -pi
    max_rad: float = pi

    pwm_m: Optional[float] = None
    pwm_b: Optional[float] = None

    gear_ratio: float = 1.0  # motor_turns / joint_turn

    def rad_to_cmd(self, q: float) -> float:
        """Geometric angle (rad) -> hardware command (PWM µs or motor turns)."""
        hw_angle_joint = self.sign * (q - self.theta0)
        if self.pwm_m is not None and self.pwm_b is not None:
            return (hw_angle_joint - self.pwm_b) / self.pwm_m
        return (self.gear_ratio * hw_angle_joint) / (2.0 * pi)

    def cmd_to_rad(self, cmd: float) -> float:
        """Hardware command -> geometric angle (rad)."""
        if self.pwm_m is not None and self.pwm_b is not None:
            hw_angle_joint = self.pwm_m * cmd + self.pwm_b
        else:
            hw_angle_joint = (cmd * 2.0 * pi) / self.gear_ratio
        return self.theta0 + self.sign * hw_angle_joint


@dataclass
class ArmParams:
    """
    Yaw–Pitch–Pitch–Pitch arm geometry and calibration.
    """
    L1: float
    L2: float
    L3: float
    L4: float = 0.0

    prefer_elbow_up: bool = False
    base_offset_x: float = 0.0
    base_offset_y: float = 0.0

    # Joint calibrations q1..q4 (yaw, shoulder, elbow, wrist)
    q1: JointCalib = field(default_factory=JointCalib)
    q2: JointCalib = field(default_factory=JointCalib)
    q3: JointCalib = field(default_factory=JointCalib)
    q4: JointCalib = field(default_factory=JointCalib)


class IKError(Exception):
    pass


def ik_yppp(
    arm: ArmParams,
    target_xyz: Tuple[float, float, float],
    target_pitch_rad: float,
) -> Dict[str, Dict[str, float]]:
    """
    Analytic IK for a yaw–pitch–pitch–pitch arm controlling (x, y, z, pitch).
    No joint-limit clamping; q1 is NOT wrapped.
    """
    x, y, z = target_xyz

    # 1) Base yaw from base (bx,by) toward (x,y)
    bx, by = arm.base_offset_x, arm.base_offset_y
    q1 = atan2(y - by, x - bx)  # [-pi, pi], y changes base rotation

    # 2) Wrist position by pulling back L4 along tool direction
    wx = x - arm.L4 * cos(q1) * cos(target_pitch_rad)
    wy = y - arm.L4 * sin(q1) * cos(target_pitch_rad)
    wz = z - arm.L4 * sin(target_pitch_rad)

    # 3) Planar reduction in the yawed plane
    r = hypot(wx - bx, wy - by)
    px = r
    pz = wz - arm.L1

    # 4) Elbow via law of cosines (numeric clamp on D only)
    D = (px*px + pz*pz - arm.L2*arm.L2 - arm.L3*arm.L3) / (2.0 * arm.L2 * arm.L3)
    D = max(-1.0, min(1.0, D))

    s = sqrt(max(0.0, 1.0 - D*D))
    q3 = atan2(+s, D) if arm.prefer_elbow_up else atan2(-s, D)

    # 5) Shoulder (two-atan form)
    k1 = arm.L2 + arm.L3 * cos(q3)
    k2 = arm.L3 * sin(q3)
    q2 = atan2(pz, px) - atan2(k2, k1)

    # 6) Wrist pitch to achieve desired tool pitch
    q4 = target_pitch_rad - (q2 + q3)

    # Outputs
    angles_rad = {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4}
    angles_deg = {k: v * RAD2DEG for k, v in angles_rad.items()}

    # q4: radians -> PWM µs (saturate to electrical bounds only)
    q4_pwm = Q4_PWM_CENTER_US + q4 * Q4_US_PER_RAD
    q4_pwm = max(Q4_PWM_MIN_US, min(Q4_PWM_MAX_US, q4_pwm))

    cmds = {
        'q1': arm.q1.rad_to_cmd(q1),  # motor turns
        'q2': arm.q2.rad_to_cmd(q2),  # motor turns
        'q3': arm.q3.rad_to_cmd(q3),  # motor turns
        'q4': int(round(q4_pwm)),     # PWM µs
    }

    return {
        'angles_rad': angles_rad,
        'angles_deg': angles_deg,
        'cmds': cmds,
        'intermediate': {
            'wrist_xyz': (wx, wy, wz),
            'planar_px': px,
            'planar_pz': pz,
            'D': D,
            'k1': k1,
            'k2': k2,
        }
    }


def max_height(arm: ArmParams) -> float:
    """Maximum vertical TCP height when the pitch stack is straight up."""
    return arm.L1 + arm.L2 + arm.L3 + arm.L4


def example_usage():
    # All geared joints are 9:1
    GR = 9.0

    arm = ArmParams(
        L1=0.176932,   # m
        L2=0.162737,   # m
        L3=0.226991,   # m
        L4=0.12047,    # m
        prefer_elbow_up=False,
        base_offset_x=0.0,
        base_offset_y=0.0,
        q1=JointCalib(theta0=0.0,    sign=+1, gear_ratio=GR),
        q2=JointCalib(theta0=pi/2.0, sign=+1, gear_ratio=GR),
        q3=JointCalib(theta0=0.0,    sign=-1, gear_ratio=GR),
        q4=JointCalib(theta0=0.0,    sign=+1, gear_ratio=GR),
    )

    # Example target (change y to test base rotation)
    target_xyz = (0.0, -0.2, 0.40)
    target_pitch = pi / 2  # tool pointing straight up

    sol = ik_yppp(arm, target_xyz, target_pitch)

    # PRINT EXACTLY: "q1 q2 q3 q4" (q1-3 two decimals, q4 int)
    c = sol['cmds']
    print(f"{c['q1']:.2f} {c['q2']:.2f} {c['q3']:.2f} {c['q4']}")

if __name__ == "__main__":
    example_usage()