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

# --- q4 servo mapping (270°, 500–2500 µs, center 1750 µs) ---
Q4_PWM_CENTER_US   = 1750
Q4_PWM_MIN_US      = 800.0
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

    IMPORTANT: Use a signed planar wrist radius so small x motions near r=L4·cos(pitch)
    don't reverse due to |r - L4 cos(pitch)|.
    """
    x, y, z = target_xyz
    bx, by = arm.base_offset_x, arm.base_offset_y

    # 1) Base yaw from base (bx,by) toward (x,y)
    q1 = atan2(y - by, x - bx)  # [-pi, pi], y changes base rotation

    # TCP planar radius from base and vertical offset relative to shoulder
    r_tcp = hypot(x - bx, y - by)         # >= 0
    z_tcp = z

    # 2) Signed planar reduction directly in the yawed plane:
    px = (r_tcp - arm.L4 * cos(target_pitch_rad))
    pz = (z_tcp - arm.L4 * sin(target_pitch_rad)) - arm.L1

    # Reconstruct wrist coordinates (for debugging/telemetry)
    wx = bx + px * cos(q1)
    wy = by + px * sin(q1)
    wz = arm.L1 + pz

    # 3) Elbow via law of cosines (numeric clamp on D only)
    D = (px*px + pz*pz - arm.L2*arm.L2 - arm.L3*arm.L3) / (2.0 * arm.L2 * arm.L3)
    D = max(-1.0, min(1.0, D))

    s = sqrt(max(0.0, 1.0 - D*D))
    q3 = atan2(+s, D) if arm.prefer_elbow_up else atan2(-s, D)

    # 4) Shoulder (two-atan form)
    k1 = arm.L2 + arm.L3 * cos(q3)
    k2 = arm.L3 * sin(q3)
    q2 = atan2(pz, px) - atan2(k2, k1)

    # 5) Wrist pitch to achieve desired tool pitch
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
            'r_tcp': r_tcp,
        }
    }


def max_height(arm: ArmParams) -> float:
    """Maximum vertical TCP height when the pitch stack is straight up (pitch = +pi/2)."""
    return arm.L1 + arm.L2 + arm.L3 + arm.L4


# -------------------- IMPORT-FRIENDLY WRAPPERS (NO IK LOGIC CHANGED) --------------------

def ik_solve(arm: ArmParams, target_xyz: Tuple[float, float, float], target_pitch_rad: float) -> Dict[str, Dict[str, float]]:
    """Return full IK solution dict."""
    return ik_yppp(arm, target_xyz, target_pitch_rad)


def ik_cmds(arm: ArmParams, target_xyz: Tuple[float, float, float], target_pitch_rad: float) -> Dict[str, float]:
    """Return only the 'cmds' dictionary."""
    return ik_yppp(arm, target_xyz, target_pitch_rad)['cmds']


def make_default_arm() -> ArmParams:
    """Build the same ArmParams used in example_usage()."""
    GR = 9.0
    return ArmParams(
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


# -------------------- BOUNDING CYLINDER GUARD (IK LOGIC UNCHANGED) --------------------

def _inside_cylinder(target_xyz: Tuple[float, float, float], radius_m: float,
                     z_min: Optional[float] = None, z_max: Optional[float] = None) -> bool:
    """
    Returns True if (x,y,z) is inside a vertical cylinder centered on the origin, radius=radius_m.
    - If z_min/z_max are None, height is unbounded (infinite cylinder).
    """
    x, y, z = target_xyz
    in_rad = (x*x + y*y) <= (radius_m * radius_m)
    if z_min is None and z_max is None:
        return in_rad
    if z_min is None:
        return in_rad and (z <= z_max)
    if z_max is None:
        return in_rad and (z >= z_min)
    return in_rad and (z_min <= z <= z_max)


def ik_cmds_bounded(
    arm: ArmParams,
    target_xyz: Tuple[float, float, float],
    target_pitch_rad: float,
    radius_m: float,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> Dict[str, float]:
    """
    If target lies inside the cylinder, return a safe zero command:
      q1=q2=q3=0.0 (motor turns), q4=servo center (1750 µs).
    Otherwise, return normal IK cmds.
    """
    if _inside_cylinder(target_xyz, radius_m, z_min, z_max):
        return {'q1': 0.0, 'q2': 0.0, 'q3': 0.0, 'q4': int(Q4_PWM_CENTER_US)}
    return ik_cmds(arm, target_xyz, target_pitch_rad)


def ik_solve_bounded(
    arm: ArmParams,
    target_xyz: Tuple[float, float, float],
    target_pitch_rad: float,
    radius_m: float,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Full solution with a 'bounded' flag.
    If inside cylinder, returns a fabricated zero-pose solution (angles_rad all 0, cmds as above).
    Otherwise, returns normal IK solution.
    """
    bounded = _inside_cylinder(target_xyz, radius_m, z_min, z_max)
    if not bounded:
        sol = ik_solve(arm, target_xyz, target_pitch_rad)
        sol['bounded'] = False
        return sol

    # Fabricate a zero-pose solution without touching IK internals.
    angles_rad = {'q1': 0.0, 'q2': 0.0, 'q3': 0.0, 'q4': 0.0}
    angles_deg = {k: v * RAD2DEG for k, v in angles_rad.items()}
    cmds = {'q1': 0.0, 'q2': 0.0, 'q3': 0.0, 'q4': int(Q4_PWM_CENTER_US)}
    return {
        'angles_rad': angles_rad,
        'angles_deg': angles_deg,
        'cmds': cmds,
        'intermediate': {
            'wrist_xyz': (0.0, 0.0, 0.0),
            'planar_px': 0.0,
            'planar_pz': 0.0,
            'D': 0.0,
            'k1': 0.0,
            'k2': 0.0,
            'r_tcp': 0.0,
        },
        'bounded': True,
    }


# ----------------------------------- EXAMPLE / CLI -----------------------------------

def example_usage():
    arm = make_default_arm()

    # Example target
    target_xyz = (0.05, 0.02, 0.10)  # inside a 0.1 m cylinder
    target_pitch = 0.0

    # Bounded call: radius 0.1 m, infinite height
    cmds = ik_cmds_bounded(arm, target_xyz, target_pitch, radius_m=0.1)
    print(f"{cmds['q1']:.2f} {cmds['q2']:.2f} {cmds['q3']:.2f} {cmds['q4']}")

if __name__ == "__main__":
    example_usage()
