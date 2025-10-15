from __future__ import annotations
from dataclasses import dataclass
from math import atan2, cos, sin, sqrt, acos, hypot, isfinite, pi
from typing import Optional, Tuple, Dict

# ========================= USER SETUP CHECKLIST =========================
# Fill these in before using IK:
# 1) ArmParams (geometry): L1, L2, L3, L4  [meters]
# 2) JointCalib for q1..q4: theta0 [rad], sign (+1 or -1), limits [rad]
#    Optional PWM mapping per joint: pwm_m (rad/µs), pwm_b (rad)
# 3) prefer_elbow_up: True/False (your default elbow branch)
# 4) In example_usage(): target_xyz (m) and target_pitch_rad (rad)
# =======================================================================

RAD2DEG = 180.0 / pi
DEG2RAD = pi / 180.0

@dataclass
class JointCalib:
    """
    Calibration for a single joint.

    theta0: geometric angle (rad) that you want to correspond to the hardware's logical zero.
            In practice: angle_in_math = sign * (angle_in_hw) + theta0
            We'll invert this when commanding.
    sign: +1 if increasing hardware command increases the positive geometric angle; else -1.
    min_rad, max_rad: joint safety limits in *geometric* radians (after applying theta0 & sign).

    Optional PWM mapping: angle (rad) <-> PWM microseconds via linear map.
      angle(rad) = pwm_m * PWM(µs) + pwm_b
      If you don't use PWM (e.g., FOC position loop), leave m/b as None.
    """
    theta0: float = 0.0
    sign: int = 1
    min_rad: float = -pi
    max_rad: float = pi

    pwm_m: Optional[float] = None
    pwm_b: Optional[float] = None

    def clamp(self, q: float) -> float:
        return max(self.min_rad, min(self.max_rad, q))

    def rad_to_cmd(self, q: float) -> float:
        """Convert geometric angle (rad) to hardware command (PWM µs) if mapping provided; else returns angle."""
        # Convert to joint's hardware-native angle first (invert zero and sign)
        hw_angle = self.sign * (q - self.theta0)
        if self.pwm_m is None or self.pwm_b is None:
            return hw_angle
        return (hw_angle - self.pwm_b) / self.pwm_m

    def cmd_to_rad(self, cmd: float) -> float:
        """Convert hardware command (PWM µs) to geometric angle (rad) if mapping provided; cmd may be angle if no mapping."""
        if self.pwm_m is None or self.pwm_b is None:
            hw_angle = cmd
        else:
            hw_angle = self.pwm_m * cmd + self.pwm_b
        return self.theta0 + self.sign * hw_angle

@dataclass
class ArmParams:
    """
    Yaw–Pitch–Pitch–Pitch arm geometry.

    L1: base height (table/world Z=0 to shoulder axis center)
    L2: shoulder->elbow planar center-to-center distance
    L3: elbow->wrist planar center-to-center distance
    L4: wrist->TCP offset along wrist-forward when pitch=0 (use 0 if unknown)

    Optional small out-of-plane offsets (ignored by the analytic solver but useful for later refinement):
      E2: lateral offset of elbow axis relative to shoulder->elbow plane (mm)
      E3: lateral offset of wrist axis relative to elbow->wrist plane (mm)

    prefer_elbow_up: True for elbow-up branch, False for elbow-down
    """
    L1: float
    L2: float
    L3: float
    L4: float = 0.0

    E2: float = 0.0
    E3: float = 0.0

    prefer_elbow_up: bool = True

    # Joint calibrations q1..q4 (yaw, shoulder, elbow, wrist)
    q1: JointCalib = JointCalib()
    q2: JointCalib = JointCalib()
    q3: JointCalib = JointCalib()
    q4: JointCalib = JointCalib()

class IKError(Exception):
    pass


def _safe_acos(x: float) -> float:
    return acos(max(-1.0, min(1.0, x)))


def ik_yppp(
    arm: ArmParams,
    target_xyz: Tuple[float, float, float],
    target_pitch_rad: float,
    *,
    clamp_to_limits: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Analytic IK for a yaw–pitch–pitch–pitch arm controlling (x,y,z,pitch).

    Inputs:
      arm: ArmParams with L1..L4 and per-joint calibration/limits
      target_xyz: (x, y, z) in world frame (base at origin, +Z up)
      target_pitch_rad: desired tool pitch angle (rad) in the vertical plane selected by yaw

    Returns a dict with angles (rad/deg) and optional hardware commands if PWM mapping is set.
    Raises IKError if unreachable and clamp_to_limits=False.
    """
    x, y, z = target_xyz

    # 1) Base yaw
    q1 = atan2(y, x)

    # 2) Compute wrist position by pulling back L4 along tool direction
    # Tool forward direction in world: [cos(q1)*cos(pitch), sin(q1)*cos(pitch), sin(pitch)]
    wx = x - arm.L4 * cos(q1) * cos(target_pitch_rad)
    wy = y - arm.L4 * sin(q1) * cos(target_pitch_rad)
    wz = z - arm.L4 * sin(target_pitch_rad)

    # 3) Reduce to planar 2-link in yawed plane
    r = hypot(wx, wy)  # horizontal distance from base axis to wrist point
    px = r  # in-plane horizontal
    pz = wz - arm.L1  # in-plane vertical relative to shoulder axis

    # 4) Law of cosines for elbow
    D = (px*px + pz*pz - arm.L2*arm.L2 - arm.L3*arm.L3) / (2.0 * arm.L2 * arm.L3)
    # Reachability check
    if D < -1.0 or D > 1.0:
        if not clamp_to_limits:
            raise IKError(f"Unreachable: wrist distance sqrt(px^2+pz^2)={sqrt(px*px+pz*pz):.2f} vs L2+L3={(arm.L2+arm.L3):.2f}")
        D = max(-1.0, min(1.0, D))

    # Choose elbow branch
    s = sqrt(max(0.0, 1.0 - D*D))
    if arm.prefer_elbow_up:
        q3 = atan2(+s, D)
    else:
        q3 = atan2(-s, D)

    # 5) Shoulder using two-atan formula for better numerics
    k1 = arm.L2 + arm.L3 * cos(q3)
    k2 = arm.L3 * sin(q3)
    q2 = atan2(pz, px) - atan2(k2, k1)

    # 6) Wrist pitch to achieve desired tool pitch
    q4 = target_pitch_rad - (q2 + q3)

    # 7) Build angle vector and clamp/check limits
    q_geom = [q1, q2, q3, q4]
    cals = [arm.q1, arm.q2, arm.q3, arm.q4]

    def _within(j: int, val: float) -> bool:
        return cals[j].min_rad - 1e-6 <= val <= cals[j].max_rad + 1e-6

    if clamp_to_limits:
        q_geom = [cals[i].clamp(q_geom[i]) for i in range(4)]
    else:
        for i in range(4):
            if not _within(i, q_geom[i]):
                raise IKError(f"Joint {i+1} out of limits: {q_geom[i]:.3f} rad not in [{cals[i].min_rad:.3f}, {cals[i].max_rad:.3f}]")

    # 8) Prepare outputs (angles + optional hardware cmds)
    angles_rad = {
        'q1': q_geom[0],
        'q2': q_geom[1],
        'q3': q_geom[2],
        'q4': q_geom[3],
    }
    angles_deg = {k: v * RAD2DEG for k, v in angles_rad.items()}

    cmds = {
        'q1': arm.q1.rad_to_cmd(q_geom[0]),
        'q2': arm.q2.rad_to_cmd(q_geom[1]),
        'q3': arm.q3.rad_to_cmd(q_geom[2]),
        'q4': arm.q4.rad_to_cmd(q_geom[3]),
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


def example_usage():
    # Fill these with YOUR measurements/calibration
    arm = ArmParams(
        L1=0.080,   # meters  <<< EDIT: base height (table -> shoulder axis) in meters
        L2=0.160,   # <<< EDIT: shoulder -> elbow planar distance (m)
        L3=0.160,   # <<< EDIT: elbow -> wrist planar distance (m)
        L4=0.050,   # <<< EDIT: wrist -> TCP along wrist-forward when pitch=0 (m)
        prefer_elbow_up=True,  # <<< EDIT: True for elbow-up branch, False for elbow-down
        q1=JointCalib(theta0=0.0,  # <<< EDIT: yaw zero-offset [rad]
                     sign=+1,     # <<< EDIT: +1 or -1 so positive yaw matches your convention
                     min_rad=-pi, # <<< EDIT: yaw min limit [rad]
                     max_rad= pi, # <<< EDIT: yaw max limit [rad]
                     pwm_m=None,  # (optional) set if using PWM: angle(rad) = m*PWM(µs)+b
                     pwm_b=None),
        q2=JointCalib(theta0=0.0,           # <<< EDIT: shoulder zero-offset [rad]
                     sign=+1,              # <<< EDIT
                     min_rad=-100*DEG2RAD, # <<< EDIT: shoulder min [rad]
                     max_rad= 100*DEG2RAD, # <<< EDIT: shoulder max [rad]
                     pwm_m=None, pwm_b=None),
        q3=JointCalib(theta0=0.0,           # <<< EDIT: elbow zero-offset [rad]
                     sign=+1,              # <<< EDIT
                     min_rad=-135*DEG2RAD, # <<< EDIT: elbow min [rad]
                     max_rad= 135*DEG2RAD, # <<< EDIT: elbow max [rad]
                     pwm_m=None, pwm_b=None),
        q4=JointCalib(theta0=0.0,           # <<< EDIT: wrist zero-offset [rad]
                     sign=+1,              # <<< EDIT
                     min_rad=-135*DEG2RAD, # <<< EDIT: wrist min [rad]
                     max_rad= 135*DEG2RAD, # <<< EDIT: wrist max [rad]
                     pwm_m=None, pwm_b=None),
    )

    target_xyz = (0.20, 0.10, 0.10)   # <<< EDIT: target (x,y,z) in meters, world frame
    target_pitch = 10 * DEG2RAD       # <<< EDIT: desired tool pitch [rad] (relative to your pitch=0°)

    sol = ik_yppp(arm, target_xyz, target_pitch)
    print("Angles (deg):", sol['angles_deg'])
    print("Commands:", sol['cmds'])


if __name__ == "__main__":
    example_usage()
