#!/usr/bin/env python3
"""
Single-file Vizzy Click-to-Move GUI (positive X shown on the LEFT)

- Uses your exact working kinematics (inlined below).
- x,y plane: [-0.67, +0.67] m; X+ is on the LEFT side of the canvas.
- Live z slider & pitch entry update the preview (no re-click needed).
- Press Enter to confirm move; Esc = Stop motors; Q = Quit.
- Multiple moves supported via an asyncio command queue.

Requires: pip install moteus
"""

from __future__ import annotations
import asyncio
import contextlib
from dataclasses import dataclass, field
from math import atan2, cos, sin, sqrt, hypot, pi
from typing import Optional, Tuple, Dict

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    raise SystemExit("Tkinter is required for this GUI")

# ========================= EXACT KINEMATICS =========================
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
    theta0: float = 0.0
    sign: int = 1
    min_rad: float = -pi
    max_rad: float = pi
    pwm_m: Optional[float] = None
    pwm_b: Optional[float] = None
    gear_ratio: float = 1.0

    def rad_to_cmd(self, q: float) -> float:
        hw_angle_joint = self.sign * (q - self.theta0)
        if self.pwm_m is not None and self.pwm_b is not None:
            return (hw_angle_joint - self.pwm_b) / self.pwm_m
        return (self.gear_ratio * hw_angle_joint) / (2.0 * pi)

    def cmd_to_rad(self, cmd: float) -> float:
        if self.pwm_m is not None and self.pwm_b is not None:
            hw_angle_joint = self.pwm_m * cmd + self.pwm_b
        else:
            hw_angle_joint = (cmd * 2.0 * pi) / self.gear_ratio
        return self.theta0 + self.sign * hw_angle_joint

@dataclass
class ArmParams:
    L1: float
    L2: float
    L3: float
    L4: float = 0.0
    prefer_elbow_up: bool = False
    base_offset_x: float = 0.0
    base_offset_y: float = 0.0
    q1: JointCalib = field(default_factory=JointCalib)
    q2: JointCalib = field(default_factory=JointCalib)
    q3: JointCalib = field(default_factory=JointCalib)
    q4: JointCalib = field(default_factory=JointCalib)

class IKError(Exception):
    pass

def ik_yppp(arm: ArmParams, target_xyz: Tuple[float, float, float], target_pitch_rad: float) -> Dict[str, Dict[str, float]]:
    x, y, z = target_xyz
    bx, by = arm.base_offset_x, arm.base_offset_y
    q1 = atan2(y - by, x - bx)
    r_tcp = hypot(x - bx, y - by)
    z_tcp = z
    px = (r_tcp - arm.L4 * cos(target_pitch_rad))
    pz = (z_tcp - arm.L4 * sin(target_pitch_rad)) - arm.L1
    wx = bx + px * cos(q1)
    wy = by + px * sin(q1)
    wz = arm.L1 + pz
    D = (px*px + pz*pz - arm.L2*arm.L2 - arm.L3*arm.L3) / (2.0 * arm.L2 * arm.L3)
    D = max(-1.0, min(1.0, D))
    s = sqrt(max(0.0, 1.0 - D*D))
    q3 = atan2(+s, D) if arm.prefer_elbow_up else atan2(-s, D)
    k1 = arm.L2 + arm.L3 * cos(q3)
    k2 = arm.L3 * sin(q3)
    q2 = atan2(pz, px) - atan2(k2, k1)
    q4 = target_pitch_rad - (q2 + q3)
    angles_rad = {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4}
    angles_deg = {k: v * RAD2DEG for k, v in angles_rad.items()}
    q4_pwm = Q4_PWM_CENTER_US + q4 * Q4_US_PER_RAD
    q4_pwm = max(Q4_PWM_MIN_US, min(Q4_PWM_MAX_US, q4_pwm))
    cmds = {
        'q1': arm.q1.rad_to_cmd(q1),
        'q2': arm.q2.rad_to_cmd(q2),
        'q3': arm.q3.rad_to_cmd(q3),
        'q4': int(round(q4_pwm)),
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
    return arm.L1 + arm.L2 + arm.L3 + arm.L4

# Arm instance (your values)
GR = 9.0
ARM = ArmParams(
    L1=0.176932, L2=0.162737, L3=0.226991, L4=0.12047,
    prefer_elbow_up=False,
    q1=JointCalib(theta0=0.0,    sign=+1, gear_ratio=GR),
    q2=JointCalib(theta0=pi/2.0, sign=+1, gear_ratio=GR),
    q3=JointCalib(theta0=0.0,    sign=-1, gear_ratio=GR),
    q4=JointCalib(theta0=0.0,    sign=+1, gear_ratio=GR),
)

# ========================= GUI / CONTROL CONFIG =========================
PLANE_RANGE_M   = 0.67
CANVAS_PX       = 720
GRID_STEP_M     = 0.10

ID1, ID2, ID3   = 1, 2, 3
GROUP_DURATION_S= 1.0
ACCEL_MULT      = 1.5
VEL_MIN         = 0.01
WATCHDOG_S      = 10

async def query_positions(m1, m2, m3):
    try:
        s1, s2, s3 = await asyncio.gather(m1.query(), m2.query(), m3.query())
        return (
            float(s1.values.get('position', 0.0)),
            float(s2.values.get('position', 0.0)),
            float(s3.values.get('position', 0.0)),
        )
    except Exception:
        return (0.0, 0.0, 0.0)

async def move_group_sync_time(m1, m2, m3, t1, t2, t3, duration=GROUP_DURATION_S):
    c1, c2, c3 = await query_positions(m1, m2, m3)
    deltas = [t1 - c1, t2 - c2, t3 - c3]
    v = [max(VEL_MIN, abs(d)/max(duration,1e-3)) for d in deltas]
    a = [max(vi * ACCEL_MULT, VEL_MIN) for vi in v]
    await asyncio.gather(
        m1.set_position(position=float(t1), velocity=float('nan'), accel_limit=a[0], velocity_limit=v[0], watchdog_timeout=WATCHDOG_S, query=False),
        m2.set_position(position=float(t2), velocity=float('nan'), accel_limit=a[1], velocity_limit=v[1], watchdog_timeout=WATCHDOG_S, query=False),
        m3.set_position(position=float(t3), velocity=float('nan'), accel_limit=a[2], velocity_limit=v[2], watchdog_timeout=WATCHDOG_S, query=False),
    )

# ========================= GUI APP =========================
class ClickToMoveApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Vizzy Click-to-Move (X+ on LEFT)")
        self.root.geometry(f"{CANVAS_PX+360}x{CANVAS_PX+70}")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=CANVAS_PX, height=CANVAS_PX, bg="#111")
        self.canvas.grid(row=0, column=0, padx=10, pady=10, rowspan=10)
        self.info = ttk.Frame(root, padding=10)
        self.info.grid(row=0, column=1, sticky="n")

        # Height
        try:
            _maxz = max_height(ARM)
        except Exception:
            _maxz = ARM.L1 + ARM.L2 + ARM.L3 + ARM.L4
        self.z_var = tk.DoubleVar(value=min(_maxz, 0.30))
        ttk.Label(self.info, text="Height z (m)").grid(row=0, column=0, sticky="w")
        self.z_scale = ttk.Scale(self.info, orient="horizontal", from_=0.0, to=_maxz, variable=self.z_var, length=280)
        self.z_scale.grid(row=1, column=0, pady=(0,8), sticky="we")
        self.z_val_lbl = ttk.Label(self.info, text=f"z = {self.z_var.get():.3f} m")
        self.z_val_lbl.grid(row=2, column=0, sticky="w")
        self.z_var.trace_add('write', lambda *a: self.z_val_lbl.config(text=f"z = {self.z_var.get():.3f} m"))

        # Pitch
        ttk.Label(self.info, text="Pitch (rad)").grid(row=3, column=0, sticky="w", pady=(8,0))
        self.pitch_entry = ttk.Entry(self.info, width=14)
        self.pitch_entry.insert(0, "0.0")
        self.pitch_entry.grid(row=4, column=0, sticky="w")

        # Preview
        self.preview = tk.StringVar(value="Click the plane to preview.")
        ttk.Label(self.info, textvariable=self.preview, justify="left", wraplength=300)\
            .grid(row=5, column=0, sticky="w", pady=(12,0))

        # Buttons
        btns = ttk.Frame(self.info)
        btns.grid(row=6, column=0, pady=(10,0), sticky="w")
        self.move_btn  = ttk.Button(btns, text="Move (Enter)",      command=self.enqueue_move)
        self.stop_btn  = ttk.Button(btns, text="Stop Motors (Esc)", command=self.stop_motors)
        self.quit_btn  = ttk.Button(btns, text="Quit (Q)",          command=self.quit)
        self.move_btn.grid(row=0, column=0, padx=(0,8))
        self.stop_btn.grid(row=0, column=1, padx=(0,8))
        self.quit_btn.grid(row=0, column=2)

        # Status
        self.status = tk.StringVar(value="Idle")
        ttk.Label(self.info, textvariable=self.status).grid(row=7, column=0, sticky="w", pady=(8,0))

        # Bindings
        self.canvas.bind('<Button-1>', self.on_click)
        self.root.bind_all('<Return>',  lambda e: self.enqueue_move())
        self.root.bind_all('<Escape>',  lambda e: self.stop_motors())
        self.root.bind_all('<q>',       lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.pitch_entry.bind('<KeyRelease>', lambda e: self._recompute_from_params())
        self.z_scale.configure(command=lambda _v: self._recompute_from_params())

        self.draw_plane()

        # Async / hardware
        self.loop = asyncio.get_event_loop()
        self.m1 = self.m2 = self.m3 = None
        self.loop.create_task(self._init_moteus())

        # Interaction state
        self._last_xy: Optional[Tuple[float,float]] = None
        self._pending_targets: Optional[Tuple[float,float,float]] = None  # (t1,t2,t3)

        # Command queue & worker (allows multiple sequential moves)
        self.cmd_q: asyncio.Queue = asyncio.Queue()
        self.loop.create_task(self._cmd_worker())

    # ----- Coord mapping (X+ on LEFT) -----
    def m_to_px(self, x: float, y: float) -> tuple[int,int]:
        scale = (CANVAS_PX/2) / PLANE_RANGE_M
        cx = CANVAS_PX//2 - int(round(x * scale))   # invert X (X+ left)
        cy = CANVAS_PX//2 - int(round(y * scale))   # Y up
        return cx, cy

    def px_to_m(self, px: int, py: int) -> tuple[float,float]:
        scale = (CANVAS_PX/2) / PLANE_RANGE_M
        x = -(px - CANVAS_PX/2) / scale             # invert X (X+ left)
        y = -(py - CANVAS_PX/2) / scale
        return x, y

    # ----- Drawing -----
    def draw_plane(self):
        c = self.canvas
        c.delete('all')
        g = GRID_STEP_M
        v = PLANE_RANGE_M
        n = int(v/g)
        for i in range(-n, n+1):
            x = i*g
            x0, y0 = self.m_to_px(x, -v)
            x1, y1 = self.m_to_px(x, +v)
            c.create_line(x0, y0, x1, y1, fill="#2a2a2a")
            y = i*g
            x0, y0 = self.m_to_px(-v, y)
            x1, y1 = self.m_to_px(+v, y)
            c.create_line(x0, y0, x1, y1, fill="#2a2a2a")
        # axes
        x0, y0 = self.m_to_px(-v, 0); x1, y1 = self.m_to_px(+v, 0)
        c.create_line(x0, y0, x1, y1, fill="#55aaff", width=2)
        x0, y0 = self.m_to_px(0, -v); x1, y1 = self.m_to_px(0, +v)
        c.create_line(x0, y0, x1, y1, fill="#55aaff", width=2)
        # border
        c.create_rectangle(*self.m_to_px(-v,-v), *self.m_to_px(+v,+v), outline="#777")

    # ----- Events / preview -----
    def on_click(self, ev):
        x, y = self.px_to_m(ev.x, ev.y)
        x = max(-PLANE_RANGE_M, min(PLANE_RANGE_M, x))
        y = max(-PLANE_RANGE_M, min(PLANE_RANGE_M, y))
        self._last_xy = (x, y)
        self._update_preview(x, y)

    def _recompute_from_params(self):
        if self._last_xy is None:
            return
        self._update_preview(*self._last_xy)

    def _update_preview(self, x: float, y: float):
        try:
            z = float(self.z_var.get())
        except Exception:
            z = 0.0
        try:
            pitch = float(self.pitch_entry.get().strip() or 0.0)
        except Exception:
            pitch = 0.0
        try:
            sol = ik_yppp(ARM, (x, y, z), pitch)
            c = sol['cmds']
            t1, t2, t3 = float(c['q1']), float(c['q2']), float(c['q3'])
            q4_pwm = int(c['q4'])
            a = sol['angles_deg']
            self._pending_targets = (t1, t2, t3)
            self.preview.set(
                f"x={x:.3f}, y={y:.3f}, z={z:.3f}, pitch={pitch:.2f}\n"
                f"Angles°: q1={a['q1']:.1f}, q2={a['q2']:.1f}, q3={a['q3']:.1f}, q4={a['q4']:.1f}\n"
                f"Cmds: q1={t1:.3f}, q2={t2:.3f}, q3={t3:.3f}, q4={q4_pwm} µs"
            )
            # marker
            self.draw_plane()
            cx, cy = self.m_to_px(x, y)
            r = 5
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#00ff88", width=2)
        except Exception as e:
            self.preview.set(f"IK failed: {e}")
            self._pending_targets = None

    # ----- Hardware init -----
    async def _init_moteus(self):
        import moteus
        self.m1 = moteus.Controller(id=ID1)
        self.m2 = moteus.Controller(id=ID2)
        self.m3 = moteus.Controller(id=ID3)
        with contextlib.suppress(Exception):
            await asyncio.gather(self.m1.set_stop(), self.m2.set_stop(), self.m3.set_stop())
        self.status.set("Hardware ready")

    # ----- Command queue & worker -----
    def enqueue_move(self):
        if self._pending_targets is None:
            self.status.set("No target — click plane or change z/pitch")
            return
        if not (self.m1 and self.m2 and self.m3):
            self.status.set("Hardware not ready")
            return
        # Push the current targets to the queue
        self.cmd_q.put_nowait(("move", self._pending_targets))
        self.status.set("Queued move")

    async def _cmd_worker(self):
        """Serializes moves so you can send multiple in a row."""
        while True:
            cmd, payload = await self.cmd_q.get()
            if cmd == "move":
                t1, t2, t3 = payload
                try:
                    self.status.set(f"Moving to q1={t1:.3f}, q2={t2:.3f}, q3={t3:.3f} …")
                    await move_group_sync_time(self.m1, self.m2, self.m3, t1, t2, t3, duration=GROUP_DURATION_S)
                    self.status.set("Arrived — ready for next target")
                except Exception as e:
                    self.status.set(f"Move error: {e}")
            elif cmd == "stop":
                # Stop immediately; drain any pending moves
                with contextlib.suppress(Exception):
                    await asyncio.gather(self.m1.set_stop(), self.m2.set_stop(), self.m3.set_stop())
                # Clear the queue
                try:
                    while True:
                        self.cmd_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self.status.set("Motors stopped (queue cleared)")
            elif cmd == "quit":
                # Stop and exit loop
                with contextlib.suppress(Exception):
                    await asyncio.gather(self.m1.set_stop(), self.m2.set_stop(), self.m3.set_stop())
                self.status.set("Exiting…")
                break

    # ----- Immediate commands -----
    def stop_motors(self):
        if not (self.m1 and self.m2 and self.m3):
            self.status.set("Stop requested (hardware not ready yet)")
            return
        self.cmd_q.put_nowait(("stop", None))

    def quit(self):
        async def _quit_async():
            # Request worker to stop; then close window
            self.cmd_q.put_nowait(("quit", None))
            await asyncio.sleep(0)  # let the worker process the quit
            try:
                if self.m1 and self.m2 and self.m3:
                    with contextlib.suppress(Exception):
                        await asyncio.gather(
                            self.m1.set_stop(),
                            self.m2.set_stop(),
                            self.m3.set_stop()
                        )
            finally:
                self.root.after(0, self.root.destroy)
        asyncio.create_task(_quit_async())


# ========================= ENTRYPOINT =========================
async def _amain(root):
    app = ClickToMoveApp(root)
    try:
        while True:
            root.update_idletasks()
            root.update()
            await asyncio.sleep(0.01)
    except tk.TclError:
        pass

if __name__ == "__main__":
    root = tk.Tk()
    asyncio.run(_amain(root))
