#!/usr/bin/python3 -B
# Requires: pip install moteus
# GUI uses built-in Tkinter (no extra install needed)

import asyncio
import contextlib
import math
import time
import moteus

# --- Tunables ---
ID1, ID2, ID3      = 1, 2, 3

# Per-motor rest positions
REST1               = 0.0
REST2               = 0.5
REST3               = 2.0

# Legacy stepper tunables (kept for wiggle / hold)
STEP_SIZE_RAD       = 0.08
STEP_DELAY_S        = 0.02
IDLE_SLEEP          = 0.02

# Default caps (used for simple holds)
ACCEL_LIMIT         = 5
VEL_LIMIT           = 10
WATCHDOG            = 10

# Group sync motion tunables
GROUP_SPEED_RAD_S   = 0.8   # target group speed (rad/s) used to compute shared duration
ACCEL_MULT          = 6.0   # accel_limit = ACCEL_MULT * per-motor velocity_limit
MIN_SPEED           = 0.05  # floor to avoid 0 velocity_limit
POS_TOL             = 0.01  # rad tolerance to consider "arrived"
SYNC_POLL_S         = 0.05  # seconds between arrival checks
SYNC_TIMEOUT_EXTRA  = 1.0   # extra slack beyond planned duration (s)

# Temperature policy (monitor motor 2 only)
TEMP_LIMIT          = 50.0
COOL_RESUME_TEMP    = 40.0

# Wiggle (motor 2) around REST2
WIGGLE_AMPLITUDE    = 0.5
WIGGLE_REPS         = 2
WIGGLE_DWELL_S      = 0.15

# GUI
GUI_UPDATE_SEC       = 0.10
ENABLE_GUI           = True
TEMP_LABEL_INTERVAL  = 1.0

# -------------------------------------------------------------------
# Helpers
def ang_err(a: float, b: float) -> float:
    """Shortest signed angular error (rad)."""
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
    """Small-step move (useful for wiggle)."""
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
    Move a set of motors so they arrive at the same time by scaling per-motor velocity_limit.

    items: list of (ctrl, current, target)
    Return: list of final positions in same order.
    """
    if not items:
        return []

    # Compute shortest deltas and the shared duration based on the farthest move
    deltas = [ang_err(t, c) for (_, c, t) in items]
    max_delta = max(abs(d) for d in deltas) if deltas else 0.0
    if max_delta < 1e-9:
        # Nothing to do, just hold at targets
        await asyncio.gather(*(hold(ctrl, tgt) for (ctrl, _, tgt) in items))
        return [t for (_, _, t) in items]

    # Shared duration from group speed
    group_speed = max(GROUP_SPEED_RAD_S, MIN_SPEED)
    duration = 1.5

    # Command final positions with per-motor velocity/accel limits so time ~= duration
    cmds = []
    for (ctrl, cur, tgt), d in zip(items, deltas):
        vel_lim = max(abs(d) / duration, MIN_SPEED)
        acc_lim = max(vel_lim * ACCEL_MULT, ACCEL_LIMIT)  # ensure accel not the limiter
        cmds.append(ctrl.set_position(
            position=float(tgt),
            velocity=float('nan'),
            accel_limit=acc_lim,
            velocity_limit=vel_lim,
            watchdog_timeout=WATCHDOG,
            query=False,
        ))
    await asyncio.gather(*cmds)

    # Wait until all are within tolerance, with a reasonable timeout
    t_end = time.monotonic() + duration + SYNC_TIMEOUT_EXTRA
    ctrls = [ctrl for (ctrl, _, _) in items]
    targets = [t for (_, _, t) in items]

    while True:
        # Timeout guard
        if time.monotonic() >= t_end:
            break

        # Query all and check position error
        try:
            states = await asyncio.gather(*(ctrl.query() for ctrl in ctrls))
            pos = [float(st.values.get('position', 0.0)) for st in states]
        except Exception:
            pos = [None] * len(ctrls)

        all_good = True
        for p, tgt in zip(pos, targets):
            if p is None:
                all_good = False
                break
            if abs(ang_err(tgt, p)) > POS_TOL:
                all_good = False
                break

        if all_good:
            break

        await asyncio.sleep(SYNC_POLL_S)

    # Final "hold" to lock exact targets (optional but tidy)
    await asyncio.gather(*(hold(ctrl, tgt) for ctrl, tgt in zip(ctrls, targets)))
    return targets

def extract_temperature(vals: dict):
    """Get FET temperature (°C) robustly from query values."""
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
    """Small wiggle (± amplitude) around center; return final pos."""
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

# -------------------------------------------------------------------
# Tkinter GUI
try:
    if ENABLE_GUI:
        import tkinter as tk
        from tkinter import ttk, messagebox
except Exception:
    ENABLE_GUI = False

_gui = None
_next_gui = 0.0

def _build_gui(cmd_queue: asyncio.Queue):
    root = tk.Tk()
    root.title("Arm Control (ID1, ID2, ID3)")
    root.geometry("440x300")
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="ID1 target (rad):").grid(row=0, column=0, sticky="w", padx=5, pady=6)
    e1 = ttk.Entry(frm, width=18); e1.grid(row=0, column=1, sticky="w", padx=5, pady=6)

    ttk.Label(frm, text="ID2 target (rad):").grid(row=1, column=0, sticky="w", padx=5, pady=6)
    e2 = ttk.Entry(frm, width=18); e2.grid(row=1, column=1, sticky="w", padx=5, pady=6)

    ttk.Label(frm, text="ID3 target (rad):").grid(row=2, column=0, sticky="w", padx=5, pady=6)
    e3 = ttk.Entry(frm, width=18); e3.grid(row=2, column=1, sticky="w", padx=5, pady=6)

    status = ttk.Label(frm, text="Status: OK")
    status.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=(8,4))

    btns = ttk.Frame(frm); btns.grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=8)

    def on_go():
        parts = []
        for ent in (e1, e2, e3):
            txt = ent.get().strip()
            if txt:
                try:
                    parts.append(float(txt))
                except ValueError:
                    messagebox.showerror("Invalid input", f"Not a number: {txt}")
                    return
        cmd_queue.put_nowait(("set", parts))

    def on_rest():
        cmd_queue.put_nowait(("rest", None))

    def on_quit():
        cmd_queue.put_nowait(("quit", None))

    ttk.Button(btns, text="Go", command=on_go).grid(row=0, column=0, padx=(0,10))
    ttk.Button(btns, text="Rest", command=on_rest).grid(row=0, column=1, padx=(0,10))
    ttk.Button(btns, text="Quit", command=on_quit).grid(row=0, column=2, padx=(0,10))

    temp_frame = ttk.LabelFrame(frm, text="Temperatures")
    temp_frame.grid(row=5, column=0, columnspan=2, sticky="we", padx=5, pady=(6,4))
    t1lbl = ttk.Label(temp_frame, text="ID1 Temp: --.- °C"); t1lbl.grid(row=0, column=0, sticky="w", padx=6, pady=4)
    t2lbl = ttk.Label(temp_frame, text="ID2 Temp: --.- °C"); t2lbl.grid(row=1, column=0, sticky="w", padx=6, pady=4)
    t3lbl = ttk.Label(temp_frame, text="ID3 Temp: --.- °C"); t3lbl.grid(row=2, column=0, sticky="w", padx=6, pady=4)

    rest_vals = ttk.Label(frm, text=f"Rest targets: [{REST1:.3f}, {REST2:.3f}, {REST3:.3f}] rad")
    rest_vals.grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=(6,2))

    return {"root": root, "status": status, "entries": (e1, e2, e3),
            "buttons": (), "temp_labels": (t1lbl, t2lbl, t3lbl),
            "all_buttons": btns.winfo_children()}

def _ensure_gui(cmd_queue: asyncio.Queue):
    global _gui
    if _gui is None and ENABLE_GUI:
        _gui = _build_gui(cmd_queue)

def _gui_set_enabled(enabled: bool):
    if _gui is None:
        return
    state = "normal" if enabled else "disabled"
    for w in _gui["entries"] + tuple(_gui["all_buttons"]):
        try: w.config(state=state)
        except Exception: pass
    _gui["status"].config(text="Status: OK" if enabled else "Status: COOLING")

def _gui_update_temps(temp1, temp2, temp3):
    if _gui is None: return
    for i, (lbl, v) in enumerate(zip(_gui["temp_labels"], (temp1, temp2, temp3)), start=1):
        lbl.config(text=f"ID{i} Temp: {v:.1f} °C" if isinstance(v,(int,float)) else f"ID{i} Temp: --.- °C")

def _gui_pulse():
    global _gui
    if _gui is None: return
    try:
        _gui["root"].update_idletasks(); _gui["root"].update()
    except Exception:
        try: _gui["root"].destroy()
        except Exception: pass
        _gui = None

# -------------------------------------------------------------------
# Main
async def main():
    global _next_gui

    m1 = moteus.Controller(id=ID1)
    m2 = moteus.Controller(id=ID2)
    m3 = moteus.Controller(id=ID3)

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

    if ENABLE_GUI:
        _ensure_gui(q)
        _next_gui = time.monotonic()

    cooling = False
    next_temp_tick = time.monotonic()
    last_temp2 = None

    try:
        while True:
            upd1 = upd2 = upd3 = False

            if not cooling:
                while not q.empty():
                    cmd, val = await q.get()
                    if cmd == "quit":
                        print("[quit] stop & exit")
                        await asyncio.gather(m1.set_stop(), m2.set_stop(), m3.set_stop())
                        return
                    if cmd == "rest":
                        t1, t2, t3 = REST1, REST2, REST3
                        upd1 = upd2 = upd3 = True
                    if cmd == "set":
                        parts = val
                        if len(parts) >= 1: t1 = float(parts[0]); upd1 = True
                        if len(parts) >= 2: t2 = float(parts[1]); upd2 = True
                        if len(parts) >= 3: t3 = float(parts[2]); upd3 = True
            else:
                while not q.empty():
                    await q.get()

            # --- Synchronized motion when there are updates ---
            group = []
            if upd1: group.append((m1, c1, t1))
            if upd2: group.append((m2, c2, t2))
            if upd3: group.append((m3, c3, t3))

            if group:
                finals = await move_group_sync_time(group)
                idx = 0
                if upd1: c1 = finals[idx]; idx += 1
                if upd2: c2 = finals[idx]; idx += 1
                if upd3: c3 = finals[idx]; idx += 1
            else:
                await asyncio.gather(hold(m1, t1), hold(m2, t2), hold(m3, t3))

            # 1 Hz temperature labels
            now = time.monotonic()
            if now >= next_temp_tick:
                try:
                    s1, s2, s3 = await asyncio.gather(m1.query(), m2.query(), m3.query())
                    temp1 = extract_temperature(s1.values)
                    temp2 = extract_temperature(s2.values)
                    temp3 = extract_temperature(s3.values)
                    last_temp2 = temp2
                except Exception:
                    temp1 = temp2 = temp3 = None
                _gui_update_temps(temp1, temp2, temp3)
                next_temp_tick = now + TEMP_LABEL_INTERVAL

            # Thermal policy
            if last_temp2 is not None:
                if (not cooling) and (last_temp2 > TEMP_LIMIT):
                    print(f"[warn] m2 > {TEMP_LIMIT:.1f} °C — returning ALL to per-motor rest (synchronized) and entering cooldown.")
                    finals = await move_group_sync_time([
                        (m1, c1, REST1),
                        (m2, c2, REST2),
                        (m3, c3, REST3),
                    ])
                    c1, c2, c3 = finals
                    t1, t2, t3 = REST1, REST2, REST3
                    cooling = True
                    _gui_set_enabled(False)
                elif cooling and last_temp2 <= COOL_RESUME_TEMP:
                    print(f"[cool] m2 <= {COOL_RESUME_TEMP:.1f} °C — wiggle m2 around REST2 then resume input.")
                    c2 = await wiggle(m2, REST2, c2)
                    t2 = REST2
                    cooling = False
                    _gui_set_enabled(True)

            # GUI pulse
            if ENABLE_GUI and _gui is not None and now >= _next_gui:
                _gui_pulse()
                _next_gui = now + GUI_UPDATE_SEC

            await asyncio.sleep(IDLE_SLEEP)

    except KeyboardInterrupt:
        print("[quit] KeyboardInterrupt -> stop")
        with contextlib.suppress(Exception):
            await asyncio.gather(m1.set_stop(), m2.set_stop(), m3.set_stop())
    finally:
        try:
            if _gui is not None and _gui.get("root"):
                _gui["root"].destroy()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            await asyncio.gather(m1.set_stop(), m2.set_stop(), m3.set_stop())

if __name__ == '__main__':
    asyncio.run(main())
