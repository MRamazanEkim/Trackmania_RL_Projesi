"""
Trajectory Recorder
====================
Drive a lap manually while this script captures XYZ positions from the
tmrl environment.  The resulting CSV feeds into TrajectoryProcessor.

How it works
------------
  The TMRL_GrabData OpenPlanet plugin streams telemetry (including car
  position) to the Python client.  This script reads that stream and
  writes x, y, z to a CSV every physics tick.

  The script sends ZERO actions each tick (coast).  In most tmrl setups
  your keyboard / gamepad input is processed by the game independently,
  so you CAN drive normally while the script records.  If the car refuses
  to respond to your input, use the --replay flag approach described below.

Usage
-----
  # Basic recording (drive a full lap, press Ctrl-C when done):
  python record_trajectory.py

  # Custom output file and auto-process immediately after:
  python record_trajectory.py --output my_track.csv --process --spacing 1.0

  # Debug: print the raw obs array to identify position indices:
  python record_trajectory.py --debug-obs

  # See all options:
  python record_trajectory.py --help

After recording
---------------
  Either use --process flag above, or run separately:
    python progress_tracker.py raw_trajectory.csv reference_trajectory.csv --spacing 1.0

  Then integrate into the main monitor:
    python telemetry_monitor.py --trajectory reference_trajectory.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

console = Console()

# ── tmrl raw data[] layout (from tmrl source: tm_gym_interfaces.py) ──────────
#   data[0]  – speed (m/s)
#   data[2]  – x position (world coords, metres)
#   data[3]  – y position
#   data[4]  – z position
#   data[9]  – gear
#   data[10] – rpm
# Position is NOT in the obs tuple; we get it by patching the interface.
DATA_IDX_X = 2
DATA_IDX_Y = 3
DATA_IDX_Z = 4


# ══════════════════════════════════════════════════════════════════════════════
# Interface patch — exposes raw data[] outside the tmrl class
# ══════════════════════════════════════════════════════════════════════════════

def _patch_interface(env) -> object:
    """
    Monkey-patch the tmrl interface so it stores the last raw data array.

    After each step, read position via:
        iface._last_data[DATA_IDX_X], [DATA_IDX_Y], [DATA_IDX_Z]

    Returns the interface object (or None if not found).
    """
    iface = getattr(env.unwrapped, "interface", None)
    if iface is None:
        return None

    original_grab = iface.grab_data_and_img

    def _patched_grab():
        data, img = original_grab()
        iface._last_data = data       # store for external readers
        return data, img

    iface.grab_data_and_img = _patched_grab
    iface._last_data = None
    return iface


def _xyz_from_iface(iface) -> tuple[float, float, float]:
    """Read x,y,z from the patched interface's last captured data."""
    d = getattr(iface, "_last_data", None)
    if d is None:
        return 0.0, 0.0, 0.0
    try:
        return float(d[DATA_IDX_X]), float(d[DATA_IDX_Y]), float(d[DATA_IDX_Z])
    except (IndexError, TypeError):
        return 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def _render(
    points: int,
    x: float, y: float, z: float,
    fps: int,
    out_path: Path,
    lap_complete: bool = False,
) -> Panel:
    status = "[bold green]LAP COMPLETE — press Ctrl-C[/]" if lap_complete else "[yellow]Recording…  drive your lap[/]"
    lines = [
        "",
        f"  {status}",
        "",
        f"  [bold]Position[/]   X {x:>10.2f}   Y {y:>10.2f}   Z {z:>10.2f}",
        "",
        f"  [bold]Points recorded[/]   {points:,}",
        f"  [bold]FPS[/]               {fps:,}",
        "",
        "  " + "─" * 50,
        f"  [dim]Output → {out_path.name}[/]",
        f"  [dim]Press Ctrl-C to stop and save.[/]",
    ]
    return Panel(
        "\n".join(lines),
        title="[bold cyan]Trajectory Recorder[/]",
        box=box.ROUNDED,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Debug mode
# ══════════════════════════════════════════════════════════════════════════════

def debug_obs(steps: int = 20):
    """
    Verify the monkey-patch works and position is being captured.
    """
    import tmrl
    console.print("[yellow]Debug mode — verifying position capture via interface patch.[/]\n")
    env   = tmrl.get_environment()
    iface = _patch_interface(env)

    if iface is None:
        console.print("[red]Could not find tmrl interface — env.unwrapped has no 'interface' attr.[/]")
        env.close()
        return

    console.print(f"[green]Interface found:[/] {type(iface).__name__}")
    result = env.reset()

    for i in range(steps):
        result = env.step(np.zeros(3, dtype=np.float32))
        x, y, z = _xyz_from_iface(iface)
        obs0 = np.asarray(result[0][0], dtype=np.float32).ravel()[0] if isinstance(result[0], (tuple, list)) else 0.0
        console.print(
            f"step {i:3d}  speed={obs0:6.2f} m/s  "
            f"X={x:9.2f}  Y={y:7.2f}  Z={z:9.2f}"
        )
        time.sleep(0.05)

    env.close()
    console.print(
        "\n[green]If X/Y/Z change as the car moves, the patch is working.[/]\n"
        "If they stay 0.0, the interface class name may differ — report the "
        f"interface type above."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main recording loop
# ══════════════════════════════════════════════════════════════════════════════

def record(
    output:  Path,
    process: bool  = False,
    spacing: float = 1.0,
):
    """Connect to tmrl and record XYZ positions until Ctrl-C."""
    import tmrl
    from progress_tracker import TrajectoryProcessor

    console.print("[yellow]Connecting to Trackmania…[/]")
    env = tmrl.get_environment()
    console.print("[bold green]✓ Connected.[/]")

    # Patch interface to expose raw data[] (x,y,z live at data[2:5])
    iface = _patch_interface(env)
    if iface is None:
        console.print("[red]Warning: could not find tmrl interface — position will be 0,0,0.[/]")

    # Open output CSV immediately so we don't lose data on crash
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(output, "w", newline="", encoding="utf-8", buffering=1)
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "x", "y", "z"])

    console.print(f"[green]Recording to:[/] {output}")
    console.print("[dim]Drive your demonstration lap.  Press Ctrl-C when finished.[/]\n")

    result = env.reset()

    points         = 0
    tick_times: list[float] = []
    x = y = z = 0.0
    lap_complete   = False
    prev_pos: tuple[float, float, float] | None = None
    MIN_MOVE = 0.05   # metres — skip stationary/duplicate frames

    try:
        with Live(
            _render(0, 0.0, 0.0, 0.0, 0, output),
            refresh_per_second=20,
            console=console,
        ) as live:
            terminated = truncated = False
            while True:
                # re-reset whenever an episode ends (map restart, finish line, etc.)
                if terminated or truncated:
                    result     = env.reset()
                    terminated = truncated = False

                result = env.step(np.zeros(3, dtype=np.float32))

                if len(result) == 5:
                    _obs, _reward, terminated, truncated, _info = result
                else:
                    _obs, _reward, terminated, _info = result
                    truncated = False

                # get position from patched interface (data[2:5])
                x, y, z = _xyz_from_iface(iface)

                # skip frames where car hasn't moved (map loading / respawn)
                if prev_pos is not None:
                    dist = np.sqrt((x - prev_pos[0])**2 +
                                   (y - prev_pos[1])**2 +
                                   (z - prev_pos[2])**2)
                    if dist < MIN_MOVE:
                        continue
                prev_pos = (x, y, z)

                # write row
                writer.writerow([f"{time.time():.6f}", f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"])
                points += 1

                # rolling FPS
                now = time.time()
                tick_times.append(now)
                tick_times = [t for t in tick_times if now - t < 1.0]

                live.update(_render(points, x, y, z, len(tick_times), output, lap_complete))

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Recording stopped.[/]  {points:,} points saved.")
    finally:
        csv_file.close()
        env.close()

    if points == 0:
        console.print("[red]No points were recorded.  "
                      "Make sure the car is on track and OBS_IDX_X/Y/Z are correct.[/]")
        return

    console.print(f"[green]Raw trajectory saved:[/] {output}  ({points:,} points)")

    # Optional immediate discretisation
    if process:
        ref_path = output.parent / (output.stem + "_reference.csv")
        console.print(f"\n[yellow]Processing trajectory (spacing={spacing}m)…[/]")
        try:
            wp = TrajectoryProcessor.process(str(output), str(ref_path), spacing=spacing)
            console.print(f"[green]Reference trajectory ready:[/] {ref_path}")
            console.print(f"  Use with:  [bold]python telemetry_monitor.py "
                          f"--trajectory {ref_path}[/]")
        except Exception as exc:
            console.print(f"[red]Processing failed:[/] {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a reference lap trajectory for ProgressTracker."
    )
    p.add_argument(
        "--output", "-o",
        default=f"trajectory_logs/raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Output CSV path  (default: trajectory_logs/raw_<timestamp>.csv)",
    )
    p.add_argument(
        "--process", action="store_true",
        help="Automatically discretize the recording when done.",
    )
    p.add_argument(
        "--spacing", type=float, default=1.0,
        help="Waypoint spacing in metres when --process is used  (default: 1.0)",
    )
    p.add_argument(
        "--debug-obs", action="store_true",
        help="Print raw observation structure for 60 steps, then exit.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.debug_obs:
        debug_obs()
        sys.exit(0)

    record(
        output  = Path(args.output),
        process = args.process,
        spacing = args.spacing,
    )
