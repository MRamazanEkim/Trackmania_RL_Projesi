"""
Trackmania Telemetry Monitor
============================
Real-time telemetry logging and dashboard for TM2020/TMNF via tmrl.

Architecture
------------
  TrackmaniaInterface  – wraps tmrl env, parses observations into TelemetryFrame
  TelemetryLogger      – writes CSV or JSONL at physics-tick frequency
  TelemetryDashboard   – rich Live console panel, refreshed ~20 Hz
  calculate_reward()   – pluggable reward/penalty hook

Usage
-----
  python telemetry_monitor.py                 # defaults: CSV logging, watch-only
  python telemetry_monitor.py --format jsonl  # JSONL instead
  python telemetry_monitor.py --episodes 5    # stop after 5 episodes
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TelemetryFrame:
    """One physics-tick snapshot of game state + computed reward."""
    timestamp: float   # Unix epoch seconds
    speed_kmh: float   # km/h
    gear: int          # 0 = Reverse, 1–5 = forward gears
    rpm: float
    steering: float    # –1.0 (full left) … +1.0 (full right)
    throttle: float    # 0.0 … 1.0
    brake: float       # 0.0 … 1.0
    reward: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Reward hook  ← customise this
# ══════════════════════════════════════════════════════════════════════════════

def calculate_reward(frame: TelemetryFrame) -> float:
    """
    Placeholder reward function.  Replace the body with your own logic.

    Ideas
    -----
    - Reward forward speed:          speed_kmh / 300
    - Penalise excessive steering:   –abs(steering) * k
    - Penalise hard braking:         –brake * k
    - Add track-progress signal from info dict in TrackmaniaInterface.step()
    """
    speed_reward    =  frame.speed_kmh / 300.0        # normalised to [0, 1]
    steering_cost   = -abs(frame.steering) * 0.10
    brake_cost      = -frame.brake         * 0.20
    return speed_reward + steering_cost + brake_cost


# ══════════════════════════════════════════════════════════════════════════════
# Logger
# ══════════════════════════════════════════════════════════════════════════════

class TelemetryLogger:
    """
    Thread-safe logger.  Writes one row per TelemetryFrame to a CSV or JSONL
    file with line-buffered I/O so data survives a crash mid-session.
    """

    def __init__(self, output_dir: str = "telemetry_logs", fmt: str = "csv"):
        self._dir        = Path(output_dir)
        self._fmt        = fmt.lower()
        self._file       = None
        self._writer     = None          # used only for CSV
        self._lock       = threading.Lock()
        self._count      = 0
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path: Optional[Path] = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"session_{self._session_id}.{self._fmt}"

        if self._fmt == "csv":
            self._file = open(self._path, "w", newline="", buffering=1, encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(TelemetryFrame.__dataclass_fields__)
            )
            self._writer.writeheader()
        elif self._fmt == "jsonl":
            self._file = open(self._path, "w", buffering=1, encoding="utf-8")
        else:
            raise ValueError(f"Unknown log format: {self._fmt!r}. Use 'csv' or 'jsonl'.")

        return self._path

    def stop(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None

    # ── I/O ───────────────────────────────────────────────────────────────

    def log(self, frame: TelemetryFrame):
        with self._lock:
            if not self._file:
                return
            if self._fmt == "csv":
                self._writer.writerow(asdict(frame))
            else:
                self._file.write(json.dumps(asdict(frame)) + "\n")
            self._count += 1

    # ── properties ────────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return self._count

    @property
    def path(self) -> Optional[Path]:
        return self._path


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

class TelemetryDashboard:
    """
    Builds a Rich Panel that `rich.Live` re-renders at ~20 Hz.
    All state is updated in the main loop; render() is pure / side-effect-free.
    """

    _GEAR_LABELS = {0: " R ", 1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}

    def __init__(self):
        self._frame:       Optional[TelemetryFrame] = None
        self._total:       int   = 0
        self._start:       float = time.time()
        self._tick_times:  list  = []   # for rolling FPS

    def update(self, frame: TelemetryFrame):
        now = time.time()
        self._tick_times.append(now)
        self._tick_times = [t for t in self._tick_times if now - t < 1.0]
        self._frame = frame
        self._total += 1

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _hbar(ratio: float, width: int = 28) -> str:
        filled = int(max(0.0, min(1.0, ratio)) * width)
        return "█" * filled + "░" * (width - filled)

    def _speed_text(self, speed: float, max_speed: float = 300.0) -> Text:
        ratio = speed / max_speed
        bar   = self._hbar(ratio)
        color = "bright_green" if ratio < 0.5 else ("yellow" if ratio < 0.8 else "bright_red")
        t = Text()
        t.append(f"[{bar}]", style=color)
        t.append(f"  {speed:6.1f} km/h", style="bold " + color)
        return t

    @staticmethod
    def _steering_ruler(steering: float, half: int = 20) -> str:
        """Draws a ruler with a pointer for steering position."""
        ruler   = ["─"] * (half * 2 + 1)
        ruler[half] = "│"          # centre mark
        pos = half + int(round(steering * half))
        pos = max(0, min(len(ruler) - 1, pos))
        ruler[pos] = "▼"
        return "".join(ruler)

    # ── render ────────────────────────────────────────────────────────────

    def render(
        self,
        log_path: Optional[Path] = None,
        log_frames: int = 0,
    ) -> Panel:
        if self._frame is None:
            return Panel(
                "\n  [yellow]Waiting for telemetry…  "
                "Is Trackmania running with the OpenPlanet plugin?[/]",
                title="[bold cyan]Trackmania Telemetry Monitor[/]",
                box=box.ROUNDED,
            )

        f       = self._frame
        fps     = len(self._tick_times)
        elapsed = time.time() - self._start
        gear    = self._GEAR_LABELS.get(f.gear, str(f.gear))

        reward_color = "bright_green" if f.reward >= 0 else "bright_red"

        lines = [
            "",
            f"  [bold cyan]SPEED[/]      {self._speed_text(f.speed_kmh)}",
            "",
            f"  [bold]GEAR[/]       [bold yellow]{gear}[/]"
            f"          [bold]RPM[/]  {f.rpm:>7,.0f}",
            "",
            f"  [bold]THROTTLE[/]   [{self._hbar(f.throttle)}]  {f.throttle:4.2f}",
            f"  [bold]BRAKE[/]      [{self._hbar(f.brake, 28)}]  {f.brake:4.2f}",
            "",
            f"  [bold]STEERING[/]   L {self._steering_ruler(f.steering)} R",
            f"             {'LEFT' if f.steering < -0.05 else ('RIGHT' if f.steering > 0.05 else 'STRAIGHT'):^41}",
            "",
            f"  [bold]REWARD[/]     [{reward_color}]{f.reward:+.4f}[/]",
            "",
            "  " + "─" * 55,
            f"  [dim]FPS {fps:3d}  │  Frames {self._total:6d}  │  Elapsed {elapsed:6.1f}s[/]",
        ]

        if log_path:
            lines.append(
                f"  [dim]● Logging → [italic]{log_path.name}[/italic]"
                f"  ({log_frames:,} rows)[/]"
            )

        return Panel(
            "\n".join(str(l) if not isinstance(l, str) else l for l in lines),
            title="[bold bright_cyan]Trackmania Telemetry Monitor[/]",
            box=box.ROUNDED,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Environment interface
# ══════════════════════════════════════════════════════════════════════════════

class TrackmaniaInterface:
    """
    Thin wrapper around tmrl's gym-like environment.

    Default TM2020 observation layout (tmrl >= 0.6)
    ------------------------------------------------
      obs[0]   speed   (m/s)
      obs[1]   gear    (0 = reverse, 1–5 = forward)
      obs[2]   rpm
      obs[3:]  LIDAR rays or image pixels (ignored here)

    Action space: [steering, gas, brake]  each in their respective ranges.

    If your tmrl config produces a different layout, adjust _parse_obs() below.
    """

    MAX_RECONNECT_TRIES = 3

    def __init__(self, reward_fn: Callable[[TelemetryFrame], float] = calculate_reward):
        self._env          = None
        self._connected    = False
        self._reward_fn    = reward_fn
        self._last_action  = np.zeros(3, dtype=np.float32)

    # ── connection ────────────────────────────────────────────────────────

    def connect(self) -> bool:
        for attempt in range(1, self.MAX_RECONNECT_TRIES + 1):
            try:
                import tmrl  # noqa: PLC0415 (lazy import to give clear error)
                console.print(
                    f"[yellow]Connecting to Trackmania via tmrl "
                    f"(attempt {attempt}/{self.MAX_RECONNECT_TRIES})…[/]"
                )
                self._env = tmrl.get_environment()
                self._connected = True
                console.print("[bold green]✓ Connected.[/]")
                return True
            except ImportError:
                console.print(
                    "[red]tmrl is not installed.[/]  "
                    "Run: [bold]pip install tmrl[/]"
                )
                return False
            except ConnectionRefusedError:
                console.print(
                    f"[red]Connection refused (attempt {attempt}).[/]  "
                    "Ensure Trackmania is running and the OpenPlanet TMRL plugin is active."
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Unexpected error: {exc}[/]")

            if attempt < self.MAX_RECONNECT_TRIES:
                time.sleep(2.0)

        return False

    # ── episode control ───────────────────────────────────────────────────

    def reset(self) -> Optional[np.ndarray]:
        if not self._connected:
            return None
        try:
            result = self._env.reset()
            # gym ≥ 0.26 returns (obs, info); older returns obs directly
            obs = result[0] if isinstance(result, tuple) else result
            self._last_action = np.zeros(3, dtype=np.float32)
            return obs
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]reset() error: {exc}[/]")
            return None

    def step(
        self, action: Optional[np.ndarray] = None
    ) -> tuple[Optional[np.ndarray], float, bool, bool, dict]:
        """
        Step the environment.  Pass action=None to coast (all zeros).
        Returns (obs, env_reward, terminated, truncated, info).
        """
        if not self._connected:
            return None, 0.0, True, False, {}

        if action is None:
            action = np.zeros(3, dtype=np.float32)

        self._last_action = action

        try:
            result = self._env.step(action)
            # unpack regardless of whether env returns 4 or 5 values
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, done, info = result
                terminated, truncated = done, False
            return obs, float(reward), terminated, truncated, info
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]step() error: {exc}[/]")
            return None, 0.0, True, False, {}

    # ── telemetry extraction ──────────────────────────────────────────────

    def parse_observation(
        self,
        obs: np.ndarray,
        action: Optional[np.ndarray] = None,
    ) -> TelemetryFrame:
        """
        Build a TelemetryFrame from a raw obs array + the action that was sent.
        Edit the index mapping here if your tmrl config differs.
        """
        if action is None:
            action = self._last_action

        speed_ms  = float(obs[0]) if obs.shape[0] > 0 else 0.0
        gear      = int(obs[1])   if obs.shape[0] > 1 else 0
        rpm       = float(obs[2]) if obs.shape[0] > 2 else 0.0

        steering  = float(np.clip(action[0], -1.0,  1.0)) if action.shape[0] > 0 else 0.0
        throttle  = float(np.clip(action[1],  0.0,  1.0)) if action.shape[0] > 1 else 0.0
        brake     = float(np.clip(action[2],  0.0,  1.0)) if action.shape[0] > 2 else 0.0

        frame = TelemetryFrame(
            timestamp = time.time(),
            speed_kmh = speed_ms * 3.6,
            gear      = gear,
            rpm       = rpm,
            steering  = steering,
            throttle  = throttle,
            brake     = brake,
        )
        frame.reward = self._reward_fn(frame)
        return frame

    # ── cleanup ───────────────────────────────────────────────────────────

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
        self._connected = False


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def run(
    log_format:   str                             = "csv",
    output_dir:   str                             = "telemetry_logs",
    reward_fn:    Callable[[TelemetryFrame], float] = calculate_reward,
    max_episodes: Optional[int]                   = None,
):
    """
    Connect, log, and display telemetry until Ctrl-C or max_episodes is reached.

    To plug in a trained policy, replace the `action = np.zeros(...)` line with
    your agent's action selection, e.g.:
        action = my_agent.act(obs)
    """
    interface = TrackmaniaInterface(reward_fn=reward_fn)
    logger    = TelemetryLogger(output_dir=output_dir, fmt=log_format)
    dashboard = TelemetryDashboard()

    if not interface.connect():
        sys.exit(1)

    log_path = logger.start()
    console.print(f"[green]Logging to:[/] {log_path}")
    console.print("[dim]Press Ctrl-C to stop.\n[/]")

    episode = 0
    try:
        with Live(
            dashboard.render(),
            refresh_per_second=20,
            screen=False,
            console=console,
        ) as live:
            while max_episodes is None or episode < max_episodes:
                obs = interface.reset()
                if obs is None:
                    console.print("[red]Could not reset environment.  Exiting.[/]")
                    break

                episode  += 1
                terminated = truncated = False

                # ── neutral starting action ──────────────────────────────
                action = np.zeros(3, dtype=np.float32)

                while not (terminated or truncated):
                    obs, _env_reward, terminated, truncated, _info = interface.step(action)
                    if obs is None:
                        break

                    frame = interface.parse_observation(obs, action)
                    logger.log(frame)
                    dashboard.update(frame)
                    live.update(dashboard.render(log_path, logger.frame_count))

                    # ╔══════════════════════════════════════════════════╗
                    # ║  PLUG YOUR POLICY / AGENT HERE                  ║
                    # ║  action = my_agent.act(obs)                     ║
                    # ╚══════════════════════════════════════════════════╝
                    action = np.zeros(3, dtype=np.float32)  # coast (no-op)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user.[/]")
    except Exception as exc:
        console.print(f"\n[red]Fatal error:[/] {exc}")
        raise
    finally:
        logger.stop()
        interface.close()
        console.print(
            f"[bold green]Session complete.[/]  "
            f"{logger.frame_count:,} frames logged → {log_path}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trackmania real-time telemetry monitor (tmrl-backed)"
    )
    p.add_argument(
        "--format", dest="fmt", choices=["csv", "jsonl"], default="csv",
        help="Log file format (default: csv)",
    )
    p.add_argument(
        "--output-dir", default="telemetry_logs",
        help="Directory for log files (default: telemetry_logs/)",
    )
    p.add_argument(
        "--episodes", type=int, default=None,
        help="Stop after N episodes (default: run indefinitely)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        log_format   = args.fmt,
        output_dir   = args.output_dir,
        max_episodes = args.episodes,
    )
