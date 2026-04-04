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
from typing import Callable, Optional, Tuple

import numpy as np
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Import AI driving logic components
try:
    from ai_driving_logic import (
        DrivingController,
        AdaptiveRewardSystem,
        FailureInfo,
    )
    AI_DRIVING_AVAILABLE = True
except ImportError:
    AI_DRIVING_AVAILABLE = False

# ── tmrl raw data[] layout ────────────────────────────────────────────────────
# Position is NOT in the obs tuple; it lives in the raw data[] array that
# tmrl's interface reads from the OpenPlanet socket.  We monkey-patch the
# interface to store the last data[] so we can read x, y, z from it.
# Indices confirmed from tmrl/custom/tm/tm_gym_interfaces.py:
DATA_IDX_X: int = 2
DATA_IDX_Y: int = 3
DATA_IDX_Z: int = 4


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
    x: float  = 0.0   # world-space position (metres)
    y: float  = 0.0
    z: float  = 0.0
    reward: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Reward hook  ← customise this
# ══════════════════════════════════════════════════════════════════════════════

def calculate_reward(frame: TelemetryFrame) -> float:
    """
    Fallback speed-based reward — used only when no ProgressTracker is active.

    When a reference trajectory is loaded (--trajectory flag), the main loop
    calls ProgressTracker.update(x, y, z) instead and ignores this function.
    """
    speed_reward  =  frame.speed_kmh / 300.0
    steering_cost = -abs(frame.steering) * 0.10
    brake_cost    = -frame.brake         * 0.20
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
        log_path:   Optional[Path] = None,
        log_frames: int = 0,
        tracker=None,                  # Optional[ProgressTracker]
        adaptive_reward=None,          # Optional[AdaptiveRewardSystem]
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
        ]

        # ── progress block (only shown when a trajectory is loaded) ──────
        if tracker is not None:
            pct      = tracker.progress_pct
            bar_done = int(pct / 100.0 * 28)
            bar      = "█" * bar_done + "░" * (28 - bar_done)
            lap_tag  = "  [bold green]LAP COMPLETE[/]" if tracker.lap_complete else ""
            lines += [
                "",
                f"  [bold magenta]PROGRESS[/]   [{bar}]  {pct:5.1f}%{lap_tag}",
                f"  [dim]waypoint {tracker.furthest_idx:,} / {tracker.total_waypoints:,}"
                f"   Σreward {tracker.cumulative_reward:,.1f}[/]",
                f"  [dim]pos  X {f.x:8.1f}  Y {f.y:8.1f}  Z {f.z:8.1f}[/]",
            ]

        # ── adaptive reward info (if AI logic enabled) ───────────────────
        if adaptive_reward is not None:
            pb = adaptive_reward.personal_best
            threshold = adaptive_reward.current_threshold
            pb_color = "green" if pct > pb else "yellow"
            lines += [
                "",
                f"  [bold blue]AI TARGETS[/]",
                f"  [dim]Personal Best:[/] [{pb_color}]{pb:5.1f}%[/]  "
                f"[dim]Threshold:[/] [cyan]{threshold:5.1f}%[/]",
            ]

        lines += [
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
        self._iface        = None   # patched tmrl interface (for raw data[])
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
                # Patch interface to expose raw data[2,3,4] = x,y,z
                self._iface = getattr(self._env.unwrapped, "interface", None)
                if self._iface is not None:
                    _orig = self._iface.grab_data_and_img
                    def _patched():
                        data, img = _orig()
                        self._iface._last_data = data
                        return data, img
                    self._iface.grab_data_and_img = _patched
                    self._iface._last_data = None
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
        obs,
        action: Optional[np.ndarray] = None,
    ) -> TelemetryFrame:
        """
        Build a TelemetryFrame from a raw obs + the action that was sent.

        tmrl 0.7+ returns obs as a tuple of arrays, e.g.:
            obs[0] = np.array([speed_ms, gear, rpm, ...])   ← scalar telemetry
            obs[1] = np.array([...])                         ← LIDAR / images
        Older versions return a single flat np.ndarray.
        Edit the index mapping here if your tmrl config differs.
        """
        if action is None:
            action = self._last_action

        # Flatten tuple obs → take the first array (scalar telemetry)
        if isinstance(obs, (tuple, list)):
            flat = np.asarray(obs[0], dtype=np.float32).ravel()
        else:
            flat = np.asarray(obs, dtype=np.float32).ravel()

        speed_ms = float(flat[0]) if len(flat) > 0 else 0.0
        gear     = int(flat[1])   if len(flat) > 1 else 0
        rpm      = float(flat[2]) if len(flat) > 2 else 0.0

        # world-space position — read from patched interface (data[2:5])
        # position is NOT in the obs tuple; tmrl uses it internally for reward only
        d = getattr(self._iface, "_last_data", None) if self._iface else None
        try:
            x = float(d[DATA_IDX_X]) if d is not None else 0.0
            y = float(d[DATA_IDX_Y]) if d is not None else 0.0
            z = float(d[DATA_IDX_Z]) if d is not None else 0.0
        except (IndexError, TypeError):
            x = y = z = 0.0

        action = np.asarray(action, dtype=np.float32).ravel()
        steering = float(np.clip(action[0], -1.0,  1.0)) if len(action) > 0 else 0.0
        throttle = float(np.clip(action[1],  0.0,  1.0)) if len(action) > 1 else 0.0
        brake    = float(np.clip(action[2],  0.0,  1.0)) if len(action) > 2 else 0.0

        frame = TelemetryFrame(
            timestamp = time.time(),
            speed_kmh = speed_ms * 3.6,
            gear      = gear,
            rpm       = rpm,
            steering  = steering,
            throttle  = throttle,
            brake     = brake,
            x         = x,
            y         = y,
            z         = z,
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
    log_format:      str                              = "csv",
    output_dir:      str                              = "telemetry_logs",
    reward_fn:       Callable[[TelemetryFrame], float] = calculate_reward,
    max_episodes:    Optional[int]                    = None,
    trajectory_path: Optional[str]                    = None,
    wp_spacing:      float                            = 1.0,
    enable_ai_logic: bool                             = True,
    history_file:    Optional[str]                    = None,
):
    """
    Connect, log, and display telemetry until Ctrl-C or max_episodes is reached.

    Args:
        trajectory_path: Path to a reference waypoints CSV (x,y,z columns).
                         When provided, reward comes from ProgressTracker instead
                         of reward_fn.  Generate one with record_trajectory.py.
        wp_spacing:      Waypoint spacing used if trajectory needs pre-processing.
        enable_ai_logic: Enable AI driving logic (failure detection + adaptive rewards).
        history_file:    Path to save episode history for adaptive reward system.
    """
    # ── optional progress tracker ────────────────────────────────────────
    tracker = None
    if trajectory_path:
        from progress_tracker import ProgressTracker, TrajectoryProcessor
        path = Path(trajectory_path)
        # auto-process raw recording if the file has a 'timestamp' column
        with open(path, newline="", encoding="utf-8") as fh:
            header = fh.readline()
        if "timestamp" in header and "waypoint" not in path.stem:
            processed_path = path.parent / (path.stem + "_reference.csv")
            console.print(f"[yellow]Pre-processing raw trajectory → {processed_path}[/]")
            TrajectoryProcessor.process(str(path), str(processed_path), spacing=wp_spacing)
            path = processed_path
        tracker = ProgressTracker(path)
        console.print(
            f"[green]ProgressTracker loaded:[/] {tracker.total_waypoints:,} waypoints  "
            f"({wp_spacing}m spacing)"
        )
        # replace reward_fn with the tracker's signal
        def reward_fn(frame: TelemetryFrame) -> float:  # noqa: F811
            return tracker.update(frame.x, frame.y, frame.z)

    # ── AI Driving Logic (failure detection + adaptive rewards) ──────────
    driving_controller = None
    adaptive_reward = None

    if enable_ai_logic and AI_DRIVING_AVAILABLE:
        driving_controller = DrivingController(
            zero_speed_threshold=5.0,    # km/h - bunun altinda "durmus" sayilir
            zero_speed_timeout=3.0,      # saniye - bu kadar duruk kalirsa fail
            reverse_threshold=-0.3,       # dot product esigi (geri gitme)
            stuck_timeout=10.0,          # saniye - ilerleme yoksa fail
        )
        adaptive_reward = AdaptiveRewardSystem(
            history_file=history_file or "episode_history.json",
            milestone_interval=50,       # Her 50 waypoint'te bonus
            milestone_bonus=5.0,
            pb_bonus_multiplier=2.0,     # Personal best asildiginda 2x bonus
        )
        console.print(
            f"[green]AI Driving Logic enabled:[/] "
            f"PB={adaptive_reward.personal_best:.1f}%, "
            f"Threshold={adaptive_reward.current_threshold:.1f}%"
        )
    elif enable_ai_logic and not AI_DRIVING_AVAILABLE:
        console.print("[yellow]Warning: ai_driving_logic module not found. Running without AI logic.[/]")

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

                if tracker is not None:
                    tracker.reset()

                # Reset AI driving components
                if driving_controller is not None:
                    driving_controller.reset()
                if adaptive_reward is not None:
                    adaptive_reward.start_episode()

                episode    += 1
                terminated  = truncated = False
                action      = np.zeros(3, dtype=np.float32)
                failure_reason = ""
                episode_reward = 0.0

                console.print(f"\n[cyan]Episode {episode} started[/]")

                while not (terminated or truncated):
                    obs, _env_reward, terminated, truncated, _info = interface.step(action)
                    if obs is None:
                        break

                    frame = interface.parse_observation(obs, action)

                    # ── AI Driving Logic: Failure Detection ──────────────
                    if driving_controller is not None:
                        progress = tracker.progress_pct if tracker else 0.0
                        failure_info = driving_controller.check_failure(
                            speed_kmh=frame.speed_kmh,
                            position=np.array([frame.x, frame.y, frame.z]),
                            progress=progress,
                        )

                        if failure_info.is_failed:
                            terminated = True
                            failure_reason = failure_info.reason
                            console.print(
                                f"[red]Episode failed: {failure_reason}[/] "
                                f"[dim]({failure_info.details})[/]"
                            )

                    # ── AI Driving Logic: Adaptive Reward ────────────────
                    if adaptive_reward is not None and tracker is not None:
                        progress = tracker.progress_pct
                        base_reward = frame.reward
                        total_reward, breakdown = adaptive_reward.calculate_reward(
                            progress=progress,
                            base_reward=base_reward,
                            is_done=terminated or truncated,
                            failure_reason=failure_reason,
                        )
                        frame.reward = total_reward
                        episode_reward += total_reward

                    logger.log(frame)
                    dashboard.update(frame)
                    live.update(dashboard.render(
                        log_path, logger.frame_count, tracker,
                        adaptive_reward=adaptive_reward
                    ))

                    # ╔══════════════════════════════════════════════════╗
                    # ║  PLUG YOUR POLICY / AGENT HERE                  ║
                    # ║  action = my_agent.act(obs)                     ║
                    # ╚══════════════════════════════════════════════════╝
                    action = np.zeros(3, dtype=np.float32)  # coast (no-op)

                # ── Episode End ──────────────────────────────────────────
                if adaptive_reward is not None:
                    max_progress = tracker.progress_pct if tracker else 0.0
                    result = adaptive_reward.end_episode(
                        max_progress=max_progress,
                        total_reward=episode_reward,
                        failure_reason=failure_reason,
                    )

                    # Episode summary
                    if result.get("pb_broken"):
                        console.print(
                            f"[bold green]NEW PERSONAL BEST![/] "
                            f"{result['personal_best']:.1f}%"
                        )
                    console.print(
                        f"[dim]Episode {episode} complete: "
                        f"Progress={result.get('max_progress', 0):.1f}%, "
                        f"Reward={result.get('total_reward', 0):.1f}, "
                        f"Duration={result.get('duration', 0):.1f}s[/]"
                    )

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user.[/]")
    except Exception as exc:
        console.print(f"\n[red]Fatal error:[/] {exc}")
        raise
    finally:
        logger.stop()
        interface.close()

        # Final summary
        if adaptive_reward is not None:
            console.print("\n" + "=" * 50)
            console.print("[bold cyan]Session Summary[/]")
            console.print(f"  Episodes completed: {episode}")
            console.print(f"  Personal Best: {adaptive_reward.personal_best:.1f}%")
            console.print(f"  Current Threshold: {adaptive_reward.current_threshold:.1f}%")
            stats = adaptive_reward.recent_stats
            if stats:
                console.print(f"  Avg Progress (recent): {stats['avg_progress']:.1f}%")
            console.print("=" * 50)

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
    p.add_argument(
        "--trajectory", default=None, metavar="CSV",
        help="Path to reference trajectory CSV for progress-based reward. "
             "Generate one with: python record_trajectory.py",
    )
    p.add_argument(
        "--spacing", type=float, default=1.0,
        help="Waypoint spacing in metres when auto-processing a raw trajectory (default: 1.0)",
    )
    p.add_argument(
        "--no-ai-logic", action="store_true",
        help="Disable AI driving logic (failure detection + adaptive rewards)",
    )
    p.add_argument(
        "--history-file", default=None, metavar="JSON",
        help="Path to episode history file for adaptive rewards (default: episode_history.json)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        log_format      = args.fmt,
        output_dir      = args.output_dir,
        max_episodes    = args.episodes,
        trajectory_path = args.trajectory,
        wp_spacing      = args.spacing,
        enable_ai_logic = not args.no_ai_logic,
        history_file    = args.history_file,
    )
