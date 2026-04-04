"""
Progress-Based Reward System
=============================
Implements the 'Clever Reward' paradigm: reward the agent for passing
equidistant waypoints on a pre-recorded reference trajectory.

Classes
-------
  TrajectoryProcessor   – load raw CSVs, discretize via linear interpolation, save
  ProgressTracker       – KDTree nearest-neighbour search + anti-backwards logic

Typical workflow
----------------
  # 1. Record a lap with record_trajectory.py  →  raw_trajectory.csv
  # 2. Pre-process once:
  from progress_tracker import TrajectoryProcessor, ProgressTracker
  TrajectoryProcessor.process("raw_trajectory.csv", "reference_trajectory.csv", spacing=1.0)

  # 3. Use every episode:
  tracker = ProgressTracker("reference_trajectory.csv")
  tracker.reset()
  reward = tracker.update(x, y, z)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import KDTree


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory pre-processing
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryProcessor:
    """
    Utilities for loading, discretizing, and saving reference trajectory CSVs.

    Input CSV format (raw recording)
    ---------------------------------
      timestamp, x, y, z   (extra columns are ignored)

    Output CSV format (discretized waypoints)
    ------------------------------------------
      x, y, z              (one row per equidistant waypoint)
    """

    # ── I/O ──────────────────────────────────────────────────────────────

    @staticmethod
    def load_raw(path: str | Path) -> np.ndarray:
        """
        Read x, y, z columns from a CSV file.

        Returns
        -------
        (N, 3) float32 array of [x, y, z] positions.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        points: list[list[float]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not {"x", "y", "z"}.issubset(reader.fieldnames):
                raise ValueError(
                    f"{path} must contain columns 'x', 'y', 'z'.  "
                    f"Found: {reader.fieldnames}"
                )
            for row in reader:
                try:
                    x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
                    # skip degenerate zero-origin frames (car not yet spawned)
                    if x == 0.0 and y == 0.0 and z == 0.0:
                        continue
                    points.append([x, y, z])
                except (ValueError, KeyError):
                    continue

        if len(points) < 2:
            raise ValueError(
                f"Need at least 2 valid points; got {len(points)} from {path}."
            )

        return np.array(points, dtype=np.float32)

    @staticmethod
    def save(points: np.ndarray, path: str | Path):
        """Write an (N, 3) waypoint array to CSV with header [x, y, z]."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x", "y", "z"])
            writer.writerows(points.tolist())

    # ── core algorithm ────────────────────────────────────────────────────

    @staticmethod
    def discretize(points: np.ndarray, spacing: float = 1.0) -> np.ndarray:
        """
        Resample a 3-D polyline to equidistant waypoints via linear interpolation.

        This makes the reward signal uniform: each waypoint represents exactly
        `spacing` metres of forward progress regardless of where the car was
        slow or fast during the demonstration lap.

        Args:
            points:  (N, 3) raw x,y,z positions.
            spacing: target distance between consecutive waypoints in metres.

        Returns:
            (M, 3) resampled waypoints.
        """
        if len(points) < 2:
            return points.copy()

        # cumulative arc-length along the raw path
        seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cum_len     = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_len   = cum_len[-1]

        if total_len < spacing:
            raise ValueError(
                f"Track total length ({total_len:.1f} m) is shorter than "
                f"requested spacing ({spacing} m)."
            )

        # positions at which to sample
        sample_positions = np.arange(0.0, total_len, spacing)

        # interpolate each spatial axis independently
        resampled = np.stack(
            [np.interp(sample_positions, cum_len, points[:, i]) for i in range(3)],
            axis=1,
        ).astype(np.float32)

        return resampled

    # ── convenience pipeline ──────────────────────────────────────────────

    @classmethod
    def process(
        cls,
        raw_csv:  str | Path,
        out_csv:  str | Path,
        spacing:  float = 1.0,
        verbose:  bool  = True,
    ) -> np.ndarray:
        """
        Full pipeline: load raw CSV → discretize → save.

        Returns the (M, 3) waypoint array for immediate use.
        """
        raw       = cls.load_raw(raw_csv)
        waypoints = cls.discretize(raw, spacing)
        cls.save(waypoints, out_csv)

        if verbose:
            seg_lens   = np.linalg.norm(np.diff(raw, axis=0), axis=1)
            total_dist = float(seg_lens.sum())
            print(
                f"Trajectory processed:\n"
                f"  raw points : {len(raw):,}\n"
                f"  waypoints  : {len(waypoints):,}\n"
                f"  track len  : {total_dist:.1f} m\n"
                f"  spacing    : {spacing} m\n"
                f"  saved to   : {Path(out_csv).resolve()}"
            )

        return waypoints

    @classmethod
    def from_tmrl_pkl(
        cls,
        pkl_path: str | Path,
        out_csv:  str | Path,
        spacing:  float = 1.0,
        verbose:  bool  = True,
    ) -> np.ndarray:
        """
        Import the trajectory that tmrl already uses for its own reward function.

        tmrl stores the reference trajectory as a numpy array of shape (N, 3)
        in  ~/TmrlData/reward/reward.pkl.  This method converts it to our CSV
        format so ProgressTracker can load it directly.

        Args:
            pkl_path:  Path to tmrl's reward.pkl  (default location below).
            out_csv:   Destination CSV path (x, y, z columns).
            spacing:   Waypoint spacing in metres for re-discretization.

        Default pkl location:
            C:/Users/<you>/TmrlData/reward/reward.pkl
        """
        import pickle

        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"reward.pkl not found at {pkl_path}")

        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)

        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError(
                f"Expected shape (N, 3) from reward.pkl, got {raw.shape}"
            )

        waypoints = cls.discretize(raw, spacing)
        cls.save(waypoints, out_csv)

        if verbose:
            seg_lens   = np.linalg.norm(np.diff(raw, axis=0), axis=1)
            total_dist = float(seg_lens.sum())
            print(
                f"tmrl trajectory imported:\n"
                f"  source     : {pkl_path}\n"
                f"  raw points : {len(raw):,}\n"
                f"  waypoints  : {len(waypoints):,}\n"
                f"  track len  : {total_dist:.1f} m\n"
                f"  spacing    : {spacing} m\n"
                f"  saved to   : {Path(out_csv).resolve()}"
            )

        return waypoints


# ══════════════════════════════════════════════════════════════════════════════
# Progress tracker
# ══════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """
    Trajectory-based progress reward for Trackmania.

    Algorithm
    ---------
    At every physics tick:
      1. Query KDTree for the waypoint nearest to the car's XYZ.
      2. reward = max(0,  nearest_idx − furthest_idx_this_lap)
         → positive only when the car passes new waypoints forward.
      3. furthest_idx = max(furthest_idx, nearest_idx)  ← monotonically increasing
      4. If furthest_idx crosses the completion threshold, award lap_bonus once.

    Anti-backwards guarantee
    ------------------------
    furthest_idx never decreases within an episode.  Driving backward reduces
    nearest_idx but furthest_idx stays put → reward = max(0, ...) = 0.
    Re-passing already-cleared waypoints also yields 0.

    Args
    ----
    trajectory_path       Path to a discretized waypoints CSV (x,y,z columns).
    lap_bonus             One-time reward added at lap completion.
    completion_threshold  Fraction of waypoints (0–1) required for lap bonus.
    """

    def __init__(
        self,
        trajectory_path:      str | Path,
        lap_bonus:            float = 50.0,
        completion_threshold: float = 0.95,
    ):
        self._waypoints: np.ndarray = TrajectoryProcessor.load_raw(trajectory_path)
        self._tree      = KDTree(self._waypoints)
        self._n         = len(self._waypoints)
        self._lap_bonus = lap_bonus
        self._thresh    = completion_threshold

        # episode state — reset() initialises these
        self._furthest_idx:  int   = 0
        self._lap_complete:  bool  = False
        self._cumulative:    float = 0.0
        self._steps:         int   = 0
        self._nearest_idx:   int   = 0   # last raw nearest (for display)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def reset(self):
        """Call at the beginning of every episode or lap restart."""
        self._furthest_idx = 0
        self._lap_complete = False
        self._cumulative   = 0.0
        self._steps        = 0
        self._nearest_idx  = 0

    # ── per-step computation ──────────────────────────────────────────────

    def update(self, x: float, y: float, z: float) -> float:
        """
        Given the car's current world-space position, return the step reward.

        Parameters
        ----------
        x, y, z : float
            Car position in Trackmania's world coordinate system (metres).

        Returns
        -------
        float  — reward for this timestep (≥ 0).
        """
        self._steps += 1

        # Nearest waypoint (KDTree query: O(log N) per call)
        _dist, raw_idx = self._tree.query([[x, y, z]], k=1)
        self._nearest_idx = int(raw_idx[0])

        # Forward-only reward
        new_points          = max(0, self._nearest_idx - self._furthest_idx)
        self._furthest_idx  = max(self._furthest_idx, self._nearest_idx)
        reward              = float(new_points)

        # Lap completion bonus (fires at most once per episode)
        if not self._lap_complete:
            if self._furthest_idx >= int(self._n * self._thresh):
                reward            += self._lap_bonus
                self._lap_complete = True

        self._cumulative += reward
        return reward

    # ── read-only properties ──────────────────────────────────────────────

    @property
    def progress_pct(self) -> float:
        """Track completion percentage (0.0 – 100.0)."""
        return (self._furthest_idx / max(self._n - 1, 1)) * 100.0

    @property
    def furthest_idx(self) -> int:
        """Highest waypoint index reached in this episode (monotone)."""
        return self._furthest_idx

    @property
    def nearest_idx(self) -> int:
        """Waypoint index nearest to the car's *current* position."""
        return self._nearest_idx

    @property
    def total_waypoints(self) -> int:
        return self._n

    @property
    def lap_complete(self) -> bool:
        return self._lap_complete

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative

    @property
    def steps(self) -> int:
        return self._steps


# ══════════════════════════════════════════════════════════════════════════════
# CLI: standalone pre-processing tool
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Pre-process a raw trajectory CSV into equidistant waypoints."
    )
    p.add_argument("raw_csv",  help="Input:  raw trajectory CSV (timestamp,x,y,z)")
    p.add_argument("out_csv",  help="Output: discretized waypoints CSV (x,y,z)")
    p.add_argument("--spacing", type=float, default=1.0,
                   help="Waypoint spacing in metres (default: 1.0)")
    args = p.parse_args()

    TrajectoryProcessor.process(args.raw_csv, args.out_csv, spacing=args.spacing)
