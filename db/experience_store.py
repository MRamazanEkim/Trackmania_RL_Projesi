"""
db/experience_store.py
======================
SQLite tabanlı deneyim deposu. Yeni pip paketi gerekmez — stdlib sqlite3 kullanır.

Her python train.py çalıştırması = bir training_run satırı.
Her episode bitişi = bir episodes satırı.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ExperienceStore:
    """
    SQLite veritabanı sarmalayıcı.

    Tipik kullanım (train.py içinde)
    ----------------------------------
    store  = ExperienceStore("training.db")
    run_id = store.create_run(
        trajectory_path = args.trajectory,
        checkpoint_dir  = args.checkpoint_dir,
        total_timesteps = args.timesteps,
        resume_path     = args.resume,
    )
    # run_id'yi EpisodeLoggerCallback'e ver
    store.close()
    """

    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at      TEXT    NOT NULL,
                trajectory_path TEXT    NOT NULL,
                resume_path     TEXT,
                checkpoint_dir  TEXT    NOT NULL,
                total_timesteps INTEGER NOT NULL,
                notes           TEXT
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            INTEGER NOT NULL REFERENCES training_runs(id),
                episode_number    INTEGER NOT NULL,
                global_step       INTEGER NOT NULL,
                cumulative_reward REAL    NOT NULL,
                steps             INTEGER NOT NULL,
                furthest_waypoint INTEGER NOT NULL,
                progress_pct      REAL    NOT NULL,
                lap_complete      INTEGER NOT NULL,
                failure_reason    TEXT    NOT NULL DEFAULT '',
                ended_at          TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON episodes(run_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_step   ON episodes(global_step);
        """)
        self._conn.commit()

    # ── Training runs ─────────────────────────────────────────────────────────

    def create_run(
        self,
        trajectory_path: str,
        checkpoint_dir:  str,
        total_timesteps: int,
        resume_path:     Optional[str] = None,
        notes:           Optional[str] = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO training_runs
                (started_at, trajectory_path, resume_path, checkpoint_dir, total_timesteps, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (_utcnow(), trajectory_path, resume_path, checkpoint_dir, total_timesteps, notes),
        )
        self._conn.commit()
        return cur.lastrowid

    # ── Episodes ──────────────────────────────────────────────────────────────

    def log_episode(
        self,
        run_id:            int,
        episode_number:    int,
        global_step:       int,
        cumulative_reward: float,
        steps:             int,
        furthest_waypoint: int,
        progress_pct:      float,
        lap_complete:      bool,
        failure_reason:    str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO episodes
                (run_id, episode_number, global_step, cumulative_reward, steps,
                 furthest_waypoint, progress_pct, lap_complete, failure_reason, ended_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, episode_number, global_step, cumulative_reward, steps,
                furthest_waypoint, progress_pct, int(lap_complete), failure_reason, _utcnow(),
            ),
        )
        self._conn.commit()

    # ── Sorgular ──────────────────────────────────────────────────────────────

    def best_progress(self, run_id: Optional[int] = None) -> Optional[dict]:
        """En yüksek progress_pct'e sahip episode'u döner."""
        if run_id is not None:
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE run_id=? ORDER BY progress_pct DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM episodes ORDER BY progress_pct DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def episode_count(self, run_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE run_id=?", (run_id,)
        ).fetchone()
        return row[0] if row else 0

    def recent_episodes(self, run_id: int, n: int = 20) -> list[dict]:
        """Son n episode'u yeniden eskiye sıralar."""
        rows = self._conn.execute(
            """
            SELECT episode_number, global_step, cumulative_reward,
                   progress_pct, lap_complete, failure_reason
            FROM episodes WHERE run_id=?
            ORDER BY episode_number DESC LIMIT ?
            """,
            (run_id, n),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_runs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, started_at, trajectory_path, resume_path, total_timesteps FROM training_runs ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
