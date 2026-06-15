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
        self._migrate()

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
                algorithm       TEXT    NOT NULL DEFAULT 'sac',
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
                algorithm         TEXT    NOT NULL DEFAULT 'sac',
                lap_time_s        REAL    NOT NULL DEFAULT 0.0,
                collision         INTEGER NOT NULL DEFAULT 0,
                ended_at          TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON episodes(run_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_step   ON episodes(global_step);

            -- Popülasyon + elitizm: her satır bir adayın bir jenerasyondaki değerlendirmesi.
            -- candidate_idx = -1 → o nesle değişmeden taşınan elit (best).
            -- is_best = 1 → o jenerasyonun kazananı (sonraki neslin ebeveyni).
            CREATE TABLE IF NOT EXISTS generations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES training_runs(id),
                gen_no          INTEGER NOT NULL,
                candidate_idx   INTEGER NOT NULL,
                mean_reward     REAL    NOT NULL,
                best_reward     REAL    NOT NULL,
                mean_progress   REAL    NOT NULL,
                best_progress   REAL    NOT NULL,
                eval_episodes   INTEGER NOT NULL,
                lap_completed   INTEGER NOT NULL DEFAULT 0,
                is_best         INTEGER NOT NULL DEFAULT 0,
                model_path      TEXT    NOT NULL DEFAULT '',
                parent_gen      INTEGER,
                algorithm       TEXT    NOT NULL DEFAULT 'sac',
                created_at      TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_gen_run  ON generations(run_id);
            CREATE INDEX IF NOT EXISTS idx_gen_no   ON generations(run_id, gen_no);
        """)
        self._conn.commit()

    def _migrate(self) -> None:
        """
        Eski DB'lere yeni kolonları geriye-uyumlu ekle (CREATE TABLE IF NOT EXISTS
        mevcut tabloya kolon eklemez). Kolon zaten varsa sessizce geç.
        """
        add = [
            ("training_runs", "algorithm", "TEXT NOT NULL DEFAULT 'sac'"),
            ("episodes",      "algorithm", "TEXT NOT NULL DEFAULT 'sac'"),
            ("episodes",      "lap_time_s", "REAL NOT NULL DEFAULT 0.0"),
            ("episodes",      "collision",  "INTEGER NOT NULL DEFAULT 0"),
            ("generations",   "algorithm", "TEXT NOT NULL DEFAULT 'sac'"),
        ]
        for table, col, decl in add:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass  # kolon zaten var
        self._conn.commit()

    # ── Training runs ─────────────────────────────────────────────────────────

    def create_run(
        self,
        trajectory_path: str,
        checkpoint_dir:  str,
        total_timesteps: int,
        resume_path:     Optional[str] = None,
        notes:           Optional[str] = None,
        algorithm:       str = "sac",
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO training_runs
                (started_at, trajectory_path, resume_path, checkpoint_dir,
                 total_timesteps, algorithm, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (_utcnow(), trajectory_path, resume_path, checkpoint_dir,
             total_timesteps, algorithm, notes),
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
        algorithm:         str = "sac",
        lap_time_s:        float = 0.0,
        collision:         bool = False,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO episodes
                (run_id, episode_number, global_step, cumulative_reward, steps,
                 furthest_waypoint, progress_pct, lap_complete, failure_reason,
                 algorithm, lap_time_s, collision, ended_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, episode_number, global_step, cumulative_reward, steps,
                furthest_waypoint, progress_pct, int(lap_complete), failure_reason,
                algorithm, lap_time_s, int(collision), _utcnow(),
            ),
        )
        self._conn.commit()

    # ── Generations (popülasyon + elitizm) ─────────────────────────────────────

    def log_candidate(
        self,
        run_id:        int,
        gen_no:        int,
        candidate_idx: int,
        mean_reward:   float,
        best_reward:   float,
        mean_progress: float,
        best_progress: float,
        eval_episodes: int,
        lap_completed: bool = False,
        is_best:       bool = False,
        model_path:    str = "",
        parent_gen:    Optional[int] = None,
        algorithm:     str = "sac",
    ) -> int:
        """Bir adayın bir jenerasyondaki değerlendirme sonucunu kaydet."""
        cur = self._conn.execute(
            """
            INSERT INTO generations
                (run_id, gen_no, candidate_idx, mean_reward, best_reward,
                 mean_progress, best_progress, eval_episodes, lap_completed,
                 is_best, model_path, parent_gen, algorithm, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, gen_no, candidate_idx, mean_reward, best_reward,
                mean_progress, best_progress, eval_episodes, int(lap_completed),
                int(is_best), model_path, parent_gen, algorithm, _utcnow(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def mark_best(self, row_id: int) -> None:
        """Bir generations satırını o jenerasyonun kazananı olarak işaretle."""
        self._conn.execute("UPDATE generations SET is_best=1 WHERE id=?", (row_id,))
        self._conn.commit()

    def best_overall(self, run_id: int) -> Optional[dict]:
        """
        Bir koşudaki TÜM jenerasyonlar arasında en iyi adayı döner.
        Sıralama: önce tur tamamlama, sonra ilerleme, sonra ödül.
        Elitizmin kalbi: sonraki nesil bunun model_path'inden türer.
        """
        row = self._conn.execute(
            """
            SELECT * FROM generations WHERE run_id=?
            ORDER BY lap_completed DESC, best_progress DESC, best_reward DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        return dict(row) if row else None

    def generation_summary(self, run_id: int) -> list[dict]:
        """
        Her jenerasyon için kazananın özeti (gen_no artan).
        'Her nesilde iyileşme'yi izlemek için: best_progress sütunu nesil
        boyunca azalmamalı (elitizm garantisi).
        """
        rows = self._conn.execute(
            """
            SELECT gen_no,
                   MAX(best_progress) AS gen_best_progress,
                   MAX(best_reward)   AS gen_best_reward,
                   MAX(lap_completed) AS any_lap
            FROM generations WHERE run_id=?
            GROUP BY gen_no ORDER BY gen_no
            """,
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Algoritma karşılaştırması (DQN vs PPO vs SAC) ──────────────────────────

    def algorithm_comparison(self) -> list[dict]:
        """
        Algoritma başına özet metrikler — özetin 'eşit koşullarda karşılaştırma'
        tablosu. episodes tablosundan algoritma bazında toplar:
          • episode sayısı
          • pist tamamlama oranı (lap_complete ortalaması)
          • ortalama/en iyi ilerleme %
          • çarpışma oranı (collision ortalaması)
          • tamamlanan turların ortalama süresi (lap_time_s, sadece lap=1)
          • ortalama ödül
        """
        rows = self._conn.execute(
            """
            SELECT algorithm,
                   COUNT(*)                                   AS episodes,
                   AVG(lap_complete)                          AS completion_rate,
                   AVG(progress_pct)                          AS mean_progress,
                   MAX(progress_pct)                          AS best_progress,
                   AVG(collision)                             AS collision_rate,
                   AVG(CASE WHEN lap_complete=1 THEN lap_time_s END) AS mean_lap_time,
                   AVG(cumulative_reward)                     AS mean_reward
            FROM episodes
            GROUP BY algorithm ORDER BY completion_rate DESC, best_progress DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

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
