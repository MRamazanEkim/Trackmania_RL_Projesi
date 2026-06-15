"""
db/callbacks.py
===============
SB3 callback'leri: episode loglama + replay buffer checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from db.experience_store import ExperienceStore


class EpisodeLoggerCallback(BaseCallback):
    """
    Her episode bitişinde SQLite'a bir satır yazar.

    Reward tespiti
    --------------
    SB3'ün off-policy döngüsünde her adımda self.locals["rewards"][0] mevcut.
    Bunu _episode_reward'a biriktirip done=True gelince kaydedip sıfırlarız.

    Episode istatistikleri
    ----------------------
    done adımında self.locals["infos"][0] şunları içerir (TrackmaniaRLEnvironment'tan):
        progress_pct, waypoint_idx, lap_complete, failure_reason
    """

    def __init__(self, store: ExperienceStore, run_id: int, verbose: int = 1):
        super().__init__(verbose)
        self._store          = store
        self._run_id         = run_id
        self._episode_reward = 0.0
        self._episode_steps  = 0
        self._episode_number = 0

    def _on_step(self) -> bool:
        self._episode_reward += float(self.locals["rewards"][0])
        self._episode_steps  += 1

        if bool(self.locals["dones"][0]):
            self._episode_number += 1
            info = self.locals["infos"][0]

            self._store.log_episode(
                run_id            = self._run_id,
                episode_number    = self._episode_number,
                global_step       = self.num_timesteps,
                cumulative_reward = self._episode_reward,
                steps             = self._episode_steps,
                furthest_waypoint = info.get("waypoint_idx", 0),
                progress_pct      = info.get("progress_pct", 0.0),
                lap_complete      = bool(info.get("lap_complete", False)),
                failure_reason    = info.get("failure_reason", ""),
            )

            if self.verbose >= 1:
                lap_tag = "LAP!" if info.get("lap_complete") else info.get("failure_reason", "")
                print(
                    f"[Ep {self._episode_number:4d}] "
                    f"adim={self.num_timesteps:7d}  "
                    f"reward={self._episode_reward:8.2f}  "
                    f"ilerleme={info.get('progress_pct', 0.0):5.1f}%  "
                    f"{lap_tag}"
                )

            self._episode_reward = 0.0
            self._episode_steps  = 0

        return True


class ReplayBufferCheckpointCallback(BaseCallback):
    """
    Model checkpoint'iyle aynı sıklıkta replay buffer'ı .pkl olarak kaydeder.

    Isimlendirme kurali
    -------------------
    CheckpointCallback -> checkpoints/sac_tm_5000_steps.zip
    Bu callback        -> checkpoints/sac_tm_5000_steps_replay.pkl

    save_freq ve name_prefix, CheckpointCallback ile ayni olmali.
    """

    def __init__(
        self,
        save_freq:   int,
        save_path:   str | Path,
        name_prefix: str = "sac_tm",
        verbose:     int = 1,
    ):
        super().__init__(verbose)
        self._save_freq   = save_freq
        self._save_path   = Path(save_path)
        self._name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self._save_freq == 0:
            replay_path = self._save_path / f"{self._name_prefix}_{self.num_timesteps}_steps_replay"
            self.model.save_replay_buffer(str(replay_path))
            if self.verbose >= 1:
                print(f"Replay buffer kaydedildi: {replay_path}.pkl")
        return True
