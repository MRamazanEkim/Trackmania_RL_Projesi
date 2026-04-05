"""
AI Driving Logic
================
Trackmania RL için gymnasium.Env wrapper ve episode failure detection.

Classes
-------
  DrivingController       – ZERO_SPEED / STUCK / REVERSE tespiti, episode'u erken bitirir
  TrackmaniaRLEnvironment – gymnasium.Env subclass, SB3 ile doğrudan kullanılabilir

Typical workflow
----------------
  from ai_driving_logic import TrackmaniaRLEnvironment

  env = TrackmaniaRLEnvironment(trajectory_path="reference.csv")
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(action)
  env.close()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import gymnasium


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureInfo:
    """Episode başarısızlık bilgisi."""
    is_failed: bool = False
    reason: str = ""
    details: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Driving Controller — Episode Failure Detection
# ══════════════════════════════════════════════════════════════════════════════

class DrivingController:
    """
    Sürüş durumunu izler ve başarısızlık koşullarını kontrol eder.

    Başarısızlık Koşulları
    ----------------------
    ZERO_SPEED : Hız belirli bir süre 0'da kalırsa
    STUCK      : Belirli süre ilerleme kaydedilemezse
    REVERSE    : Araç geri gitmeye başlarsa

    Args
    ----
    zero_speed_threshold : Hız bu değerin altındaysa "durmuş" sayılır (km/h)
    zero_speed_timeout   : Bu süre boyunca duruk kalırsa fail (saniye)
    stuck_timeout        : İlerleme olmadan geçen maks süre (saniye)
    reverse_threshold    : Geri gitme için dot-product eşiği (negatif)
    """

    def __init__(
        self,
        zero_speed_threshold: float = 5.0,
        zero_speed_timeout: float = 3.0,
        stuck_timeout: float = 10.0,
        reverse_threshold: float = -0.3,
    ):
        self._zero_speed_threshold = zero_speed_threshold
        self._zero_speed_timeout = zero_speed_timeout
        self._stuck_timeout = stuck_timeout
        self._reverse_threshold = reverse_threshold

        self._zero_speed_start: Optional[float] = None
        self._last_progress_time: float = 0.0
        self._last_progress_value: float = 0.0
        self._last_position: Optional[np.ndarray] = None
        self._active: bool = False

    def reset(self):
        """Her episode başında çağrılır."""
        self._zero_speed_start = None
        self._last_progress_time = time.time()
        self._last_progress_value = 0.0
        self._last_position = None
        self._active = True

    def check_failure(
        self,
        speed_kmh: float,
        position: np.ndarray,
        progress: float = 0.0,
        forward_direction: Optional[np.ndarray] = None,
    ) -> FailureInfo:
        """
        Mevcut durumu kontrol et, başarısızlık varsa bildir.

        Args
        ----
        speed_kmh         : Aracın anlık hızı (km/h)
        position          : Dünya koordinatları [x, y, z]
        progress          : İlerleme yüzdesi (0–100)
        forward_direction : Aracın ileri yön vektörü (opsiyonel)
        """
        if not self._active:
            return FailureInfo()

        now = time.time()
        position = np.asarray(position, dtype=np.float32)

        # ── Hız Sıfır Kontrolü ────────────────────────────────────────────────
        if speed_kmh < self._zero_speed_threshold:
            if self._zero_speed_start is None:
                self._zero_speed_start = now
            elif (now - self._zero_speed_start) >= self._zero_speed_timeout:
                return FailureInfo(
                    is_failed=True,
                    reason="ZERO_SPEED",
                    details={"speed": speed_kmh, "duration": now - self._zero_speed_start},
                )
        else:
            self._zero_speed_start = None

        # ── Geri Gitme Kontrolü ───────────────────────────────────────────────
        if (
            self._last_position is not None
            and forward_direction is not None
        ):
            movement = position - self._last_position
            mag = np.linalg.norm(movement)
            if mag > 0.01:
                fwd = np.asarray(forward_direction, dtype=np.float32)
                fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
                dot = float(np.dot(movement / mag, fwd))
                if dot < self._reverse_threshold:
                    return FailureInfo(
                        is_failed=True,
                        reason="REVERSE",
                        details={"dot_product": dot},
                    )

        # ── Takılı Kalma Kontrolü (İlerleme Yok) ─────────────────────────────
        if progress > self._last_progress_value:
            self._last_progress_time = now
            self._last_progress_value = progress
        elif (now - self._last_progress_time) >= self._stuck_timeout:
            return FailureInfo(
                is_failed=True,
                reason="STUCK",
                details={
                    "no_progress_duration": now - self._last_progress_time,
                    "last_progress": self._last_progress_value,
                },
            )

        self._last_position = position.copy()
        return FailureInfo()


# ══════════════════════════════════════════════════════════════════════════════
# Trackmania RL Environment
# ══════════════════════════════════════════════════════════════════════════════

class TrackmaniaRLEnvironment(gymnasium.Env):
    """
    tmrl ortamını saran gymnasium.Env subclass.

    SB3 (SAC, TD3, PPO, …) ile doğrudan kullanılabilir.

    Observation
    -----------
    tmrl'ın tuple obs'u (scalar telemetri + LIDAR) tek bir flat Box'a
    dönüştürülür. Boyut runtime'da tmrl'dan alınır; hardcode edilmez.

    Reward
    ------
    ProgressTracker'dan gelir: araç yeni waypoint'ler geçtikçe pozitif reward.
    tmrl'ın kendi reward'ı kullanılmaz.

    Episode sonu
    ------------
    tmrl terminated/truncated VEYA DrivingController failure (ZERO_SPEED /
    STUCK / REVERSE) episode'u bitirir.

    Args
    ----
    trajectory_path   : Referans waypoint CSV (x,y,z sütunları)
    wp_spacing        : Waypoint aralığı, metre (ham kayıt işlenirken)
    failure_detection : DrivingController'ı etkinleştir/devre dışı bırak
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trajectory_path: str,
        wp_spacing: float = 1.0,
        failure_detection: bool = True,
    ):
        super().__init__()

        self._trajectory_path = str(trajectory_path)
        self._wp_spacing = wp_spacing
        self._failure_detection = failure_detection

        # Bileşenler — connect() / reset() içinde somutlaştırılır
        self._interface = None
        self._tracker = None
        self._controller = DrivingController() if failure_detection else None

        # Spaces — _setup_spaces() içinde set edilir
        self._spaces_ready = False
        self._flatten_obs = False

        # Placeholder; SB3 bazen constructor'da erişir
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ── Setup ──────────────────────────────────────────────────────────────

    def _connect(self):
        """tmrl bağlantısını kur (ilk reset'te lazy init)."""
        from telemetry_monitor import TrackmaniaInterface
        self._interface = TrackmaniaInterface()
        if not self._interface.connect():
            raise RuntimeError(
                "tmrl bağlantısı kurulamadı. "
                "Trackmania açık ve OpenPlanet plugin'leri aktif mi?"
            )

    def _setup_spaces(self):
        """
        observation_space ve action_space'i tmrl'den al.
        tmrl'ın Tuple obs space'ini SB3 için flat Box'a dönüştür.
        """
        if self._spaces_ready:
            return

        tmrl_env = self._interface._env
        tmrl_obs_space = tmrl_env.observation_space

        # Action space: tmrl'dekini kullan
        self.action_space = tmrl_env.action_space

        # Observation space: Tuple ise flatten et
        if isinstance(tmrl_obs_space, gymnasium.spaces.Tuple):
            total_dim = sum(
                int(np.prod(s.shape)) for s in tmrl_obs_space.spaces
            )
            self.observation_space = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_dim,),
                dtype=np.float32,
            )
            self._flatten_obs = True
        else:
            self.observation_space = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=tmrl_obs_space.shape,
                dtype=np.float32,
            )
            self._flatten_obs = False

        self._spaces_ready = True

    def _setup_tracker(self):
        """ProgressTracker'ı yükle (ham CSV ise otomatik işle)."""
        from progress_tracker import ProgressTracker, TrajectoryProcessor

        path = Path(self._trajectory_path)
        # Ham kayıt ise (timestamp sütunu var, waypoint adı yok) → işle
        with open(path, newline="", encoding="utf-8") as fh:
            header = fh.readline()
        if "timestamp" in header and "waypoint" not in path.stem:
            processed = path.parent / (path.stem + "_reference.csv")
            print(f"Ham trajectory işleniyor → {processed}")
            TrajectoryProcessor.process(
                str(path), str(processed), spacing=self._wp_spacing
            )
            path = processed

        self._tracker = ProgressTracker(path)

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # İlk çağrıda bağlan
        if self._interface is None:
            self._connect()
            self._setup_spaces()

        # İlk çağrıda tracker'ı yükle
        if self._tracker is None:
            self._setup_tracker()

        raw_obs = self._interface.reset()
        if raw_obs is None:
            raise RuntimeError("tmrl reset() başarısız.")

        if self._controller is not None:
            self._controller.reset()
        self._tracker.reset()

        obs = self._flatten(raw_obs)
        info = {
            "trajectory": self._trajectory_path,
            "total_waypoints": self._tracker.total_waypoints,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        raw_obs, _tmrl_reward, terminated, truncated, info = (
            self._interface.step(action)
        )
        if raw_obs is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, info

        # Telemetri çıkar (x, y, z ve speed için)
        frame = self._interface.parse_observation(raw_obs, action)
        position = np.array([frame.x, frame.y, frame.z], dtype=np.float32)

        # Progress-based reward
        reward = self._tracker.update(frame.x, frame.y, frame.z)
        progress = self._tracker.progress_pct

        # Failure detection
        failure_reason = ""
        if self._controller is not None:
            failure = self._controller.check_failure(
                speed_kmh=frame.speed_kmh,
                position=position,
                progress=progress,
            )
            if failure.is_failed:
                terminated = True
                failure_reason = failure.reason

        info.update({
            "progress_pct": progress,
            "waypoint_idx": self._tracker.furthest_idx,
            "total_waypoints": self._tracker.total_waypoints,
            "lap_complete": self._tracker.lap_complete,
            "failure_reason": failure_reason,
        })

        obs = self._flatten(raw_obs)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._interface is not None:
            self._interface.close()

    # ── Internal ───────────────────────────────────────────────────────────

    def _flatten(self, obs) -> np.ndarray:
        """tmrl obs'unu flat float32 array'e dönüştür."""
        if self._flatten_obs and isinstance(obs, (tuple, list)):
            parts = [np.asarray(o, dtype=np.float32).ravel() for o in obs]
            return np.concatenate(parts)
        return np.asarray(obs, dtype=np.float32).ravel()
