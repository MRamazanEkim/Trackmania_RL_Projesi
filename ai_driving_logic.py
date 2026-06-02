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
    grace_period         : Episode başında bu süre boyunca hiçbir başarısızlık
                           kontrolü yapılmaz. Araca kalkış/hızlanma payı verir;
                           durağan spawn yüzünden ilk saniyede reset olmasını
                           engeller (saniye).
    """

    def __init__(
        self,
        zero_speed_threshold: float = 2.0,
        zero_speed_timeout: float = 3.0,
        stuck_timeout: float = 10.0,
        reverse_threshold: float = -0.3,
        grace_period: float = 2.0,
    ):
        self._zero_speed_threshold = zero_speed_threshold
        self._zero_speed_timeout = zero_speed_timeout
        self._stuck_timeout = stuck_timeout
        self._reverse_threshold = reverse_threshold
        self._grace_period = grace_period

        self._zero_speed_start: Optional[float] = None
        self._last_progress_time: float = 0.0
        self._last_progress_value: float = 0.0
        self._last_position: Optional[np.ndarray] = None
        self._episode_start: float = 0.0
        self._active: bool = False

    def reset(self):
        """Her episode başında çağrılır."""
        self._zero_speed_start = None
        self._last_progress_time = time.time()
        self._last_progress_value = 0.0
        self._last_position = None
        self._episode_start = time.time()
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

        # Kalkış payı: episode başında araca hızlanması için süre tanı.
        if (now - self._episode_start) < self._grace_period:
            self._last_position = position.copy()
            return FailureInfo()

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
    speed_reward_coef : Hıza orantılı ödül katsayısı. reward += coef * hız(km/h).
                        Araç ne kadar hızlı giderse o kadar çok ödül — yavaş
                        sallanarak ilerlemeyi caydırır (sadece katedilen mesafe
                        değil, hız da ödüllendirilir).
    disable_brake     : True ise fren aksiyonu (idx1) hep kapalı gönderilir;
                        araç yalnız gaz (idx0) + direksiyon (idx2) kullanır.
                        Aracın kasıtlı yavaşlayıp ceza/çarpmadan kaçmasını önler.
    crash_penalty     : Episode başarısızlıkla biterse verilen terminal ceza.
                        Küçük tutulur: büyük olursa "sür + çarp" denemesini,
                        "yerinde sallan + hayatta kal" seçeneğinden kötü yapıp
                        aracın denemeyi bırakmasına yol açıyordu. Küçük ceza
                        "duvara çarpma kötü" sinyalini korur, denemeyi öldürmez.
    idle_speed_kmh    : Hız bu değerin altındaysa araç "duruyor" sayılır (km/h).
    idle_penalty      : Yerinde durmanın/sürünmenin adım başına büyük cezası.
                        Aracı yerinde sallanma lokal minimumundan çıkarıp hareket
                        etmeye zorlar.
    low_speed_kmh     : "Yavaş sürüş" eşiği (km/h). Bu değerin altında uzun süre
                        gidilirse ceza verilir.
    low_speed_timeout : Hız low_speed_kmh altında bu süreden uzun kalırsa ceza
                        başlar (saniye). Kısa yavaşlamalara (viraj, kalkış) izin
                        verir; sürekli yavaş sürünmeyi cezalandırır.
    low_speed_penalty : low_speed_timeout aşıldıktan sonra adım başına ceza.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trajectory_path: str,
        wp_spacing: float = 1.0,
        failure_detection: bool = True,
        speed_reward_coef: float = 0.02,
        crash_penalty: float = 2.0,
        idle_speed_kmh: float = 5.0,
        idle_penalty: float = 0.5,
        low_speed_kmh: float = 30.0,
        low_speed_timeout: float = 3.0,
        low_speed_penalty: float = 0.3,
        disable_brake: bool = True,
    ):
        super().__init__()

        self._trajectory_path = str(trajectory_path)
        self._wp_spacing = wp_spacing
        self._failure_detection = failure_detection
        self._speed_reward_coef = speed_reward_coef
        self._crash_penalty = crash_penalty
        self._idle_speed_kmh = idle_speed_kmh
        self._idle_penalty = idle_penalty
        self._low_speed_kmh = low_speed_kmh
        self._low_speed_timeout = low_speed_timeout
        self._low_speed_penalty = low_speed_penalty
        self._low_speed_start: Optional[float] = None
        self._disable_brake = disable_brake

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
        self._low_speed_start = None

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
        # tmrl klavye mapping: aksiyon = [gaz(idx0), fren(idx1), direksiyon(idx2)].
        # control[0]>0→ileri, control[1]>0→fren/geri, control[2]>0.5→sağ <-0.5→sol.
        # Fren iptal: idx1'i -1'e zorla → fren/geri hiç tetiklenmez
        # (araç yalnız gaz idx0 + direksiyon idx2 kullanır).
        if self._disable_brake:
            action = np.asarray(action, dtype=np.float32).copy()
            if action.shape[-1] > 1:
                action[1] = -1.0

        raw_obs, _tmrl_reward, terminated, truncated, info = (
            self._interface.step(action)
        )
        if raw_obs is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, info

        # Telemetri çıkar (x, y, z ve speed için)
        frame = self._interface.parse_observation(raw_obs, action)
        position = np.array([frame.x, frame.y, frame.z], dtype=np.float32)

        # Araç henüz spawn olmadıysa pozisyon (0,0,0) gelir. Bu frame'i nötr
        # geç: aksi halde origin referans çizgiye ~800m uzak görünür, sahte
        # off-track cezası + anında reset olur.
        if abs(frame.x) < 1e-3 and abs(frame.y) < 1e-3 and abs(frame.z) < 1e-3:
            info.update({
                "progress_pct": self._tracker.progress_pct,
                "waypoint_idx": self._tracker.furthest_idx,
                "total_waypoints": self._tracker.total_waypoints,
                "lap_complete": self._tracker.lap_complete,
                "failure_reason": "",
                "dist_from_line": 0.0,
            })
            return self._flatten(raw_obs), 0.0, terminated, truncated, info

        # İlerleme ödülü (+ parkurda kalma) + küçük hız ipucu
        reward = self._tracker.update(frame.x, frame.y, frame.z)
        reward += self._speed_reward_coef * max(0.0, frame.speed_kmh)
        # Yerinde durma/sürünme büyük cezası — sallanma tuzağına karşı
        if frame.speed_kmh < self._idle_speed_kmh:
            reward -= self._idle_penalty
        # Sürekli yavaş sürüş cezası — low_speed_kmh altında low_speed_timeout'tan
        # uzun kalırsa ceza (kısa yavaşlamalara izin var). Aracı hızlı tutmaya iter.
        _now = time.time()
        if frame.speed_kmh < self._low_speed_kmh:
            if self._low_speed_start is None:
                self._low_speed_start = _now
            elif (_now - self._low_speed_start) >= self._low_speed_timeout:
                reward -= self._low_speed_penalty
        else:
            self._low_speed_start = None
        progress = self._tracker.progress_pct

        # Failure detection — duvara çarpma (STUCK/ZERO_SPEED), geri gitme,
        # veya racing line'dan çok sapma episode'u bitirir.
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
        if self._tracker.off_track and not failure_reason:
            terminated = True
            failure_reason = "OFF_TRACK"

        # Çarpma/başarısızlık terminal cezası — parkuru tamamlamadan biterse.
        # Aracın duvara çarpmaktansa dönmeyi tercih etmesini sağlar.
        if terminated and failure_reason and not self._tracker.lap_complete:
            reward -= self._crash_penalty

        info.update({
            "progress_pct": progress,
            "waypoint_idx": self._tracker.furthest_idx,
            "total_waypoints": self._tracker.total_waypoints,
            "lap_complete": self._tracker.lap_complete,
            "failure_reason": failure_reason,
            "dist_from_line": self._tracker.last_dist,
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
