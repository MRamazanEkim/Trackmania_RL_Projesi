"""
AI Driving Logic - Gelismis Surus ve Odul Sistemi
==================================================
Bu modul, Trackmania RL icin gelismis suruş mantigi saglar:

1. Basarisizlik Kosullari (done = True):
   - Aracin hizi 0'a duserse
   - Arac geri gitmeye baslarsa (ters yon)
   - Belirli sure ilerleme kaydedilemezse (stuck detection)

2. Adaptive Threshold Reward System:
   - Her seansta elde edilen en yuksek skoru hedef olarak belirle
   - Yapay zekayi bu skoru gecmesi icin tesvik et
   - Progressive difficulty: Basari arttikca hedefler yukselir

Kullanim
--------
  from ai_driving_logic import DrivingController, AdaptiveRewardSystem

  controller = DrivingController()
  reward_system = AdaptiveRewardSystem()

  # Her adimda:
  done, failure_reason = controller.check_failure(speed, position, direction)
  reward = reward_system.calculate_reward(progress, done)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DrivingState:
    """Aracin anlik suruş durumu."""
    speed_kmh: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    forward_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    timestamp: float = 0.0


@dataclass
class FailureInfo:
    """Basarisizlik bilgisi."""
    is_failed: bool = False
    reason: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class EpisodeStats:
    """Bir episode'un istatistikleri."""
    episode_id: int = 0
    max_progress: float = 0.0
    total_reward: float = 0.0
    duration: float = 0.0
    failure_reason: str = ""
    timestamp: float = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════════════════
# Driving Controller - Basarisizlik Kontrolu
# ══════════════════════════════════════════════════════════════════════════════

class DrivingController:
    """
    Suruş durumunu izler ve basarisizlik kosullarini kontrol eder.

    Basarisizlik Kosullari:
    1. ZERO_SPEED: Hiz belirli bir sure 0'da kalirsa
    2. REVERSE: Arac geri gitmeye baslarsa
    3. STUCK: Belirli sure ilerleme kaydedilemezse
    4. OFF_TRACK: Arac pistten cikarsa (opsiyonel)

    Args:
        zero_speed_threshold: Hiz bu degerin altindaysa "durmus" sayilir (km/h)
        zero_speed_timeout: Bu sure boyunca duruk kalirsa fail (saniye)
        reverse_threshold: Negatif hiz esigi (geri gitme tespiti)
        stuck_timeout: Ilerleme olmadan gecen max sure (saniye)
        min_progress_rate: Minimum ilerleme orani (waypoint/saniye)
    """

    def __init__(
        self,
        zero_speed_threshold: float = 5.0,
        zero_speed_timeout: float = 3.0,
        reverse_threshold: float = -2.0,
        stuck_timeout: float = 10.0,
        min_progress_rate: float = 0.1,
    ):
        self._zero_speed_threshold = zero_speed_threshold
        self._zero_speed_timeout = zero_speed_timeout
        self._reverse_threshold = reverse_threshold
        self._stuck_timeout = stuck_timeout
        self._min_progress_rate = min_progress_rate

        # Internal state
        self._zero_speed_start: Optional[float] = None
        self._last_progress_time: float = time.time()
        self._last_progress_value: float = 0.0
        self._last_position: Optional[np.ndarray] = None
        self._position_history: List[np.ndarray] = []
        self._is_active: bool = False

    def reset(self):
        """Episode basinda cagrilir - tum durumlari sifirla."""
        self._zero_speed_start = None
        self._last_progress_time = time.time()
        self._last_progress_value = 0.0
        self._last_position = None
        self._position_history = []
        self._is_active = True

    def check_failure(
        self,
        speed_kmh: float,
        position: np.ndarray,
        progress: float = 0.0,
        forward_direction: Optional[np.ndarray] = None,
    ) -> FailureInfo:
        """
        Mevcut durumu kontrol et ve basarisizlik varsa bildir.

        Args:
            speed_kmh: Aracin anlık hızı (km/h)
            position: Aracın dünya koordinatları [x, y, z]
            progress: Mevcut ilerleme yüzdesi (0-100)
            forward_direction: Aracın ileri yön vektörü (opsiyonel)

        Returns:
            FailureInfo: Başarısızlık durumu ve nedeni
        """
        if not self._is_active:
            return FailureInfo(is_failed=False)

        now = time.time()
        position = np.asarray(position, dtype=np.float32)

        # ── 1. Hiz Sifir Kontrolu ─────────────────────────────────────────────
        if speed_kmh < self._zero_speed_threshold:
            if self._zero_speed_start is None:
                self._zero_speed_start = now
            elif (now - self._zero_speed_start) >= self._zero_speed_timeout:
                return FailureInfo(
                    is_failed=True,
                    reason="ZERO_SPEED",
                    details={
                        "speed": speed_kmh,
                        "duration": now - self._zero_speed_start,
                        "threshold": self._zero_speed_threshold,
                    }
                )
        else:
            self._zero_speed_start = None

        # ── 2. Geri Gitme Kontrolu ────────────────────────────────────────────
        if self._last_position is not None and forward_direction is not None:
            movement = position - self._last_position
            movement_mag = np.linalg.norm(movement)

            if movement_mag > 0.01:  # Anlamlı hareket var
                forward_dir = np.asarray(forward_direction, dtype=np.float32)
                forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-8)

                # Hareketin ileri yönle açısını hesapla
                dot_product = np.dot(movement / movement_mag, forward_dir)

                # Negatif dot product = geri gitme
                if dot_product < self._reverse_threshold:
                    return FailureInfo(
                        is_failed=True,
                        reason="REVERSE",
                        details={
                            "dot_product": float(dot_product),
                            "movement": movement.tolist(),
                            "threshold": self._reverse_threshold,
                        }
                    )

        # ── 3. Stuck Kontrolu (Ilerleme Yok) ──────────────────────────────────
        if progress > self._last_progress_value:
            self._last_progress_time = now
            self._last_progress_value = progress
        elif (now - self._last_progress_time) >= self._stuck_timeout:
            # Uzun sure ilerleme yok
            return FailureInfo(
                is_failed=True,
                reason="STUCK",
                details={
                    "no_progress_duration": now - self._last_progress_time,
                    "last_progress": self._last_progress_value,
                    "timeout": self._stuck_timeout,
                }
            )

        # ── Durum Guncelle ────────────────────────────────────────────────────
        self._last_position = position.copy()
        self._position_history.append(position.copy())
        if len(self._position_history) > 100:
            self._position_history.pop(0)

        return FailureInfo(is_failed=False)

    def deactivate(self):
        """Kontrolleri devre disi birak."""
        self._is_active = False


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive Reward System - Threshold Bazli Odul
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveRewardSystem:
    """
    Dinamik eşik (threshold) tabanlı ödül sistemi.

    Her başarılı episode'da en yüksek skor hedef olarak belirlenir.
    Yapay zeka bu hedefi geçtiğinde bonus ödül alır.

    Özellikler:
    - Personal Best Tracking: Her seansın en iyi skorunu sakla
    - Progressive Difficulty: Başarı arttıkça hedefler yükselir
    - Milestone Bonuses: Belirli ilerleme noktalarında ekstra ödül
    - Anti-Stagnation: Uzun sure gelişme olmazsa hedefi hafifce düşür

    Args:
        history_file: Geçmiş episode verilerinin saklandığı dosya
        milestone_interval: Bonus için waypoint aralığı
        milestone_bonus: Her milestone'da verilen bonus
        pb_bonus_multiplier: Personal best aşıldığında çarpan
        stagnation_threshold: Gelişme olmadan geçen max episode sayısı
        stagnation_decay: Her stagnation'da hedefin düşürülme oranı
    """

    DEFAULT_HISTORY_FILE = "episode_history.json"

    def __init__(
        self,
        history_file: Optional[str] = None,
        milestone_interval: int = 50,
        milestone_bonus: float = 5.0,
        pb_bonus_multiplier: float = 2.0,
        stagnation_threshold: int = 20,
        stagnation_decay: float = 0.95,
    ):
        self._history_file = Path(history_file or self.DEFAULT_HISTORY_FILE)
        self._milestone_interval = milestone_interval
        self._milestone_bonus = milestone_bonus
        self._pb_bonus_multiplier = pb_bonus_multiplier
        self._stagnation_threshold = stagnation_threshold
        self._stagnation_decay = stagnation_decay

        # State
        self._personal_best: float = 0.0
        self._current_threshold: float = 0.0
        self._episodes_without_improvement: int = 0
        self._episode_history: List[EpisodeStats] = []
        self._current_episode: Optional[EpisodeStats] = None
        self._milestones_reached: set = set()
        self._episode_count: int = 0

        # Load history if exists
        self._load_history()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_history(self):
        """Geçmiş episode verilerini yükle."""
        if self._history_file.exists():
            try:
                with open(self._history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._personal_best = data.get("personal_best", 0.0)
                    self._current_threshold = data.get("current_threshold", 0.0)
                    self._episode_count = data.get("episode_count", 0)
                    self._episodes_without_improvement = data.get(
                        "episodes_without_improvement", 0
                    )
                    # Load recent history (last 100 episodes)
                    history_data = data.get("recent_history", [])
                    self._episode_history = [
                        EpisodeStats(**ep) for ep in history_data[-100:]
                    ]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Start fresh if file is corrupted

    def _save_history(self):
        """Episode verilerini kaydet."""
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "personal_best": self._personal_best,
            "current_threshold": self._current_threshold,
            "episode_count": self._episode_count,
            "episodes_without_improvement": self._episodes_without_improvement,
            "recent_history": [
                {
                    "episode_id": ep.episode_id,
                    "max_progress": ep.max_progress,
                    "total_reward": ep.total_reward,
                    "duration": ep.duration,
                    "failure_reason": ep.failure_reason,
                    "timestamp": ep.timestamp,
                }
                for ep in self._episode_history[-100:]
            ],
        }
        with open(self._history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ── Episode Lifecycle ─────────────────────────────────────────────────────

    def start_episode(self) -> int:
        """Yeni episode başlat. Episode ID'sini döndürür."""
        self._episode_count += 1
        self._current_episode = EpisodeStats(
            episode_id=self._episode_count,
            timestamp=time.time(),
        )
        self._milestones_reached = set()
        return self._episode_count

    def end_episode(
        self,
        max_progress: float,
        total_reward: float,
        failure_reason: str = "",
    ) -> dict:
        """
        Episode'u sonlandır ve istatistikleri hesapla.

        Returns:
            dict: Episode sonuç bilgileri (pb_broken, new_threshold, etc.)
        """
        if self._current_episode is None:
            return {}

        duration = time.time() - self._current_episode.timestamp
        self._current_episode.max_progress = max_progress
        self._current_episode.total_reward = total_reward
        self._current_episode.duration = duration
        self._current_episode.failure_reason = failure_reason

        self._episode_history.append(self._current_episode)

        # Personal best kontrolü
        pb_broken = max_progress > self._personal_best
        if pb_broken:
            old_pb = self._personal_best
            self._personal_best = max_progress
            self._current_threshold = max_progress
            self._episodes_without_improvement = 0
        else:
            self._episodes_without_improvement += 1

        # Anti-stagnation: Uzun süre gelişme olmazsa hedefi düşür
        if self._episodes_without_improvement >= self._stagnation_threshold:
            self._current_threshold *= self._stagnation_decay
            self._episodes_without_improvement = 0

        self._save_history()

        result = {
            "episode_id": self._current_episode.episode_id,
            "max_progress": max_progress,
            "total_reward": total_reward,
            "duration": duration,
            "pb_broken": pb_broken,
            "personal_best": self._personal_best,
            "current_threshold": self._current_threshold,
            "episodes_without_improvement": self._episodes_without_improvement,
        }

        self._current_episode = None
        return result

    # ── Reward Calculation ────────────────────────────────────────────────────

    def calculate_reward(
        self,
        progress: float,
        base_reward: float,
        is_done: bool = False,
        failure_reason: str = "",
    ) -> Tuple[float, dict]:
        """
        Adaptif ödül hesapla.

        Args:
            progress: Mevcut ilerleme yüzdesi (0-100)
            base_reward: Temel ödül (progress tracker'dan)
            is_done: Episode bitti mi?
            failure_reason: Başarısızlık nedeni (varsa)

        Returns:
            Tuple[float, dict]: (toplam_odul, odul_detaylari)
        """
        total_reward = base_reward
        reward_breakdown = {
            "base": base_reward,
            "milestone": 0.0,
            "threshold_bonus": 0.0,
            "pb_bonus": 0.0,
            "failure_penalty": 0.0,
        }

        # ── Milestone Bonuslari ───────────────────────────────────────────────
        milestone_idx = int(progress // self._milestone_interval)
        for m in range(milestone_idx + 1):
            if m > 0 and m not in self._milestones_reached:
                self._milestones_reached.add(m)
                milestone_bonus = self._milestone_bonus * (1 + m * 0.1)
                reward_breakdown["milestone"] += milestone_bonus
                total_reward += milestone_bonus

        # ── Threshold Bonusu ──────────────────────────────────────────────────
        if self._current_threshold > 0 and progress > self._current_threshold:
            excess = progress - self._current_threshold
            threshold_bonus = excess * 0.5  # Eşiği aşan her % için bonus
            reward_breakdown["threshold_bonus"] = threshold_bonus
            total_reward += threshold_bonus

        # ── Personal Best Bonusu ──────────────────────────────────────────────
        if progress > self._personal_best and self._personal_best > 0:
            pb_bonus = (progress - self._personal_best) * self._pb_bonus_multiplier
            reward_breakdown["pb_bonus"] = pb_bonus
            total_reward += pb_bonus

        # ── Başarısızlık Cezası ───────────────────────────────────────────────
        if is_done and failure_reason:
            # Başarısızlık tipine göre ceza
            penalty_map = {
                "ZERO_SPEED": -10.0,
                "REVERSE": -15.0,
                "STUCK": -5.0,
                "OFF_TRACK": -20.0,
            }
            penalty = penalty_map.get(failure_reason, -5.0)
            reward_breakdown["failure_penalty"] = penalty
            total_reward += penalty

        # Episode stats güncelle
        if self._current_episode is not None:
            self._current_episode.max_progress = max(
                self._current_episode.max_progress, progress
            )
            self._current_episode.total_reward += total_reward

        return total_reward, reward_breakdown

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def personal_best(self) -> float:
        """En yüksek ilerleme skoru."""
        return self._personal_best

    @property
    def current_threshold(self) -> float:
        """Mevcut hedef eşik değeri."""
        return self._current_threshold

    @property
    def episode_count(self) -> int:
        """Toplam episode sayısı."""
        return self._episode_count

    @property
    def recent_stats(self) -> dict:
        """Son 10 episode'un istatistikleri."""
        recent = self._episode_history[-10:]
        if not recent:
            return {}

        return {
            "avg_progress": np.mean([ep.max_progress for ep in recent]),
            "avg_reward": np.mean([ep.total_reward for ep in recent]),
            "avg_duration": np.mean([ep.duration for ep in recent]),
            "best_recent": max(ep.max_progress for ep in recent),
            "count": len(recent),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Integrated Environment Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class TrackmaniaRLEnvironment:
    """
    tmrl ortamını saran ve gelişmiş sürüş mantığı ekleyen wrapper.

    Bu sınıf:
    - DrivingController ile başarısızlık kontrolü yapar
    - AdaptiveRewardSystem ile akıllı ödül hesaplar
    - Episode yönetimini otomatikleştirir

    Kullanım:
        env = TrackmaniaRLEnvironment()
        obs = env.reset()

        while True:
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
    """

    def __init__(
        self,
        trajectory_path: Optional[str] = None,
        history_file: Optional[str] = None,
        enable_failure_detection: bool = True,
    ):
        self._trajectory_path = trajectory_path
        self._enable_failure_detection = enable_failure_detection

        # Components
        self._controller = DrivingController()
        self._reward_system = AdaptiveRewardSystem(history_file=history_file)
        self._tracker = None
        self._interface = None

        # State
        self._episode_active = False
        self._last_obs = None
        self._forward_direction = np.array([1.0, 0.0, 0.0])

    def connect(self) -> bool:
        """tmrl ortamına bağlan."""
        from telemetry_monitor import TrackmaniaInterface

        self._interface = TrackmaniaInterface()
        connected = self._interface.connect()

        if connected and self._trajectory_path:
            from progress_tracker import ProgressTracker
            self._tracker = ProgressTracker(self._trajectory_path)

        return connected

    def reset(self) -> Optional[np.ndarray]:
        """Episode'u sıfırla."""
        if self._interface is None:
            return None

        obs = self._interface.reset()
        if obs is None:
            return None

        self._controller.reset()
        self._reward_system.start_episode()
        if self._tracker:
            self._tracker.reset()

        self._episode_active = True
        self._last_obs = obs
        return obs

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float, bool, bool, dict]:
        """
        Bir adım ilerle.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        if not self._episode_active or self._interface is None:
            return None, 0.0, True, False, {}

        # tmrl step
        obs, env_reward, terminated, truncated, info = self._interface.step(action)
        if obs is None:
            return None, 0.0, True, False, info

        # Telemetri çıkar
        frame = self._interface.parse_observation(obs, action)
        position = np.array([frame.x, frame.y, frame.z])

        # Progress hesapla
        progress = 0.0
        base_reward = 0.0
        if self._tracker:
            base_reward = self._tracker.update(frame.x, frame.y, frame.z)
            progress = self._tracker.progress_pct

        # Başarısızlık kontrolü
        failure_info = FailureInfo(is_failed=False)
        if self._enable_failure_detection:
            failure_info = self._controller.check_failure(
                speed_kmh=frame.speed_kmh,
                position=position,
                progress=progress,
                forward_direction=self._forward_direction,
            )

        # Ödül hesapla
        is_done = terminated or truncated or failure_info.is_failed
        total_reward, reward_breakdown = self._reward_system.calculate_reward(
            progress=progress,
            base_reward=base_reward,
            is_done=is_done,
            failure_reason=failure_info.reason if failure_info.is_failed else "",
        )

        # Episode bitti mi?
        if is_done:
            self._episode_active = False
            episode_result = self._reward_system.end_episode(
                max_progress=progress if self._tracker else 0.0,
                total_reward=total_reward,
                failure_reason=failure_info.reason,
            )
            info["episode_result"] = episode_result

        # Info zenginleştir
        info.update({
            "progress": progress,
            "reward_breakdown": reward_breakdown,
            "failure": failure_info.__dict__ if failure_info.is_failed else None,
            "personal_best": self._reward_system.personal_best,
            "current_threshold": self._reward_system.current_threshold,
        })

        self._last_obs = obs
        return obs, total_reward, is_done, truncated, info

    def close(self):
        """Kaynakları temizle."""
        if self._interface:
            self._interface.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from rich.console import Console
    from rich.table import Table

    console = Console()

    parser = argparse.ArgumentParser(description="AI Driving Logic Demo")
    parser.add_argument(
        "--trajectory",
        default=None,
        help="Path to reference trajectory CSV",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    args = parser.parse_args()

    console.print("[bold cyan]AI Driving Logic - Demo[/]")
    console.print("=" * 50)

    # Sistem durumunu göster
    reward_system = AdaptiveRewardSystem()

    table = Table(title="Adaptive Reward System Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Personal Best", f"{reward_system.personal_best:.1f}%")
    table.add_row("Current Threshold", f"{reward_system.current_threshold:.1f}%")
    table.add_row("Total Episodes", str(reward_system.episode_count))

    stats = reward_system.recent_stats
    if stats:
        table.add_row("Avg Progress (last 10)", f"{stats['avg_progress']:.1f}%")
        table.add_row("Avg Reward (last 10)", f"{stats['avg_reward']:.1f}")

    console.print(table)

    console.print("\n[yellow]Kullanim:[/]")
    console.print("  from ai_driving_logic import TrackmaniaRLEnvironment")
    console.print("  env = TrackmaniaRLEnvironment(trajectory_path='reference.csv')")
    console.print("  env.connect()")
    console.print("  obs = env.reset()")
    console.print("  obs, reward, done, _, info = env.step(action)")
