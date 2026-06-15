"""
Trackmania RL — Training Script
================================
Stable-Baselines3 SAC ile RL ajanını eğitir.

Kullanım
--------
  # İlk eğitim:
  python train.py --trajectory reference.csv

  # Checkpoint'ten devam:
  python train.py --trajectory reference.csv --resume checkpoints/sac_tm_50000_steps.zip

  # TensorBoard ile izleme (ayrı terminalde):
  tensorboard --logdir logs/tb

Tüm seçenekler:
  python train.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Windows cp1254 konsol unicode'u (…, →) encode edemez → stdout'u utf-8'e çevir.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env

from ai_driving_logic import TrackmaniaRLEnvironment
from async_sac import AsyncSAC
from db.experience_store import ExperienceStore
from db.callbacks import EpisodeLoggerCallback, ReplayBufferCheckpointCallback


# ══════════════════════════════════════════════════════════════════════════════
# Stop condition: süre dolunca VEYA araç parkuru tamamlayınca eğitimi bitir
# ══════════════════════════════════════════════════════════════════════════════

class StopOnTimeOrCompletion(BaseCallback):
    """
    Eğitimi şu durumlarda erken durdurur:
      • Araç parkuru tamamlarsa (info['lap_complete'] == True), VEYA
      • Belirtilen süre (saat) dolduysa.

    Args
    ----
    max_hours        : Maksimum eğitim süresi (saat). 0 → süre sınırı yok.
    stop_on_finish   : Araç bitişe ulaşınca dur.
    """

    def __init__(self, max_hours: float = 7.0, stop_on_finish: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self._max_seconds = max_hours * 3600.0 if max_hours and max_hours > 0 else None
        self._stop_on_finish = stop_on_finish
        self._start_time = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Tamamlama kontrolü
        if self._stop_on_finish:
            for info in self.locals.get("infos", []):
                if info.get("lap_complete"):
                    if self.verbose:
                        print(f"\n🏁 Araç parkuru tamamladı! ({self.num_timesteps:,} adım) — eğitim durduruluyor.")
                    return False
        # Süre kontrolü
        if self._max_seconds is not None:
            elapsed = time.time() - self._start_time
            if elapsed >= self._max_seconds:
                if self.verbose:
                    print(f"\n⏰ Süre doldu ({elapsed/3600:.2f} saat, {self.num_timesteps:,} adım) — eğitim durduruluyor.")
                return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Environment factory
# ══════════════════════════════════════════════════════════════════════════════

def make_env(trajectory: str, wp_spacing: float, verify: bool = False) -> TrackmaniaRLEnvironment:
    """
    TrackmaniaRLEnvironment oluştur ve bağlantı kur.

    Args
    ----
    trajectory : Referans waypoint CSV yolu
    wp_spacing : Waypoint aralığı (metre)
    verify     : gymnasium.check_env() çalıştır (geliştirme sırasında faydalı)
    """
    env = TrackmaniaRLEnvironment(
        trajectory_path=trajectory,
        wp_spacing=wp_spacing,
        failure_detection=True,
    )

    # Spaces'ı başlatmak için reset() çağır
    # (tmrl bağlantısı burada kurulur)
    print("Trackmania'ya bağlanılıyor…")
    env.reset()
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")

    if verify:
        print("Ortam doğrulanıyor (check_env)…")
        check_env(env, warn=True)
        print("Ortam doğrulandı.")

    return env


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace):
    # ── Ortam ────────────────────────────────────────────────────────────────
    env = make_env(
        trajectory=args.trajectory,
        wp_spacing=args.wp_spacing,
        verify=args.verify_env,
    )

    # ── Deneyim Veritabanı ───────────────────────────────────────────────────
    # (merge sonrası kaybolmuştu; episode_logger_cb store/run_id'ye bağımlı)
    store  = ExperienceStore(args.db_path)
    run_id = store.create_run(
        trajectory_path = args.trajectory,
        checkpoint_dir  = args.checkpoint_dir,
        total_timesteps = args.timesteps,
        resume_path     = args.resume,
    )
    print(f"DB: {args.db_path}  (run_id={run_id})")

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="sac_tm",
        verbose=1,
    )
    episode_logger_cb = EpisodeLoggerCallback(store, run_id, verbose=1)
    replay_cb = ReplayBufferCheckpointCallback(
        save_freq   = args.save_freq,
        save_path   = checkpoint_dir,
        name_prefix = "sac_tm",
        verbose     = 1,
    )

    callbacks = [checkpoint_cb, episode_logger_cb, replay_cb]

    # Canlı dashboard — ajanın aksiyon/telemetri/LIDAR'ını eğitim sırasında göster.
    # Ayrı tmrl bağlantısı açmaz; eğitim akışındaki veriyi kullanır.
    dashboard = None
    if args.dashboard:
        from ui_dashboard import DashboardCallback
        dashboard = DashboardCallback(draw_every=args.dashboard_every, verbose=1)
        callbacks.append(dashboard.sb3_callback)

    # Süre/tamamlama ile erken durdurma
    if args.max_hours > 0 or args.stop_on_finish:
        callbacks.append(
            StopOnTimeOrCompletion(
                max_hours=args.max_hours,
                stop_on_finish=args.stop_on_finish,
                verbose=1,
            )
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Async modda toplama (oyun) ve öğrenme (gradient) ayrı thread'lerde
    # örtüşür → oyun beklerken CPU boşta kalmaz. tmrl'in worker/trainer
    # paralelizminin tek-süreç taklidi. Bkz. async_sac.py.
    ModelCls = AsyncSAC if args.async_train else SAC
    extra_kwargs = {"max_grad_per_step": args.max_grad_per_step} if args.async_train else {}

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Hata: checkpoint bulunamadı: {resume_path}")
            sys.exit(1)
        print(f"Checkpoint yükleniyor: {resume_path}" + ("  [async]" if args.async_train else ""))
        model = ModelCls.load(str(resume_path), env=env, verbose=1, **extra_kwargs)
        reset_timesteps = False
    else:
        model = ModelCls(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=args.log_dir,
            # SAC hiperparametreleri — hızlı/sample-efficient öğrenme için ayarlı
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=2_000,   # İlk N step random aksiyon (keşif) — buffer'ı ileri sürüş örnekleriyle doldur. Oyun yavaş olduğu için erken öğrenmeye başla.
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=4,        # her gerçek adımda 4 güncelleme → aynı oyun süresinde daha çok öğren (CPU'da güncelleme yavaş ama oyun darboğaz)
            ent_coef="auto",         # entropy katsayısı otomatik → otomatik keşif dengesi
            **extra_kwargs,
        )
        reset_timesteps = True

    # ── Eğitim ───────────────────────────────────────────────────────────────
    print(f"\nEğitim başlıyor: {args.timesteps:,} adım")
    print(f"Checkpoint: {checkpoint_dir}/ (her {args.save_freq:,} adımda)")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_timesteps,
        )
    except KeyboardInterrupt:
        print("\nEğitim kullanıcı tarafından durduruldu.")

    # ── Kaydet ───────────────────────────────────────────────────────────────
    final_path = checkpoint_dir / "sac_tm_final"
    model.save(str(final_path))
    print(f"\nModel kaydedildi: {final_path}.zip")
    print(f"Devam etmek için: python train.py --trajectory {args.trajectory} --resume {final_path}.zip")

    if dashboard is not None:
        dashboard.close()
    store.close()
    env.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trackmania RL — SAC eğitimi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--trajectory", required=True, metavar="CSV",
        help="Referans waypoint CSV dosyası (record_trajectory.py ile oluşturulur)",
    )
    p.add_argument(
        "--timesteps", type=int, default=2_000_000,
        help="Maks toplam eğitim adımı (genelde süre/tamamlama önce durdurur)",
    )
    p.add_argument(
        "--max-hours", type=float, default=7.0,
        help="Maksimum eğitim süresi (saat). 0 → süre sınırı yok",
    )
    p.add_argument(
        "--stop-on-finish", action="store_true", default=True,
        help="Araç parkuru tamamlayınca eğitimi durdur",
    )
    p.add_argument(
        "--no-stop-on-finish", dest="stop_on_finish", action="store_false",
        help="Araç tamamlasa bile eğitime devam et",
    )
    p.add_argument(
        "--resume", default=None, metavar="ZIP",
        help="Devam edilecek checkpoint .zip dosyası",
    )
    p.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Checkpoint kayıt klasörü",
    )
    p.add_argument(
        "--db-path", default="experience.db",
        help="Deneyim/episode kayıt veritabanı (SQLite) yolu",
    )
    p.add_argument(
        "--log-dir", default="logs/tb",
        help="TensorBoard log klasörü",
    )
    p.add_argument(
        "--save-freq", type=int, default=5_000,
        help="Checkpoint kayıt sıklığı (adım)",
    )
    p.add_argument(
        "--wp-spacing", type=float, default=1.0,
        help="Waypoint aralığı, metre (ham trajectory işlenirken)",
    )
    p.add_argument(
        "--verify-env", action="store_true",
        help="Başlamadan önce gymnasium.check_env() çalıştır",
    )
    p.add_argument(
        "--async", dest="async_train", action="store_true",
        help="Eşzamanlı toplama+öğrenme (AsyncSAC). Oyun beklerken CPU öğrenir "
             "→ aynı duvar-saatinde daha çok gradient. tmrl worker/trainer taklidi.",
    )
    p.add_argument(
        "--max-grad-per-step", type=float, default=4.0,
        help="Async modda: gerçek oyun adımı başına maks gradient adımı oranı "
             "(tmrl MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP). Bayat-veri aşırı uyumunu sınırlar",
    )
    p.add_argument(
        "--dashboard", action="store_true",
        help="Eğitim sırasında canlı pygame dashboard'u aç (aksiyon/telemetri/LIDAR). "
             "Ayrı tmrl bağlantısı açmaz; eğitim akışındaki veriyi gösterir",
    )
    p.add_argument(
        "--dashboard-every", type=int, default=1,
        help="Dashboard'u her N adımda bir çiz (1=her adım). CPU yükü için artırılabilir",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
