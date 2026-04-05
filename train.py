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
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env

from ai_driving_logic import TrackmaniaRLEnvironment


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

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="sac_tm",
        verbose=1,
    )

    callbacks = [checkpoint_cb]

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Hata: checkpoint bulunamadı: {resume_path}")
            sys.exit(1)
        print(f"Checkpoint yükleniyor: {resume_path}")
        model = SAC.load(str(resume_path), env=env, verbose=1)
        reset_timesteps = False
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=args.log_dir,
            # SAC hiperparametreleri — sonraki aşamada özelleştirilebilir
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,   # İlk N step random aksiyon (keşif)
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
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
        "--timesteps", type=int, default=100_000,
        help="Toplam eğitim adımı",
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
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
