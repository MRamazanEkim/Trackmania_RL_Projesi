"""
compare_train.py — DQN / PPO / SAC Karşılaştırmalı Eğitim
========================================================
Projenin özgün katkısı: aynı Trackmania ortamında (EŞİT KOŞUL) üç RL
algoritmasını ayrı ayrı eğitip karşılaştırmak.

Eşit koşul
----------
Üç algoritma da AYNI:
  • ortam (TrackmaniaRLEnvironment) ve ödül fonksiyonu,
  • observation (tmrl LIDAR),
  • eğitim adımı bütçesi (--timesteps),
  • metrikler (tur süresi, çarpışma oranı, pist tamamlama oranı, ilerleme, ödül)
ile değerlendirilir. Sadece ALGORİTMA değişir.

Aksiyon uzayı
-------------
  • SAC, PPO → sürekli [gaz, fren, direksiyon] (env'in doğal uzayı)
  • DQN      → ayrık; DiscreteActionWrapper ile 5 aksiyona indirgenir
    (DQN sürekli aksiyon çalıştıramaz)

Kullanım
--------
  python compare_train.py --algo sac --trajectory ..._reference.csv --async
  python compare_train.py --algo ppo --trajectory ..._reference.csv
  python compare_train.py --algo dqn --trajectory ..._reference.csv

  # Üçünü sırayla aynı bütçeyle eğit (karşılaştırma için):
  python compare_train.py --algo all --trajectory ..._reference.csv --timesteps 50000

  # Sonuç tablosu:
  python compare_train.py --report

  # Popülasyon+elitizm (yalnız sac/ppo):
  python compare_train.py --algo sac --trajectory ... --population
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

from stable_baselines3 import SAC, PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from ai_driving_logic import TrackmaniaRLEnvironment
from async_sac import AsyncSAC
from discrete_action import DiscreteActionWrapper
from db.experience_store import ExperienceStore

# tmrl fizik adımı (config.json RTGYM_CONFIG.time_step_duration) → tur süresi hesabı
TIME_STEP_DURATION = 0.05  # saniye/adım (20 Hz)

# Çarpışma sayılan başarısızlık sebepleri (env DrivingController/OFF_TRACK)
COLLISION_REASONS = {"STUCK", "OFF_TRACK", "ZERO_SPEED"}


# ══════════════════════════════════════════════════════════════════════════════
# Episode logger — eşit metriklerle DB'ye yazar
# ══════════════════════════════════════════════════════════════════════════════

class AlgoEpisodeLogger(BaseCallback):
    """
    Her episode bitişinde DB'ye bir satır yazar; algoritma adı + tur süresi +
    çarpışma bayrağı dahil. Üç algoritma için aynı metrikler → adil karşılaştırma.
    """

    def __init__(self, store: ExperienceStore, run_id: int, algorithm: str, verbose: int = 1):
        super().__init__(verbose)
        self._store = store
        self._run_id = run_id
        self._algo = algorithm
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_number = 0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_steps += 1
        if bool(self.locals["dones"][0]):
            self._ep_number += 1
            info = self.locals["infos"][0]
            reason = info.get("failure_reason", "")
            lap = bool(info.get("lap_complete", False))
            self._store.log_episode(
                run_id=self._run_id,
                episode_number=self._ep_number,
                global_step=self.num_timesteps,
                cumulative_reward=self._ep_reward,
                steps=self._ep_steps,
                furthest_waypoint=info.get("waypoint_idx", 0),
                progress_pct=info.get("progress_pct", 0.0),
                lap_complete=lap,
                failure_reason=reason,
                algorithm=self._algo,
                lap_time_s=(self._ep_steps * TIME_STEP_DURATION if lap else 0.0),
                collision=(reason in COLLISION_REASONS),
            )
            if self.verbose:
                tag = "LAP!" if lap else reason
                print(f"  [{self._algo} ep{self._ep_number:4d}] adım={self.num_timesteps:7d} "
                      f"ödül={self._ep_reward:8.1f} ilerleme={info.get('progress_pct',0.0):5.1f}% {tag}")
            self._ep_reward = 0.0
            self._ep_steps = 0
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Ortam + model fabrikası
# ══════════════════════════════════════════════════════════════════════════════

def build_env(args, algorithm: str):
    env = TrackmaniaRLEnvironment(
        trajectory_path=args.trajectory,
        wp_spacing=args.wp_spacing,
        failure_detection=True,
    )
    print("Trackmania'ya bağlanılıyor…")
    env.reset()
    # DQN ayrık aksiyon ister → wrapper
    if algorithm == "dqn":
        env = DiscreteActionWrapper(env)
        print("DQN için DiscreteActionWrapper uygulandı (5 ayrık aksiyon).")
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    return env


def build_model(algorithm: str, env, args):
    """Eşit koşul: ortak hiperparametreler; sadece algoritma sınıfı değişir."""
    common = dict(policy="MlpPolicy", env=env, verbose=0,
                  tensorboard_log=args.log_dir, learning_rate=3e-4, gamma=0.99)
    if algorithm == "sac":
        cls = AsyncSAC if args.async_train else SAC
        extra = {"max_grad_per_step": args.max_grad_per_step} if args.async_train else {}
        return cls(buffer_size=200_000, learning_starts=2_000, batch_size=256,
                   tau=0.005, train_freq=1, gradient_steps=4, ent_coef="auto",
                   **common, **extra)
    if algorithm == "ppo":
        return PPO(n_steps=2048, batch_size=64, n_epochs=10,
                   gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, **common)
    if algorithm == "dqn":
        return DQN(buffer_size=200_000, learning_starts=2_000, batch_size=256,
                   tau=1.0, train_freq=4, target_update_interval=1_000,
                   exploration_fraction=0.2, exploration_final_eps=0.05, **common)
    raise ValueError(f"Bilinmeyen algoritma: {algorithm}")


# ══════════════════════════════════════════════════════════════════════════════
# Tek algoritma eğitimi
# ══════════════════════════════════════════════════════════════════════════════

def train_one(algorithm: str, args, store: ExperienceStore):
    print(f"\n{'='*60}\nALGORİTMA: {algorithm.upper()}\n{'='*60}")
    env = build_env(args, algorithm)

    run_id = store.create_run(
        trajectory_path=args.trajectory,
        checkpoint_dir=args.checkpoint_dir,
        total_timesteps=args.timesteps,
        algorithm=algorithm,
        notes=f"compare algo={algorithm}",
    )
    print(f"DB: {args.db_path}  (run_id={run_id}, algo={algorithm})")

    model = build_model(algorithm, env, args)

    ckpt_dir = Path(args.checkpoint_dir) / algorithm
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        AlgoEpisodeLogger(store, run_id, algorithm, verbose=1),
        CheckpointCallback(save_freq=args.save_freq, save_path=str(ckpt_dir),
                           name_prefix=algorithm, verbose=0),
    ]
    if args.dashboard:
        from ui_dashboard import DashboardCallback
        dash = DashboardCallback(draw_every=args.dashboard_every, verbose=0)
        callbacks.append(dash.sb3_callback)

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print(f"\n{algorithm} eğitimi durduruldu.")
    dt = time.time() - t0

    model.save(str(ckpt_dir / f"{algorithm}_final"))
    print(f"{algorithm} bitti ({dt:.0f}s). Model: {ckpt_dir}/{algorithm}_final.zip")
    env.close()


# ══════════════════════════════════════════════════════════════════════════════
# Karşılaştırma raporu
# ══════════════════════════════════════════════════════════════════════════════

def print_report(store: ExperienceStore):
    rows = store.algorithm_comparison()
    if not rows:
        print("Henüz veri yok. Önce eğitim çalıştır.")
        return
    print("\n" + "=" * 78)
    print("ALGORİTMA KARŞILAŞTIRMASI  (eşit koşullarda)")
    print("=" * 78)
    hdr = f"{'Algo':>5} | {'Ep':>5} | {'Tamamlama':>9} | {'Ort.İlerl':>9} | {'En İyi':>7} | {'Çarpışma':>8} | {'Ort.Tur(s)':>10} | {'Ort.Ödül':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lap_t = f"{r['mean_lap_time']:.1f}" if r['mean_lap_time'] is not None else "  —"
        print(f"{r['algorithm']:>5} | {r['episodes']:>5} | "
              f"{r['completion_rate']*100:>8.1f}% | {r['mean_progress']:>8.1f}% | "
              f"{r['best_progress']:>6.1f}% | {r['collision_rate']*100:>7.1f}% | "
              f"{lap_t:>10} | {r['mean_reward']:>9.1f}")
    print("=" * 78)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    store = ExperienceStore(args.db_path)

    if args.report:
        print_report(store)
        store.close()
        return

    # Popülasyon opsiyonu (yalnız sürekli aksiyonlu: sac/ppo) → population_train'e devret
    if args.population:
        if args.algo == "dqn":
            print("Hata: --population DQN ile kullanılamaz (ayrık aksiyon). sac/ppo seç.")
            store.close(); sys.exit(1)
        store.close()
        import population_train
        pop_args = argparse.Namespace(
            trajectory=args.trajectory, pop_size=args.pop_size, generations=args.generations,
            learn_steps=args.timesteps, eval_episodes=2, mutation_std=0.0, resume=args.resume,
            checkpoint_dir=str(Path(args.checkpoint_dir) / "pop"), db_path=args.db_path,
            log_dir=args.log_dir, wp_spacing=args.wp_spacing, async_train=args.async_train,
            max_grad_per_step=args.max_grad_per_step, dashboard=args.dashboard,
            dashboard_every=args.dashboard_every,
        )
        population_train.population_train(pop_args)
        return

    algos = ["dqn", "ppo", "sac"] if args.algo == "all" else [args.algo]
    for algo in algos:
        train_one(algo, args, store)

    print_report(store)
    store.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DQN/PPO/SAC karşılaştırmalı eğitim (eşit koşullar)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo", choices=["dqn", "ppo", "sac", "all"], default="sac",
                   help="Eğitilecek algoritma ('all' = üçü sırayla, aynı bütçeyle)")
    p.add_argument("--trajectory", metavar="CSV",
                   help="Referans waypoint CSV (--report dışında zorunlu)")
    p.add_argument("--timesteps", type=int, default=50_000,
                   help="Algoritma başına eğitim adımı (eşit koşul için aynı tut)")
    p.add_argument("--report", action="store_true",
                   help="Sadece karşılaştırma tablosunu yazdır (eğitim yok)")
    p.add_argument("--population", action="store_true",
                   help="Popülasyon+elitizm modu (yalnız sac/ppo)")
    p.add_argument("--pop-size", type=int, default=4, help="Popülasyon aday sayısı")
    p.add_argument("--generations", type=int, default=10, help="Jenerasyon sayısı")
    p.add_argument("--resume", default=None, metavar="ZIP", help="Devam edilecek checkpoint")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint kök klasörü")
    p.add_argument("--db-path", default="experience.db", help="SQLite veritabanı")
    p.add_argument("--log-dir", default="logs/tb", help="TensorBoard log klasörü")
    p.add_argument("--save-freq", type=int, default=5_000, help="Checkpoint sıklığı")
    p.add_argument("--wp-spacing", type=float, default=1.0, help="Waypoint aralığı (m)")
    p.add_argument("--async", dest="async_train", action="store_true",
                   help="SAC için AsyncSAC (eşzamanlı toplama+öğrenme)")
    p.add_argument("--max-grad-per-step", type=float, default=4.0,
                   help="Async modda gradient/adım oranı tavanı")
    p.add_argument("--dashboard", action="store_true", help="Canlı pygame dashboard")
    p.add_argument("--dashboard-every", type=int, default=1, help="Dashboard çizim sıklığı")
    args = p.parse_args()
    if not args.report and not args.trajectory:
        p.error("--trajectory gerekli (--report hariç)")
    return args


if __name__ == "__main__":
    main(_parse_args())
