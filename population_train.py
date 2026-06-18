"""
Trackmania RL — Popülasyon + Elitizm Eğitimi
=============================================
"Her jenerasyonda iyileşme" mekanizması.

Fikir
-----
Trackmania tek oyun penceresinde aynı anda tek araç koşturabilir. Bu yüzden
"birden fazla araç" SIRAYLA değerlendirilir: her jenerasyonda N aday model,
oyunda sırayla birkaç tur atar, puanlanır. En iyi aday SAKLANIR (elit) ve
sonraki neslin ebeveyni olur.

Elitizm garantisi
-----------------
Her neslin kazananı `best.zip` olarak korunur. Sonraki nesil:
  • candidate 0 = elitin değişmemiş kopyası (best'in kendisi)
  • candidate 1..N-1 = elitten türeyip kısa SAC eğitimi + keşifle farklaşan kopyalar
Değerlendirme sonunda yeni en iyi, TÜM nesiller arası en iyiyle kıyaslanır
(ExperienceStore.best_overall). Bu yüzden best_progress nesil boyunca asla
azalmaz — "her jenerasyon ≥ önceki" garantisi buradan gelir.

Aday üretimi
------------
Her aday best'ten başlar, `--learn-steps` kadar SAC (veya AsyncSAC) ile öğrenir.
SAC'in entropy tabanlı keşfi adayları doğal olarak farklılaştırır; ayrıca
isteğe bağlı ağırlık gürültüsü (--mutation-std) ekstra çeşitlilik verir.
Sonra `--eval-episodes` kadar değerlendirme turu atılıp puanlanır.

Kullanım
--------
  python population_train.py --trajectory ..._reference.csv
  python population_train.py --trajectory ... --pop-size 4 --generations 10 \
      --learn-steps 3000 --eval-episodes 2 --async --dashboard

  # Devam et (mevcut best'ten):
  python population_train.py --trajectory ... --resume checkpoints/pop/best.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

import numpy as np
import torch as th
from stable_baselines3 import SAC

from ai_driving_logic import TrackmaniaRLEnvironment
from async_sac import AsyncSAC
from db.experience_store import ExperienceStore


# ══════════════════════════════════════════════════════════════════════════════
# Mutasyon — model ağırlıklarına Gaussian gürültü
# ══════════════════════════════════════════════════════════════════════════════

def mutate(model, std: float) -> None:
    """
    Actor ağ ağırlıklarına std'lik Gaussian gürültü ekle (yerinde).
    std=0 → mutasyon yok (sadece SAC öğrenmesiyle farklılaşma).
    """
    if std <= 0:
        return
    with th.no_grad():
        for p in model.policy.actor.parameters():
            p.add_(th.randn_like(p) * std)


def _evaluate(model, env, n_episodes: int, dashboard=None) -> dict:
    """
    Saf değerlendirme: gradient/öğrenme YOK, warm-up rastgeleliği YOK.
    Modeli n_episodes tur sürer, ödül + ilerleme + tur-tamamlama toplar.
    SAC stochastic policy → deterministic=True ile en iyi (kararlı) sürüş.
    """
    rewards, progress, laps = [], [], []
    details = []   # her episode için: bitiş sebebi + metrikler (failure analizi için)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1
            last_info = info
            done = terminated or truncated
            if dashboard is not None:
                dashboard.tick_external(env)
        rewards.append(ep_reward)
        progress.append(float(last_info.get("progress_pct", 0.0)))
        laps.append(bool(last_info.get("lap_complete", False)))
        # Episode neden bitti? truncated (süre/tmrl) ise failure_reason="" → "TIMEOUT/CLEAN"
        reason = str(last_info.get("failure_reason", "")) or ("LAP" if last_info.get("lap_complete") else "TIMEOUT")
        details.append({
            "reward":   ep_reward,
            "steps":    steps,
            "furthest": int(last_info.get("waypoint_idx", 0)),
            "progress": float(last_info.get("progress_pct", 0.0)),
            "lap":      bool(last_info.get("lap_complete", False)),
            "reason":   reason,
        })

    if not rewards:
        return {"mean_reward": 0.0, "best_reward": 0.0, "mean_progress": 0.0,
                "best_progress": 0.0, "eval_episodes": 0, "lap": False, "details": []}
    return {
        "mean_reward":   float(np.mean(rewards)),
        "best_reward":   float(np.max(rewards)),
        "mean_progress": float(np.mean(progress)),
        "best_progress": float(np.max(progress)),
        "eval_episodes": len(rewards),
        "lap":           any(laps),
        "details":       details,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Popülasyon eğitimi
# ══════════════════════════════════════════════════════════════════════════════

def population_train(args: argparse.Namespace):
    pop_dir = Path(args.checkpoint_dir)
    pop_dir.mkdir(parents=True, exist_ok=True)
    best_path = pop_dir / "best.zip"

    # ── Ortam (tek bağlantı, tüm adaylar paylaşır) ───────────────────────────
    env = TrackmaniaRLEnvironment(
        trajectory_path=args.trajectory,
        wp_spacing=args.wp_spacing,
        failure_detection=True,
        adaptive_zones=getattr(args, "adaptive_zones", True),
    )
    print("Trackmania'ya bağlanılıyor…")
    env.reset()
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")

    # ── DB ───────────────────────────────────────────────────────────────────
    store = ExperienceStore(args.db_path)
    run_id = store.create_run(
        trajectory_path = args.trajectory,
        checkpoint_dir  = args.checkpoint_dir,
        total_timesteps = args.generations * args.pop_size * args.learn_steps,
        resume_path     = args.resume,
        notes           = f"population pop={args.pop_size} gen={args.generations}",
    )
    print(f"DB: {args.db_path}  (run_id={run_id})")

    ModelCls = AsyncSAC if args.async_train else SAC
    extra_kwargs = {"max_grad_per_step": args.max_grad_per_step} if args.async_train else {}

    def new_model():
        return ModelCls(
            policy="MlpPolicy", env=env, verbose=0,
            tensorboard_log=args.log_dir,
            learning_rate=3e-4, buffer_size=100_000, learning_starts=1_000,
            batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
            gradient_steps=4, ent_coef="auto", **extra_kwargs,
        )

    # ── Dashboard (opsiyonel) ────────────────────────────────────────────────
    dashboard = None
    dash_cb = None
    if args.dashboard:
        from ui_dashboard import DashboardCallback
        dashboard = DashboardCallback(draw_every=args.dashboard_every, verbose=0)
        dash_cb = dashboard.sb3_callback

    # ── Başlangıç eliti ──────────────────────────────────────────────────────
    # resume verildiyse ya da best.zip varsa ondan başla; yoksa sıfırdan model.
    elite = new_model()
    seed_path = args.resume or (str(best_path) if best_path.exists() else None)
    if seed_path and Path(seed_path).exists():
        print(f"Başlangıç eliti yükleniyor: {seed_path}")
        elite = ModelCls.load(seed_path, env=env, **extra_kwargs)
        # Tüm önceki koşuların en iyisini göster — zincirin nereden devam ettiği net olsun.
        prev_global = store.best_overall_global()
        if prev_global:
            print(f"  ↳ Geçmiş en iyi (tüm koşular): "
                  f"ilerleme={prev_global['best_progress']:.1f}%  "
                  f"ödül={prev_global['best_reward']:.1f}  "
                  f"tur={'Evet' if prev_global['lap_completed'] else 'Hayır'}")
        trend = store.global_generation_trend()
        if trend:
            t_str = "  ".join(
                f"run{r['run_id']}/g{r['gen_no']}:{r['gen_best_progress']:.1f}%"
                for r in trend[-6:]  # son 6 nesil
            )
            print(f"  ↳ Geçmiş nesil trendi: {t_str}")
    else:
        print("Başlangıç eliti: sıfırdan rastgele model.")
    elite.save(str(best_path))

    # ── Jenerasyon döngüsü ───────────────────────────────────────────────────
    ep_counter = 0   # episodes tablosu için artan episode sayacı (tüm koşu boyunca)
    try:
        for gen in range(args.generations):
            # Jenerasyon başında mevcut en iyiyi göster → "öncekinden öğreniyorum" zinciri
            _cur_best = store.best_overall(run_id)
            _cur_prog = f"{_cur_best['best_progress']:.1f}%" if _cur_best else "—"
            print(f"\n{'='*60}")
            print(f"JENERASYON {gen}  (pop={args.pop_size})")
            print(f"  Başlangıç eliti: {_cur_prog} ilerleme  "
                  f"— aday {1}-{args.pop_size - 1} bu modelden öğrenir")
            print(f"{'='*60}")
            gen_rows = []   # (db_row_id, score_tuple, model_path)

            for cand in range(args.pop_size):
                # Aday modeli: cand 0 = elit kopyası (değişmez), diğerleri mutasyonlu
                model = ModelCls.load(str(best_path), env=env, **extra_kwargs)
                is_elite_clone = (cand == 0)
                if not is_elite_clone:
                    mutate(model, args.mutation_std)

                t0 = time.time()

                # 1) Öğrenme fazı (elit kopyası öğrenmez → saf elit performansı korunur)
                steps = 0 if is_elite_clone else args.learn_steps
                if steps > 0:
                    model.learn(total_timesteps=steps,
                                callback=([dash_cb] if dash_cb else None),
                                reset_num_timesteps=True)

                # 2) Değerlendirme fazı — saf rollout, gradient YOK, warm-up YOK.
                #    Adayı puanlamak için eval_episodes kadar tur at.
                s = _evaluate(model, env, args.eval_episodes, dashboard)
                dt = time.time() - t0
                cand_path = pop_dir / f"gen{gen}_cand{cand}.zip"
                model.save(str(cand_path))

                row_id = store.log_candidate(
                    run_id=run_id, gen_no=gen,
                    candidate_idx=(-1 if is_elite_clone else cand),
                    mean_reward=s["mean_reward"], best_reward=s["best_reward"],
                    mean_progress=s["mean_progress"], best_progress=s["best_progress"],
                    eval_episodes=s["eval_episodes"], lap_completed=s["lap"],
                    model_path=str(cand_path),
                    parent_gen=(gen - 1 if gen > 0 else None),
                )

                # Her değerlendirme episode'unu episodes tablosuna yaz → episode'ların
                # NEDEN bittiğini (OFF_TRACK / STUCK / ZERO_SPEED / TIMEOUT / LAP)
                # sonradan sorgulayabilmek için. failure_reason analizi buradan gelir.
                for d in s.get("details", []):
                    ep_counter += 1
                    store.log_episode(
                        run_id=run_id, episode_number=ep_counter, global_step=ep_counter,
                        cumulative_reward=d["reward"], steps=d["steps"],
                        furthest_waypoint=d["furthest"], progress_pct=d["progress"],
                        lap_complete=d["lap"], failure_reason=d["reason"],
                    )
                reasons = Counter(d["reason"] for d in s.get("details", []))
                reason_str = " ".join(f"{k}×{v}" for k, v in reasons.most_common())

                # Skor: önce tur, sonra ilerleme, sonra ödül
                score = (int(s["lap"]), s["best_progress"], s["best_reward"])
                gen_rows.append((row_id, score, cand_path))

                tag = "ELİT" if is_elite_clone else f"aday{cand}"
                print(f"  [{tag}] ilerleme={s['best_progress']:5.1f}%  "
                      f"ödül={s['best_reward']:8.1f}  ep={s['eval_episodes']}  "
                      f"{'LAP!' if s['lap'] else ''}  [{reason_str}]  ({dt:.0f}s)")

            # ── Neslin kazananı ──────────────────────────────────────────────
            gen_rows.sort(key=lambda r: r[1], reverse=True)
            win_row_id, win_score, win_path = gen_rows[0]
            store.mark_best(win_row_id)

            # Elitizm: best.zip = TÜM nesiller arası en iyi (yeni kazanan dahil).
            # best_overall yeni kazananı da içerir; daha kötüyse önceki en iyi korunur
            # → best.zip asla geriye gitmez.
            global_best = store.best_overall(run_id)
            shutil.copyfile(global_best["model_path"], str(best_path))
            print(f"\n  → Nesil {gen} kazananı: ilerleme={win_score[1]:.1f}% ödül={win_score[2]:.1f}")
            print(f"  → best.zip = global en iyi (gen{global_best['gen_no']}, "
                  f"ilerleme={global_best['best_progress']:.1f}%, ödül={global_best['best_reward']:.1f})")

            # İlerleme özeti (her nesilde iyileşme görünür)
            summ = store.generation_summary(run_id)
            trend = "  ".join(f"g{r['gen_no']}:{r['gen_best_progress']:.1f}%" for r in summ)
            print(f"  İlerleme trendi: {trend}")

    except KeyboardInterrupt:
        print("\nPopülasyon eğitimi kullanıcı tarafından durduruldu.")

    finally:
        if dashboard is not None:
            dashboard.close()
        bo = store.best_overall(run_id)
        if bo:
            print(f"\nEn iyi model: {bo['model_path']}  "
                  f"(ilerleme={bo['best_progress']:.1f}%, ödül={bo['best_reward']:.1f})")
            print(f"best.zip: {best_path}")
        store.close()
        env.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trackmania RL — Popülasyon + Elitizm eğitimi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trajectory", required=True, metavar="CSV",
                   help="Referans waypoint CSV")
    p.add_argument("--pop-size", type=int, default=4,
                   help="Jenerasyon başına aday sayısı (1'i elit kopyası)")
    p.add_argument("--generations", type=int, default=10,
                   help="Jenerasyon sayısı")
    p.add_argument("--learn-steps", type=int, default=3_000,
                   help="Her adayın elitten türeyip öğreneceği adım sayısı")
    p.add_argument("--eval-episodes", type=int, default=2,
                   help="Her adayın puanlanacağı değerlendirme turu sayısı")
    p.add_argument("--mutation-std", type=float, default=0.0,
                   help="Aday ağırlıklarına eklenen Gaussian gürültü std'si "
                        "(0 → sadece SAC keşfiyle farklılaşma)")
    p.add_argument("--resume", default=None, metavar="ZIP",
                   help="Başlangıç eliti olarak yüklenecek checkpoint")
    p.add_argument("--checkpoint-dir", default="checkpoints/pop",
                   help="Popülasyon checkpoint klasörü (best.zip burada)")
    p.add_argument("--db-path", default="experience.db",
                   help="SQLite veritabanı yolu")
    p.add_argument("--log-dir", default="logs/tb",
                   help="TensorBoard log klasörü")
    p.add_argument("--wp-spacing", type=float, default=1.0,
                   help="Waypoint aralığı (metre)")
    p.add_argument("--async", dest="async_train", action="store_true",
                   help="AsyncSAC kullan (eşzamanlı toplama+öğrenme)")
    p.add_argument("--max-grad-per-step", type=float, default=4.0,
                   help="Async modda gradient/adım oranı tavanı")
    p.add_argument("--no-adaptive-zones", dest="adaptive_zones",
                   action="store_false", default=True,
                   help="Viraj/düzlük bölgesine göre uyarlanan dinamik ceza sistemini kapat "
                        "(varsayılan: açık)")
    p.add_argument("--dashboard", action="store_true",
                   help="Canlı pygame dashboard")
    p.add_argument("--dashboard-every", type=int, default=1,
                   help="Dashboard'u her N adımda çiz")
    return p.parse_args()


if __name__ == "__main__":
    population_train(_parse_args())
