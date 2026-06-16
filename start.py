"""
start.py
========
Trackmania RL — tek-komut başlatıcı (sürekli SAC eğitimi).

    python start.py                # tam otomatik: en iyiden devam, sürekli döngü
    python start.py --steps 10000  # tur başına adım (her tur sonunda kaydeder)
    python start.py --no-async     # AsyncSAC yerine standart SAC
    python start.py --no-dashboard # UI penceresi açmadan sadece eğit

  Otomatik döngü: her tur (--steps adım) bitince kaydedip yeni tura geçer,
  Ctrl+C ile durdurana kadar sürekli öğrenir. Gece bırakılabilir; her turda
  kayıt alındığı için durdurunca ilerleme korunur.

Ne yapar?
---------
1. trajectory_logs/ içinden referans CSV otomatik seçilir.
2. En iyi SAC modeli bulunur (checkpoints/sac_continue/best.zip → yoksa
   checkpoints/sac/sac_final.zip) ve ondan DEVAM edilir — baştan başlamaz.
3. Varsa deneyim hafızası (replay buffer) yüklenir.
4. Pygame dashboard + SAC eğitimi aynı anda çalışır.
5. Bitince (Ctrl+C veya --steps) model + replay buffer
   checkpoints/sac_continue/best.zip'e yazılır.

Sürekli iyileşme
----------------
  • Her çalıştırma öncekinin bıraktığı yerden sürer → öğrenme birikir.
  • İlerleme zinciri: %12 → %25 → ... → %100, sonra tur süresini kısaltma.
  • Sen de bir başkası da aynı 'python start.py' ile aynı modeli ilerletir.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

# Hangi klasörden çalıştırılırsa çalıştırılsın (ör. üst klasörden
# `python Trackmania_RL_Projesi/start.py`), tüm göreli yollar (trajectory_logs,
# experience.db, checkpoints, logs) script'in bulunduğu proje klasörüne göre
# çözülsün diye çalışma dizinini buraya sabitle.
_PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(_PROJECT_DIR)


def _ensure_venv() -> None:
    """
    Bağımlılıklar (stable_baselines3 vb.) projenin venv'inde kurulu. Script global
    Python ile başlatılırsa (ör. VS Code), kendini otomatik venv Python'ı ile
    yeniden çalıştır — böylece tek komut her ortamda çalışır.
    """
    if "_TMRL_VENV_REEXEC" in os.environ:
        return  # zaten yeniden başlatıldık → sonsuz döngüyü önle

    if sys.platform == "win32":
        venv_py = _PROJECT_DIR / "venv" / "Scripts" / "python.exe"
    else:
        venv_py = _PROJECT_DIR / "venv" / "bin" / "python"

    if not venv_py.exists():
        return  # venv yok → mevcut Python ile devam (kullanıcı kendi kurmuş olabilir)

    # Zaten venv Python'ı ile mi çalışıyoruz?
    try:
        same = Path(sys.executable).resolve() == venv_py.resolve()
    except OSError:
        same = False
    if same:
        return

    # Bağımlılık zaten mevcut Python'da varsa yeniden başlatmaya gerek yok.
    import importlib.util
    if importlib.util.find_spec("stable_baselines3") is not None:
        return

    print(f"[start] Bağımlılıklar venv'de — venv Python'ı ile yeniden başlatılıyor:\n"
          f"        {venv_py}\n", flush=True)  # execv buffer'ı temizler → flush şart
    os.environ["_TMRL_VENV_REEXEC"] = "1"
    os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])


_ensure_venv()

# ── Varsayılanlar ─────────────────────────────────────────────────────────────
_DEFAULT_DB       = "experience.db"
_DEFAULT_CHKDIR   = "checkpoints/pop"
# Sürekli SAC ajanının kalıcı modeli burada tutulur (best.zip + replay buffer).
# Her start.py bitişinde güncellenir → sonraki başlatış buradan devam eder.
CONTINUE_DIR      = "checkpoints/sac_continue"
_DEFAULT_LOGDIR   = "logs/tb"
_DEFAULT_WP_SPC   = 1.0


# ── Yardımcı: trajectory CSV seçimi ──────────────────────────────────────────

def _find_trajectory() -> str | None:
    """
    trajectory_logs/ içinden en uygun referans CSV'yi döner.
    Öncelik sırası:
      1. *_reference*.csv  — işlenmiş, waypoint atanmış (boyuta göre büyükten küçüğe)
      2. reference_trajectory.csv
      3. Herhangi büyük CSV (>10 KB)
    """
    tdir = Path("trajectory_logs")
    if not tdir.exists():
        return None

    refs = sorted(
        [p for p in tdir.glob("*_reference*.csv") if p.stat().st_size > 1000],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if refs:
        return str(refs[0])

    rt = tdir / "reference_trajectory.csv"
    if rt.exists() and rt.stat().st_size > 1000:
        return str(rt)

    big = sorted(
        [p for p in tdir.glob("*.csv") if p.stat().st_size > 10_000],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return str(big[0]) if big else None


# ── Yardımcı: DB'den geçmiş okuma ────────────────────────────────────────────

def _db_history(db_path: str) -> tuple[list[dict], dict | None]:
    """
    DB'den tüm koşu özetini ve global en iyi aday modelini döner.
    Returns: (run_summary_list, global_best_candidate_or_None)
    """
    if not Path(db_path).exists():
        return [], None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Koşu özeti episodes tablosundan (SAC-continue episodes'a yazar).
        runs = conn.execute("""
            SELECT r.id,
                   r.started_at,
                   r.notes,
                   r.algorithm,
                   COUNT(e.id)              AS gen_count,
                   MAX(e.progress_pct)      AS best_progress,
                   MAX(e.cumulative_reward) AS best_reward,
                   MAX(e.lap_complete)      AS any_lap
            FROM   training_runs r
            LEFT JOIN episodes e ON e.run_id = r.id
            GROUP  BY r.id
            ORDER  BY r.id DESC
        """).fetchall()
        run_list = [dict(r) for r in runs]

        # Tüm koşuların en iyi ilerlemesi (banner ★ satırı için).
        row = conn.execute("""
            SELECT MAX(progress_pct) AS best_progress,
                   MAX(cumulative_reward) AS best_reward,
                   MAX(lap_complete) AS lap_completed
            FROM   episodes
        """).fetchone()
        best_cand = dict(row) if row and row["best_progress"] is not None else None
    except sqlite3.Error:
        run_list, best_cand = [], None

    conn.close()
    return run_list, best_cand


# ── Yardımcı: resume checkpoint seçimi ───────────────────────────────────────

def _find_resume(best_cand: dict | None, chk_dir: str) -> str | None:
    """
    Devam edilecek SAC modelini döner — "her başlatışta en iyiden devam".
    Öncelik:
      1. checkpoints/sac_continue/best.zip → sürekli SAC ajanının kalıcı modeli
         (start.py her bitişte buraya yazar → sonraki başlatış buradan sürer)
      2. checkpoints/sac/sac_final.zip → ilk SAC compare_train modeli (%12 başlangıcı)
      3. checkpoints/ altındaki herhangi bir .zip (son değiştirilenler önce)
    """
    # 1. sürekli SAC ajanı (asıl devam noktası)
    cont_best = Path(CONTINUE_DIR) / "best.zip"
    if cont_best.exists():
        return str(cont_best)

    # 2. ilk SAC modeli (compare_train'den, %12 başlangıç)
    sac_final = Path("checkpoints/sac/sac_final.zip")
    if sac_final.exists():
        return str(sac_final)

    # 3. herhangi bir .zip (son değiştirilenler önce)
    zips = sorted(
        Path("checkpoints").rglob("*.zip") if Path("checkpoints").exists() else [],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(zips[0]) if zips else None


# ── Banner: başlamadan önce geçmişi özetle ───────────────────────────────────

def _print_banner(
    run_list:  list[dict],
    best_cand: dict | None,
    resume:    str | None,
    traj:      str | None,
    args:      argparse.Namespace,
) -> None:
    W = 68
    print("=" * W)
    print("  TRACKMANIA RL — Başlatıcı")
    print("=" * W)

    # Önceki koşular
    print()
    if run_list:
        print(f"  Önceki eğitim koşuları ({len(run_list)} adet):")
        for r in run_list[:8]:
            prog = f"{r['best_progress']:.1f}%" if r["best_progress"] is not None else "—"
            gen  = r["gen_count"] or 0
            lap  = " ✓ TURLADΙ" if r["any_lap"] else ""
            alg  = f"[{r['algorithm']}]" if r["algorithm"] else ""
            print(f"    run#{r['id']:3d}  {alg:<6}  gen:{gen:<3}  "
                  f"en-iyi:{prog:>7}{lap}  ({r['started_at'][:10]})")
    else:
        print("  DB'de önceki kayıt yok — sıfırdan başlıyor.")

    # Global en iyi
    if best_cand:
        print()
        print(f"  ★ Global en iyi (DB):  ilerleme={best_cand['best_progress']:.1f}%  "
              f"ödül={best_cand['best_reward']:.1f}  "
              f"tur={'Evet' if best_cand['lap_completed'] else 'Hayır'}")

    print()
    print(f"  Checkpoint  : {resume or '(yok — sıfırdan başlıyor)'}")
    print(f"  Trajectory  : {traj or '(BULUNAMADI!)'}")
    print()
    print(f"  Ayarlar     : steps={args.steps:,}  "
          f"async={'Evet' if args.async_train else 'Hayır'}  "
          f"dashboard={'Evet' if args.dashboard else 'Hayır'}")
    print()
    print(f"  Çalışma mantığı: Tek SAC ajanı en iyi modelden DEVAM eder; baştan")
    print(f"  başlamaz. Öğrenmesi birikir (%12 → ... → %100), sonra tur süresini")
    print(f"  kısaltmak için optimizasyon. Ctrl+C ile durdur — ilerleme kaydedilir.")
    print("=" * W)
    print()


# ── Sürekli SAC eğitimi ──────────────────────────────────────────────────────

def sac_continue(args) -> None:
    """
    Tek, sürekli iyileşen SAC ajanı. Her start.py çalıştırışında:
      • en iyi modelden (args.resume) DEVAM eder — baştan başlamaz,
      • varsa deneyim hafızasını (replay buffer) yükler,
      • kaldığı yerden öğrenir (Ctrl+C veya --steps'e kadar),
      • bitince modeli + replay buffer'ı CONTINUE_DIR/best.zip'e yazar
        → sonraki başlatış buradan sürer. Böylece %12 → %25 → ... → %100
        ve %100 sonrası tur-süresi optimizasyonu kesintisiz birikir.
    """
    import time
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from ai_driving_logic import TrackmaniaRLEnvironment
    from async_sac import AsyncSAC
    from db.experience_store import ExperienceStore
    from db.callbacks import EpisodeLoggerCallback

    cont_dir = Path(CONTINUE_DIR)
    cont_dir.mkdir(parents=True, exist_ok=True)
    best_path = cont_dir / "best.zip"
    replay_path = cont_dir / "best_replay.pkl"

    # ── Ortam ────────────────────────────────────────────────────────────────
    env = TrackmaniaRLEnvironment(
        trajectory_path=args.trajectory,
        wp_spacing=args.wp_spacing,
        failure_detection=True,
    )
    print("Trackmania'ya bağlanılıyor…")
    env.reset()

    ModelCls = AsyncSAC if args.async_train else SAC
    extra = {"max_grad_per_step": args.max_grad_per_step} if args.async_train else {}

    # ── DB ───────────────────────────────────────────────────────────────────
    store = ExperienceStore(args.db_path)
    run_id = store.create_run(
        trajectory_path=args.trajectory, checkpoint_dir=str(cont_dir),
        total_timesteps=args.steps, resume_path=args.resume,
        algorithm="sac", notes="sac_continue (sürekli devam)",
    )

    # ── Model: en iyiden DEVAM ─────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        print(f"Devam ediliyor: {args.resume}")
        model = ModelCls.load(args.resume, env=env, tensorboard_log=args.log_dir, **extra)
        # Deneyim hafızası varsa yükle (off-policy SAC için sağlıklı devam)
        if replay_path.exists():
            try:
                model.load_replay_buffer(str(replay_path.with_suffix("")))
                print(f"Deneyim hafızası yüklendi: {model.replay_buffer.size():,} adım")
            except Exception as exc:  # noqa: BLE001
                print(f"Replay buffer yüklenemedi (yok sayıldı): {exc}")
    else:
        print("Sıfırdan yeni SAC modeli (devam edilecek model bulunamadı).")
        model = ModelCls(
            policy="MlpPolicy", env=env, verbose=0, tensorboard_log=args.log_dir,
            learning_rate=3e-4, buffer_size=200_000, learning_starts=2_000,
            batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
            gradient_steps=4, ent_coef="auto", **extra,
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────
    callbacks = [
        EpisodeLoggerCallback(store, run_id, verbose=1),
        CheckpointCallback(save_freq=args.save_freq, save_path=str(cont_dir),
                           name_prefix="sac_step", verbose=0),
    ]
    dashboard = None
    if args.dashboard:
        from ui_dashboard import DashboardCallback
        dashboard = DashboardCallback(draw_every=args.dashboard_every, verbose=0)
        callbacks.append(dashboard.sb3_callback)

    # ── Kaydetme yardımcısı (her tur sonunda çağrılır → ilerleme korunur) ───────
    def _save():
        model.save(str(best_path.with_suffix("")))
        try:
            model.save_replay_buffer(str(replay_path.with_suffix("")))
        except Exception as exc:  # noqa: BLE001
            print(f"Replay buffer kaydedilemedi (yok sayıldı): {exc}")

    # ── Eğit — otomatik döngü ───────────────────────────────────────────────────
    # Her tur args.steps kadar öğrenir, kaydeder, sonra yeni tura geçer.
    # Sen Ctrl+C ile durdurana kadar sürekli döner; her tur sonunda kayıt
    # alındığı için durdurduğunda ilerleme korunur (gece bırakılabilir).
    print(f"\nOtomatik döngü başladı (tur başına {args.steps:,} adım). "
          f"Ctrl+C ile durdur — ilerleme her turda kaydedilir.")
    t0 = time.time()
    round_no = 0
    try:
        while True:
            round_no += 1
            print(f"\n── Tur {round_no} (toplam adım: {model.num_timesteps:,}) ──")
            model.learn(total_timesteps=args.steps, callback=callbacks,
                        reset_num_timesteps=False)
            _save()
            best = store.best_progress(run_id)
            best_pct = best["progress_pct"] if best else 0.0
            print(f"  Tur {round_no} bitti, kaydedildi. En iyi ilerleme: {best_pct:.1f}%")
    except KeyboardInterrupt:
        print("\nDöngü durduruldu (Ctrl+C). Son durum kaydediliyor…")
        _save()
    dt = time.time() - t0

    if dashboard is not None:
        dashboard.close()

    best = store.best_progress(run_id)
    best_pct = best["progress_pct"] if best else 0.0
    print(f"\nBitti ({dt:.0f}s, {round_no} tur). En iyi ilerleme: {best_pct:.1f}%")
    print(f"Model kaydedildi: {best_path}  → bir sonraki 'python start.py' buradan devam eder.")

    store.close()
    env.close()
    # DB kaybolma ihtimaline karşı local metin log yedeği
    try:
        from export_logs import export
        export(args.db_path, "training_logs")
    except Exception as exc:  # noqa: BLE001
        print(f"Log yedeği alınamadı (yok sayıldı): {exc}")


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Trackmania RL — tek-komut başlatıcı",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trajectory",
                   default=None, metavar="CSV",
                   help="Referans waypoint CSV (otomatik bulunur)")
    p.add_argument("--steps",
                   type=int, default=10_000,
                   help="Tur başına adım sayısı; her tur sonunda kaydedip döngüye devam eder "
                        "(Ctrl+C ile durdur)")
    p.add_argument("--save-freq",
                   type=int, default=5_000,
                   help="Ara checkpoint kayıt sıklığı (adım)")
    p.add_argument("--resume",
                   default=None, metavar="ZIP",
                   help="Devam edilecek model (otomatik bulunur — en iyi SAC modeli)")
    p.add_argument("--checkpoint-dir",
                   default=_DEFAULT_CHKDIR,
                   help="(kullanılmıyor; sürekli SAC modeli checkpoints/sac_continue/ altında)")
    p.add_argument("--db-path",
                   default=_DEFAULT_DB,
                   help="SQLite veritabanı yolu")
    p.add_argument("--log-dir",
                   default=_DEFAULT_LOGDIR,
                   help="TensorBoard log klasörü")
    p.add_argument("--wp-spacing",
                   type=float, default=_DEFAULT_WP_SPC,
                   help="Waypoint interpolasyon aralığı (metre)")
    p.add_argument("--no-async",
                   dest="async_train", action="store_false", default=True,
                   help="AsyncSAC yerine standart SAC kullan")
    p.add_argument("--max-grad-per-step",
                   type=float, default=4.0,
                   help="Async modda collector/gradient adım oranı tavanı")
    p.add_argument("--no-dashboard",
                   dest="dashboard", action="store_false", default=True,
                   help="Pygame dashboard penceresini açma")
    p.add_argument("--dashboard-every",
                   type=int, default=1,
                   help="Dashboard'u her N adımda bir çiz (1=her adım)")
    args = p.parse_args()

    # ── Trajectory ────────────────────────────────────────────────────────────
    if not args.trajectory:
        args.trajectory = _find_trajectory()
    if not args.trajectory:
        print("[HATA] Referans trajectory CSV bulunamadı.")
        print("       Önce 'python record_trajectory.py --process' çalıştır.")
        sys.exit(1)

    # ── DB geçmişi ────────────────────────────────────────────────────────────
    run_list, best_cand = _db_history(args.db_path)

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    # Kullanıcı açıkça vermediyse otomatik bul.
    # Bu sayede her start.py çalıştırması önceki neslin en iyi modelinden devam eder.
    if not args.resume:
        args.resume = _find_resume(best_cand, args.checkpoint_dir)

    # ── Özet banner ───────────────────────────────────────────────────────────
    _print_banner(run_list, best_cand, args.resume, args.trajectory, args)

    # ── Sürekli SAC eğitimine devret ──────────────────────────────────────────
    # Tek SAC ajanı: en iyiden devam eder, öğrenmesi birikir. Dashboard
    # sac_continue içinde oluşturulur (--dashboard varsayılan açık).
    sac_continue(args)


if __name__ == "__main__":
    main()
