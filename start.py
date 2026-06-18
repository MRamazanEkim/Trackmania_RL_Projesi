"""
start.py
========
Trackmania RL — tek-komut başlatıcı.

    python start.py                          # tam otomatik
    python start.py --generations 20         # jenerasyon sayısını geçersiz kıl
    python start.py --pop-size 3             # popülasyon boyutunu geçersiz kıl
    python start.py --learn-steps 5000       # aday başına öğrenme adımı
    python start.py --no-async               # AsyncSAC olmadan (daha yavaş ama daha kararlı)
    python start.py --no-dashboard           # UI penceresi açmadan sadece eğit

Ne yapar?
---------
1. experience.db okunur → önceki koşuların en iyi modeli ve geçmiş bulunur.
2. En iyi checkpoint (best.zip veya DB'deki/checkpoints'teki en iyi aday)
   --resume olarak kullanılır. Her yeni jenerasyon öncekinin üzerine öğrenir.
3. trajectory_logs/ içinden referans CSV otomatik seçilir (en büyük _reference.csv).
4. Pygame UI dashboard + popülasyon eğitimi aynı anda başlar (DashboardCallback).

Elitizm garantisi
-----------------
  • Her jenerasyon içinde aday 0 = değişmemiş elit kopyası (regresyon yok).
  • Jenerasyon kazananı TÜM nesiller arası en iyiyle kıyaslanır → best.zip asla
    geriye gitmez. Yani: her jenerasyon ≥ önceki.
  • Bu script yeniden çalıştırıldığında best.zip'ten devam → zincir kopmaz.
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
_DEFAULT_LOGDIR   = "logs/tb"
_DEFAULT_POP      = 4
_DEFAULT_GENS     = 10
_DEFAULT_STEPS    = 3_000
_DEFAULT_EVAL_EP  = 2
_DEFAULT_MUT_STD  = 0.0
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
        runs = conn.execute("""
            SELECT r.id,
                   r.started_at,
                   r.notes,
                   r.algorithm,
                   COUNT(DISTINCT g.gen_no)  AS gen_count,
                   MAX(g.best_progress)       AS best_progress,
                   MAX(g.best_reward)         AS best_reward,
                   MAX(g.lap_completed)       AS any_lap
            FROM   training_runs r
            LEFT JOIN generations g ON g.run_id = r.id
            GROUP  BY r.id
            ORDER  BY r.id DESC
        """).fetchall()
        run_list = [dict(r) for r in runs]

        # Global en iyi aday — model_path dosyası gerçekten varlığı kontrol edilir
        cands = conn.execute("""
            SELECT * FROM generations
            WHERE  model_path != ''
            ORDER  BY lap_completed DESC, best_progress DESC, best_reward DESC
        """).fetchall()
        best_cand = None
        for c in cands:
            d = dict(c)
            if Path(d["model_path"]).exists():
                best_cand = d
                break
    except sqlite3.Error:
        run_list, best_cand = [], None

    conn.close()
    return run_list, best_cand


# ── Yardımcı: resume checkpoint seçimi ───────────────────────────────────────

def _find_resume(best_cand: dict | None, chk_dir: str) -> str | None:
    """
    Başlangıç eliti olarak kullanılacak checkpoint'i döner.
    Öncelik:
      1. checkpoints/pop/best.zip   → popülasyon eğitiminin tüm-zamanlar en iyisi
      2. DB'deki global en iyi adayın model_path'i
      3. checkpoints/sac/sac_final.zip → SAC compare_train'den gelen son model
      4. checkpoints/ altındaki herhangi bir .zip (son değiştirilenler önce)
    """
    # 1. population best
    pop_best = Path(chk_dir) / "best.zip"
    if pop_best.exists():
        return str(pop_best)

    # 2. DB global best
    if best_cand:
        p = Path(best_cand["model_path"])
        if p.exists():
            return str(p)

    # 3. SAC final
    sac_final = Path("checkpoints/sac/sac_final.zip")
    if sac_final.exists():
        return str(sac_final)

    # 4. Herhangi bir .zip
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
    print(f"  Ayarlar     : pop={args.pop_size}  gen={args.generations}  "
          f"steps={args.learn_steps}  eval={args.eval_episodes}")
    print(f"                async={'Evet' if args.async_train else 'Hayır'}  "
          f"dashboard={'Evet' if args.dashboard else 'Hayır'}  "
          f"mutation_std={args.mutation_std}")
    print()
    print(f"  Çalışma mantığı: Her jenerasyon önceki neslin en iyi modelinden")
    print(f"  başlar ve SAC öğrenmesiyle ilerler. Elit kopyası ile regresyon engellenir.")
    print("=" * W)
    print()


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Trackmania RL — tek-komut başlatıcı",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trajectory",
                   default=None, metavar="CSV",
                   help="Referans waypoint CSV (otomatik bulunur)")
    p.add_argument("--pop-size",
                   type=int, default=_DEFAULT_POP,
                   help="Jenerasyon başına aday sayısı (1'i daima elit kopyası)")
    p.add_argument("--generations",
                   type=int, default=_DEFAULT_GENS,
                   help="Toplam jenerasyon sayısı")
    p.add_argument("--learn-steps",
                   type=int, default=_DEFAULT_STEPS,
                   help="Her adayın elitten türeyip SAC ile öğreneceği adım sayısı")
    p.add_argument("--eval-episodes",
                   type=int, default=_DEFAULT_EVAL_EP,
                   help="Her adayın puanlanacağı değerlendirme turu sayısı")
    p.add_argument("--mutation-std",
                   type=float, default=_DEFAULT_MUT_STD,
                   help="Aday ağırlıklarına eklenen Gaussian gürültü std (0=kapalı)")
    p.add_argument("--resume",
                   default=None, metavar="ZIP",
                   help="Başlangıç eliti checkpoint (otomatik bulunur)")
    p.add_argument("--checkpoint-dir",
                   default=_DEFAULT_CHKDIR,
                   help="Popülasyon checkpoint klasörü (best.zip burada tutulur)")
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
    p.add_argument("--no-adaptive-zones",
                   dest="adaptive_zones", action="store_false", default=True,
                   help="Viraj/düzlük bölgesine göre uyarlanan dinamik ceza sistemini kapat")
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

    # ── population_train'e devret ─────────────────────────────────────────────
    # DashboardCallback population_train içinde oluşturulur (--dashboard=True).
    # Aynı process — UI + eğitim aynı anda çalışır, ayrı thread gerekmez.
    from population_train import population_train
    population_train(args)


if __name__ == "__main__":
    main()
