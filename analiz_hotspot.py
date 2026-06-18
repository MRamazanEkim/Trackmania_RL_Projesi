"""
analiz_hotspot.py
=================
"Araç nerede takılıyor?" analizi — DB'deki episode bitişlerini kullanır.

Her değerlendirme episode'u episodes tablosuna (progress_pct + failure_reason ile)
yazılıyor. Bu araç:
  1. Episode'ların NEREDE bittiğini %'lik dilimlere göre histogramlar.
  2. Her dilimde hangi sebebin baskın olduğunu (OFF_TRACK / STUCK / ZERO_SPEED /
     TIMEOUT / LAP) gösterir.
  3. O parkur dilimi GERÇEKTEN viraj mı (trajectory eğriliği) — geometriyle çapraz
     doğrular. Böylece "araç %20'de takılıyor, orası viraj" hipotezini test ederiz.

Kullanım:
  python analiz_hotspot.py                 # en son episode'lu koşuyu analiz et
  python analiz_hotspot.py --run 11        # belirli run
  python analiz_hotspot.py --bins 20       # dilim sayısı
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _corner_factors_for_run(db_path: str, run_id: int, bins: int):
    """O koşunun trajectory'sinden, her %'lik dilim için ortalama viraj faktörü."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT trajectory_path FROM training_runs WHERE id=?", (run_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    traj = row[0]
    # işlenmiş referans dosyasını dene
    cand = [traj]
    p = Path(traj)
    cand.append(str(p.parent / (p.stem + "_reference.csv")))
    path = next((c for c in cand if Path(c).exists()), None)
    if path is None:
        return None
    try:
        from progress_tracker import ProgressTracker
        tr = ProgressTracker(path)
        cf = tr._corner_factor
        n = len(cf)
        # her dilim için ortalama corner_factor
        out = []
        for b in range(bins):
            lo = int(n * b / bins)
            hi = max(lo + 1, int(n * (b + 1) / bins))
            out.append(float(np.mean(cf[lo:hi])))
        return out
    except Exception as e:  # noqa: BLE001
        print(f"(viraj haritası yüklenemedi: {e})")
        return None


def main():
    ap = argparse.ArgumentParser(description="Episode bitiş hotspot analizi")
    ap.add_argument("--db", default="experience.db")
    ap.add_argument("--run", type=int, default=None, help="run_id (varsayılan: episode'lu son koşu)")
    ap.add_argument("--bins", type=int, default=20, help="parkur %'lik dilim sayısı")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"DB yok: {args.db}")
        return

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    run_id = args.run
    if run_id is None:
        r = conn.execute(
            "SELECT run_id, COUNT(*) c FROM episodes GROUP BY run_id ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        if not r:
            print("episodes tablosunda kayıt yok — önce yeni logging'li bir eğitim koş.")
            conn.close()
            return
        run_id = r["run_id"]

    rows = conn.execute(
        "SELECT progress_pct, failure_reason FROM episodes WHERE run_id=?", (run_id,)
    ).fetchall()
    conn.close()

    if not rows:
        print(f"run{run_id} için episode kaydı yok.")
        return

    bins = max(4, args.bins)
    step = 100.0 / bins
    # dilim x sebep sayacı
    reasons = sorted({(r["failure_reason"] or "TIMEOUT") for r in rows})
    counts = {b: {} for b in range(bins)}
    bin_total = [0] * bins
    for r in rows:
        pct = max(0.0, min(99.999, float(r["progress_pct"])))
        b = int(pct / step)
        reason = r["failure_reason"] or "TIMEOUT"
        counts[b][reason] = counts[b].get(reason, 0) + 1
        bin_total[b] += 1

    cf_bins = _corner_factors_for_run(args.db, run_id, bins)
    total = len(rows)

    print(f"\n=== HOTSPOT ANALİZİ — run{run_id}  ({total} episode) ===")
    print(f"Her satır {step:.0f}%'lik bir parkur dilimi. '#' = o dilimde biten episode sayısı.\n")
    maxbar = max(bin_total) or 1
    for b in range(bins):
        lo, hi = b * step, (b + 1) * step
        bar = "#" * int(round(40 * bin_total[b] / maxbar))
        # baskın sebep
        dom = max(counts[b].items(), key=lambda kv: kv[1])[0] if counts[b] else "-"
        cf = cf_bins[b] if cf_bins else None
        zone = ""
        if cf is not None:
            zone = "  VİRAJ" if cf >= 0.5 else ("  ~viraj" if cf >= 0.2 else "  düzlük")
        print(f"  %{lo:3.0f}-{hi:3.0f} | {bar:<40} {bin_total[b]:4d}  baskın:{dom:<10}{zone}")

    # en çok ölünen dilim
    worst = int(np.argmax(bin_total))
    print(f"\n→ EN ÇOK takılınan dilim: %{worst*step:.0f}-{(worst+1)*step:.0f} "
          f"({bin_total[worst]} episode, baskın sebep: "
          f"{max(counts[worst].items(), key=lambda kv: kv[1])[0] if counts[worst] else '-'})")
    if cf_bins:
        wc = cf_bins[worst]
        print(f"  Bu dilim viraj faktörü={wc:.2f} → "
              f"{'EVET viraj (hipotez doğrulandı)' if wc >= 0.5 else 'viraj değil gibi'}")
    print("\nSebep dağılımı (tüm parkur):")
    allr = {}
    for r in rows:
        k = r["failure_reason"] or "TIMEOUT"
        allr[k] = allr.get(k, 0) + 1
    for k, v in sorted(allr.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12} {v:4d}  (%{100*v/total:.0f})")


if __name__ == "__main__":
    main()
