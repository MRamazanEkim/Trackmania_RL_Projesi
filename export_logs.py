"""
export_logs.py — DB → metin log yedeği (yalnız localde)
=======================================================
experience.db kaybolma/bozulma ihtimaline karşı, içeriğini insan-okur ve
kolay incelenebilir metin dosyalarına aktarır. Çıktılar training_logs/ altında
ve .gitignore'da → REPOYA COMMITLENMEZ, sadece local yedek.

Üretilenler (training_logs/)
----------------------------
  runs.csv      — her training_run bir satır (algoritma, tarih, bütçe)
  episodes.csv  — her episode bir satır (ödül, ilerleme, çarpışma, tur süresi)
  generations.csv — popülasyon adayları (varsa)
  SUMMARY.md    — algoritma karşılaştırma tablosu (insan okur)

Kullanım
--------
  python export_logs.py                      # experience.db → training_logs/
  python export_logs.py --db other.db --out yedek/

Otomatik: compare_train.py / population_train.py / start.py eğitim sonunda
çağırabilir (opsiyonel). Manuel de çalıştırılabilir.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _dump_table(conn: sqlite3.Connection, table: str, out_path: Path) -> int:
    """Bir tabloyu CSV'ye yaz. Satır sayısını döner."""
    cur = conn.execute(f"SELECT * FROM {table} ORDER BY id")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    return len(rows)


def _write_summary(conn: sqlite3.Connection, out_path: Path) -> None:
    """Algoritma karşılaştırma tablosunu Markdown olarak yaz."""
    rows = conn.execute(
        """
        SELECT algorithm,
               COUNT(*)                                          AS episodes,
               AVG(lap_complete)                                 AS completion_rate,
               AVG(progress_pct)                                 AS mean_progress,
               MAX(progress_pct)                                 AS best_progress,
               AVG(collision)                                    AS collision_rate,
               AVG(CASE WHEN lap_complete=1 THEN lap_time_s END) AS mean_lap_time,
               AVG(cumulative_reward)                            AS mean_reward
        FROM episodes
        GROUP BY algorithm
        ORDER BY completion_rate DESC, best_progress DESC
        """
    ).fetchall()

    from datetime import datetime
    lines = [
        "# Eğitim Özeti — Algoritma Karşılaştırması",
        "",
        f"_Üretim: {datetime.now().isoformat(timespec='seconds')} — experience.db yedeği_",
        "",
        "| Algo | Episode | Tamamlama | Ort.İlerleme | En İyi | Çarpışma | Ort.Tur(s) | Ort.Ödül |",
        "|------|--------:|----------:|-------------:|-------:|---------:|-----------:|---------:|",
    ]
    for r in rows:
        algo, eps, comp, mprog, bprog, coll, lapt, mrew = r
        lapt_s = f"{lapt:.1f}" if lapt is not None else "—"
        lines.append(
            f"| {algo} | {eps} | {comp*100:.1f}% | {mprog:.1f}% | "
            f"{bprog:.1f}% | {coll*100:.1f}% | {lapt_s} | {mrew:.1f} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export(db_path: str, out_dir: str) -> None:
    db = Path(db_path)
    if not db.exists():
        print(f"DB bulunamadı: {db}")
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db))
    try:
        n_runs = _dump_table(conn, "training_runs", out / "runs.csv")
        n_eps = _dump_table(conn, "episodes", out / "episodes.csv")
        # generations tablosu olmayabilir (eski DB) → varsa yaz
        try:
            n_gen = _dump_table(conn, "generations", out / "generations.csv")
        except sqlite3.OperationalError:
            n_gen = 0
        _write_summary(conn, out / "SUMMARY.md")
    finally:
        conn.close()

    print(f"Log yedeği yazıldı → {out}/")
    print(f"  runs.csv        : {n_runs} koşu")
    print(f"  episodes.csv    : {n_eps} episode")
    print(f"  generations.csv : {n_gen} aday")
    print(f"  SUMMARY.md      : algoritma karşılaştırma tablosu")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="experience.db → metin log yedeği (local)")
    p.add_argument("--db", default="experience.db", help="Kaynak SQLite veritabanı")
    p.add_argument("--out", default="training_logs", help="Çıktı klasörü (gitignore'da)")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    export(a.db, a.out)
