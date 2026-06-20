"""
stagnation_monitor.py
======================
Uzun (örn. 24 saat) popülasyon eğitimlerinde "tıkanma" (stagnation) tespiti +
otomatik ödül/ceza ayarı.

Sorun
-----
Elitizm garantisi sayesinde `best_progress` nesil boyunca asla azalmaz; ama
azalmaması "ilerliyor" demek değildir — saatlerce AYNI değerde takılı kalabilir
(plato). Araç her seferinde aynı yerde aynı şekilde başarısız oluyordur
(hep OFF_TRACK, hep STUCK, hep TIMEOUT…). O zaman daha fazla nesil beklemek
zaman kaybıdır; ödül sinyalini değiştirmek gerekir.

Bu modül üç parçadan oluşur
---------------------------
  StagnationMonitor  – nesil-başına en iyi ilerlemeyi ve başarısızlık
                       nedenlerini izler, plato olup olmadığına karar verir.
  analyze_window     – plato penceresindeki başarısızlık nedenlerinin dağılımını
                       ve baskın nedeni çıkarır (teşhis).
  RewardAutoTuner    – baskın başarısızlık nedenine göre hangi ödül/ceza
                       parametresinin nasıl değişeceğine kural-tabanlı karar verir
                       ve ortama (TrackmaniaRLEnvironment) canlı uygular.

Ayrıca write_stagnation_report() teşhisi markdown olarak `analysis/` altına yazar.

Kullanım (population_train.py içinde)
-------------------------------------
  monitor = StagnationMonitor(patience=5, min_delta=0.5)
  tuner   = RewardAutoTuner()
  ...
  # her nesil sonunda:
  monitor.record(gen_no, gen_best_progress, gen_reason_counter)
  if monitor.is_stagnant():
      diag = monitor.diagnose()
      write_stagnation_report(diag, env.reward_params(), out_dir="analysis")
      if mode == "adapt":
          deltas  = tuner.suggest(diag)
          changes = env.apply_reward_adjustment(**deltas)
          monitor.note_adaptation(changes)   # platoyu sıfırla, devam et
      elif mode == "stop":
          break
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Teşhis veri sınıfı
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StagnationDiagnosis:
    """Bir plato anının teşhisi."""
    gen_no: int                       # tıkanmanın tespit edildiği nesil
    plateau_value: float              # platonun takıldığı en iyi ilerleme (%)
    plateau_len: int                  # kaç nesildir iyileşme yok
    reason_counts: dict               # {failure_reason: adet} (pencere boyunca)
    dominant_reason: str              # en sık görülen başarısızlık nedeni
    dominant_share: float             # baskın nedenin oranı (0..1)
    total_episodes: int               # penceredeki toplam değerlendirme episode'u


# ══════════════════════════════════════════════════════════════════════════════
# Stagnation Monitor
# ══════════════════════════════════════════════════════════════════════════════

class StagnationMonitor:
    """
    Nesil-başına en iyi ilerlemeyi izler; `patience` nesil boyunca iyileşme
    `min_delta`'dan küçükse "tıkandı" der.

    Args
    ----
    patience  : Kaç nesil iyileşme olmazsa plato sayılacağı.
    min_delta : "İyileşme" sayılması için en iyi ilerlemenin (%) artması gereken
                en küçük miktar. Gürültüye karşı eşik (örn. 35.0 → 35.2 iyileşme
                DEĞİLDİR).
    window    : Teşhis için son kaç neslin başarısızlık nedenlerine bakılacağı.
                Verilmezse `patience` kullanılır.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.5,
                 window: Optional[int] = None):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.window = int(window) if window else int(patience)

        self._best_so_far: float = float("-inf")
        self._gens_since_improve: int = 0
        self._last_gen: int = -1
        # son nesillerin (gen_no, best_progress, reason_counter) geçmişi
        self._history: list[tuple[int, float, Counter]] = []

    # ── Kayıt ───────────────────────────────────────────────────────────────

    def record(self, gen_no: int, best_progress: float,
               reason_counts: Counter | dict | None = None) -> None:
        """Bir neslin sonucunu kaydet. reason_counts: o neslin failure nedenleri."""
        rc = Counter(reason_counts) if reason_counts else Counter()
        self._history.append((gen_no, float(best_progress), rc))
        self._last_gen = gen_no

        if best_progress > self._best_so_far + self.min_delta:
            self._best_so_far = float(best_progress)
            self._gens_since_improve = 0
        else:
            self._best_so_far = max(self._best_so_far, float(best_progress))
            self._gens_since_improve += 1

    def is_stagnant(self) -> bool:
        """patience nesildir anlamlı iyileşme yoksa True."""
        return self._gens_since_improve >= self.patience

    @property
    def gens_since_improve(self) -> int:
        return self._gens_since_improve

    @property
    def best_so_far(self) -> float:
        return self._best_so_far

    # ── Teşhis ──────────────────────────────────────────────────────────────

    def diagnose(self) -> StagnationDiagnosis:
        """Plato penceresindeki başarısızlık nedenlerini özetler."""
        window_hist = self._history[-self.window:]
        agg = analyze_window(window_hist)
        return StagnationDiagnosis(
            gen_no=self._last_gen,
            plateau_value=self._best_so_far,
            plateau_len=self._gens_since_improve,
            **agg,
        )

    def note_adaptation(self, changes: dict | None = None) -> None:
        """
        Ödül ayarı uygulandıktan sonra çağrılır: plato sayacını sıfırlar ki
        yeni parametrelere bir `patience` penceresi daha şans verilsin.
        Geçmişi de temizler (eski nedenler yeni ödülü teşhis ederken yanıltmasın).
        """
        self._gens_since_improve = 0
        self._history.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Başarısızlık analizi
# ══════════════════════════════════════════════════════════════════════════════

def analyze_window(history: list[tuple[int, float, Counter]]) -> dict:
    """
    (gen_no, best_progress, reason_counter) listesinden başarısızlık nedeni
    dağılımını çıkarır. dominant_reason = en sık görülen neden.
    """
    total = Counter()
    for _gen, _prog, rc in history:
        total.update(rc)
    n = sum(total.values())
    if n == 0:
        return {
            "reason_counts": {},
            "dominant_reason": "UNKNOWN",
            "dominant_share": 0.0,
            "total_episodes": 0,
        }
    dom, dom_n = total.most_common(1)[0]
    return {
        "reason_counts": dict(total),
        "dominant_reason": dom,
        "dominant_share": dom_n / n,
        "total_episodes": n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Reward Auto-Tuner — baskın başarısızlık nedeni → ödül/ceza değişikliği
# ══════════════════════════════════════════════════════════════════════════════

class RewardAutoTuner:
    """
    Baskın başarısızlık nedenine göre hangi ödül/ceza parametresini nasıl
    değiştireceğine kural-tabanlı karar verir.

    Çarpanlar TrackmaniaRLEnvironment.apply_reward_adjustment()'a verilir; orada
    sınırlar (clamp) içinde uygulanır → otomatik ayar kontrolden çıkamaz.

    Kural mantığı
    -------------
    TIMEOUT     Araç çarpmıyor ama ilerlemiyor — düzlükte sürünüp süre doluyor.
                → hızı ve ilerlemeyi daha çok ödüllendir/zorla:
                  speed_reward_coef↑, no_progress_penalty↑, low_speed_penalty↑.
    STUCK /     Araç duruyor / duvara yapışıp kalıyor.
    ZERO_SPEED  → yerinde durmayı daha pahalı yap, hareketi ödüllendir:
                  idle_penalty↑, speed_reward_coef↑.
    OFF_TRACK   Araç parkuru terk ediyor — büyük ihtimalle viraja çok hızlı girip
                savruluyor. → virajda yavaşlamaya daha çok izin ver, ham hız
                teşvikini biraz kıs: corner_relief↑, speed_reward_coef↓,
                crash_penalty↑ (parkurdan çıkmak biraz daha caydırıcı olsun).
    REVERSE     Araç geri gidiyor → ileri hareketi ödüllendir, durmayı cezalandır:
                  speed_reward_coef↑, idle_penalty↑.
    LAP/diğer   Net teşhis yok → ufak genel itki: speed_reward_coef↑.

    Args
    ----
    up   : "Artır" adımının çarpanı (>1).
    down : "Azalt" adımının çarpanı (<1).
    """

    def __init__(self, up: float = 1.40, down: float = 0.75):
        self.up = float(up)
        self.down = float(down)

    def suggest(self, diag: StagnationDiagnosis) -> dict:
        """Teşhisten {param_adı: çarpan} sözlüğü üretir."""
        up, down = self.up, self.down
        r = diag.dominant_reason

        if r == "TIMEOUT":
            return {"speed_reward_coef": up,
                    "no_progress_penalty": up,
                    "low_speed_penalty": up}
        if r in ("STUCK", "ZERO_SPEED"):
            return {"idle_penalty": up,
                    "speed_reward_coef": up}
        if r == "OFF_TRACK":
            return {"corner_relief": up,
                    "speed_reward_coef": down,
                    "crash_penalty": up}
        if r == "REVERSE":
            return {"speed_reward_coef": up,
                    "idle_penalty": up}
        # LAP / UNKNOWN / diğer: çarpmıyor ama ilerlemiyor → hafif hız itkisi
        return {"speed_reward_coef": up}


# ══════════════════════════════════════════════════════════════════════════════
# Markdown rapor
# ══════════════════════════════════════════════════════════════════════════════

def write_stagnation_report(
    diag: StagnationDiagnosis,
    reward_params_before: dict,
    out_dir: str = "analysis",
    changes: Optional[dict] = None,
    suggested: Optional[dict] = None,
) -> Path:
    """
    Teşhisi (+ varsa uygulanan değişiklikleri) markdown olarak yazar ve yolunu döner.
    changes : env.apply_reward_adjustment()'ın döndürdüğü {param: (eski, yeni)}.
    suggested: tuner.suggest()'ın döndürdüğü ham çarpanlar (uygulanmadıysa da yazılır).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).astimezone()
    path = out / f"stagnation_gen{diag.gen_no}_{ts:%Y%m%d_%H%M%S}.md"

    lines: list[str] = []
    lines.append(f"# Tıkanma (Stagnation) Teşhisi — Nesil {diag.gen_no}")
    lines.append("")
    lines.append(f"- **Tarih:** {ts:%Y-%m-%d %H:%M:%S %z}")
    lines.append(f"- **Takılı kalınan en iyi ilerleme:** {diag.plateau_value:.1f}%")
    lines.append(f"- **Plato uzunluğu:** {diag.plateau_len} nesildir iyileşme yok")
    lines.append(f"- **Pencere içi değerlendirme episode'u:** {diag.total_episodes}")
    lines.append("")

    lines.append("## Başarısızlık nedeni dağılımı")
    lines.append("")
    lines.append("| Neden | Adet | Oran |")
    lines.append("|---|---:|---:|")
    total = max(1, diag.total_episodes)
    for reason, cnt in sorted(diag.reason_counts.items(),
                              key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {reason} | {cnt} | {cnt / total * 100:.0f}% |")
    lines.append("")
    lines.append(f"**Baskın neden:** `{diag.dominant_reason}` "
                 f"(%{diag.dominant_share * 100:.0f})")
    lines.append("")

    lines.append("## Ödül/ceza parametreleri (ayar öncesi)")
    lines.append("")
    lines.append("| Parametre | Değer |")
    lines.append("|---|---:|")
    for k, v in reward_params_before.items():
        lines.append(f"| {k} | {v:.3f} |")
    lines.append("")

    if suggested:
        lines.append("## Önerilen değişiklikler (çarpan)")
        lines.append("")
        lines.append("| Parametre | Çarpan |")
        lines.append("|---|---:|")
        for k, v in suggested.items():
            lines.append(f"| {k} | ×{v:.2f} |")
        lines.append("")

    if changes:
        lines.append("## Uygulanan değişiklikler")
        lines.append("")
        lines.append("| Parametre | Eski | Yeni |")
        lines.append("|---|---:|---:|")
        for k, (old, new) in changes.items():
            lines.append(f"| {k} | {old:.3f} | {new:.3f} |")
        lines.append("")
    elif changes is not None:
        lines.append("## Uygulanan değişiklikler")
        lines.append("")
        lines.append("_Sınırlar (clamp) nedeniyle parametre değişmedi — "
                     "ilgili ceza/ödül zaten uç değerinde._")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
