"""
ui_dashboard.py
===============
Pygame tabanli gercek zamanli LIDAR radar + telemetri dashboardu.

Kullanim
--------
  python ui_dashboard.py
  python ui_dashboard.py --trajectory trajectory_logs/..._reference.csv
  python ui_dashboard.py --delay 3

Egitimden bagimsiz calisir: tmrl'a baglanir, sifir aksiyon gonderir,
anlık arac verilerini ve LIDAR'i gosterir.
Trackmania acik ve OpenPlanet plugin'leri aktif olmali.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    print("pygame kurulu degil.  Kurmak icin:  pip install pygame")
    sys.exit(1)

# ── LIDAR konfigurasyonu ──────────────────────────────────────────────────────
#
# tmrl TM20LIDAR: 19 isin, range(90, 280, 10) derece (goruntu koordinatlari).
# Goruntu koordinatlari: 0=sag, 90=asagi, 180=sol, 270=yukari (ileri).
# Ekran gosterimi icin 270° = ileri olacak sekilde donusturuluyor:
#   screen_deg = angle - 270   →  [-180, -170, …, 0]
#   Sonra simetri: +/- 90 araligina almak icin:
#   Dizinin ilk yarisi geriye dogru gider; son eleman (270°) tam ileridir.
#
# Pratik gosterim: 19 isini soldan saga yayin, merkez = ileri.
# Gercek fiziksel acilar tmrl kaynak koduna gore yukarda aciklanmistir.
# Gorsel olarak duzeltmek icin LIDAR_FLIP = True/False deneyin.

N_RAYS       = 19
LIDAR_MAX_PX = 60.0        # 64x64 goruntu icin maks. mesafe (piksel)
LIDAR_FLIP   = False       # Ray sirasini tersine cevir (gerekirse True yap)

# Ekrandaki gosterim acilari: soldan saga -90° -> +90°, merkez = ileri
_SCREEN_ANGLES = np.linspace(-90.0, 90.0, N_RAYS)

# ── Pencere boyutlari ─────────────────────────────────────────────────────────
WIN_W, WIN_H = 1120, 680

# Radar parametreleri
RADAR_CX = 290          # radar merkezi x
RADAR_CY = 370          # radar merkezi y
RADAR_R  = 250          # radar yarıcapi (px)

# ── Renk paleti ───────────────────────────────────────────────────────────────
C_BG        = ( 13,  13,  26)   # koyu lacivert arka plan
C_PANEL     = ( 18,  18,  38)
C_RADAR_BG  = (  8,   8,  20)
C_RADAR_RNG = ( 28,  55,  38)   # radar halka/grid
C_GRID      = ( 30,  50,  36)
C_CAR       = (190, 200, 255)   # arac sembol rengi
C_TEXT      = (195, 210, 228)
C_DIM       = ( 80,  92, 112)
C_ACCENT    = ( 45, 195, 140)   # yesil vurgu
C_WHITE     = (255, 255, 255)
C_RED       = (215,  55,  55)
C_ORANGE    = (215, 125,  45)
C_YELLOW    = (215, 195,  45)
C_GREEN     = ( 55, 195,  95)
C_DIVIDER   = ( 38,  42,  68)


# ── Yardimci fonksiyonlar ─────────────────────────────────────────────────────

def ray_color(norm: float) -> tuple:
    """0 = yakin/tehlike (kirmizi) … 1 = uzak/guvenli (yesil)"""
    if norm < 0.20:
        return C_RED
    if norm < 0.45:
        return C_ORANGE
    if norm < 0.70:
        return C_YELLOW
    return C_GREEN


def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_bar(surf, x, y, w, h, val, color, bg=(38, 38, 58)):
    val = max(0.0, min(1.0, val))
    pygame.draw.rect(surf, bg, (x, y, w, h), border_radius=4)
    if val > 0:
        pygame.draw.rect(surf, color, (x, y, int(w * val), h), border_radius=4)


def draw_text(surf, text, font, color, x, y):
    surf.blit(font.render(text, True, color), (x, y))


# ── LIDAR Radar ───────────────────────────────────────────────────────────────

def draw_radar(surf, lidar_norm: np.ndarray):
    # Arka plan
    pygame.draw.circle(surf, C_RADAR_BG, (RADAR_CX, RADAR_CY), RADAR_R)

    # Mesafe halkalari (% 25, 50, 75, 100)
    for frac in (0.25, 0.50, 0.75, 1.00):
        r = int(RADAR_R * frac)
        pygame.draw.circle(surf, C_RADAR_RNG, (RADAR_CX, RADAR_CY), r, 1)

    # Eksen cizgileri
    pygame.draw.line(surf, C_GRID,
                     (RADAR_CX - RADAR_R, RADAR_CY),
                     (RADAR_CX + RADAR_R, RADAR_CY), 1)
    pygame.draw.line(surf, C_GRID,
                     (RADAR_CX, RADAR_CY - RADAR_R),
                     (RADAR_CX, RADAR_CY + RADAR_R), 1)
    # Ceyrek cizgiler (45°)
    off45 = int(RADAR_R * 0.707)
    for dx, dy in ((off45, off45), (off45, -off45),
                   (-off45, off45), (-off45, -off45)):
        pygame.draw.line(surf, C_RADAR_RNG,
                         (RADAR_CX, RADAR_CY),
                         (RADAR_CX + dx, RADAR_CY + dy), 1)

    rays = lidar_norm[::-1] if LIDAR_FLIP else lidar_norm

    # LIDAR isınları
    for angle_deg, norm in zip(_SCREEN_ANGLES, rays):
        rad   = math.radians(angle_deg)
        dx    =  math.sin(rad)          # sag pozitif
        dy    = -math.cos(rad)          # yukari (ileri) negatif y
        color = ray_color(norm)

        ray_px = norm * RADAR_R
        ex = RADAR_CX + int(dx * ray_px)
        ey = RADAR_CY + int(dy * ray_px)

        # Isin cizgisi (gradyan etki icin iki kat)
        dim_c = lerp_color(C_RADAR_BG, color, 0.35)
        pygame.draw.line(surf, dim_c, (RADAR_CX, RADAR_CY), (ex, ey), 3)
        pygame.draw.line(surf, color, (RADAR_CX, RADAR_CY), (ex, ey), 1)

        # Uc nokta dairesi
        pygame.draw.circle(surf, color, (ex, ey), 4)

    # Dolgu alanı (tüm ısınların iç yüzeyi — polygon)
    pts = [(RADAR_CX, RADAR_CY)]
    for angle_deg, norm in zip(_SCREEN_ANGLES, rays):
        rad = math.radians(angle_deg)
        r   = norm * RADAR_R
        pts.append((RADAR_CX + int(math.sin(rad) * r),
                    RADAR_CY - int(math.cos(rad) * r)))
    if len(pts) > 2:
        fill_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        avg_norm  = float(np.mean(lidar_norm))
        fill_col  = (*lerp_color(C_RED, C_GREEN, avg_norm), 28)
        pygame.draw.polygon(fill_surf, fill_col, pts)
        surf.blit(fill_surf, (0, 0))

    # Arac sembolü (merkez ucgen)
    sz = 14
    car_pts = [
        (RADAR_CX,      RADAR_CY - sz),           # on
        (RADAR_CX - sz//2 + 1, RADAR_CY + sz//2), # sol arka
        (RADAR_CX + sz//2 - 1, RADAR_CY + sz//2), # sag arka
    ]
    pygame.draw.polygon(surf, C_CAR, car_pts)
    pygame.draw.polygon(surf, C_ACCENT, car_pts, 1)

    # Dis cerceve
    pygame.draw.circle(surf, C_ACCENT, (RADAR_CX, RADAR_CY), RADAR_R, 2)


# ── Telemetri Paneli ──────────────────────────────────────────────────────────

_GEAR_MAP = {0: "GERI", 1: "1.", 2: "2.", 3: "3.", 4: "4.", 5: "5."}


def draw_telemetry(surf, frame, lidar_norm, tracker, fonts):
    fL, fM, fS = fonts
    px = 620
    y  = 48

    draw_text(surf, "TELEMETRI", fL, C_ACCENT, px, y);  y += 46

    # ── Hiz ──────────────────────────────────────────────────────────────────
    spd_n  = min(frame.speed_kmh / 300.0, 1.0)
    spd_c  = lerp_color(C_GREEN, C_RED, spd_n)
    draw_text(surf, "HIZ", fM, C_DIM, px, y)
    draw_text(surf, f"{frame.speed_kmh:6.1f} km/h", fM, spd_c, px + 60, y)
    y += 26
    draw_bar(surf, px, y, 450, 16, spd_n, spd_c);  y += 28

    # ── Vites / RPM ──────────────────────────────────────────────────────────
    gear_str = _GEAR_MAP.get(frame.gear, str(frame.gear))
    draw_text(surf, f"VITES  {gear_str:<5}   RPM  {frame.rpm:>7,.0f}", fM, C_TEXT, px, y)
    y += 38

    # ── Gaz ──────────────────────────────────────────────────────────────────
    draw_text(surf, "GAZ",  fM, C_DIM, px, y)
    draw_text(surf, f"{frame.throttle:.2f}", fM, C_GREEN, px + 55, y)
    y += 24
    draw_bar(surf, px, y, 450, 14, frame.throttle, C_GREEN);  y += 28

    # ── Fren ─────────────────────────────────────────────────────────────────
    draw_text(surf, "FREN", fM, C_DIM, px, y)
    draw_text(surf, f"{frame.brake:.2f}", fM, C_RED, px + 60, y)
    y += 24
    draw_bar(surf, px, y, 450, 14, frame.brake, C_RED);  y += 36

    # ── Direksiyon ───────────────────────────────────────────────────────────
    draw_text(surf, "DIREKSIYON", fM, C_DIM, px, y);  y += 24
    rw, rh = 450, 20
    pygame.draw.rect(surf, (38, 38, 58), (px, y, rw, rh), border_radius=5)
    pygame.draw.line(surf, C_DIM, (px + rw//2, y), (px + rw//2, y + rh), 1)
    nx = px + int((frame.steering + 1.0) / 2.0 * rw)
    nx = max(px + 3, min(px + rw - 3, nx))
    pygame.draw.rect(surf, C_ACCENT, (nx - 4, y + 2, 8, rh - 4), border_radius=3)
    y += rh + 4
    draw_text(surf, "SOL", fS, C_DIM, px, y)
    lbl_r = fS.render("SAG", True, C_DIM)
    surf.blit(lbl_r, (px + rw - lbl_r.get_width(), y))
    y += 28

    # ── Pozisyon ─────────────────────────────────────────────────────────────
    pygame.draw.line(surf, C_DIVIDER, (px, y), (px + 450, y), 1);  y += 10
    draw_text(surf, f"X {frame.x:9.1f}   Y {frame.y:9.1f}   Z {frame.z:9.1f}",
              fM, C_DIM, px, y);  y += 34

    # ── Ilerleme (varsa) ─────────────────────────────────────────────────────
    if tracker is not None:
        pygame.draw.line(surf, C_DIVIDER, (px, y), (px + 450, y), 1);  y += 10
        pct = tracker.progress_pct
        draw_text(surf, f"ILERLEME   {pct:5.1f}%", fM, C_ACCENT, px, y);  y += 24
        draw_bar(surf, px, y, 450, 14, pct / 100.0, C_ACCENT);  y += 26
        if tracker.lap_complete:
            draw_text(surf, "TUR TAMAMLANDI!", fM, C_GREEN, px, y)
        y += 30

    # ── LIDAR ozeti ──────────────────────────────────────────────────────────
    pygame.draw.line(surf, C_DIVIDER, (px, y), (px + 450, y), 1);  y += 10
    min_n  = float(np.min(lidar_norm)) if len(lidar_norm) else 1.0
    min_px = min_n * LIDAR_MAX_PX
    lc     = ray_color(min_n)
    draw_text(surf, f"LIDAR MIN   {min_n*100:.0f}%  ({min_px:.0f} px)",
              fM, lc, px, y)


# ── Obs parcalama ─────────────────────────────────────────────────────────────

def _extract_lidar(raw_obs) -> np.ndarray:
    """raw_obs'tan 19 LIDAR ray'ini cıkar."""
    if isinstance(raw_obs, (tuple, list)) and len(raw_obs) >= 2:
        return np.asarray(raw_obs[1], dtype=np.float32).ravel()[:N_RAYS]
    flat = np.asarray(raw_obs, dtype=np.float32).ravel()
    return flat[1 : 1 + N_RAYS]   # flat: [speed, lidar x19, ...]


# ── Ana dongu ────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    from telemetry_monitor import TrackmaniaInterface, calculate_reward

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Trackmania RL — Canli Dashboard")
    clock = pygame.time.Clock()

    # Fontlar (Consolas → fallback None)
    def _font(size, bold=False):
        try:
            return pygame.font.SysFont("consolas", size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size + 6)

    fL = _font(21, bold=True)
    fM = _font(17)
    fS = _font(13)
    fonts = (fL, fM, fS)

    # Baslangiç gecikmesi — oyuna odak ver
    for i in range(args.delay, 0, -1):
        screen.fill(C_BG)
        msg = fL.render(f"Trackmania penceresine tıklayin...  {i}", True, C_YELLOW)
        screen.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2 - 20))
        sub = fS.render("ESC veya pencereyi kapat = cikis", True, C_DIM)
        screen.blit(sub, (WIN_W // 2 - sub.get_width() // 2, WIN_H // 2 + 20))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
        time.sleep(1)

    # tmrl baglantisi
    interface = TrackmaniaInterface(reward_fn=calculate_reward)
    connecting = True
    while connecting:
        screen.fill(C_BG)
        msg = fM.render("Trackmania'ya baglaniliyor...", True, C_ACCENT)
        screen.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
        if interface.connect():
            connecting = False
        else:
            screen.fill(C_BG)
            msg = fM.render("Baglanti basarisiz!  Trackmania acik mi?  5s sonra tekrar...", True, C_RED)
            screen.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2))
            pygame.display.flip()
            time.sleep(5)

    # Trajectory / tracker (opsiyonel)
    tracker = None
    if args.trajectory:
        try:
            from progress_tracker import ProgressTracker, TrajectoryProcessor
            path = Path(args.trajectory)
            with open(path, newline="", encoding="utf-8") as fh:
                header = fh.readline()
            if "timestamp" in header and "waypoint" not in path.stem:
                processed = path.parent / (path.stem + "_reference.csv")
                TrajectoryProcessor.process(str(path), str(processed))
                path = processed
            tracker = ProgressTracker(path)
        except Exception as exc:
            print(f"Trajectory yuklenemedi: {exc}")

    raw_obs = interface.reset()
    if tracker:
        tracker.reset()

    lidar_norm  = np.ones(N_RAYS) * 0.5
    frame       = None
    terminated  = truncated = False
    action      = np.zeros(3, dtype=np.float32)

    running = True
    while running:
        # Olaylar
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Step
        if raw_obs is not None and not (terminated or truncated):
            result      = interface.step(action)
            raw_obs, _, terminated, truncated, _ = result

        if raw_obs is not None:
            frame       = interface.parse_observation(raw_obs, action)
            lidar_rays  = _extract_lidar(raw_obs)
            lidar_norm  = np.clip(lidar_rays / LIDAR_MAX_PX, 0.0, 1.0)
            if tracker:
                tracker.update(frame.x, frame.y, frame.z)

        if terminated or truncated:
            raw_obs = interface.reset()
            if tracker:
                tracker.reset()
            terminated = truncated = False

        # ── Cizim ────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        # Baslik
        title = fL.render("TRACKMANIA RL  —  CANLI DASHBOARD", True, C_ACCENT)
        screen.blit(title, (WIN_W // 2 - title.get_width() // 2, 10))

        # Dikey ayirici
        pygame.draw.line(screen, C_DIVIDER, (580, 44), (580, WIN_H - 16), 1)

        # Radar baslik
        draw_text(screen, "LIDAR RADAR", fM, C_DIM,
                  RADAR_CX - fM.size("LIDAR RADAR")[0] // 2, 46)

        # Radar
        draw_radar(screen, lidar_norm)

        # Radar etiketleri
        def _lbl(text, x, y):
            s = fS.render(text, True, C_DIM)
            screen.blit(s, (x - s.get_width() // 2, y))

        _lbl("ILERI",   RADAR_CX,                RADAR_CY - RADAR_R - 18)
        _lbl("GERI",    RADAR_CX,                RADAR_CY + RADAR_R + 6)
        _lbl("SOL",     RADAR_CX - RADAR_R - 28, RADAR_CY - 8)
        _lbl("SAG",     RADAR_CX + RADAR_R + 6,  RADAR_CY - 8)

        # Mesafe etiketleri (25%, 50%, 75%)
        for frac, label in ((0.25, "25%"), (0.5, "50%"), (0.75, "75%")):
            r = int(RADAR_R * frac)
            lbl_s = fS.render(label, True, C_RADAR_RNG)
            screen.blit(lbl_s, (RADAR_CX + 4, RADAR_CY - r - 14))

        # Telemetri
        if frame is not None:
            draw_telemetry(screen, frame, lidar_norm, tracker, fonts)
        else:
            draw_text(screen, "Baglanti bekleniyor...", fM, C_DIM, 640, 310)

        pygame.display.flip()
        clock.tick(20)   # 20 FPS — oyun fizik hizina uyum

    interface.close()
    pygame.quit()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trackmania RL — Canli LIDAR + Telemetri Dashboardu"
    )
    p.add_argument(
        "--trajectory", default=None, metavar="CSV",
        help="Referans trajectory CSV (ilerleme gostermek icin)",
    )
    p.add_argument(
        "--delay", type=int, default=5, metavar="SEC",
        help="Baslangic gecikmesi saniye — oyuna focus icin (varsayilan: 5)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
