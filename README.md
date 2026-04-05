# Trackmania 2020 — Pekiştirmeli Öğrenme ile Otonom Araç Sürüşü

**Bitirme Projesi** | Takım: MRamazanEkim, MRjogurtbey

Trackmania 2020 yarış oyununda bir aracın pekiştirmeli öğrenme (RL) ile otonom olarak sürmesini sağlamak amaçlanmaktadır. Proje; gerçek zamanlı telemetri toplama, referans tur kaydı, ilerleme tabanlı ödül sistemi ve SB3 SAC ile eğitim döngüsünden oluşmaktadır.

---

## Sistem Mimarisi

```
train.py  (SB3 SAC eğitim döngüsü)
  └─► TrackmaniaRLEnvironment       gymnasium.Env
        ├─► TrackmaniaInterface      tmrl üzerinden oyun bağlantısı
        ├─► ProgressTracker          KDTree waypoint ödül sistemi
        └─► DrivingController        Episode başarısızlık tespiti
```

Yapay zeka ajanı klavye veya fare simülasyonu kullanmaz; aksiyonlar (`[direksiyon, gaz, fren]`) tmrl → OpenPlanet soketi üzerinden doğrudan Trackmania fizik motoruna iletilir.

---

## Proje İlerlemesi

### Aşama 1 — Telemetri Altyapısı

tmrl kütüphanesi aracılığıyla Trackmania'dan gerçek zamanlı telemetri verisi okunmaktadır.

**Toplanan veriler:**
- Araç hızı (km/h), vites, RPM
- Direksiyon, gaz, fren değerleri
- Dünya koordinatları (X, Y, Z) — tmrl interface monkey-patch ile elde edilir

**Loglama:** Her seans `telemetry_logs/session_YYYYMMDD_HHMMSS.csv` formatında CSV veya JSONL olarak kaydedilmektedir.

**Dashboard:** `rich` kütüphanesi ile gerçek zamanlı terminal arayüzü (hız çubuğu, direksiyon ruler, ilerleme göstergesi).

---

### Aşama 2 — Referans Tur Kaydı ve İlerleme Ödülü

RL ajanına anlamlı bir ödül sinyali verebilmek için "Clever Reward" paradigması uygulanmıştır.

**Yöntem:**
1. Geliştirici `record_trajectory.py` ile bir turu manuel sürerek kaydeder.
2. Ham kayıt (timestamp, x, y, z) `TrajectoryProcessor` ile eşit aralıklı waypoint'lere dönüştürülür (varsayılan: 1 metre).
3. `ProgressTracker`, KDTree ile her adımda aracın en yakın waypoint'ini bulur; yeni waypoint geçildiğinde +1 ödül verir.
4. Geri gitme koruması: `furthest_idx` monoton artar, geri dönüş sıfır ödül üretir.
5. Tur tamamlama bonusu: %95 eşiği geçilince +50 ödül.

---

### Aşama 3 — Episode Başarısızlık Tespiti

Ajan takılıp kaldığında veya geri gittiğinde episodu otomatik sonlandıran `DrivingController` geliştirilmiştir.

| Koşul | Parametre | Açıklama |
|---|---|---|
| `ZERO_SPEED` | < 5 km/h, 3 saniye | Araç durdu |
| `STUCK` | 10 saniye ilerleme yok | Waypoint geçilemiyor |
| `REVERSE` | dot product < −0.3 | Araç geri gidiyor |

---

### Aşama 4 — Gymnasium Ortamı ve SB3 Entegrasyonu

`TrackmaniaRLEnvironment`, Stable-Baselines3 ile doğrudan çalışan bir `gymnasium.Env` subclass'ı olarak geliştirilmiştir.

**Tasarım kararları:**
- tmrl'ın `Tuple(Box, Box)` observation space'i, SB3 `MlpPolicy` uyumu için tek flat `Box`'a dönüştürülür (boyut runtime'da tmrl'dan alınır, hardcode edilmez).
- tmrl'ın kendi reward fonksiyonu devre dışı bırakılmış; reward tamamen `ProgressTracker`'dan üretilmektedir.
- tmrl konfigürasyonu: `TM20LIDAR` — 19 LIDAR ışını + hız + action buffer (~26 float). Görüntü tabanlı konfigürasyon (`TM20IMAGES`, 16393 float) `MlpPolicy` ile çalışmaz.

**Eğitim:**
- Algoritma: SAC (Soft Actor-Critic) — sürekli aksiyon uzayı için uygun
- Kütüphane: Stable-Baselines3 2.x
- Her 5.000 adımda checkpoint kaydı
- TensorBoard ile metrik izleme
- `--resume` flag ile eğitim devamı

---

## Kurulum

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt --no-cache-dir
```

### tmrl Konfigürasyonu

`C:\Users\<kullanıcı>\TmrlData\config\config.json` içinde:

```json
"RTGYM_INTERFACE": "TM20LIDAR"
```

### OpenPlanet Plugin'leri

- [MLFeed Race Data](https://openplanet.dev/plugin/mlfeedracedata)
- [MLHook](https://openplanet.dev/plugin/mlhook)

---

## Kullanım

```bat
# 1. Referans tur kaydet (Trackmania açıkken)
python record_trajectory.py --process

# 2. Eğitimi başlat
python train.py --trajectory trajectory_logs/raw_..._reference.csv

# 3. TensorBoard ile izle (ayrı terminal)
tensorboard --logdir logs/tb

# 4. Checkpoint'ten devam
python train.py --trajectory trajectory_logs/raw_..._reference.csv --resume checkpoints/sac_tm_5000_steps.zip

# 5. Sadece telemetri dashboard (eğitim olmadan)
python telemetry_monitor.py --trajectory trajectory_logs/raw_..._reference.csv
```

---

## Sonraki Adımlar

- [ ] Kendi özel RL ajan mimarimizi geliştirmek (SB3 SAC baseline olarak kullanılıyor)
- [ ] LIDAR gözlemlerinin görselleştirilmesi
- [ ] Farklı haritalar için genelleme testi

---

## Bağımlılıklar

| Kütüphane | Kullanım |
|---|---|
| `tmrl` | Trackmania ↔ Python bağlantısı (OpenPlanet soketi) |
| `stable-baselines3` | SAC implementasyonu |
| `gymnasium` | RL ortam standardı |
| `torch` | Sinir ağı backend |
| `scipy` | KDTree (waypoint arama) |
| `rich` | Terminal dashboard |
| `tensorboard` | Eğitim metrik görselleştirme |
