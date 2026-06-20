# Trackmania RL — Proje Rehberi

Trackmania 2020'de otonom araç sürüşü için RL ajanı geliştirme projesi.
Bitirme projesi. Amaç: kayıtlı verilerle RL algoritması geliştirmek ve aracı otonom sürmek.

## Mimari

```
population_train.py  (popülasyon + elitizm; ana eğitim girişi)
  ├─► AsyncSAC               (async_sac.py)         ← eşzamanlı toplama+öğrenme
  ├─► ExperienceStore        (db/experience_store.py) ← generations/episodes (SQLite)
  ├─► DashboardCallback      (ui_dashboard.py)      ← canlı LIDAR+telemetri (opsiyonel)
  └─► TrackmaniaRLEnvironment (ai_driving_logic.py) ← gymnasium.Env
        ├─► TrackmaniaInterface  (telemetry_monitor.py) ← tmrl wrapper
        ├─► ProgressTracker      (progress_tracker.py)  ← waypoint reward
        └─► DrivingController    (ai_driving_logic.py)  ← failure detection
```

**Eğitim modeli:** Popülasyon + elitizm. Her jenerasyonda N aday SIRAYLA tur atar
(TM tek oyun penceresi), en iyi aday `best.zip` olarak korunur ve sonraki neslin
ebeveyni olur. best asla geriye gitmez → her nesil ≥ önceki.

**Observation:** tmrl LIDAR (~26 float, flat Box) — hız + 19 LIDAR ışını + action buffer  
**Action:** `[gas, brake, steering]` — tmrl üzerinden doğrudan oyun motoruna gönderilir
(tmrl mapping: idx0=ileri/gaz, idx1=geri/fren, idx2=direksiyon)  
**Reward:** Araç yeni waypoint geçtikçe +1, tamamlama bonusu +50  
**Episode sonu:** tmrl terminated/truncated VEYA DrivingController: ZERO_SPEED / STUCK / REVERSE

## Kurulum

### Gereksinimler
- Python 3.10+
- Trackmania 2020 (Steam)
- OpenPlanet: **MLFeed Race Data** + **MLHook** plugin'leri kurulu ve aktif
- tmrl OpenPlanet plugin'i kurulu (tmrl dokümantasyonuna bak)

### Adımlar

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt --no-cache-dir
```

### tmrl Konfigürasyonu

`C:\Users\<kullanıcı>\TmrlData\config\config.json` dosyasında şu ayar olmalı:

```json
"RTGYM_INTERFACE": "TM20LIDAR"
```

Image tabanlı (`TM20IMAGES`) ise observation space 16000+ boyuta çıkar, MlpPolicy çalışmaz.
Referans LIDAR config: `C:\Users\<kullanıcı>\TmrlData\resources\config_lidars.json`

## Kullanım

### 1. Referans Tur Kaydet

Trackmania açıkken, haritada araçla bir tur sürerek kayıt al:

```bat
python record_trajectory.py --process
```

`trajectory_logs/` klasörüne `raw_..._reference.csv` dosyası oluşur.

### 2. Eğitimi Başlat (popülasyon + elitizm)

```bat
python population_train.py --trajectory trajectory_logs/raw_..._reference.csv --async --dashboard
```

Faydalı bayraklar: `--pop-size N` (aday sayısı), `--generations G`, `--learn-steps S`
(aday başına öğrenme adımı), `--eval-episodes E`, `--mutation-std`, `--async`
(eşzamanlı toplama+öğrenme), `--dashboard` (canlı pencere).
En iyi model her zaman `checkpoints/pop/best.zip` olarak korunur.

### 3. Devam Et (mevcut best'ten)

```bat
python population_train.py --trajectory trajectory_logs/raw_..._reference.csv --resume checkpoints/pop/best.zip --async
```

### 4. TensorBoard ile İzle

```bat
tensorboard --logdir logs/tb
```

Tarayıcıda: `http://localhost:6006`

### 5. Sadece Telemetri İzle (eğitim olmadan)

```bat
python telemetry_monitor.py --trajectory trajectory_logs/raw_..._reference.csv
```

## Dosya Yapısı

| Dosya | Açıklama |
|---|---|
| `population_train.py` | Popülasyon + elitizm eğitimi — **buradan başla** |
| `async_sac.py` | AsyncSAC — eşzamanlı toplama+öğrenme (oyun beklerken öğren) |
| `ai_driving_logic.py` | `TrackmaniaRLEnvironment` (gymnasium.Env) + `DrivingController` |
| `db/experience_store.py` | SQLite: `generations` (elitizm) + `episodes` tabloları |
| `ui_dashboard.py` | Canlı LIDAR radar + telemetri (gömülü `DashboardCallback` veya bağımsız) |
| `progress_tracker.py` | KDTree tabanlı waypoint ilerleme ödülü — değiştirme |
| `stagnation_monitor.py` | Uzun koşularda tıkanma tespiti + otomatik ödül ayarı (bkz. `STAGNASYON_TESPITI.md`) |
| `record_trajectory.py` | Manuel tur kaydı aracı |
| `telemetry_monitor.py` | tmrl wrapper (`TrackmaniaInterface`) + telemetri loglama |
| `requirements.txt` | Python bağımlılıkları |

## Kendi Ajanını Geliştirme

Şu an SB3 SAC kullanılıyor. Kendi RL ajanını entegre etmek için:

1. `TrackmaniaRLEnvironment` değişmez — gymnasium.Env arayüzü sabit kalır
2. `population_train.py` içindeki `new_model()` / `ModelCls` kısmında `AsyncSAC`
   yerine kendi modelini kullan:

```python
# population_train.py içinde model oluşturmayı değiştir:
from my_agent import MyAgent
model = MyAgent(env)
model.learn(total_timesteps=...)
```

Ortam `reset()` → `(obs, info)`, `step(action)` → `(obs, reward, terminated, truncated, info)` döndürür.

## Sık Karşılaşılan Sorunlar

| Hata | Çözüm |
|---|---|
| `ModuleNotFoundError: stable_baselines3` | `venv\Scripts\activate` çalıştırılmadı |
| `Connection refused` | Trackmania açık değil veya OpenPlanet plugin'leri aktif değil |
| Observation space `(16393,)` | `config.json`'da `RTGYM_INTERFACE` → `TM20LIDAR` yap |
| `Time-step timed out` | Normal, ilk bağlantıda görünebilir; eğitim başlarsa sorun yok |
| `pip install` permission error | `pip install -r requirements.txt --no-cache-dir` kullan |
