# Trackmania RL — Proje Rehberi

Trackmania 2020'de otonom araç sürüşü için RL ajanı geliştirme projesi.
Bitirme projesi. Amaç: kayıtlı verilerle RL algoritması geliştirmek ve aracı otonom sürmek.

## Mimari

```
train.py
  └─► TrackmaniaRLEnvironment  (ai_driving_logic.py)   ← gymnasium.Env
        ├─► TrackmaniaInterface  (telemetry_monitor.py) ← tmrl wrapper
        ├─► ProgressTracker      (progress_tracker.py)  ← waypoint reward
        └─► DrivingController    (ai_driving_logic.py)  ← failure detection
```

**Observation:** tmrl LIDAR (~26 float, flat Box) — hız + 19 LIDAR ışını + action buffer  
**Action:** `[steering, gas, brake]` — her biri float, tmrl üzerinden doğrudan oyun motoruna gönderilir  
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

### 2. Eğitimi Başlat

```bat
python train.py --trajectory trajectory_logs/raw_..._reference.csv
```

Her 5000 adımda `checkpoints/` klasörüne kayıt yapılır.

### 3. Checkpoint'ten Devam Et

```bat
python train.py --trajectory trajectory_logs/raw_..._reference.csv --resume checkpoints/sac_tm_5000_steps.zip
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
| `train.py` | SB3 SAC eğitim scripti — buradan başla |
| `ai_driving_logic.py` | `TrackmaniaRLEnvironment` (gymnasium.Env) + `DrivingController` |
| `progress_tracker.py` | KDTree tabanlı waypoint ilerleme ödülü — değiştirme |
| `record_trajectory.py` | Manuel tur kaydı aracı |
| `telemetry_monitor.py` | Gerçek zamanlı telemetri dashboard + loglama |
| `requirements.txt` | Python bağımlılıkları |

## Kendi Ajanını Geliştirme

Şu an SB3 SAC kullanılıyor. Kendi RL ajanını entegre etmek için:

1. `TrackmaniaRLEnvironment` değişmez — gymnasium.Env arayüzü sabit kalır
2. `train.py`'da `SAC` yerine kendi modelini kullan:

```python
# train.py içinde, model oluşturma kısmını değiştir:
from my_agent import MyAgent
model = MyAgent(env)
model.learn(total_timesteps=args.timesteps)
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
