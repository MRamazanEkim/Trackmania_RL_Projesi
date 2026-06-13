# Trackmania RL — Proje Rehberi (AI için)

Trackmania 2020'de SAC algoritmasıyla otonom araç sürüşü geliştiren bitirme projesi.
Bu dosya hem insan hem de AI asistanlar için yazılmıştır — branch'i ilk açan AI buradan başlamalı.

---

## Aktif Branch: `feature/experience-database`

Bu branch `main`'e ek olarak şunları getirir:

| Eklenen | Ne işe yarar |
|---|---|
| `db/experience_store.py` | SQLite tabanlı deneyim deposu (training_runs + episodes tabloları) |
| `db/callbacks.py` | SB3 callback'leri: episode loglama + replay buffer checkpoint |
| `ui_dashboard.py` | Pygame penceresi: canlı LIDAR radar + telemetri görselleştirmesi |
| `train.py` güncellendi | DB entegrasyonu, replay buffer kaydet/yükle, `--db-path` argümanı |

**Temel kazanım:** `--resume` ile devam edilince önceki oturumun replay buffer'ı otomatik yüklenir
→ ajan her seferinde sıfırdan başlamaz.

---

## Mimari

```
train.py
  ├── ExperienceStore (db/experience_store.py)   ← SQLite: run + episode kayıtları
  ├── EpisodeLoggerCallback (db/callbacks.py)    ← her episode sonu DB'ye yaz
  ├── ReplayBufferCheckpointCallback             ← checkpoint ile .pkl kaydet
  └── TrackmaniaRLEnvironment (ai_driving_logic.py)   ← gymnasium.Env
        ├── TrackmaniaInterface (telemetry_monitor.py) ← tmrl oyun bağlantısı
        ├── ProgressTracker (progress_tracker.py)      ← KDTree waypoint ödülü
        └── DrivingController (ai_driving_logic.py)    ← ZERO_SPEED/STUCK/REVERSE

ui_dashboard.py  ← bağımsız pygame penceresi (eğitimden ayrı çalışır)
```

**Observation:** tmrl TM20LIDAR — tuple formatı:
- `obs[0]` → hız (1 float, m/s)
- `obs[1]` → 19 LIDAR ışını (piksel mesafesi, 64×64 görüntü, maks ~60px)
- `obs[2:]` → action buffer (act_buf_len=2, her biri 3 float)
- Flat hale getirince: `[speed, lidar×19, action_buf×6]` = 26 float

**Action:** `[steering, gas, brake]` — float, tmrl socket üzerinden oyun motoruna

**Reward:** `ProgressTracker.update()` → yeni waypoint +1, adım başı −0.05, tur bonusu +50

**Episode sonu:** tmrl terminated/truncated **veya** DrivingController tetiklenir:
- `ZERO_SPEED`: < 5 km/h, 3 saniye
- `STUCK`: 10 saniyede waypoint ilerlemesi yok
- `REVERSE`: hareket vektörü ile ileri yön dot < −0.3

---

## Kurulum (sıfırdan)

### 1. Sistem gereksinimleri

- Python 3.10+
- Trackmania 2020 (Steam, lisanslı)
- **OpenPlanet** (oyun içi mod çalıştırıcı) — F3 ile açılır
  - Plugin: **MLFeed Race Data** (aktif olmalı)
  - Plugin: **MLHook** (aktif olmalı)
  - Plugin: **TMRL_GrabData** / tmrl OpenPlanet plugin'i (aktif olmalı)

### 2. Python ortamı

```powershell
cd Trackmania_RL_Projesi
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

> SSL hatası alırsan `--trusted-host` eklerini kullan (kurumsal ağ / antivirus)

### 3. tmrl konfigürasyonu

`C:\Users\<kullanici>\TmrlData\config\config.json` içinde:

```json
"RTGYM_INTERFACE": "TM20LIDAR"
```

`TM20IMAGES` bırakırsan observation space 16000+ boyuta çıkar → `MlpPolicy` çalışmaz.

---

## Kullanım — Adım Adım

### Adım 1 — Referans tur kaydet

Trackmania **açık**, haritada araç **pistte** iken:

```powershell
.\venv\Scripts\activate
python record_trajectory.py --process
```

- 5 saniyelik geri sayım: Trackmania penceresine tıkla
- Bir tam tur sür, `Ctrl+C` ile bitir
- `trajectory_logs/raw_YYYYMMDD_HHMMSS_reference.csv` oluşur

### Adım 2 — Eğitimi başlat

```powershell
python train.py --trajectory trajectory_logs/raw_YYYYMMDD_HHMMSS_reference.csv
```

- `training.db` oluşur (SQLite, episode istatistikleri)
- Her 5000 adımda `checkpoints/sac_tm_NNNNN_steps.zip` + `..._replay.pkl`
- TensorBoard: `tensorboard --logdir logs/tb` → `http://localhost:6006`

### Adım 3 — Kaldığı yerden devam et

```powershell
python train.py \
  --trajectory trajectory_logs/raw_YYYYMMDD_HHMMSS_reference.csv \
  --resume checkpoints/sac_tm_5000_steps.zip
```

→ `sac_tm_5000_steps_replay.pkl` otomatik bulunup yüklenir.
→ Terminal: `Replay buffer yuklendi: XXXX transition`

### Adım 4 — Canlı dashboard (eğitimden bağımsız)

```powershell
python ui_dashboard.py
# veya trajectory ile:
python ui_dashboard.py --trajectory trajectory_logs/raw_YYYYMMDD_HHMMSS_reference.csv
```

- Sol panel: LIDAR radar (19 ışın, renk kodlu)
- Sağ panel: hız, vites/RPM, gaz/fren, direksiyon, konum, ilerleme
- ESC veya pencereyi kapat → çıkar

---

## Dosya Yapısı

```
Trackmania_RL_Projesi/
├── train.py                  ← SAC eğitim scripti — buradan başla
├── ai_driving_logic.py       ← TrackmaniaRLEnvironment + DrivingController
├── progress_tracker.py       ← KDTree waypoint ödülü — değiştirme
├── record_trajectory.py      ← Manuel tur kayıt aracı
├── telemetry_monitor.py      ← Terminal tabanlı telemetri dashboard
├── ui_dashboard.py           ← Pygame LIDAR + telemetri penceresi (YENİ)
├── requirements.txt          ← Python bağımlılıkları
├── db/
│   ├── experience_store.py   ← ExperienceStore (SQLite wrapper) (YENİ)
│   └── callbacks.py          ← EpisodeLoggerCallback + ReplayBufferCheckpointCallback (YENİ)
├── checkpoints/              ← model .zip + replay .pkl dosyaları (git ignore)
├── trajectory_logs/          ← ham ve referans CSV'ler (git ignore)
├── logs/tb/                  ← TensorBoard logları (git ignore)
└── training.db               ← SQLite veritabanı (git ignore)
```

---

## Veritabanı Şeması

```sql
-- Her python train.py çağrısı = bir satır
training_runs (id, started_at, trajectory_path, resume_path, checkpoint_dir, total_timesteps)

-- Her episode sonu = bir satır
episodes (id, run_id, episode_number, global_step,
          cumulative_reward, steps, furthest_waypoint,
          progress_pct, lap_complete, failure_reason, ended_at)
```

DB'yi sorgulamak için:

```python
from db.experience_store import ExperienceStore
s = ExperienceStore("training.db")
print(s.recent_episodes(run_id=1, n=10))  # son 10 episode
print(s.best_progress())                   # tüm zamanların rekoru
s.close()
```

---

## SAC Hiperparametreleri (train.py)

| Parametre | Değer | Not |
|---|---|---|
| `learning_rate` | 3e-4 | |
| `buffer_size` | 200 000 | ~46 MB .pkl |
| `learning_starts` | 5 000 | ilk N adım rastgele |
| `batch_size` | 256 | |
| `gamma` | 0.99 | |
| `gradient_steps` | 2 | adım başı 2 güncelleme |
| `ent_coef` | "auto" | otomatik entropi dengesi |
| `save_freq` | 5 000 | checkpoint aralığı |

---

## Kendi Ajanını Geliştirme

`TrackmaniaRLEnvironment` gymnasium arayüzü sabittir — değiştirme.
`train.py` içinde sadece model bloğunu değiştir:

```python
# SAC yerine kendi modelini koy:
from my_agent import MyAgent
model = MyAgent(env)
model.learn(total_timesteps=args.timesteps, callback=callbacks)
```

---

## LIDAR Teknik Detaylar

- **19 ışın**, image koordinatlarında `range(90, 280, 10)` derece
- Kaynak: `venv/lib/site-packages/tmrl/custom/tm/utils/tools.py`
- Değerler **piksel cinsinden mesafe** (64×64 görüntüde maks ~60px)
- UI'da normalize: `lidar_px / 60.0` → renk: kırmızı (yakın) → yeşil (uzak)
- LIDAR ışınları ters görünürse `ui_dashboard.py` içinde `LIDAR_FLIP = True`

---

## Sık Karşılaşılan Sorunlar

| Hata | Çözüm |
|---|---|
| `ModuleNotFoundError: stable_baselines3` | `.\venv\Scripts\activate` çalıştırılmadı |
| `Connection refused` | Trackmania açık değil veya OpenPlanet plugin'leri aktif değil |
| `OpenPlanet stopped sending data` | Trackmania'da aktif bir harita yüklü değil (menüde değil, pistte ol) |
| `Observation space (16393,)` | `config.json`'da `RTGYM_INTERFACE` → `TM20LIDAR` yap |
| `FileNotFoundError: raw_..._reference.csv` | Önce `record_trajectory.py --process` çalıştır |
| `pip install` SSL hatası | `--trusted-host pypi.org --trusted-host files.pythonhosted.org` ekle |
| `Replay buffer bulunamadi` uyarısı | Normal: ilk `--resume`'de .pkl yoksa sıfırdan başlar |
| LIDAR ışınları ters | `ui_dashboard.py` içinde `LIDAR_FLIP = True` yap |
