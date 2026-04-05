# Trackmania 2020 — Pekiştirmeli Öğrenme ile Otonom Araç Sürüşü

**Bitirme Projesi** | Takım: MRamazanEkim, MRjogurtbey, kayracavli

---

## Proje Hakkında

Bu proje, Trackmania 2020 yarış oyununda bir aracın pekiştirmeli öğrenme (Reinforcement Learning) algoritmaları kullanılarak otonom biçimde sürmesini sağlamayı amaçlamaktadır.

Geleneksel otonom sürüş çalışmalarının aksine bu projede gerçek bir simülasyon ortamı yerine ticari bir yarış oyunu kullanılmaktadır. Oyunun fizik motoru, araç dinamiği ve pist geometrisi; RL ajanının gerçekçi ve zorlu bir ortamda eğitilmesine olanak tanımaktadır.

**Temel hedefler:**

- Araç telemetrisini gerçek zamanlı olarak okuyup kaydetmek
- İnsan sürüşünden elde edilen referans verilerle ödül sinyali üretmek
- Eğitilen RL ajanının aracı başarıyla pistte ilerletmesini sağlamak
- İlerleyen aşamalarda kendi RL mimarimizi geliştirmek

---

## Sistem Mimarisi

```
train.py                         ← Eğitim döngüsü (SB3 SAC)
  └─► TrackmaniaRLEnvironment    ← gymnasium.Env (ai_driving_logic.py)
        ├─► TrackmaniaInterface  ← tmrl oyun bağlantısı (telemetry_monitor.py)
        ├─► ProgressTracker      ← Waypoint tabanlı ödül (progress_tracker.py)
        └─► DrivingController    ← Episode başarısızlık tespiti (ai_driving_logic.py)
```

### Bileşenler

**TrackmaniaInterface**
tmrl kütüphanesi ve OpenPlanet eklentileri üzerinden Trackmania 2020'ye bağlanır. Her fizik tick'inde (~20 Hz) araç gözlemlerini okur, aksiyonları oyuna iletir. Aksiyonlar klavye/fare simülasyonu değil; doğrudan oyun motoruna socket üzerinden gönderilir.

**TrackmaniaRLEnvironment**
Stable-Baselines3 ile uyumlu `gymnasium.Env` subclass'ı. tmrl'ın tuple observation space'ini SB3 `MlpPolicy` için flat `Box`'a dönüştürür. Reward tmrl'ın kendi reward'ı yerine `ProgressTracker`'dan üretilir.

**ProgressTracker**
İnsan sürüşünden kaydedilen referans tur, eşit aralıklı waypoint'lere ayrıştırılır. KDTree ile her adımda aracın konumu kontrol edilir; yeni waypoint geçilince +1 ödül verilir. Geri gitme koruması: `furthest_idx` monoton artar, geri dönüş sıfır ödül üretir. Tur tamamlama bonusu: +50.

**DrivingController**
Ajan takılıp kaldığında, durduğunda veya geri gittiğinde episodu erken sonlandırır:

| Koşul        | Parametre                                        |
| ------------ | ------------------------------------------------ |
| `ZERO_SPEED` | < 5 km/h, 3 saniye boyunca                       |
| `STUCK`      | 10 saniye boyunca waypoint ilerlemesi yok        |
| `REVERSE`    | Hareket vektörü ile ileri yön dot product < −0.3 |

---

## Gözlem ve Aksiyon Uzayı

**Gözlem (Observation):** tmrl `TM20LIDAR` konfigürasyonu

- 19 LIDAR ışını (aracın çevresini ölçer)
- Araç hızı (m/s)
- Action buffer (önceki aksiyonlar)
- Toplam: ~26 boyutlu flat float32 dizisi

**Aksiyon (Action):** `[direksiyon, gaz, fren]`

- Direksiyon: −1.0 (tam sol) … +1.0 (tam sağ)
- Gaz: 0.0 … 1.0
- Fren: 0.0 … 1.0

---

## Eğitim Yöntemi

Başlangıç algoritması olarak **SAC (Soft Actor-Critic)** seçilmiştir. SAC, sürekli aksiyon uzayları için etkin bir off-policy algoritmasıdır; entropi düzenlileştirmesi sayesinde keşif-sömürü dengesini otomatik olarak yönetir.

Eğitim akışı:

1. İnsan sürüşüyle referans tur kaydedilir (`record_trajectory.py`)
2. Ham kayıt eşit aralıklı waypoint'lere dönüştürülür
3. SAC ajanı, ProgressTracker ödülüyle eğitilir
4. Her 5.000 adımda checkpoint kaydedilir, TensorBoard ile izlenir

---

## Kurulum

### Gereksinimler

- Python 3.10+
- Trackmania 2020 (Steam)
- OpenPlanet: **MLFeed Race Data** ve **MLHook** eklentileri
- tmrl OpenPlanet eklentisi

### Adımlar

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

---

## Kullanım

```bat
# 1. Referans tur kaydet (Trackmania açıkken, bir tur sür)
python record_trajectory.py --process

# 2. Eğitimi başlat
python train.py --trajectory trajectory_logs/raw_..._reference.csv

# 3. TensorBoard ile izle (ayrı terminalde)
tensorboard --logdir logs/tb

# 4. Checkpoint'ten devam et
python train.py --trajectory trajectory_logs/raw_..._reference.csv --resume checkpoints/sac_tm_5000_steps.zip

# 5. Sadece telemetri dashboard
python telemetry_monitor.py --trajectory trajectory_logs/raw_..._reference.csv
```

---

## Dosya Yapısı

| Dosya                  | Açıklama                                        |
| ---------------------- | ----------------------------------------------- |
| `train.py`             | SAC eğitim scripti                              |
| `ai_driving_logic.py`  | `TrackmaniaRLEnvironment` + `DrivingController` |
| `progress_tracker.py`  | Waypoint kaydı ve ilerleme ödülü                |
| `record_trajectory.py` | Manuel tur kayıt aracı                          |
| `telemetry_monitor.py` | Gerçek zamanlı telemetri dashboard + loglama    |
| `requirements.txt`     | Python bağımlılıkları                           |

---

## Kullanılan Teknolojiler

| Kütüphane                                                        | Kullanım                       |
| ---------------------------------------------------------------- | ------------------------------ |
| [tmrl](https://github.com/trackmania-rl/tmrl)                    | Trackmania ↔ Python bağlantısı |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | SAC implementasyonu            |
| [Gymnasium](https://gymnasium.farama.org)                        | RL ortam standardı             |
| [PyTorch](https://pytorch.org)                                   | Sinir ağı backend              |
| [SciPy](https://scipy.org)                                       | KDTree (waypoint arama)        |
| [Rich](https://github.com/Textualize/rich)                       | Terminal dashboard             |
| [TensorBoard](https://www.tensorflow.org/tensorboard)            | Eğitim metrik görselleştirme   |

---

## Sonraki Adımlar

- [ ] Kendi RL ajan mimarimizi geliştirmek (şu an SAC baseline)
- [ ] LIDAR gözlem görselleştirmesi
- [ ] Farklı pistte genelleme testi
- [ ] Performans karşılaştırması (SAC vs. kendi ajan)
