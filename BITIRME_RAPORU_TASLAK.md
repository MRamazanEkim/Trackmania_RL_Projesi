# Bitirme Projesi Raporu — Çalışma Taslağı

> Trackmania 2020 ortamında pekiştirmeli öğrenme (RL) ile otonom sürüş ajanı.
> Bu dosya tez/rapor metnini doldurmak için ham içerik sağlar: **ne yaptık**,
> **sistem nasıl gerçekleşti**, **bulgular**. Tablolardaki sayılar projedeki
> `experience.db` + `experience_archive.db` verisinden alınmıştır
> (üretim tarihi: 2026-06-16).

---

## 1. Yapılan Çalışmalar (Özet)

Bu projede Trackmania 2020 oyununda gerçek zamanlı karar verebilen bir otonom
sürüş ajanı geliştirildi. Çalışmanın kapsamı:

1. **Oyunla etkileşim katmanı** — tmrl (TrackMania Reinforcement Learning)
   kütüphanesi ve OpenPlanet plugin'leri (MLFeed Race Data, MLHook) üzerinden
   oyundan sensör benzeri telemetri (pozisyon, hız, yön, LIDAR ışınları) okuyan
   ve oyuna aksiyon (gaz/fren/direksiyon) gönderen bir arayüz kuruldu.
2. **Gymnasium uyumlu RL ortamı** — `TrackmaniaRLEnvironment` sınıfı standart
   `reset()` / `step()` arayüzünü sağlayacak şekilde yazıldı. Böylece herhangi
   bir RL algoritması (kendi yazılan veya hazır) ortama takılabilir.
3. **Ödül fonksiyonu tasarımı** — önceden kaydedilmiş bir referans tur üzerinden
   waypoint tabanlı ilerleme ödülü + hız teşviki + çarpma/durma cezaları.
4. **Üç RL algoritmasının karşılaştırmalı eğitimi** — DQN, PPO ve SAC, **aynı
   ortam, aynı ödül, aynı eğitim bütçesi** ile ayrı ayrı eğitildi.
5. **Deneyim/metrik kayıt altyapısı** — SQLite veritabanı; her episode için
   ödül, ilerleme, tur süresi, çarpışma bayrağı, başarısızlık nedeni kaydedildi.
6. **Canlı izleme ve görselleştirme** — pygame tabanlı LIDAR + telemetri
   dashboard'u ve TensorBoard öğrenme eğrileri.

---

## 2. Sistemin Gerçekleştirilmesi (Yöntem ve Mimari)

### 2.1 Genel Mimari

```
population_train.py / compare_train.py / start.py  (eğitim girişleri)
  ├─► RL algoritması (SAC / PPO / DQN, SB3)         ← öğrenen ajan
  ├─► ExperienceStore (db/experience_store.py)      ← SQLite metrik deposu
  ├─► DashboardCallback (ui_dashboard.py)           ← canlı LIDAR + telemetri
  └─► TrackmaniaRLEnvironment (ai_driving_logic.py) ← gymnasium.Env
        ├─► TrackmaniaInterface (telemetry_monitor.py)  ← tmrl wrapper
        ├─► ProgressTracker (progress_tracker.py)       ← waypoint ödülü (KDTree)
        └─► DrivingController (ai_driving_logic.py)     ← başarısızlık tespiti
```

### 2.2 Durum Uzayı (Observation)

Ajan, görüntü yerine tmrl'in **LIDAR** arayüzünü kullanır (`TM20LIDAR`).
Gözlem vektörü düşük boyutlu, flat bir `Box`:

- Araç hızı,
- 19 LIDAR ışını (pist kenarına olan mesafeler),
- son aksiyonların tamponu (action buffer).

> Görüntü tabanlı arayüz (`TM20IMAGES`) 16000+ boyut üretip MLP politikasını
> kullanılamaz hale getirdiği için bilinçli olarak LIDAR tercih edildi. Bu,
> eğitimi CPU üzerinde uygulanabilir kıldı.

### 2.3 Aksiyon Uzayı (Action)

- **SAC / PPO** → sürekli `[gaz, fren, direksiyon]` (`Box`, ortamın doğal uzayı).
- **DQN** → DQN sürekli aksiyon çalıştıramaz; `DiscreteActionWrapper` ile 5 ayrık
  aksiyona indirgenir (örn. düz, hafif sol, sert sol, hafif sağ, sert sağ).

Bir tasarım kararı olarak fren aksiyonu varsayılan kapalı (`disable_brake=True`)
tutuldu: ajanın "yavaşla → cezadan/çarpmadan kaç → yerinde sallan" yerel
minimumuna sıkışmasını önlemek için araç yalnız gaz + direksiyon kullanır.

### 2.4 Ödül-Ceza Fonksiyonu

Ödül, tmrl'in kendi ödülü yerine projeye özgü bir **waypoint ilerleme** sistemine
dayanır (`ProgressTracker`, KDTree en-yakın-komşu araması):

| Bileşen | Değer | Amaç |
|---|---|---|
| Yeni waypoint geçme | +1 | Pist boyunca ilerlemeyi ödüllendir |
| Tur tamamlama bonusu | +50 | Bitirmeyi güçlü teşvik et |
| Hız ödülü | +0.05 × hız(km/h) | Hızlı sürmeyi teşvik (sadece mesafe değil) |
| Yerinde durma cezası | −1.0 / adım (hız < 5 km/h) | Sallanma tuzağından çıkar |
| Sürekli yavaş sürüş | −0.3 / adım (30 km/h altında >3 s) | Sürünmeyi cezalandır |
| İlerleme yok cezası | −0.3 / adım (>2 s yeni waypoint yok) | Yerinde bekleyeni yakala |
| Çarpma/başarısızlık | −0.3 (terminal) | "Çarpmak kötü" ama denemeyi öldürmeyen küçük ceza |

> Ceza katsayıları deneysel olarak ayarlandı. Özellikle çarpma cezası bilinçle
> küçük tutuldu: büyük olduğunda "sür + çarp" denemesi "yerinde sallan + hayatta
> kal"dan daha kötü puanlanıp ajanın denemekten vazgeçmesine yol açıyordu.

### 2.5 Episode (Bölüm) Sonu

Bir bölüm şu koşullarda biter:
- tmrl `terminated` / `truncated`, **veya**
- `DrivingController` başarısızlık tespiti: `ZERO_SPEED` (araç durdu),
  `STUCK` (sıkıştı), `REVERSE` (geri gidiyor), `OFF_TRACK` (racing line'dan saptı).

### 2.6 Algoritmaların Eşit Koşulda Karşılaştırılması

`compare_train.py` projenin özgün katkısıdır. Üç algoritma da **aynı**:
ortam + ödül fonksiyonu + gözlem + eğitim adımı bütçesi (50.000 adım) +
ölçülen metrikler ile değerlendirilir. **Sadece algoritma sınıfı değişir.**

Ortak hiperparametreler: `MlpPolicy`, `learning_rate=3e-4`, `gamma=0.99`.

| Algo | Aksiyon | Önemli hiperparametreler |
|---|---|---|
| SAC | sürekli | buffer 200k, batch 256, τ=0.005, gradient_steps=4, ent_coef=auto |
| PPO | sürekli | n_steps=2048, batch 64, n_epochs=10, gae_λ=0.95, clip 0.2 |
| DQN | ayrık (5) | buffer 200k, batch 256, exploration_fraction=0.2, final_eps=0.05 |

### 2.7 İleri Eğitim Mekanizmaları

- **AsyncSAC** (`async_sac.py`): Oyun gerçek-zaman 20 Hz çalıştığı için standart
  SB3 döngüsü her adımda ~50 ms CPU'yu boşta bekletir. AsyncSAC, deneyim
  toplamayı (collector thread) ve gradient öğrenmeyi (ana thread) ayrı thread'lerde
  örtüştürür; oyun beklerken CPU öğrenir. tmrl'in dağıtık RolloutWorker↔Trainer
  mimarisinin tek-süreçli taklidi.
- **Popülasyon + Elitizm** (`population_train.py`): Her jenerasyonda N aday
  sırayla tur atar, en iyisi `best.zip` olarak korunur ve sonraki neslin
  ebeveyni olur. `best` asla geriye gitmez → her nesil ≥ önceki nesil.

### 2.8 Veri ve Metrik Kaydı

SQLite (`db/experience_store.py`) üç tablo tutar:
- `training_runs` — her eğitim çalıştırması,
- `episodes` — her bölüm: ödül, adım sayısı, ilerleme %, tur süresi (s),
  çarpışma bayrağı, başarısızlık nedeni, algoritma adı,
- `generations` — popülasyon/elitizm jenerasyon kayıtları.

Bu sayede rapordaki **tüm değerlendirme metrikleri** (ortalama ödül, başarı/
tamamlama oranı, çarpışma oranı, tur süresi, ilerleme) tek sorguyla üretilebilir.

---

## 3. Bulgular ve Deneysel Sonuçlar

### 3.1 Algoritma Karşılaştırması (eşit koşul, ~50k adım/algoritma)

| Algoritma | Episode | Tamamlama | Ort. İlerleme | En İyi İlerleme | Çarpışma Oranı | Ort. Ödül |
|---|--:|--:|--:|--:|--:|--:|
| **SAC** | 389 | %0 | %3.8 | **%12.0** | %75.3 | **576.6** |
| **DQN** | 489 | %0 | %3.4 | %10.1 | %78.5 | 499.0 |
| **PPO** | 269 | %0 | %4.0 | %7.1 | %74.3 | 503.2 |

> Not: Tablo verisi `experience.db` (SAC) + `experience_archive.db` (DQN, PPO)
> birleşimidir. Raporu yazarken `compare_train.py --report
> --db-path experience_archive.db` ile güncel tabloyu yeniden üretmek mümkün.

### 3.2 Yorum

- **Hiçbir algoritma tam tur tamamlamadı (tamamlama %0).** Pisti en uzağa
  taşıyan SAC oldu (%12 ilerleme). Bu, 50k adımlık bütçenin tam tur için yetersiz
  kaldığını gösterir; SAC ayrı bir koşuda 310k adıma kadar eğitilmiştir.
- **Öğrenme verimliliği:** SAC en yüksek en-iyi-ilerleme (%12) ve en yüksek
  ortalama ödülü (576) elde etti → off-policy + sürekli aksiyon bu görevde
  örneklem-verimli. DQN ayrık aksiyona indirgenmenin maliyetini taşıdı.
  PPO en az episode'da (on-policy, daha yavaş veri tüketimi) benzer ilerleme
  yakaladı ama en-iyi ilerlemesi düşük kaldı.
- **Çarpışma oranı üç algoritmada da yüksek (%74–79).** Ajanlar güvenli sürüşü
  henüz yeterince öğrenmedi; bu, kısa eğitim bütçesi ve zorlu sürekli-kontrol
  görevi ile tutarlı.
- **Sıralama (bu görev + bu bütçe için):** ilerleme ve ödülde **SAC > DQN ≈ PPO**.

### 3.3 Niteliksel Gözlemler

- Ödül fonksiyonunun erken sürümlerinde ajan "yerinde sallanıp hayatta kalma"
  yerel minimumuna takıldı; idle/no-progress cezaları ve küçük çarpma cezası
  bunu çözdü. **Bulgu:** ödül şekillendirmesi (reward shaping) bu görevde
  algoritma seçiminden daha belirleyici oldu.
- Fren aksiyonunu kapatmak erken eğitimde ilerlemeyi belirgin hızlandırdı.
- AsyncSAC, gerçek-zaman bekleme süresini öğrenmeye çevirerek aynı duvar-saati
  süresinde daha çok gradient adımı sağladı.

---

## 4. Rapor Hedefleri ile Karşılaştırma (Kapsam Değerlendirmesi)

| Rapor hedefi | Durum | Not |
|---|---|---|
| RL ile otonom karar verme modellemek | ✅ | Çalışan prototip |
| Ödül-ceza fonksiyonu tasarlamak | ✅ | Waypoint + hız + ceza sistemi |
| DQN/PPO/SAC karşılaştırması | ✅ | Eşit koşul, tek tablo |
| Hız/dönüş/kaza stratejisi öğrenme | ◐ | Kısmen; ilerleme var, tam tur yok |
| **Farklı pistlerde genelleme** | ❌ | **Tek pist kaydı var; çok-pist eval yapılmadı** |
| Performans grafikleri / öğrenme eğrileri | ✅ | TensorBoard + SQLite/CSV |
| Algoritma + hiperparametre karşılaştırması | ✅ | compare_train.py |
| Deneysel analiz raporu | ◐ | Bu doküman taslağı |

### 4.1 Bilinen Eksik ve Öneriler

1. **Genelleme (en önemli açık):** `trajectory_logs/` yalnız 1 pist içerir. Rapor
   "farklı pist yapılarında genelleme" istiyor. En az 2. pist kaydedilip "Pist A'da
   eğit → Pist B'de değerlendir" senaryosu çalıştırılmalı.
2. **Tam tur için eğitim bütçesi artırılmalı** (50k → 150k+), tamamlama %0'dan
   kurtarılmalı.
3. **Karşılaştırma tablosu** tez metnine eklenirken doğru DB'den
   (`experience_archive.db`) yeniden üretilmeli; aktif `experience.db` yalnız son
   SAC-devam koşusunu içerir.

---

## 5. Kullanılan Teknolojiler

- Python 3.10+, Gymnasium, Stable-Baselines3 (SAC/PPO/DQN), PyTorch
- tmrl + OpenPlanet (MLFeed Race Data, MLHook) — oyun arayüzü
- SQLite (stdlib) — metrik deposu
- pygame — canlı dashboard; TensorBoard — öğrenme eğrileri
- SciPy KDTree — waypoint en-yakın-komşu ödülü

---

_Not: Tablolardaki sayıları teze geçirmeden önce `python compare_train.py
--report --db-path experience_archive.db` çıktısıyla doğrula; eğitim devam
ederse değerler değişir._
