# Trackmania Otonom Sürüş — Proje Özeti

> Bitirme Projesi · Pekiştirmeli Öğrenme (Reinforcement Learning) ile otonom araç sürüşü
> Güncelleme: 2026-06-03

---

## 1. Proje Amacı

Trackmania 2020 oyununda bir aracın, **pekiştirmeli öğrenme (RL)** ile kendi kendine
sürmeyi öğrenmesi. Araç, **ödül/ceza** mekanizmasıyla deneme-yanılma yaparak:

1. Parkurda ilerlemeyi,
2. Virajları dönmeyi (duvara çarpmadan),
3. Parkuru tamamlamayı

öğrenir. Her başarılı denemede kendini geliştirir. Hedef sade tutulmuştur: **araç
parkuru tamamlasın ve her seferinde gelişsin** — kusursuz yarış performansı şart değil.

---

## 2. Kullanılan Teknolojiler

| Katman | Teknoloji | Görevi |
|---|---|---|
| **Oyun / Simülasyon** | Trackmania 2020 | Fizik motoru, gerçek sürüş ortamı |
| **Veri köprüsü** | OpenPlanet + MLFeed / MLHook | Oyundan telemetri (hız, konum) çekme |
| **Gerçek-zaman arayüzü** | tmrl + rtgym | Oyunu RL ortamına bağlayan gerçek-zamanlı katman; LIDAR algılama |
| **RL ortam standardı** | Gymnasium | Standart `reset()` / `step()` arayüzü |
| **RL algoritması** | Stable-Baselines3 — **SAC** | Sürekli kontrol için derin pekiştirmeli öğrenme |
| **Sinir ağı motoru** | PyTorch | SAC'ın altındaki derin öğrenme kütüphanesi |
| **Sayısal hesap** | NumPy / SciPy | Waypoint ve ödül hesapları |
| **İzleme / Görselleştirme** | TensorBoard | Öğrenme eğrisi grafikleri |
| **Arayüz / Loglama** | rich + CSV/JSONL | Canlı telemetri paneli, veri kaydı |
| **Dil** | Python 3.10+ | Tüm proje |

---

## 3. Neden SAC (Soft Actor-Critic)?

Araç kontrolü **sürekli (continuous)** değerlerden oluşur: direksiyon, gaz, fren.
Bu yüzden ayrık-aksiyon algoritmaları (ör. DQN) uygun değildir. SAC seçildi çünkü:

- **Sürekli aksiyon** uzayında çalışır (direksiyon −1…1, gaz/fren 0…1).
- **Off-policy** + replay buffer → **örnek-verimli**: az veriyle çok öğrenir
  (gerçek zamanlı oyunda veri toplamak yavaş olduğu için kritik).
- **Maksimum entropi** ilkesi → keşif/sömürü dengesini **otomatik** kurar, erken
  takılmayı azaltır.

> Karşılaştırma: tmrl projesinin kendisi de Trackmania için **REDQ-SAC** kullanır.
> Biz bitirme kapsamında daha sade olan Stable-Baselines3 **SAC** ile ilerledik.

---

## 4. Sistem Mimarisi

```
train.py  (eğitim döngüsü, SAC)
  └─► TrackmaniaRLEnvironment   (Gymnasium ortamı)
        ├─► TrackmaniaInterface  → oyundan LIDAR + hız + KONUM al, aksiyon gönder (tmrl)
        ├─► ProgressTracker      → ilerleme + parkurda kalma ödülünü hesapla
        └─► DrivingController    → başarısızlık tespiti (durdu / takıldı / geri gitti)
```

| Bileşen | Girdi | Çıktı |
|---|---|---|
| **Gözlem (observation)** | — | 83 sayı: hız + 19 LIDAR ışını + aksiyon geçmişi |
| **Aksiyon (action)** | SAC ağı | `[direksiyon, gaz, fren]` (sürekli) |
| **Ödül (reward)** | konum + hız | ProgressTracker'dan (aşağıda) |

---

## 5. Ödül / Ceza Sistemi

Araç ne yaparsa ne kazanır/kaybeder:

| Durum | Sonuç | Değer |
|---|---|---|
| Referans tur boyunca **ileri** gider (yeni waypoint) | ÖDÜL | **+2** / waypoint |
| **Hızlı** gider (gaza basmayı öğrensin diye küçük ipucu) | ödül | **+0.002 × hız** |
| Parkuru **tamamlar** (%95) | büyük ÖDÜL | **+50** |
| Her adım (zaman baskısı) | ceza | **−0.05** |
| **Yerinde durur / sürünür** (<5 km/h) | **büyük ceza** | **−0.5** / adım |
| Racing line'dan **>5 m sapar** | ceza | sapma arttıkça artan ceza |
| **>15 m sapar** / duvara çarpar | episode biter + küçük ceza | **−1** |
| Durur (uzun) / takılır / geri gider | episode biter | (başarısızlık) |

**Mantık:** İleri ilerleme ödülü (×2) baskındır → araç çizgide kalıp **virajı
dönmek zorunda**. **Yerinde durma büyük cezası** aracı "sallanma" lokal
minimumundan çıkarır (oturup hayatta kalmak artık pahalı). Çarpma cezası **küçük**
tutulur: büyük olunca "sür+çarp" denemesi, "yerinde sallan" seçeneğinden kötü
görünüp aracın denemeyi bırakmasına yol açıyordu.

**İlerleme ölçümü (ProgressTracker):** İnsan tarafından kaydedilmiş bir referans tur
(~2300 eşit aralıklı waypoint) baz alınır. Aracın konumu, bu çizgide ileri-pencere
araması ile eşleştirilir; geçtiği waypoint sayısı kadar ödül verilir. Bu yöntem
**loop (kapalı) parkurlara** ve kestirmelere dayanıklıdır.

---

## 6. Aktif Eğitim Nasıl Çalışıyor?

Komut:
```bat
python train.py --trajectory trajectory_logs/raw_20260405_134820_reference.csv --max-hours 7
```

Döngü (her ~50 ms = 20 Hz):
1. Oyundan gözlem alınır (hız + LIDAR + konum).
2. SAC ağı bir aksiyon üretir `[direksiyon, gaz, fren]`.
3. Aksiyon oyuna gönderilir, araç hareket eder.
4. Yeni konuma göre **ödül** hesaplanır (ilerleme − cezalar).
5. (gözlem, aksiyon, ödül) replay buffer'a yazılır; SAC ağları güncellenir.
6. Araç durur/çarpar/saparsa episode biter, parkur başa sarılır.

**Durma koşulu (otomatik):**
- **7 saat** dolunca, **veya**
- Araç **parkuru tamamlarsa** (erken durur) — `StopOnTimeOrCompletion` callback.

**Kayıt:** Her 5000 adımda `checkpoints/` klasörüne ara model; sonunda
`sac_tm_final.zip`. İzleme: `tensorboard --logdir logs/tb` (eğri: `rollout/ep_rew_mean`).

---

## 7. Mevcut Durum

| Ölçüt | Değer |
|---|---|
| Aktif koşu | `logs/tb/SAC_11` |
| Adım | ~7.000 / hedef ~350.000 (7 saat) |
| ep_rew_mean | ~−2 (erken keşif fazı; tırmanması bekleniyor) |
| Hız (FPS) | ~14 adım/sn (CPU üzerinde) |

**Önemli teknik kazanım:** Eğitim sırasında, aracın **konum telemetrisinin hiç
okunmadığı** bir hata bulundu ve düzeltildi (yanlış arayüz metodu sarılıyordu →
konum hep (0,0,0) geliyordu → parkur-takip ödülü ölüydü). Düzeltmeden sonra araç
gerçekten konumuna göre ödül alıyor; viraj dönme öğrenimi artık mümkün.

Eğitim şu an **aktif** ve 7 saatlik koşuda. Sonuç modeli yarın test edilebilir.

---

## 8. Sonraki Adımlar

- 7 saatlik eğitimin tamamlanması ve **modelin test edilmesi** (araç parkuru ne
  kadar tamamlıyor?).
- Eğitilmiş modeli deterministik çalıştıran **eval / play** scripti.
- **GPU** ile çok daha hızlı eğitim (şu an CPU).
- Ödül ince ayarı ve farklı parkurlarda **genelleme** testi.

---

## Dosya Yapısı (özet)

| Dosya | Açıklama |
|---|---|
| `train.py` | SAC eğitim döngüsü + süre/tamamlama ile durdurma |
| `ai_driving_logic.py` | Gymnasium ortamı + başarısızlık tespiti + ödül birleştirme |
| `progress_tracker.py` | Referans tura göre ilerleme + sapma ödül/cezası |
| `telemetry_monitor.py` | tmrl arayüzü, telemetri okuma (hız/konum), canlı panel |
| `record_trajectory.py` | Manuel referans tur kaydı |
