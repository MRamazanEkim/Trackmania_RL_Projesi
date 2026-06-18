# Dinamik Bölge Sistemi (Adaptive Zones)

Parkurun **viraj** ve **düzlük** bölgelerini otomatik tespit edip ödül/cezayı
bölgeye göre uyarlayan sistem. Amaç: araç düzlükte hızlı ve çizgiye sadık sürsün,
virajda ise yavaşlamaya / geniş hat çizmeye **cezalandırılmadan** izin verilsin.

> **Neden?** Sabit cezalarla araç zor virajları (özellikle U-dönüşleri) dönemiyordu:
> yavaşlama cezası viraj için gereken fren/yavaşlamayı engelliyor, dar off-track
> sınırı virajda savrulunca aracı anında resetliyordu. Dinamik bölgeler bunu çözer.

---

## Nasıl çalışır?

### 1. Viraj tespiti (geometri tabanlı, baştan hesaplanır)

Referans çizgisinin (kaydedilmiş tur) **eğriliği** kullanılır. Ardışık waypoint'ler
arasındaki yön değişimi açısı = yerel eğrilik. Her waypoint'e `corner_factor ∈ [0,1]`
atanır:

- `0.0` → düz yol
- `1.0` → keskin viraj

Açı `corner_angle_deg` (varsayılan 8°/waypoint) eşiğine ulaşınca `corner_factor=1`
olur. Ardından `corner_dilate` (varsayılan ±12 waypoint) kadar **genişletilir** —
böylece virajın **girişi ve çıkışı** da "viraj" sayılır (apex'ten önce yavaşlamaya,
sonra hızlanmaya izin).

> Bu sinyal **failure verisinden değil, geometriden** gelir. Yani eğitimin ilk
> saniyesinden itibaren doğrudur (sıfırdan eğitimde bile). "Araç nerede takılıyor"
> verisi ise ayrı bir **doğrulama** aracıdır → `analiz_hotspot.py`.

### 2. Bölgeye göre uyarlanan ödül/ceza

`straight_scale = 1 - corner_factor * corner_relief` çarpanı hız-ilişkili tüm
terimlere uygulanır:

| Terim | Düzlük (cf≈0) | Viraj (cf≈1) |
|---|---|---|
| Hız ödülü | tam | kalkar (hızı virajda ödüllemez) |
| Idle cezası (<5 km/h) | tam | hafifler (yavaşlamaya izin) |
| Yavaş sürüş cezası (<30 km/h) | tam | hafifler |
| Off-track toleransı | dar (5 m) | geniş (5 + `corner_extra_tol`) |
| `max_stray` (reset sınırı) | dar (20 m) | geniş (20 + `corner_extra_stray`) |

Sonuç: **düzlükte sıkı** (hızlı git, çizgiyi tut), **virajda esnek** (yavaşla,
geniş hat çiz, savrulsan da resetlenme).

---

## Açma / Kapama

Sistem **varsayılan olarak AÇIK**. Kapatmak (A/B karşılaştırma) için:

```bat
python start.py --no-adaptive-zones
python population_train.py --trajectory ... --no-adaptive-zones
```

---

## Parametreler

`ProgressTracker` (progress_tracker.py) — viraj tespiti + off-track:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `adaptive_offtrack` | `True` | Virajda off-track tolerans/max_stray genişlemesi |
| `corner_angle_deg` | `8.0` | Waypoint başına bu açı (derece) → corner_factor=1 |
| `corner_dilate` | `12` | Viraj bölgesini ±N waypoint genişlet (giriş+çıkış) |
| `corner_extra_tol` | `4.0` | Tam virajda off-track toleransına eklenen metre |
| `corner_extra_stray` | `15.0` | Tam virajda max_stray'e eklenen metre |

`TrackmaniaRLEnvironment` (ai_driving_logic.py) — ceza ölçekleme:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `adaptive_zones` | `True` | Dinamik bölge ceza ölçeklemesini aç/kapat |
| `corner_relief` | `1.0` | Tam virajda hız-ilişkili cezaların ne kadarı kalkar (1.0 = tamamen) |

### Ayar ipuçları
- Viraj bölgeleri **çok geniş** çıkıyorsa → `corner_dilate` azalt veya `corner_angle_deg` artır.
- Viraj **algılanmıyorsa** → `corner_angle_deg` azalt (ör. 5).
- Virajda hâlâ çok resetliyorsa → `corner_extra_stray` artır.

---

## Doğrulama: araç nerede takılıyor?

Eğitim sırasında her değerlendirme episode'unun **bitiş sebebi ve yeri** DB'ye
(`episodes` tablosu) yazılır: `OFF_TRACK / STUCK / ZERO_SPEED / TIMEOUT / LAP`.

```bat
python analiz_hotspot.py                # en son koşuyu analiz et
python analiz_hotspot.py --run 11       # belirli run
python analiz_hotspot.py --bins 20      # parkur dilim sayısı
```

Çıktı: parkurun %'lik dilimlerine göre **nerede en çok episode bittiği**, baskın
sebep, ve o dilimin **gerçekten viraj olup olmadığı** (geometriyle çapraz doğrulama).
Böylece "araç %20'de takılıyor, orası viraj mı?" hipotezini test edersin.

---

## Curriculum ile kullanım (önerilen akış)

1. **Aşama 1 — Bitir:** Cezalar gevşek + dinamik bölgeler açık → araç önce **turu
   tamamlamayı** öğrensin.
2. **Aşama 2 — İyileştir:** Tur tamamlanınca cezaları **kademeli** sıkılaştır
   (düzlük cezalarını artır). Dinamik sistem bunu otomatik viraj-farkında yapar:
   sıkma düzlükleri vurur, virajları boğmaz → daha hızlı, daha temiz tur.

> Cezaları **tek seferde** sertleştirme — politika geçici çöker. Adım adım sık,
> her seferinde tur-atan checkpoint'ten devam et.
