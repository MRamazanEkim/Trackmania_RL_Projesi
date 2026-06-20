# Tıkanma Tespiti + Otomatik Ödül Ayarı (Stagnation Auto-Tuning)

> **Branch:** `gelistirme/stagnasyon-tespit-2026-06-20`
> Bu branch'e yeni başlayan biri için: aşağıdaki "30 saniyede özet" yeterli;
> ayrıntı ve ödül/ceza mantığı sonraki bölümlerde.

## 30 saniyede özet

Uzun (örn. **24 saat**) popülasyon eğitimlerinde araç bir noktada **takılıp
kalıyordu**: en iyi ilerleme saatlerce aynı değerde (örn. %35) duruyor, araç her
seferinde aynı yerde aynı şekilde başarısız oluyordu. Önceden bunu fark etmek için
insanın başında durması gerekiyordu.

Bu branch, eğitime bir **tıkanma izleyici** (`StagnationMonitor`) ekler:

1. Her nesil sonunda en iyi ilerlemeyi izler.
2. Belirli sayıda nesil (varsayılan **5**) boyunca iyileşme olmazsa **"tıkandı"** der.
3. O an aracın **neden** başarısız olduğunu teşhis eder (baskın `failure_reason`).
4. Teşhise göre **ödül/cezayı canlı ayarlar** (`RewardAutoTuner`) ve eğitime devam
   eder — ya da `--stagnation-mode stop` ile rapor yazıp durur.
5. Her tıkanma için `analysis/` klasörüne bir markdown **teşhis raporu** yazar.

Böylece eğitim kendini izleyip "ufak ufak" düzelterek gece boyunca tek başına
ilerleyebilir.

---

## Neden gerekliydi?

Eğitim **popülasyon + elitizm** modelinde çalışıyor (bkz. `CLAUDE.md`): en iyi aday
`best.zip` olarak korunur ve `best_progress` **asla geriye gitmez**. Bu güzel bir
garanti ama bir kör nokta yaratır:

> `best_progress`'in **azalmaması**, "ilerliyor" demek değildir.
> Saatlerce **aynı** değerde takılı kalabilir (plato).

Gerçek koşulardan örnek (run 13): araç %38.5'e ulaştıktan sonra episode'ların büyük
çoğunluğu ilk %10'da **sürünüp TIMEOUT** oluyordu. Elitizm "geriye gitmediği" için
trend grafiği düz bir çizgi gibi görünür; ama aslında öğrenme durmuştur. İnsan
başında olmadan bunu fark edip ödül sinyalini değiştirmek mümkün değildi.

---

## Nasıl çalışır?

### 1. Tıkanma tespiti — `StagnationMonitor`

Her nesil sonunda `monitor.record(gen, best_progress, failure_nedenleri)` çağrılır.

- En iyi ilerleme **`min_delta`** (varsayılan %0.5) kadar artarsa → "iyileşme",
  sayaç sıfırlanır.
- Aksi halde "iyileşme yok" sayacı artar.
- Sayaç **`patience`**'a (varsayılan 5 nesil) ulaşınca → **tıkanma**.

`min_delta` gürültü eşiğidir: 35.0% → 35.2% iyileşme **sayılmaz** (elitizmin küçük
dalgalanmalarını plato saymamak için).

### 2. Teşhis — baskın başarısızlık nedeni

Tıkanınca son `patience` neslin **tüm değerlendirme episode'larının** bitiş nedenleri
toplanır ve **baskın neden** bulunur. Olası nedenler (`ai_driving_logic.py`'den):

| Neden | Anlamı |
|---|---|
| `TIMEOUT` | Araç çarpmıyor ama ilerlemiyor — süre/tmrl truncation ile bitiyor (sürünme) |
| `STUCK` | Belirli süre yeni waypoint geçilemedi (genelde duvara yapışma) |
| `ZERO_SPEED` | Hız bir süre ~0'da kaldı (durdu) |
| `OFF_TRACK` | Araç referans çizgisinden çok saptı (parkuru terk etti) |
| `REVERSE` | Araç geri gitmeye başladı |
| `LAP` | Tur tamamlandı (başarı) |

### 3. Otomatik ödül ayarı — `RewardAutoTuner`

Baskın nedene göre **kural-tabanlı** olarak hangi ödül/ceza parametresinin nasıl
değişeceğine karar verilir ve ortama **canlı** uygulanır (`apply_reward_adjustment`).
Değerler `±` çarpanla değişir (artır ×1.40, azalt ×0.75) ve **sınırlar (clamp)**
içinde tutulur → otomatik ayar kontrolden çıkamaz.

| Baskın neden | Teşhis | Yapılan ayar |
|---|---|---|
| `TIMEOUT` | Çarpmıyor, sürünüyor, süre doluyor | `speed_reward_coef`↑, `no_progress_penalty`↑, `low_speed_penalty`↑ |
| `STUCK` / `ZERO_SPEED` | Duruyor / duvara yapışıyor | `idle_penalty`↑, `speed_reward_coef`↑ |
| `OFF_TRACK` | Viraja çok hızlı girip savruluyor | `corner_relief`↑, `speed_reward_coef`↓, `crash_penalty`↑ |
| `REVERSE` | Geri gidiyor | `speed_reward_coef`↑, `idle_penalty`↑ |
| diğer/belirsiz | Net teşhis yok | `speed_reward_coef`↑ (hafif itki) |

Ayar sonrası plato sayacı **sıfırlanır**: yeni parametrelere bir `patience` penceresi
daha şans verilir. En fazla `--stagnation-max-adapts` (varsayılan 6) kez ayar yapılır;
sonra eğitim durur (sonsuz döngüye girmesin diye).

### 4. Teşhis raporu

Her tıkanmada `analysis/stagnation_genN_<zaman>.md` yazılır: plato değeri, başarısızlık
nedeni dağılımı (tablo), ayar öncesi parametreler, önerilen çarpanlar ve uygulanan
değişiklikler. Bu klasör `.gitignore`'dadır (runtime çıktısı).

---

## Ödül / Ceza sistemi (tam referans)

Otomatik ayarın oynadığı parametreler `TrackmaniaRLEnvironment` (`ai_driving_logic.py`)
içinde tanımlıdır. Hepsi **her adımda canlı okunur**, bu yüzden eğitim sürerken
değiştirilebilirler.

| Parametre | Varsayılan | Sınır | Ne yapar? |
|---|---:|---|---|
| `speed_reward_coef` | 0.08 | 0.02–0.25 | Adım ödülüne `katsayı × hız(km/h)` ekler — hızlı sürmeyi ödüllendirir (düzlükte) |
| `idle_penalty` | 1.00 | 0.20–4.00 | Hız `idle_speed_kmh` (5) altındaysa adım başına ceza — yerinde sallanmayı kırar |
| `low_speed_penalty` | 0.30 | 0.00–2.00 | `low_speed_kmh` (30) altında `low_speed_timeout` (3sn)'den uzun kalınırsa ceza — sürünmeyi kırar |
| `no_progress_penalty` | 0.30 | 0.00–2.00 | Yeni waypoint geçilmeden geçen her adıma (grace sonrası) ceza |
| `crash_penalty` | 0.00 | 0.00–5.00 | Episode başarısızlıkla biterse terminal ceza (varsayılan kapalı; OFF_TRACK tıkanmasında açılır) |
| `corner_relief` | 1.00 | 0.00–1.00 | Virajda hız-ilişkili cezaların ne kadarının kalkacağı (1.0 = tamamen) |

> **Önemli:** Bunların hepsi **dinamik bölge** sistemiyle (`straight_scale`) çarpılır:
> hız-ilişkili cezalar **düzlükte tam**, **virajda hafif** uygulanır. Ayrıntı:
> [DINAMIK_BOLGELER.md](DINAMIK_BOLGELER.md). Auto-tuner bu yapıyı bozmaz, sadece
> temel katsayıları kaydırır.

Bunların dışında sabit (auto-tune edilmeyen) terimler: waypoint başına +1 ilerleme
ödülü, tur tamamlama +50 bonusu, `ProgressTracker` off-track tespiti.

---

## Kullanım

Mevcut akış değişmez; tıkanma izleyici **varsayılan açıktır** (`adapt` modu).

```bat
:: best.zip'ten devam, tıkanınca ödülü otomatik ayarla (varsayılan)
python population_train.py --trajectory trajectory_logs/raw_..._reference.csv ^
    --resume checkpoints/pop/best.zip --async

:: Tıkanınca AYARLAMA, sadece rapor yazıp DUR (insan müdahalesi için)
python population_train.py --trajectory ... --resume checkpoints/pop/best.zip ^
    --stagnation-mode stop

:: İzleyiciyi tamamen kapat (eski davranış)
python population_train.py --trajectory ... --stagnation-mode off
```

### İlgili bayraklar

| Bayrak | Varsayılan | Açıklama |
|---|---:|---|
| `--stagnation-mode` | `adapt` | `adapt` (ayarla+devam) / `stop` (rapor+dur) / `off` (kapalı) |
| `--stagnation-patience` | 5 | Kaç nesil iyileşme olmazsa tıkanma sayılır |
| `--stagnation-min-delta` | 0.5 | İyileşme sayılması için en iyi ilerlemenin (%) artması gereken miktar |
| `--stagnation-max-adapts` | 6 | `adapt` modunda en fazla kaç kez ödül ayarı (sonra durur) |
| `--analysis-dir` | `analysis` | Teşhis raporlarının yazılacağı klasör |

---

## Dosyalar

| Dosya | Bu branch'teki değişiklik |
|---|---|
| `stagnation_monitor.py` | **YENİ** — `StagnationMonitor`, `RewardAutoTuner`, rapor üretimi |
| `ai_driving_logic.py` | `reward_params()` + `apply_reward_adjustment()` (canlı ödül güncelleme) |
| `population_train.py` | Nesil döngüsüne tıkanma kontrolü + 5 yeni CLI bayrağı |
| `STAGNASYON_TESPITI.md` | **YENİ** — bu doküman |

---

## Sınırlar / dikkat

- Auto-tuner **kural-tabanlıdır**, öğrenen bir meta-optimize edici değildir. Baskın
  başarısızlık nedeni net olmayan durumlarda (karışık nedenler) etkisi sınırlıdır.
- Ödül sinyalini değiştirmek SAC'in replay buffer'ındaki eski deneyimlerle yeni
  ödülü kısa süre tutarsız yapabilir; bu yüzden adımlar küçük (×1.4 / ×0.75) ve
  ayar sayısı sınırlıdır.
- `adapt` modunu açık bırakıp gece çalıştırın; sabah `analysis/` klasöründeki
  raporlardan aracın **nerede ve neden** takıldığını ve hangi ayarların yapıldığını
  görebilirsiniz.
