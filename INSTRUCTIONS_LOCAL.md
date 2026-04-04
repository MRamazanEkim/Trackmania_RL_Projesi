# Trackmania RL — Yerel Kurulum Talimatları

Bu proje **Windows yerel ortamında (venv)** çalıştırılmak üzere yapılandırılmıştır.
Docker kurulumu repoda bulunmakla birlikte aktif eğitim için tavsiye edilmez.

---

## Neden Docker Değil, Yerel Ortam?

| Sorun | Açıklama |
|-------|----------|
| **uinput / sanal gamepad** | `vgamepad` kütüphanesi Linux'ta `/dev/uinput` kernel aygıtına ihtiyaç duyar. Docker Desktop for Windows'un WSL2 kernel'i bu modülü içermediğinden konteyner içinden sanal gamepad oluşturulamaz. |
| **Gerçek zamanlı gecikme (latency)** | tmrl, Trackmania ile 20 Hz fizik döngüsünde senkronize çalışır. Docker'ın ağ katmanı ve WSL2 overhead'i bu senkronizasyonu bozabilir. |
| **Ekran erişimi** | `pyautogui` ve `mss` (ekran görüntüsü) Windows display stack'ine doğrudan erişim gerektirir; Docker container'ı bunu sağlayamaz. |

**Sonuç:** Trainer ve telemetry monitor doğrudan Windows'ta, aynı makinede Trackmania ile birlikte çalışmalıdır.
TensorBoard için Docker hâlâ kullanılabilir (`docker compose up tensorboard`).

---

## Gerekli Sürücü — ViGEmBus

`vgamepad` kütüphanesinin Windows'ta çalışması için **ViGEmBus** sanal gamepad sürücüsü gereklidir.

**İndirme:** https://github.com/nefarius/ViGEmBus/releases

En son `ViGEmBus_Setup_x64.exe` dosyasını indirip kurun. Kurulum sonrasında bilgisayarı yeniden başlatın.

---

## OpenPlanet Eklentileri

Trackmania'nın tmrl ile haberleşmesi için aşağıdaki iki eklentinin **manuel olarak** kurulması gerekmektedir.

### 1. MLFeed: Race Data

Araç telemetrisini (hız, konum, vites, gaz, fren) OpenPlanet üzerinden dışarı aktarır.

**İndirme:** https://openplanet.dev/plugin/mlfeedracedata

### 2. MLHook

ML araçlarının OpenPlanet ile iletişim kurmasını sağlayan köprü eklentisidir.

**İndirme:** https://openplanet.dev/plugin/mlhook

### Kurulum Adımları

1. Yukarıdaki linklerden `.op` veya `.zip` dosyalarını indirin.
2. İndirilen dosyaları şu klasöre kopyalayın:

   ```
   Documents\OpenplanetNext\Plugins\
   ```

3. Trackmania'yı başlatın (veya yeniden başlatın). OpenPlanet menüsünde eklentilerin aktif olduğunu doğrulayın.

---

## Python Sanal Ortam (venv) Kurulumu

Gereksinimler: **Python 3.10+** ve **Git**

### Adım 1 — Repoyu Klonla

```bat
git clone https://github.com/MRjogurtbey/Trackmania_RL_Projesi.git
cd Trackmania_RL_Projesi
```

### Adım 2 — Sanal Ortam Oluştur ve Aktif Et

```bat
python -m venv venv
venv\Scripts\activate
```

### Adım 3 — Bağımlılıkları Kur

```bat
pip install -r requirements.txt vgamepad
```

> `torch` ve diğer büyük paketler nedeniyle kurulum birkaç dakika sürebilir.

---

## Telemetry Monitor'ı Çalıştırma

**Ön koşullar:**
- Trackmania 2020 açık ve bir haritada bekliyor olmalı.
- OpenPlanet eklentileri (MLFeed + MLHook) aktif olmalı.
- ViGEmBus kurulu olmalı.

### Temel Kullanım

```bat
venv\Scripts\activate
python telemetry_monitor.py --format csv
```

### Ek Seçenekler

```bat
# Belirli sayıda episode çalıştır
python telemetry_monitor.py --episodes 10

# Referans trajectory ile progress-based ödül
python telemetry_monitor.py --trajectory trajectories/raw_trajectory.csv

# AI mantığını devre dışı bırak (saf telemetri)
python telemetry_monitor.py --no-ai-logic

# JSONL formatında logla
python telemetry_monitor.py --format jsonl
```

Loglar `telemetry_logs/` klasörüne, `session_YYYYMMDD_HHMMSS.csv` formatında kaydedilir.

---

## TensorBoard (Opsiyonel)

Eğitim metriklerini görselleştirmek için Docker üzerinden çalıştırılabilir:

```bat
docker compose up tensorboard
```

Ardından tarayıcıda: `http://localhost:6006`

Veya doğrudan yerel ortamdan:

```bat
venv\Scripts\activate
python -m tensorboard.main --logdir=telemetry_logs --host=0.0.0.0 --port=6006
```
