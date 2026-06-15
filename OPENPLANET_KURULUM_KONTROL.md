# OpenPlanet Kurulum Kontrol — Arkadaşın PC'si

Bu dosya, eğitimi başka bir PC'de çalıştırmak için OpenPlanet + TMRL plugin
kurulumunu doğrulamak içindir. Claude'a bu dosyayı ver ve **"bu adımları kontrol et"**
de — Claude komutları çalıştırıp sonuçları benim çalışan PC'min ayarlarıyla
karşılaştıracak.

> **NOT:** Proje CLAUDE.md'sinde "MLFeed Race Data + MLHook" yazıyor ama gerçek
> kurulum bunları KULLANMIYOR. Eğitimi sağlayan tek plugin **TMRL_GrabData**.
> Aşağıdaki referans, çalışan PC'den (shebn) alınmıştır.

---

## Çalışan PC'nin referans değerleri (hedef)

| Şey | Beklenen değer |
|---|---|
| OpenPlanet sürümü | `1.29.5 (next, Public)` veya yenisi |
| OpenPlanet klasörü | `%USERPROFILE%\OpenplanetNext\` |
| Gerekli plugin | **TMRL_GrabData v0.3** → `Plugins\TMRL_GrabData.op` |
| (opsiyonel) | `TMRL_SaveGhost v0.2`, `TMDojo` |
| Plugin TCP portu | `127.0.0.1:9000` (plugin server açar, tmrl bağlanır) |
| Signature mode | Public yeterli (TMRL_GrabData imzalı) |
| tmrl config | `%USERPROFILE%\TmrlData\config\config.json` |
| `RTGYM_INTERFACE` | `TM20LIDAR` (KESİNLİKLE — IMAGES değil) |

> MLFeed / MLHook **gerekmez**. Kurulu değilse sorun değil.

---

## Kontrol Adımları (Claude bunları sırayla çalıştırsın)

### 1. OpenPlanet klasörü var mı + sürüm
```bash
ls "$USERPROFILE/OpenplanetNext/"
grep -i "Openplanet for" "$USERPROFILE/OpenplanetNext/Openplanet.log" | head -1
```
Beklenen: klasör mevcut, `Openplanet for Trackmania Next x64` satırı görünür.

### 2. TMRL_GrabData plugin kurulu mu (EN ÖNEMLİ)
```bash
ls -la "$USERPROFILE/OpenplanetNext/Plugins/" | grep -i tmrl
```
Beklenen: `TMRL_GrabData.op` dosyası var.
**Yoksa:** OpenPlanet içi Plugin Manager'dan "TMRL grab data" (siteid 343) kur,
veya çalışan PC'deki `.op` dosyasını kopyala.

### 3. Plugin gerçekten yüklendi + socket açtı mı
```bash
grep -i -E "TMRL_GrabData|Server socket|Could not initiate" "$USERPROFILE/OpenplanetNext/Openplanet.log" | tail -15
```
Beklenen (oyun açık + plugin aktifken):
```
Loaded zipped plugin 'TMRL_GrabData' (version 0.3)
[TMRL_GrabData] ...: Created server socket
[TMRL_GrabData] ...: Server socket ready
```
**`Could not initiate server socket` görürsen:** port 9000 başka uygulama
tarafından tutulmuş. Aşağı bak (Sorun Giderme).

### 4. tmrl config doğru mu
```bash
python -c "import json,os;d=json.load(open(os.path.expanduser('~/TmrlData/config/config.json')));print('RTGYM_INTERFACE =', d['ENV']['RTGYM_INTERFACE'])"
```
Beklenen: `RTGYM_INTERFACE = TM20LIDAR`
**`TM20IMAGES` ise:** config.json'da `ENV.RTGYM_INTERFACE` değerini `TM20LIDAR`
yap (observation space 16000+ olmasın, MlpPolicy bozulur).

### 5. Pencere ayarları çakışmasın
```bash
python -c "import json,os;e=json.load(open(os.path.expanduser('~/TmrlData/config/config.json')))['ENV'];print('Window', e['WINDOW_WIDTH'],'x',e['WINDOW_HEIGHT'])"
```
Beklenen: `Window 958 x 488`. tmrl ekran yakalaması bu boyuta göre çalışır;
Trackmania penceresi bu boyutta + sol-üst köşeye yapışık olmalı (windowed mode).

### 6. Port 9000 boş mu (oyun kapalıyken kontrol)
```bash
netstat -ano | grep ":9000" || echo "port 9000 bos - OK"
```
Oyun KAPALIYKEN bir şey çıkmamalı. Çıkıyorsa başka process portu tutuyor.

---

## Sık Karşılaşılan Plugin Hataları

| Belirti | Sebep | Çözüm |
|---|---|---|
| `Plugin is not suitable for the current signature mode` | İLGİSİZ plugin (ör. EditorDeveloper). TMRL_GrabData'yı etkilemez | Yok say |
| TMRL_GrabData log'da yok | Plugin kurulu değil / aktif değil | Plugin Manager'dan kur, OpenPlanet overlay'de aktif et |
| `Could not initiate server socket` | Port 9000 dolu (eski tmrl/oyun açık) | Tüm Trackmania + python tmrl processlerini kapat, tekrar dene |
| `Disconnected, could not send data` | tmrl client erken kapandı — normal, eğitim durunca olur | Eğitim çalışıyorsa yok say |
| `Connection refused` (python tarafı) | Oyun kapalı VEYA plugin socket açmadı | Önce oyunu aç, haritaya gir, sonra train.py çalıştır |
| Observation space `(16393,)` | `RTGYM_INTERFACE=TM20IMAGES` | Adım 4'e bak, LIDAR yap |

---

## Doğru Başlatma Sırası (her seferinde)

1. Trackmania 2020 aç, bir haritaya gir (araç sürülebilir durumda).
2. OpenPlanet overlay (F3) → TMRL_GrabData aktif, log'da `Server socket ready`.
3. venv aktifleştir: `venv\Scripts\activate`
4. `python train.py --trajectory trajectory_logs\raw_..._reference.csv`

> Sıra önemli: oyun + plugin ÖNCE, python SONRA. Plugin socket'i açmadan
> python `Connection refused` alır.

---

## Hâlâ Çalışmıyorsa — Claude'a Topla

Claude şu çıktıları toplayıp çalışan PC ile farkı raporlasın:
```bash
echo "=== plugins ==="; ls "$USERPROFILE/OpenplanetNext/Plugins/"
echo "=== son log (tmrl) ==="; grep -i tmrl "$USERPROFILE/OpenplanetNext/Openplanet.log" | tail -20
echo "=== interface ==="; python -c "import json,os;print(json.load(open(os.path.expanduser('~/TmrlData/config/config.json')))['ENV']['RTGYM_INTERFACE'])"
echo "=== port ==="; netstat -ano | grep ":9000"
```
