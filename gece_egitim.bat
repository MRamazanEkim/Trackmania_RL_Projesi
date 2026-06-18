@echo off
REM ============================================================
REM  Trackmania RL - Gece Egitimi (detached / bagimsiz calisma)
REM ------------------------------------------------------------
REM  Bu .bat dosyasi Windows Gorev Zamanlayici tarafindan
REM  calistirilir. Boylece egitim; terminale, VS Code'a veya
REM  Claude oturumuna BAGLI OLMADAN calisir.
REM
REM  Cikti: logs\gece_egitim.log  (her baslatmada uzerine yazar)
REM  Model: --resume YOK. start.py otomatik checkpoints\pop\best.zip arar;
REM         su an checkpoints temiz oldugu icin ILK kosu SIFIRDAN baslar.
REM         Sonraki yeniden baslatmalarda olusan best.zip'ten DEVAM eder.
REM  Dashboard kapali (sadece egitim).
REM ============================================================

cd /d "%~dp0"

echo ==== Egitim baslangic: %DATE% %TIME% ==== > "logs\gece_egitim.log"
REM --generations 100000: pratikte sonsuz; sen "schtasks /End" ile durdurana
REM kadar TEK kosuda kesintisiz egitir. Tek kosu icinde best.zip asla geriye gitmez.
"venv\Scripts\python.exe" -u start.py --no-dashboard --generations 100000 >> "logs\gece_egitim.log" 2>&1
echo ==== Egitim bitis: %DATE% %TIME% (cikis kodu %ERRORLEVEL%) ==== >> "logs\gece_egitim.log"
