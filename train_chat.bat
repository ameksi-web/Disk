@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)
%PY% -m delta1flash.train_chat --data data\chat_intents.jsonl --out runs\delta1flash_chat_model.json --epochs 100 --learning-rate 0.35
echo.
echo Chat intent model trained. Now run chat_demo.bat.
pause
