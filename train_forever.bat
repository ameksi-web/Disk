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
echo Infinite training started. Stop with Ctrl+C.
%PY% -m delta1flash.train --corpus data\demo_focused_corpus.txt --out runs\delta1flash_demo_model.json --model-name "Delta 1 Flash" --token-mode char --max-vocab 0 --steps 500 --batch-size 64 --context-length 6 --learning-rate 0.6 --warm-start-counts --forever --save-every-cycles 1 --metrics-out runs\delta1flash_forever_metrics.jsonl
pause
