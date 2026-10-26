@echo off
rem Stops anything running from this folder (the demo app and its portable Ollama).
rem Normally closing the "Health Coach Demo" window is enough; this is a backup.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$d = (Resolve-Path '%~dp0').Path; Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($d) } | Stop-Process -Force; Write-Host 'Demo stopped.'"
timeout /t 3 >nul
