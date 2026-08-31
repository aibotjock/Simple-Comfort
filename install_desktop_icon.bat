@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: Simple-Comfort — Desktop Icon Installer (Windows)
:: Creates a shortcut on the Desktop that launches python app.py
:: ─────────────────────────────────────────────────────────────────────────────
setlocal EnableDelayedExpansion

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Simple-Comfort  —  Desktop Installer
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

:: Resolve the directory this .bat lives in
set "APP_DIR=%~dp0"
:: Remove trailing backslash
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"

set "ICON_PATH=%APP_DIR%\static\icon.ico"
set "SHORTCUT=%USERPROFILE%\Desktop\Simple-Comfort.lnk"
set "LAUNCH_SCRIPT=%APP_DIR%\launch.bat"

:: ── Create a clean launch wrapper ────────────────────────────────────────────
echo Creating launch wrapper...
(
echo @echo off
echo cd /d "%APP_DIR%"
echo echo Starting Simple-Comfort...
echo echo Open http://localhost:7788 in your browser.
echo python app.py
echo pause
) > "%LAUNCH_SCRIPT%"

:: ── Create Desktop shortcut via PowerShell VBScript bridge ───────────────────
echo Creating Desktop shortcut...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell; ^
   $s  = $ws.CreateShortcut('%SHORTCUT%'); ^
   $s.TargetPath       = '%LAUNCH_SCRIPT%'; ^
   $s.WorkingDirectory = '%APP_DIR%'; ^
   $s.IconLocation     = '%ICON_PATH%'; ^
   $s.Description      = 'Simple-Comfort - Local AI Video Generator'; ^
   $s.WindowStyle      = 1; ^
   $s.Save()"

if exist "%SHORTCUT%" (
    echo   OK  Shortcut created: %SHORTCUT%
) else (
    echo   WARN Could not create shortcut automatically.
    echo        Manually create a shortcut to: %LAUNCH_SCRIPT%
    echo        and set the icon to:           %ICON_PATH%
)

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Done.
echo   Double-click Simple-Comfort on your Desktop,
echo   then open http://localhost:7788
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pause
