@echo off
rem Shannon - double-click, or drag a recording onto this file.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "AUDIO=%~1"

if "%AUDIO%"=="" (
  rem No file dragged on. Ask for one. PowerShell is used for the dialogs
  rem because the embeddable Python runtime has no tkinter.
  for /f "usebackq delims=" %%F in (`powershell -NoProfile -STA -Command ^
    "Add-Type -AssemblyName System.Windows.Forms;" ^
    "$d = New-Object System.Windows.Forms.OpenFileDialog;" ^
    "$d.Title = 'Choose a recording to transcribe';" ^
    "$d.Filter = 'Recordings|*.mp3;*.m4a;*.wav;*.mp4;*.m4v;*.aac;*.flac;*.ogg;*.wma|All files|*.*';" ^
    "if ($d.ShowDialog() -eq 'OK') { $d.FileName }"`) do set "AUDIO=%%F"
)

if "%AUDIO%"=="" (
  echo No file chosen.
  pause
  exit /b 1
)

set "SPEAKERS="
for /f "usebackq delims=" %%N in (`powershell -NoProfile -Command ^
  "Add-Type -AssemblyName Microsoft.VisualBasic;" ^
  "[Microsoft.VisualBasic.Interaction]::InputBox('How many people are speaking on this recording?', 'Shannon', '2')"`) do set "SPEAKERS=%%N"

if "%SPEAKERS%"=="" set "SPEAKERS=2"

echo.
echo   File     : %AUDIO%
echo   Speakers : %SPEAKERS%
echo.
echo Working. A one-hour recording takes about an hour - you can leave this running.
echo.

"%~dp0python\python.exe" "%~dp0transcribe.py" "%AUDIO%" --speakers %SPEAKERS% --format both
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo Done. Opening the folder.
  for %%I in ("%AUDIO%") do explorer /select,"%%~dpnI.docx"
) else (
  echo Something went wrong ^(exit code %RC%^). The message above says what.
)
pause
