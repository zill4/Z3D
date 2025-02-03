@echo off
set BLENDER_PATH="C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"

echo Creating blend file...
%BLENDER_PATH% --background --python scripts/blender_viewer.py -- output/generation_39
if errorlevel 1 (
    echo Error creating blend file
    exit /b 1
)

echo.
echo Opening blend file...
timeout /t 2 /nobreak > nul
start "" %BLENDER_PATH% "output/generation_39/viewer.blend"