@echo off
setlocal

REM Установите зависимости заранее:
REM pip install pyinstaller joblib librosa numpy pandas scipy scikit-learn soundfile audioread

pyinstaller --noconfirm --clean --onefile --windowed ^
  --name VoiceClassifierGUI ^
  gui_voice_classifier_app.py

REM После сборки положите файл модели рядом с exe:
REM dist\VoiceClassifierGUI.exe
REM dist\voice_random_forest_model_bundle.joblib
REM либо:
REM dist\model\voice_random_forest_model_bundle.joblib

echo.
echo Готово. exe будет в папке dist.
pause
