Файлы:
1. gui_voice_classifier_app.py — готовый интерфейс для классификации аудиофайла.
2. build_exe_windows.bat — сборка exe через PyInstaller.

Как использовать:
1. Убедитесь, что у вас уже есть обученная модель voice_random_forest_model_bundle.joblib.
2. Положите ее:
   - рядом с exe
   или
   - в подпапку model рядом с exe.
3. Запустите программу.
4. Нажмите "Загрузить аудиофайл".
5. Нажмите "Запустить анализ".
6. Программа покажет итоговый класс и вероятности по 4 классам.

Как собрать exe на Windows:
1. Установите Python.
2. Установите зависимости:
   pip install pyinstaller joblib librosa numpy pandas scipy scikit-learn soundfile audioread
3. Положите gui_voice_classifier_app.py и build_exe_windows.bat в одну папку.
4. Запустите build_exe_windows.bat.
5. Заберите exe из папки dist.

Важно:
- В этом окружении я подготовил исходники, но не собрал настоящий Windows .exe.
- Для получения именно .exe его нужно собрать на Windows через PyInstaller.
