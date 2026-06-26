SmartVisionX GUI Upgrade
========================

Files:
1) voice_agent.py
2) smart_dashboard.py

Where to put them:
Put both files in the same folder as main.py.

Do NOT delete or replace:
- main.py
- scene_pipeline.py
- object_pipeline.py
- ocr_pipeline.py
- navigation_pipeline.py
- decision_engine.py

Install GUI library:
    pip install PySide6 pyttsx3

Run CPU:
    python smart_dashboard.py --camera 0 --device cpu

Run CUDA:
    python smart_dashboard.py --camera 0 --device cuda

What changed:
- main.py is untouched.
- Object Detection is untouched.
- New professional GUI dashboard added.
- New Voice Agent added.
- GUI shows:
    Scene
    Objects
    Danger level
    Navigation
    OCR
    FPS
    Latency
    Voice Agent history

Buttons:
- Read Text: forces OCR.
- Repeat Alert: repeats last spoken message.
- Mute Voice: toggles voice.
- Stop: closes the app.

If import error happens:
Make sure smart_dashboard.py and voice_agent.py are beside main.py.
Make sure the Pipelines folder and Models folder are in the same project folder.
