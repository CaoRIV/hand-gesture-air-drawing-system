# Hand Gesture Air Drawing System

Phase 1 sets up the project foundation for webcam-based hand tracking. It opens the default webcam, detects one hand with MediaPipe Hands, draws hand landmarks on the live frame, and exits safely with `q` or `Esc`.

## Requirements

- Python 3.10+
- A working webcam

Install dependencies:

```powershell
D:\Python3\python.exe -m pip install -r requirements.txt
```

## Run Phase 1

```powershell
D:\Python3\python.exe main.py
```

Controls:

- `q`: quit
- `Esc`: quit

## Current Scope

Implemented:

- Webcam capture with mirrored preview
- MediaPipe hand landmark detection
- Local MediaPipe Tasks model at `models/hand_landmarker.task`
- Debug landmark drawing
- Safe camera/window cleanup

Not implemented yet:

- Drawing canvas
- Gesture-based drawing mode
- Color, eraser, brush size, or save image tools
