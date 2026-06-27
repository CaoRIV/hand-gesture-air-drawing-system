# Hand Gesture Air Drawing System

The app opens the default webcam, detects one hand with MediaPipe Hands, follows a smoothed index fingertip point, draws strokes only in draw mode, and supports a gesture toolbar for colors, eraser, brush size, clear, and save.

## Requirements

- Python 3.10+
- A working webcam

Install dependencies:

```powershell
D:\Python3\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
D:\Python3\python.exe main.py
```

Controls:

- Index finger only: draw
- Index + middle fingers: move cursor without drawing and select toolbar items
- `c`: clear drawing canvas
- `q`: quit
- `Esc`: quit

Toolbar:

- `Red`, `Green`, `Blue`, `Yellow`, `White`: change brush color
- `Erase`: erase parts of the canvas
- `Thin`, `Thick`: change brush size
- `Clear`: clear the canvas
- `Save`: save the drawing to `outputs/saved_drawings`

## Current Scope

Implemented:

- Webcam capture with mirrored preview
- MediaPipe hand landmark detection
- Local MediaPipe Tasks model at `models/hand_landmarker.task`
- Debug landmark drawing
- Smoothed index fingertip drawing on a separate virtual canvas
- Gesture modes for draw, move, and idle
- Gesture toolbar for colors, eraser, brush size, clear, and save
- Clear canvas with `c`
- 16:9 preview frame that preserves camera aspect ratio
- Basic status overlay with hand detection state and FPS
- Safe camera/window cleanup

Not implemented yet:

- Smart shape recognition
- Handwriting or digit recognition
