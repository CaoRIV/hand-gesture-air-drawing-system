from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import cv2

from camera import Camera, CameraConfig
from canvas import DrawingCanvas
from display import DisplayConfig, draw_app_overlay, fit_frame_to_display, frame_point_to_display
from gesture_controller import GestureController, GestureMode
from hand_tracker import HandTracker, HandTrackerConfig
from smoothing import PointSmoother, SmoothingConfig
from toolbar import GestureToolbar, ToolbarAction, draw_toolbar


WINDOW_NAME = "Hand Gesture Air Drawing - Gesture Toolbar"
OUTPUT_DIR = Path("outputs") / "saved_drawings"
TOOL_COLORS = {
    ToolbarAction.RED: (40, 80, 255),
    ToolbarAction.GREEN: (70, 220, 90),
    ToolbarAction.BLUE: (255, 140, 60),
    ToolbarAction.YELLOW: (50, 230, 245),
    ToolbarAction.WHITE: (245, 245, 245),
}


def should_quit(key_code: int) -> bool:
    return key_code in (27, ord("q"), ord("Q"))


def should_clear(key_code: int) -> bool:
    return key_code in (ord("c"), ord("C"))


def save_filename() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"drawing_{timestamp}.png"


def main() -> int:
    display_config = DisplayConfig(width=1280, height=720)
    camera_config = CameraConfig(
        camera_index=0,
        mirror=True,
        width=display_config.width,
        height=display_config.height,
        fps=30,
    )
    hand_tracker_config = HandTrackerConfig(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    camera = Camera(camera_config)
    drawing_canvas = DrawingCanvas()
    gesture_controller = GestureController()
    toolbar = GestureToolbar()
    point_smoother = PointSmoother(SmoothingConfig(alpha=0.35))
    current_color_action = ToolbarAction.RED
    active_toolbar_action = ToolbarAction.RED
    erasing = False
    previous_draw_point: tuple[int, int] | None = None
    drawing_canvas.set_brush_color(TOOL_COLORS[current_color_action])

    if not camera.open():
        print(
            "Error: Could not open webcam at index 0. "
            "Check that a webcam is connected and not being used by another app."
        )
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        with HandTracker(hand_tracker_config) as hand_tracker:
            previous_time = time.perf_counter()
            smoothed_fps = 0.0

            while True:
                success, frame = camera.read()
                if not success:
                    print("Error: Could not read frame from webcam.")
                    return 1

                drawing_canvas.ensure_size(frame.shape)
                results = hand_tracker.detect(frame)
                gesture_state = gesture_controller.analyze(results, frame.shape)
                index_tip = point_smoother.update(gesture_state.index_tip)
                drawing_active = gesture_state.mode == GestureMode.DRAW

                if not drawing_active or index_tip is None:
                    previous_draw_point = None
                else:
                    if previous_draw_point is not None:
                        if erasing:
                            drawing_canvas.erase_line(previous_draw_point, index_tip)
                        else:
                            drawing_canvas.draw_line(previous_draw_point, index_tip)
                    previous_draw_point = index_tip

                frame = drawing_canvas.compose(frame)
                hand_tracker.draw_landmarks(frame, results)
                if index_tip is not None:
                    drawing_canvas.draw_cursor(frame, index_tip, erasing=erasing)

                current_time = time.perf_counter()
                instant_fps = 1.0 / max(current_time - previous_time, 0.0001)
                previous_time = current_time
                smoothed_fps = (
                    instant_fps if smoothed_fps == 0.0 else smoothed_fps * 0.9 + instant_fps * 0.1
                )

                display_frame, frame_bounds = fit_frame_to_display(frame, display_config)
                cursor_display_point = frame_point_to_display(
                    index_tip,
                    frame.shape,
                    frame_bounds,
                )
                hovered_action = toolbar.hit_test(
                    cursor_display_point if gesture_state.mode == GestureMode.MOVE else None,
                    display_frame.shape[1],
                )
                selected_action = toolbar.select(hovered_action, current_time)

                if selected_action in TOOL_COLORS:
                    current_color_action = selected_action
                    active_toolbar_action = selected_action
                    erasing = False
                    drawing_canvas.set_brush_color(TOOL_COLORS[selected_action])
                    previous_draw_point = None
                elif selected_action == ToolbarAction.ERASER:
                    active_toolbar_action = ToolbarAction.ERASER
                    erasing = True
                    previous_draw_point = None
                elif selected_action == ToolbarAction.THIN:
                    drawing_canvas.set_brush_thickness(5)
                    previous_draw_point = None
                elif selected_action == ToolbarAction.THICK:
                    drawing_canvas.set_brush_thickness(13)
                    previous_draw_point = None
                elif selected_action == ToolbarAction.CLEAR:
                    drawing_canvas.clear()
                    point_smoother.reset()
                    previous_draw_point = None
                elif selected_action == ToolbarAction.SAVE:
                    drawing_canvas.save(OUTPUT_DIR, save_filename())
                    previous_draw_point = None

                hand_detected = bool(results.hand_landmarks)
                draw_app_overlay(
                    display_frame,
                    frame_bounds,
                    hand_detected=hand_detected,
                    mode=gesture_state.mode.value,
                    fps=smoothed_fps,
                )
                draw_toolbar(
                    display_frame,
                    toolbar,
                    active_toolbar_action,
                    hovered_action,
                )

                cv2.imshow(WINDOW_NAME, display_frame)
                key_code = cv2.waitKey(1) & 0xFF
                if should_quit(key_code):
                    break
                if should_clear(key_code):
                    drawing_canvas.clear()
                    point_smoother.reset()
                    previous_draw_point = None
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
