from __future__ import annotations

from enum import Enum
import time

import cv2

from camera import Camera, CameraConfig
from game_capture_gesture import TwoHandSpreadCaptureGesture
from game_gesture import PinchGesture
from game_hud import (
    draw_capture_gesture,
    draw_capture_hud,
    draw_cursor,
    draw_play_hud,
    draw_victory_hud,
)
from hand_tracker import HandTracker, HandTrackerConfig
from puzzle_board import PuzzleBoard, PuzzleBoardConfig
from smoothing import PointSmoother, SmoothingConfig


WINDOW_NAME = "Gesture Puzzle Game"
INDEX_TIP = 8
THUMB_TIP = 4


class GameState(Enum):
    CAPTURE = "capture"
    PLAYING = "playing"
    VICTORY = "victory"


def should_quit(key_code: int) -> bool:
    return key_code in (27, ord("q"), ord("Q"))


def should_capture(key_code: int) -> bool:
    return key_code in (ord(" "), 10, 13, ord("c"), ord("C"))


def should_restart(key_code: int) -> bool:
    return key_code in (ord("r"), ord("R"))


def board_top_left(frame_shape, board_size: int) -> tuple[int, int]:
    height, width = frame_shape[:2]
    return (width - board_size) // 2, max(92, (height - board_size) // 2 + 24)


def main() -> int:
    camera = Camera(CameraConfig(camera_index=0, mirror=True, width=1280, height=720, fps=30))
    if not camera.open():
        print(
            "Error: Could not open webcam at index 0. "
            "Check that a webcam is connected and not being used by another app."
        )
        return 1

    hand_tracker_config = HandTrackerConfig(
        max_num_hands=2,
        min_detection_confidence=0.45,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    cursor_smoother = PointSmoother(SmoothingConfig(alpha=0.32))
    pinch_detector = PinchGesture()
    capture_gesture = TwoHandSpreadCaptureGesture()
    board_config = PuzzleBoardConfig(grid_size=3, board_size=540)
    game_state = GameState.CAPTURE
    board: PuzzleBoard | None = None
    selected_tile: int | None = None
    moves = 0
    started_at = 0.0
    victory_elapsed = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        with HandTracker(hand_tracker_config) as hand_tracker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Error: Could not read frame from webcam.")
                    return 1

                capture_frame = frame.copy()
                results = hand_tracker.detect(frame)
                hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0
                index_tip = hand_tracker.get_landmark_pixel(results, frame.shape, INDEX_TIP)
                thumb_tip = hand_tracker.get_landmark_pixel(results, frame.shape, THUMB_TIP)
                cursor = cursor_smoother.update(index_tip)
                pinch = pinch_detector.update(thumb_tip, index_tip)

                if game_state == GameState.CAPTURE:
                    capture_result = capture_gesture.update(results, frame.shape)
                    hand_tracker.draw_landmarks(frame, results)
                    draw_cursor(frame, cursor, pinch.active)
                    draw_capture_gesture(frame, capture_result.points, capture_result.bounds)
                    draw_capture_hud(
                        frame,
                        hand_count,
                        capture_result.message,
                        capture_result.progress,
                    )
                    cv2.imshow(WINDOW_NAME, frame)
                    key_code = cv2.waitKey(1) & 0xFF
                    if should_quit(key_code):
                        break
                    if capture_result.captured or should_capture(key_code):
                        board = PuzzleBoard.from_frame(capture_frame, board_config)
                        game_state = GameState.PLAYING
                        selected_tile = None
                        moves = 0
                        started_at = time.perf_counter()
                        cursor_smoother.reset()
                        pinch_detector = PinchGesture()
                        capture_gesture.reset()
                    continue

                elif game_state == GameState.PLAYING:
                    if board is None:
                        game_state = GameState.CAPTURE
                        continue

                    top_left = board_top_left(frame.shape, board_config.board_size)
                    hovered_tile = board.tile_at(cursor, top_left)

                    if pinch.started and hovered_tile is not None:
                        selected_tile = hovered_tile
                    elif pinch.released:
                        if selected_tile is not None and hovered_tile is not None:
                            if board.swap(selected_tile, hovered_tile):
                                moves += 1
                        selected_tile = None

                    board.render(
                        frame,
                        top_left,
                        selected_position=selected_tile,
                        hovered_position=hovered_tile,
                    )
                    draw_cursor(frame, cursor, pinch.active)
                    elapsed = time.perf_counter() - started_at
                    draw_play_hud(frame, elapsed, moves, pinch.active, selected_tile)

                    if board.solved:
                        victory_elapsed = elapsed
                        game_state = GameState.VICTORY
                    cv2.imshow(WINDOW_NAME, frame)
                    key_code = cv2.waitKey(1) & 0xFF
                    if should_quit(key_code):
                        break
                    continue

                elif game_state == GameState.VICTORY:
                    if board is not None:
                        top_left = board_top_left(frame.shape, board_config.board_size)
                        board.render(frame, top_left)
                    draw_victory_hud(frame, victory_elapsed, moves)
                    cv2.imshow(WINDOW_NAME, frame)
                    key_code = cv2.waitKey(1) & 0xFF
                    if should_quit(key_code):
                        break
                    if should_restart(key_code):
                        board = None
                        selected_tile = None
                        moves = 0
                        victory_elapsed = 0.0
                        cursor_smoother.reset()
                        pinch_detector = PinchGesture()
                        capture_gesture.reset()
                        game_state = GameState.CAPTURE
                    continue
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
