from __future__ import annotations

import cv2

from camera import Camera, CameraConfig
from hand_tracker import HandTracker, HandTrackerConfig


WINDOW_NAME = "Hand Gesture Air Drawing - Phase 1"


def should_quit(key_code: int) -> bool:
    return key_code in (27, ord("q"), ord("Q"))


def main() -> int:
    camera_config = CameraConfig(camera_index=0, mirror=True)
    hand_tracker_config = HandTrackerConfig(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    camera = Camera(camera_config)
    if not camera.open():
        print(
            "Error: Could not open webcam at index 0. "
            "Check that a webcam is connected and not being used by another app."
        )
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        with HandTracker(hand_tracker_config) as hand_tracker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Error: Could not read frame from webcam.")
                    return 1

                results = hand_tracker.detect(frame)
                hand_tracker.draw_landmarks(frame, results)

                cv2.imshow(WINDOW_NAME, frame)
                key_code = cv2.waitKey(1) & 0xFF
                if should_quit(key_code):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
