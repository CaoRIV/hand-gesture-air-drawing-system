from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot


class CapturePhase(Enum):
    WAITING = "Waiting"
    CENTER = "Center"
    SPREAD = "Spread"
    HOLD = "Hold"


@dataclass(frozen=True)
class CaptureGestureResult:
    captured: bool
    phase: CapturePhase
    message: str
    points: list[tuple[int, int]]
    bounds: tuple[int, int, int, int] | None
    progress: float


class TwoHandSpreadCaptureGesture:
    """Detects two hands pinched near center, spreading outward, then holding still."""

    THUMB_TIP = 4
    INDEX_TIP = 8

    def __init__(
        self,
        center_frames_required: int = 3,
        stable_frames_required: int = 8,
        centered_radius_ratio: float = 0.23,
        cluster_radius_ratio: float = 0.20,
        spread_ratio_required: float = 0.52,
        stable_motion_threshold: float = 24.0,
    ) -> None:
        self.center_frames_required = center_frames_required
        self.stable_frames_required = stable_frames_required
        self.centered_radius_ratio = centered_radius_ratio
        self.cluster_radius_ratio = cluster_radius_ratio
        self.spread_ratio_required = spread_ratio_required
        self.stable_motion_threshold = stable_motion_threshold
        self._phase = CapturePhase.WAITING
        self._center_frames = 0
        self._stable_frames = 0
        self._last_bounds: tuple[int, int, int, int] | None = None

    def reset(self) -> None:
        self._phase = CapturePhase.WAITING
        self._center_frames = 0
        self._stable_frames = 0
        self._last_bounds = None

    def update(self, results, frame_shape) -> CaptureGestureResult:
        points = self._finger_points(results, frame_shape)
        if len(points) < 4:
            self.reset()
            return CaptureGestureResult(
                False,
                CapturePhase.WAITING,
                "Show both hands: thumbs + index fingers",
                points,
                None,
                0.0,
            )

        bounds = self._bounds(points)
        frame_height, frame_width = frame_shape[:2]
        frame_min = min(frame_width, frame_height)
        center = self._center(points)
        frame_center = (frame_width / 2.0, frame_height / 2.0)
        center_distance = self._distance(center, frame_center)
        cluster_radius = max(self._distance(point, center) for point in points)
        spread_ratio = max(bounds[2] / frame_width, bounds[3] / frame_height)

        centered = center_distance <= frame_min * self.centered_radius_ratio
        clustered = cluster_radius <= frame_min * self.cluster_radius_ratio
        spread_enough = spread_ratio >= self.spread_ratio_required

        captured = False
        if self._phase == CapturePhase.WAITING:
            if centered and clustered:
                self._center_frames += 1
                self._phase = CapturePhase.CENTER
            else:
                self._center_frames = 0
        elif self._phase == CapturePhase.CENTER:
            if centered and clustered:
                self._center_frames += 1
                if self._center_frames >= self.center_frames_required:
                    self._phase = CapturePhase.SPREAD
            else:
                self.reset()
        elif self._phase == CapturePhase.SPREAD:
            if spread_enough:
                self._phase = CapturePhase.HOLD
                self._stable_frames = 0
            elif clustered and centered:
                self._stable_frames = 0
            elif not centered and spread_ratio < 0.25:
                self.reset()
        elif self._phase == CapturePhase.HOLD:
            if not spread_enough:
                self._phase = CapturePhase.SPREAD
                self._stable_frames = 0
            elif self._is_stable(bounds):
                self._stable_frames += 1
                if self._stable_frames >= self.stable_frames_required:
                    captured = True
                    self.reset()
            else:
                self._stable_frames = 0

        self._last_bounds = bounds
        return CaptureGestureResult(
            captured,
            self._phase,
            self._message(self._phase),
            points,
            bounds,
            min(1.0, spread_ratio / self.spread_ratio_required),
        )

    def _finger_points(self, results, frame_shape) -> list[tuple[int, int]]:
        if not results.hand_landmarks or len(results.hand_landmarks) < 2:
            return []

        frame_height, frame_width = frame_shape[:2]
        points = []
        for hand_landmarks in results.hand_landmarks[:2]:
            for landmark_index in (self.THUMB_TIP, self.INDEX_TIP):
                landmark = hand_landmarks[landmark_index]
                x = min(max(int(landmark.x * frame_width), 0), frame_width - 1)
                y = min(max(int(landmark.y * frame_height), 0), frame_height - 1)
                points.append((x, y))
        return points

    def _bounds(self, points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return min_x, min_y, max_x - min_x, max_y - min_y

    def _center(self, points: list[tuple[int, int]]) -> tuple[float, float]:
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    def _is_stable(self, bounds: tuple[int, int, int, int]) -> bool:
        if self._last_bounds is None:
            return False

        x, y, width, height = bounds
        last_x, last_y, last_width, last_height = self._last_bounds
        motion = (
            abs(x - last_x)
            + abs(y - last_y)
            + abs(width - last_width)
            + abs(height - last_height)
        )
        return motion <= self.stable_motion_threshold

    def _distance(self, first, second) -> float:
        return hypot(first[0] - second[0], first[1] - second[1])

    def _message(self, phase: CapturePhase) -> str:
        if phase == CapturePhase.CENTER:
            return "Hold 4 fingertips together at screen center"
        if phase == CapturePhase.SPREAD:
            return "Move both hands outward to frame the shot"
        if phase == CapturePhase.HOLD:
            return "Hold still to capture"
        return "Show both hands: thumbs + index fingers"
