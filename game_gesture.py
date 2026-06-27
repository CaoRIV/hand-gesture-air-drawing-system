from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class PinchState:
    active: bool
    started: bool
    released: bool
    distance: float | None


class PinchGesture:
    """Hysteresis-based pinch detector using thumb and index fingertips."""

    def __init__(
        self,
        pinch_threshold: float = 48.0,
        release_threshold: float = 72.0,
    ) -> None:
        self.pinch_threshold = pinch_threshold
        self.release_threshold = release_threshold
        self._active = False

    def update(
        self,
        thumb_tip: tuple[int, int] | None,
        index_tip: tuple[int, int] | None,
    ) -> PinchState:
        if thumb_tip is None or index_tip is None:
            was_active = self._active
            self._active = False
            return PinchState(False, False, was_active, None)

        distance = hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
        started = False
        released = False

        if not self._active and distance <= self.pinch_threshold:
            self._active = True
            started = True
        elif self._active and distance >= self.release_threshold:
            self._active = False
            released = True

        return PinchState(self._active, started, released, distance)
