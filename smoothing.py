from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmoothingConfig:
    alpha: float = 0.35


class PointSmoother:
    """Exponential moving average smoother for fingertip coordinates."""

    def __init__(self, config: SmoothingConfig | None = None) -> None:
        self.config = config or SmoothingConfig()
        self._point: tuple[float, float] | None = None

    def reset(self) -> None:
        self._point = None

    def update(self, point: tuple[int, int] | None) -> tuple[int, int] | None:
        if point is None:
            self.reset()
            return None

        if self._point is None:
            self._point = float(point[0]), float(point[1])
            return point

        alpha = self.config.alpha
        smoothed_x = self._point[0] * (1.0 - alpha) + point[0] * alpha
        smoothed_y = self._point[1] * (1.0 - alpha) + point[1] * alpha
        self._point = smoothed_x, smoothed_y
        return int(round(smoothed_x)), int(round(smoothed_y))
