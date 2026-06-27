from __future__ import annotations

from dataclasses import dataclass
import random

import cv2
import numpy as np


@dataclass(frozen=True)
class PuzzleBoardConfig:
    grid_size: int = 3
    board_size: int = 540
    gap: int = 4


class PuzzleBoard:
    """Tile-swap puzzle board built from a webcam snapshot."""

    def __init__(self, image, config: PuzzleBoardConfig | None = None) -> None:
        self.config = config or PuzzleBoardConfig()
        self._source = self._prepare_square_image(image)
        self._tiles = self._make_tiles()
        self._order = list(range(len(self._tiles)))
        self.shuffle()

    @classmethod
    def from_frame(
        cls,
        frame,
        config: PuzzleBoardConfig | None = None,
    ) -> "PuzzleBoard":
        return cls(frame.copy(), config)

    @property
    def solved(self) -> bool:
        return self._order == list(range(len(self._tiles)))

    def shuffle(self) -> None:
        self._order = list(range(len(self._tiles)))
        while self.solved:
            random.shuffle(self._order)

    def swap(self, first_position: int, second_position: int) -> bool:
        if first_position == second_position:
            return False
        if not self._is_valid_position(first_position) or not self._is_valid_position(second_position):
            return False

        self._order[first_position], self._order[second_position] = (
            self._order[second_position],
            self._order[first_position],
        )
        return True

    def tile_at(self, point: tuple[int, int] | None, top_left: tuple[int, int]) -> int | None:
        if point is None:
            return None

        x, y = point
        board_x, board_y = top_left
        local_x = x - board_x
        local_y = y - board_y
        if local_x < 0 or local_y < 0:
            return None
        if local_x >= self.config.board_size or local_y >= self.config.board_size:
            return None

        tile_size = self.config.board_size // self.config.grid_size
        column = min(local_x // tile_size, self.config.grid_size - 1)
        row = min(local_y // tile_size, self.config.grid_size - 1)
        return int(row * self.config.grid_size + column)

    def render(
        self,
        frame,
        top_left: tuple[int, int],
        selected_position: int | None = None,
        hovered_position: int | None = None,
    ) -> None:
        board_x, board_y = top_left
        tile_size = self.config.board_size // self.config.grid_size
        panel_padding = 12
        cv2.rectangle(
            frame,
            (board_x - panel_padding, board_y - panel_padding),
            (
                board_x + self.config.board_size + panel_padding,
                board_y + self.config.board_size + panel_padding,
            ),
            (18, 20, 26),
            -1,
        )

        for position, tile_id in enumerate(self._order):
            row, column = divmod(position, self.config.grid_size)
            x = board_x + column * tile_size
            y = board_y + row * tile_size
            tile = self._tiles[tile_id]
            frame[y : y + tile_size, x : x + tile_size] = tile

            border_color = (45, 48, 56)
            border_thickness = 2
            if position == hovered_position:
                border_color = (0, 210, 255)
                border_thickness = 4
            if position == selected_position:
                border_color = (0, 255, 120)
                border_thickness = 5

            cv2.rectangle(
                frame,
                (x, y),
                (x + tile_size - 1, y + tile_size - 1),
                border_color,
                border_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                str(tile_id + 1),
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _prepare_square_image(self, image):
        height, width = image.shape[:2]
        crop_size = min(height, width)
        x = (width - crop_size) // 2
        y = (height - crop_size) // 2
        cropped = image[y : y + crop_size, x : x + crop_size]
        return cv2.resize(
            cropped,
            (self.config.board_size, self.config.board_size),
            interpolation=cv2.INTER_AREA,
        )

    def _make_tiles(self) -> list[np.ndarray]:
        tile_size = self.config.board_size // self.config.grid_size
        tiles = []
        for row in range(self.config.grid_size):
            for column in range(self.config.grid_size):
                x = column * tile_size
                y = row * tile_size
                tiles.append(self._source[y : y + tile_size, x : x + tile_size].copy())
        return tiles

    def _is_valid_position(self, position: int) -> bool:
        return 0 <= position < len(self._tiles)
