from typing import Tuple

import numpy as np
from PIL import Image


def one_color(mode: str, size: Tuple[int, int], color: Tuple[int, int, int]) -> Image:
    return Image.new(mode, size, color)


def two_tone_horizontal(mode: str, w: int, h: int, split: int, color1: Tuple[int, int, int],
                        color2: Tuple[int, int, int]) -> Image:
    arr = np.full((h, w, 3), color1, dtype=np.uint8)
    arr[:, split:] = color2
    return Image.fromarray(arr, mode)


def two_tone_vertical(mode: str, w: int, h: int, split: int, color1: Tuple[int, int, int],
                      color2: Tuple[int, int, int]) -> Image:
    arr = np.full((h, w, 3), color1, dtype=np.uint8)
    arr[split:, :] = color2
    return Image.fromarray(arr, mode)


def gradient_horizontal(mode: str, w: int, h: int, color1: Tuple[int, int, int],
                        color2: Tuple[int, int, int]) -> Image:
    gradient = np.array([
        np.linspace(color_component1, color_component2, w)
        for color_component1, color_component2 in zip(color1, color2)
    ], dtype=np.uint8).T
    image = np.repeat(gradient[np.newaxis, :], h, 0)
    return Image.fromarray(image, mode)


def gradient_vertical(mode: str, w: int, h: int, color1: Tuple[int, int, int],
                      color2: Tuple[int, int, int]) -> Image:
    gradient = np.array([
        np.linspace(color_component1, color_component2, h)
        for color_component1, color_component2 in zip(color1, color2)
    ], dtype=np.uint8).T
    image = np.repeat(gradient[:, np.newaxis], w, 1)
    return Image.fromarray(image, mode)
