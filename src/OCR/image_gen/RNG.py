from functools import lru_cache
from typing import List, Tuple, Any, Union, Generator

import numpy as np
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont
from numpy import ndarray, array

from src.OCR.image_gen import Images
from src.OCR.image_gen.model.Language import Language, FontCache
from src.OCR.image_gen.model.Text import Text, TextImage, Character

rng = np.random.default_rng(42)


# not quite sure whether needed
class RNG:
    def __init__(self, values: List[Any], dtype=np.uint16):
        self.values = values
        self.calls = np.zeros(len(values), dtype)

    def __next__(self, num=1):
        idx = rng.choice(len(self.values), num, p=1 - self.calls, shuffle=False)
        self.calls[idx] += 1
        return self.values[idx]


def random_value(min_value: int, max_value: int, step: int) -> int:
    return rng.choice(np.arange(min_value, max_value, step), shuffle=False)


def random_color() -> Tuple[int, int, int]:
    return tuple(rng.integers(0, 256, 3))


def random_bool():
    return rng.integers(2) == 0


def choice(arr: Union[ndarray, List], size: int = 1, p: Union[List[float], ndarray] = None):
    return rng.choice(arr, size, shuffle=False, p=p)


def random_elem(arr: Union[ndarray, List]):
    idx = rng.choice(len(arr), shuffle=False)
    return idx, arr[idx]


class RandomCharacterGenerator:
    def __init__(self, language: Language, min_rand_height: int, max_rand_height: int, step: int,
                 max_actual_height: int):
        self.language = language
        self.font_caches = {font: FontCache(font) for font in self.language.fonts}
        self.min_rand_size = min_rand_height
        self.max_rand_sizes = {font: max_rand_height for font in self.language.fonts}
        self.step = step
        self.max_actual_height = max_actual_height

    def __call__(self) -> Tuple[Character, FreeTypeFont]:
        char_idx, char = random_elem(self.language.chars)
        font_idx, font = random_elem(self.language.fonts)
        size, sized_font = self.find_acceptable_font_size(char, font)
        return Character(self.language, char_idx, font_idx, size), sized_font

    def find_acceptable_font_size(self, char: str, font: str) -> Tuple[int, FreeTypeFont]:
        font_cache = self.font_caches[font]
        while True:
            picked_size = random_value(self.min_rand_size, self.max_rand_sizes[font], self.step)
            picked_font: FreeTypeFont = font_cache.get(picked_size)

            if self._char_height(char, picked_font) < self.max_actual_height:
                font_cache.cache_font(picked_size, picked_font)
                return picked_size, picked_font
            else:
                self.max_rand_sizes[font_cache] = picked_size

    @staticmethod
    def _char_height(char: str, font: FreeTypeFont) -> int:
        return font.getsize(char)[1]


class RandomTextGenerator:
    def __init__(self, languages: List[Language], max_actual_height: int, char_num: int = 15,
                 random_number_of_chars: bool = True, min_rand_size: int = 20, max_rand_size: int = 1000,
                 size_step: int = 5):
        self.languages = np.array(languages, dtype=object)
        self.char_num = char_num
        self.random_num = random_number_of_chars
        self.char_rngs = {
            language: RandomCharacterGenerator(language, min_rand_size, max_rand_size, size_step, max_actual_height) for
            language in self.languages
        }

    @lru_cache(maxsize=None)
    def char_language_distribution(self) -> ndarray:
        num_chars_per_language = array([len(chars) for chars in self.languages])
        return num_chars_per_language / sum(num_chars_per_language)

    def __call__(self) -> Generator[Text, None, None]:
        if self.random_num:
            while True:
                languages = choice(self.languages, rng.integers(self.char_num), self.char_language_distribution())
                yield Text([self.char_rngs[lang]() for lang in languages])
        else:
            while True:
                languages = choice(self.languages, self.char_num, self.char_language_distribution())
                yield Text([self.char_rngs[lang]() for lang in languages])


class RandomTextImageGenerator:
    def __init__(self, size: Tuple[int, int], text_rng: RandomTextGenerator, mode: str = 'RGB'):
        self.size = size
        self.text_rng = text_rng
        self.mode = mode

    def __call__(self) -> Generator[TextImage, None, None]:
        text_gen = self.text_rng()
        for text in text_gen:
            if text.width > self.size[0] or text.height > self.size[1]:
                continue

            img_type, img = self.random_image(self.mode, self.size)
            text.draw_to_image_centered(img, random_color())
            yield TextImage(text, img, img_type)

    @staticmethod
    def random_image(mode: str, size: Tuple[int, int]) -> Tuple[str, Image]:
        case = rng.integers(3)

        if case == 0:
            return 'one-color', Images.one_color(mode, size, random_color())

        w, h = size

        color1 = random_color()
        color2 = random_color()

        if case == 1:
            if random_bool():
                return 'two-tone-h', Images.two_tone_horizontal(mode, w, h, rng.integers(w), color1, color2)
            else:
                return 'two-tone-v', Images.two_tone_vertical(mode, w, h, rng.integers(h), color1, color2)
        if case == 2:
            if random_bool():
                return 'gradient-h', Images.gradient_horizontal(mode, w, h, color1, color2)
            else:
                return 'gradient-v', Images.gradient_vertical(mode, w, h, color1, color2)
