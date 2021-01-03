import math
from functools import lru_cache
from typing import List, Tuple, Any, Union, Generator

import numpy as np
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont
from numpy import ndarray, array

from OCR.image_gen.model import Images
from OCR.image_gen.model.Language import Language, FontCache
from OCR.image_gen.model.Text import Character, TextImage, ImageText

# rng = np.random.default_rng(42)
rng = np.random.default_rng()


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
                 min_actual_height: int, max_actual_height: int, max_actual_width: int, fixed_size: bool = False):
        self.language = language
        self.min_actual_height = min_actual_height
        self.max_actual_height = max_actual_height
        self.max_actual_width = max_actual_width
        self.font_caches = {font: FontCache(font) for font in self.language.fonts}

        self.fixed_size = fixed_size
        if fixed_size:
            self.fitting_size = {
                char:
                    {font: self.find_fitting_font_size(char, font) for font in language.fonts}
                for char in language
            }
            self.font_caches = None  # delete caches afterwards, because they're bloated
        else:
            self.min_rand_size = min_rand_height
            self.max_rand_sizes = {font: max_rand_height for font in self.language.fonts}
            self.step = step

    def __call__(self) -> Tuple[Character, FreeTypeFont]:
        char_idx, char = random_elem(self.language.chars)
        font_idx, font = random_elem(self.language.fonts)
        if self.fixed_size:
            size, sized_font = self.fitting_size[char][font]
        else:
            size, sized_font = self.find_acceptable_font_size(char, font)
        return Character(self.language, char_idx, font_idx, size), sized_font

    def find_acceptable_font_size(self, char: str, font: str) -> Tuple[int, FreeTypeFont]:
        font_cache = self.font_caches[font]
        while True:
            picked_size = random_value(self.min_rand_size, self.max_rand_sizes[font], self.step)
            picked_font: FreeTypeFont = font_cache.get(picked_size)

            char_width, char_height = self._char_size(char, picked_font)
            if char_height < self.max_actual_height and char_width < self.max_actual_width:
                font_cache.cache_font(picked_size, picked_font)
                return picked_size, picked_font
            else:
                self.max_rand_sizes[font_cache] = picked_size

    def find_fitting_font_size(
            self, char: str, font: str, min_size: int = 0, max_size: int = math.inf
    ) -> Tuple[int, FreeTypeFont]:
        font_cache = self.font_caches[font]
        cur_size = 10
        idx = 0
        while True:
            cur_font = font_cache.get(cur_size)
            font_cache.cache_font(cur_size, cur_font)

            char_width, char_height = self._char_size(char, cur_font)

            if char_width + 1 < self.max_actual_width:
                min_size = cur_size
                cur_size = min(cur_size * 2, max_size - 1)
            elif self.max_actual_width < char_width - 1:
                max_size = cur_size
                cur_size = max(min_size + 1, int((max_size - min_size) / 2) + min_size)
            else:
                return cur_size, cur_font
            idx += 1
            if idx > 200:
                raise ValueError('Couldn\'t find a matching size')

    @staticmethod
    def _char_size(char: str, font: FreeTypeFont) -> int:
        return font.getsize(char)


class RandomTextGenerator:
    def __init__(self, languages: List[Language], max_actual_height: int, max_actual_width: int, max_chars: int = 15,
                 min_chars: int = 0,
                 random_number_of_chars: bool = True, min_rand_size: int = 20, max_rand_size: int = 1000,
                 size_step: int = 5, min_actual_height: int = 10, fixed_size: bool = False):
        self.languages = np.array(languages, dtype=object)
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.random_num = random_number_of_chars
        self.char_rngs = {
            language: RandomCharacterGenerator(
                language,
                min_rand_size, max_rand_size, size_step,
                min_actual_height, max_actual_height, max_actual_width,
                fixed_size
            ) for language in self.languages
        }

    @lru_cache(maxsize=None)
    def char_language_distribution(self) -> ndarray:
        num_chars_per_language = array([len(chars) for chars in self.languages])
        return num_chars_per_language / sum(num_chars_per_language)

    def __call__(self) -> Generator[ImageText, None, None]:
        if self.random_num:
            while True:
                yield self.pick_random_text(rng.integers(self.min_chars, self.max_chars, endpoint=True))
        else:
            while True:
                yield self.pick_random_text(self.max_chars)

    def create_evenly(self, num: int, size: Tuple[int, int]):
        num_per_num = math.ceil(num / (self.max_chars - self.min_chars + 1))
        for char_num in range(self.min_chars, self.max_chars + 1):
            idx = 0
            while idx < num_per_num:
                text = self.pick_random_text(char_num)
                if text.width > size[0] or text.height > size[1]:
                    continue
                idx += 1
                yield text

    def pick_random_text(self, char_num):
        random_languages = choice(self.languages, char_num, self.char_language_distribution())
        random_characters = zip(*[self.char_rngs[lang]() for lang in random_languages])
        return ImageText(*random_characters)


class RandomTextImageGenerator:
    def __init__(self, size: Tuple[int, int], text_rng: RandomTextGenerator, mode: str = 'RGB'):
        self.size = size
        self.text_rng = text_rng
        self.mode = mode

    def __call__(self, num: int = None) -> Generator[TextImage, None, None]:
        if num is None:
            text_gen = self.text_rng()
        else:
            text_gen = self.text_rng.create_evenly(num, self.size)

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
