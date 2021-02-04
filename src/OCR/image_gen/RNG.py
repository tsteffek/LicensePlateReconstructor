import math
from functools import lru_cache
from typing import List, Tuple, Any, Union, Generator

import numpy as np
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont
from numpy import ndarray, array

from src.OCR.GeneratedImages.model.Image import TypedImageWithText
from src.OCR.image_gen.model import Images
from src.OCR.image_gen.model.Language import Language, FontCache
from src.base.model import Character
from src.base.model.Texts import ImageText

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
    def __init__(
            self,
            language: Language,
            min_rand_height: int,
            max_rand_height: int,
            step: int,
            min_actual_height: int,
            max_actual_height: int,
            min_actual_width: int,
            max_actual_width: int,
            fixed_width: bool = False,
            fixed_height: bool = False
    ):
        self.language = language
        self.min_actual_height = min_actual_height
        self.max_actual_height = max_actual_height
        self.min_actual_width = min_actual_width
        self.max_actual_width = max_actual_width
        self.font_caches = {font: FontCache(font) for font in self.language.fonts}

        if fixed_width and fixed_height:
            raise ValueError('Can\'t fix both height and width!')
        elif fixed_width:
            self.build_fixed_size_dict(max_actual_width, fixed_width=True)
        elif fixed_height:
            self.build_fixed_size_dict(max_actual_height, fixed_width=False)
        else:
            self.min_rand_sizes = {font: min_rand_height for font in self.language.fonts}
            self.max_rand_sizes = {font: max_rand_height for font in self.language.fonts}
            self.step = step
            self.find_size = self._find_acceptable_font_size

    def build_fixed_size_dict(self, max_char_size: int, fixed_width: bool = False):
        fitting_size = {
            char:
                {
                    font: self._find_fitting_font_size(char, font, max_char_size, fixed_width)
                    for font in self.language.fonts
                }
            for char in self.language
        }
        self.font_caches = None  # delete caches afterwards, because they're bloated and unnecessary
        self.find_size = lambda char, font: fitting_size[char][font]

    def _find_fitting_font_size(
            self, char: str, font: str, max_char_size: int, fixed_width: bool = False,
            start_size: int = 10, min_size: int = 0, max_size: int = math.inf
    ) -> Tuple[int, FreeTypeFont]:
        font_cache = self.font_caches[font]
        idx = 0
        size_elem = 0 if fixed_width else 1
        while True:
            cur_font = font_cache.get(start_size)
            font_cache.cache_font(start_size, cur_font)

            # width, height
            char_size = self._char_size(char, cur_font)[size_elem]

            if char_size + 1 < max_char_size:
                min_size = start_size
                start_size = min(start_size * 2, max_size - 1)
            elif max_char_size < char_size - 1:
                max_size = start_size
                start_size = max(min_size + 1, int((max_size - min_size) / 2) + min_size)
            else:
                return start_size, cur_font
            idx += 1
            if idx > 200:
                raise ValueError('Couldn\'t find a matching size')

    def __call__(self) -> Tuple[Character, FreeTypeFont]:
        char_idx, char = random_elem(self.language.chars)
        font_idx, font = random_elem(self.language.fonts)
        size, sized_font = self.find_size(char, font)

        return Character(self.language, char_idx, font_idx, size), sized_font

    def _find_acceptable_font_size(self, char: str, font: str) -> Tuple[int, FreeTypeFont]:
        font_cache = self.font_caches[font]
        while True:
            picked_size = random_value(self.min_rand_sizes[font], self.max_rand_sizes[font], self.step)
            picked_font: FreeTypeFont = font_cache.get(picked_size)

            char_width, char_height = self._char_size(char, picked_font)
            if self.min_actual_height > char_height or self.min_actual_width > char_width:
                self.min_rand_sizes[font_cache] = picked_size
            elif char_height > self.max_actual_height or char_width > self.max_actual_width:
                self.max_rand_sizes[font_cache] = picked_size
            else:
                font_cache.cache_font(picked_size, picked_font)
                return picked_size, picked_font

    @staticmethod
    def _char_size(char: str, font: FreeTypeFont) -> Tuple[int, int]:
        return font.getbbox(char, anchor='lt')[-2:]


class RandomTextGenerator:
    def __init__(
            self,
            languages: List[Language],
            max_actual_height: int,
            max_actual_width: int,
            min_actual_height: int = 0,
            min_actual_width: int = 0,
            min_chars: int = 0,
            max_chars: int = 15,
            random_number_of_chars: bool = True,
            min_rand_size: int = 20,
            max_rand_size: int = 1000,
            size_step: int = 5,
            fixed_width: bool = False,
            fixed_height: bool = False
    ):
        self.languages = np.array(languages, dtype=object)
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.random_num = random_number_of_chars
        self.char_rngs = {
            language: RandomCharacterGenerator(
                language,
                min_rand_size, max_rand_size, size_step,
                min_actual_height, max_actual_height,
                min_actual_width, max_actual_width,
                fixed_width, fixed_height
            ) for language in self.languages
        }

    @lru_cache(maxsize=None)
    def char_language_distribution(self) -> ndarray:
        num_chars_per_language = array([len(chars) for chars in self.languages])
        return num_chars_per_language / sum(num_chars_per_language)

    def __call__(self, size: Tuple[int, int]) -> Generator[ImageText, None, None]:
        while True:
            text = self.pick_random_text(
                rng.integers(self.min_chars, self.max_chars, endpoint=True) if self.random_num else self.max_chars
            )
            if text.width > size[0] or text.height > size[1]:
                continue
            yield text

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
    def __init__(
            self,
            size: Tuple[int, int],
            text_rng: RandomTextGenerator,
            fit_to_text: bool = False,
            margin_x: int = 0,
            margin_y: int = 0,
            mode: str = 'RGB'
    ):
        self.size = size
        self.text_rng = text_rng
        self.fit_to_text = fit_to_text
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.size_without_margin = size[0] + margin_x, size[1] + margin_y
        if self.size_without_margin[0] <= 0 or self.size_without_margin[1] <= 0:
            raise ValueError('Margins too big for the image!')
        self.mode = mode

    def __call__(self, num: int = None) -> Generator[TypedImageWithText, None, None]:
        if num is None:
            text_gen = self.text_rng(
                self.size_without_margin if not self.fit_to_text else (math.inf, math.inf)
            )
        else:
            text_gen = self.text_rng.create_evenly(
                num, self.size_without_margin if not self.fit_to_text else (math.inf, math.inf)
            )

        size = self.size
        for text in text_gen:
            if self.fit_to_text:
                size = text.width + self.margin_x, text.height + self.margin_y

            img_type, img = self.random_image(self.mode, size)
            text.draw_to_image_centered(img, random_color())
            yield TypedImageWithText(img, text, img_type)

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
