import csv
from time import process_time
from typing import List, Tuple, Generator, Any

import numpy as np
from PIL import Image
from PIL.ImageFont import ImageFont
from numpy import ndarray

from src.OCR.data.Fonts_and_Languages import Languages, char_language_distribution, characters
from src.OCR.data.Text import Text
from src.OCR.data._io import read_fonts_for_language

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


def random_weighted_language(num, char_language_distribution) -> List[Languages]:
    return rng.choice(len(Languages), num, p=char_language_distribution, shuffle=False)


def random_character(language: Languages) -> str:
    return rng.choice(len(characters[language]), shuffle=False)


def random_value(min_value: int, max_value: int, step: int) -> int:
    return rng.choice(np.arange(min_value, max_value, step), shuffle=False)


def random_color():
    return tuple(rng.integers(0, 256, 3))


def random_image(mode, size):
    case = rng.integers(3)

    if case == 0:
        return Image.new(mode, size, random_color())

    w, h = size

    if case == 1:
        arr = np.full((h, w, 3), random_color(), dtype=np.uint8)
        if rng.integers(2) == 0:
            arr[:, rng.integers(w):] = random_color()
            return Image.fromarray(arr, mode)
        else:
            arr[rng.integers(h):, :] = random_color()
            return Image.fromarray(arr, mode)
    if case == 2:
        color1 = random_color()
        color2 = random_color()
        if rng.integers(2) == 0:
            gradient = np.array([
                np.linspace(color_component1, color_component2, w)
                for color_component1, color_component2 in zip(color1, color2)
            ], dtype=np.uint8).T
            image = np.repeat(gradient[np.newaxis, :], h, 0)
            return Image.fromarray(image, mode)
        else:
            gradient = np.array([
                np.linspace(color_component1, color_component2, h)
                for color_component1, color_component2 in zip(color1, color2)
            ], dtype=np.uint8).T
            image = np.repeat(gradient[:, np.newaxis], w, 1)
            return Image.fromarray(image, mode)


def choice(arr: ndarray):
    return rng.choice(arr)


class LanguageFontRandomizer:
    def __init__(
            self,
            language: Languages, min_rand_height: int, max_rand_height: int, step: int, root=None
    ):
        self.language = language
        self.fonts = read_fonts_for_language(language, root)
        self.min_rand_size = min_rand_height
        self.max_rand_sizes = {font: max_rand_height for font in self.fonts}
        self.step = step

    def __call__(self, max_actual_height: int) -> ImageFont:
        font = choice(self.fonts)
        while True:
            picked_value = random_value(self.min_rand_size, self.max_rand_sizes[font], self.step)
            picked_font = font.instantiate(picked_value)

            if self._actual_font_size_ok(picked_font, max_actual_height):
                font.cache_font(picked_value, picked_font)
                return picked_font
            else:
                self.max_rand_sizes[font] = picked_value

    @staticmethod
    def _actual_font_size_ok(font: ImageFont, max_actual_size: int):
        return font.getsize('AgÄ')[1] < max_actual_size


class RandomTextGenerator:
    def __init__(self, char_num: int = 10, languages: List[Languages] = Languages, random_number_of_chars: bool = False,
                 min_rand_size: int = 20, max_rand_size: int = 1000, size_step: int = 5):
        self.char_num = char_num
        self.languages = languages
        self.random_num = random_number_of_chars
        self.min_rand_size = min_rand_size
        self.max_rand_size = max_rand_size
        self.size_step = size_step
        self.lf_rngs = np.array([
            LanguageFontRandomizer(language, self.min_rand_size, self.max_rand_size, self.size_step) for
            language in self.languages
        ])

    def __call__(self, max_actual_height: int) -> Generator[Text, None, None]:
        if self.random_num:
            while True:
                yield Text([
                    (language, random_character(language), self.lf_rngs[language](max_actual_height))
                    for language in random_weighted_language(rng.integers(self.char_num), char_language_distribution)
                ])
        else:
            while True:
                yield Text([
                    (language, random_character(language), self.lf_rngs[language](max_actual_height))
                    for language in random_weighted_language(self.char_num, char_language_distribution)
                ])


class RandomTextImageGenerator:
    def __init__(self, size: Tuple[int, int], text_rng: RandomTextGenerator, mode: str = 'RGB'):
        self.size = size
        self.text_rng = text_rng
        self.mode = mode

    def __call__(self):
        text_gen = self.text_rng(self.size[1])
        for text in text_gen:
            if text.width > self.size[0] or text.height > self.size[1]:
                print(text.width, text.height, self.size)
                continue

            img = random_image(self.mode, self.size)
            text.draw_to_image_centered(img, random_color())
            yield text, img


if __name__ == '__main__':
    text_rng = RandomTextGenerator()
    img_rng_gen = RandomTextImageGenerator((800*4, 128*4), text_rng)()
    root = '/mnt/f/Data/OCR_Generated/'
    print('writing to', root)

    idx = 0
    with open(f'{root}idx.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)

        start = process_time()
        for text, img in img_rng_gen:
            img.save(f'{root}{idx}.jpg')
            writer.writerow(text.to_string())
            idx += 1
            if idx % 100 == 0:
                elapsed = process_time() - start
                print(idx, elapsed, elapsed / idx)

            if idx == 1000:
                break
