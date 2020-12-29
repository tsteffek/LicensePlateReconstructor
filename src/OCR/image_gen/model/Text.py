from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from PIL import ImageDraw
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont

from src.OCR import IO
from src.OCR.image_gen.model.Language import Language


@dataclass
class Character:
    language: Language
    char_idx: int
    font_idx: int
    size: int

    @property
    def char(self) -> str:
        return self.language[self.char_idx]

    def __repr__(self):
        return self.char

    @property
    def font(self) -> FreeTypeFont:
        return self.language[self.font_idx]

    def to_string(self) -> str:
        return f'{self.language.id}-{self.char_idx}-{self.font_idx}-{self.size}'

    @classmethod
    def from_string(cls, string: str, languages: Dict[int, Language]):
        language_id, char_idx, font_idx, size = map(int, string.split('-'))
        return cls(languages[language_id], char_idx, font_idx, size)


class Text:
    def __init__(self, chars: List[Character] = None):
        if chars is None:
            chars = []
        self.chars = chars

    def to_string(self):
        return '_'.join([char.to_string() for char in self.chars])

    @classmethod
    def from_string(cls, string: str, languages: Dict[int, Language]):
        if len(string) == 0:
            return cls()

        char_strings = string.split('_')
        chars = [Character.from_string(char_string, languages) for char_string in char_strings]
        return cls(chars)

    def __len__(self):
        return len(self.chars)

    def __repr__(self):
        return ''.join([str(char) for char in self.chars])


class ImageText(Text):
    def __init__(self, chars: List[Character] = None, fonts: List[FreeTypeFont] = None):
        super().__init__(chars)
        if fonts is None:
            fonts = []
        self.fonts = fonts

        if len(self.chars) == 0:
            self.widths = []
            self.heights = []
            self.width = 0
            self.height = 0
            self.offsets = [0]
        else:
            self.widths, self.heights = list(zip(
                *(font.getsize(char.char) for (char, font) in zip(self.chars, self.fonts))
            ))

            offsets = [0]
            for idx, width in enumerate(self.widths):
                offsets.append(width + offsets[idx] + 1)

            self.offsets = np.array(offsets)
            self.height = max(self.heights)
            self.width = self.offsets[-1]

    def draw_to_image_centered(self, img: Image, fill: Tuple[int, int, int]):
        W, H = img.size
        draw_ctx = ImageDraw.Draw(img)
        text_start = (W - self.width) / 2
        for char, font, offset, height in zip(self.chars, self.fonts, self.offsets, self.heights):
            left_upper_point = (text_start + offset, (H - height) / 2)
            draw_ctx.text(left_upper_point, char.char, fill=fill, font=font)


@dataclass
class TextImage:
    text: Text
    img: Image
    img_type: str

    def to_string(self):
        return f'{self.img_type}_{self.text.to_string()}'

    @classmethod
    def load(cls, path: str, languages: Dict[int, Language]):
        file_name = IO.file_name(path)
        img_type, text_str = file_name.split('_', 1)

        img = IO.load_image(path)
        text = Text.from_string(text_str, languages)

        return cls(text, img, img_type)
