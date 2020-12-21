from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import ImageDraw
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont

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

    @property
    def font(self) -> FreeTypeFont:
        return self.language[self.font_idx]

    def to_string(self) -> str:
        return f'{self.language.id}-{self.char_idx}-{self.font_idx}-{self.size}'


class Text:
    def __init__(self, chars_with_fonts: List[Tuple[Character, FreeTypeFont]]):
        self.chars_with_fonts = chars_with_fonts
        if len(self.chars_with_fonts) == 0:
            self.width = 0
            self.height = 0
            self.offsets = [0]
            self.heights = []
            self.widths = []
            return

        self.widths, self.heights = zip(
            *(font.getsize(char.char) for (char, font) in self.chars_with_fonts))

        offsets = [0]
        for idx, width in enumerate(self.widths):
            offsets.append(width + offsets[idx] + 1)

        self.offsets = np.array(offsets)
        self.height = max(self.heights)
        self.width = self.offsets[-1]

    def draw_to_image_centered(self, img: Image, fill):
        W, H = img.size
        draw_ctx = ImageDraw.Draw(img)
        text_start = (W - self.width) / 2
        for (char, font), offset, height in zip(self.chars_with_fonts, self.offsets, self.heights):
            left_upper_point = (text_start + offset, (H - height) / 2)
            draw_ctx.text(left_upper_point, char.char, fill=fill, font=font)

    def to_string(self):
        return '_'.join([char.to_string() for char, _ in self.chars_with_fonts])

    def __len__(self):
        return len(self.chars_with_fonts)


@dataclass
class TextImage:
    text: Text
    img: Image
    img_type: str

    def to_string(self):
        return f'{self.img_type}_{self.text.to_string()}'
