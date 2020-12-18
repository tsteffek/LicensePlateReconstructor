from typing import List, Tuple

import numpy as np
from PIL import ImageDraw
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont

from src.OCR.data.Fonts_and_Languages import Languages, characters


class Text:
    def __init__(self, chars_with_fonts: List[Tuple[Languages, str, FreeTypeFont]]):
        self.chars_with_fonts = chars_with_fonts
        if len(self.chars_with_fonts) == 0:
            return

        self.widths, self.heights = zip(*(font.getsize(self.get_char(lang, char)) for lang, char, font in chars_with_fonts))

        offsets = [0]
        for idx, width in enumerate(self.widths):
            offsets.append(width + offsets[idx] + 1)

        self.offsets = np.array(offsets)
        self.height = max(self.heights)
        self.width = self.offsets[-1]

    def draw_to_image_centered(self, img: Image, fill):
        W, H = img.size
        draw_ctx = ImageDraw.Draw(img)
        for idx, (lang, char_idx, font) in enumerate(self.chars_with_fonts):
            draw_ctx.text(((W - self.width) / 2 + self.offsets[idx], (H - self.heights[idx]) / 2),
                          self.get_char(lang, char_idx), fill=fill,
                          font=font)

    def get_char(self, language, char_idx):
        return characters[language][char_idx]

    def to_string(self):
        return [''.join(
            [self.get_char(lang, char_idx) for lang, char_idx, font in self.chars_with_fonts])] + self.chars_with_fonts
