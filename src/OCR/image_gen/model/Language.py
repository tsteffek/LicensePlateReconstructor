from dataclasses import dataclass, field
from typing import List

from PIL import ImageFont
from PIL.ImageFont import FreeTypeFont


def id_gen():
    id = 0
    while True:
        yield id
        id += 1


ID_GEN = id_gen()


@dataclass
class Language:
    id: int = field(init=False)
    name: str
    chars: List[str]
    fonts: List[str]

    def __post_init__(self):
        self.id = next(ID_GEN)

    def __len__(self):
        return len(self.chars)

    def __iter__(self):
        return self.chars.__iter__()

    def __getitem__(self, item):
        return self.chars[item]

    def __hash__(self):
        return hash(self.name)


class FontCache:
    def __init__(self, path: str):
        self.path = path
        self.cache = {}

    def get(self, size: int) -> FreeTypeFont:
        if size in self.cache:
            return self.cache[size]
        return ImageFont.truetype(self.path, size)

    def cache_font(self, size: int, font: FreeTypeFont):
        self.cache[size] = font
