from enum import IntEnum

from PIL import ImageFont
from numpy import array as np_array


class Languages(IntEnum):
    Chinese = 0
    Latin = 1
    Special = 2


characters = {
    Languages.Chinese: np_array(
        ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
         "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", ' ']),
    Languages.Latin: np_array(
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
         'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']),
    Languages.Special: np_array(
        ['⬤', '●', '-', '▊', '▗', '▁', '▂', '▃', '▅', '▆', '▇', '▉', '▊', '▋', '▎', '▏', '▔', '▕', '▖', '▗', '▘', '▙',
         '▚', '▛', '▜', '▝', '▞', '▟', '/', '\\'])
}

char_language_distribution = np_array([len(chars) for chars in characters.values()])
char_language_distribution = char_language_distribution / sum(char_language_distribution)


class Font:
    def __init__(self, path: str):
        self.path = path
        self.cache = {}

    def instantiate(self, size: int) -> ImageFont:
        if size in self.cache:
            return self.cache[size]
        return ImageFont.truetype(self.path, size)

    def cache_font(self, size: int, font: ImageFont):
        self.cache[size] = font
