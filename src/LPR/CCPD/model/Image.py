from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple, List

from src.base import IO
from src.base.model import ImageWithText

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def read_lp_indices(indices: List[int]) -> List[str]:
    return [provinces[indices[0]]] + [alphabets[indices[1]]] + [ads[idx] for idx in indices[2:]]


Coordinate = namedtuple('Coordinate', 'x y')


@dataclass
class CCPDImage(ImageWithText):
    area: int
    tilt: Tuple[int, int]
    bb_coords: Tuple[Coordinate, ...]
    four_vertices: Tuple[Coordinate, ...]
    brightness: int
    blurriness: int

    @classmethod
    def load(cls, path: str, cropped: bool = True):
        file_name = IO.file_name(path)
        area, tilt, bb_coords_raw, four_vertices_raw, lp_num_raw, brightness, blurriness = \
            split(file_name, ['-', '_', '&'])

        bb_coords = tuple([Coordinate(*raw) for raw in bb_coords_raw])
        four_vertices = tuple([Coordinate(*raw) for raw in four_vertices_raw])
        text = read_lp_indices(lp_num_raw)

        img = IO.load_image(path)

        if cropped:
            left_upper, right_lower = bb_coords
            img = img.crop((*left_upper, *right_lower))

        return cls(img, text, area, tilt, bb_coords, four_vertices, brightness, blurriness)


def split(string: str, splits: List[str], idx: int = 0):
    if len(splits) > idx and splits[idx] in string:
        return tuple(split(s, splits, idx + 1) for s in string.split(splits[idx]))
    else:
        return int(string)
