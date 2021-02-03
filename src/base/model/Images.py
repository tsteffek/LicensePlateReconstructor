from dataclasses import dataclass
from typing import Union, List

from PIL.Image import Image

from src.base.model.Texts import Text


@dataclass
class ImageWithText:
    img: Image
    text: Union[List[str], Text]
