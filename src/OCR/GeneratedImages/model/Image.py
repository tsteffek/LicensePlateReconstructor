from dataclasses import dataclass
from typing import List

from src.OCR.image_gen.model.Language import Language
from src.base import IO
from src.base.model import ImageWithText, Text


@dataclass
class TypedImageWithText(ImageWithText):
    img_type: str

    def to_string(self):
        return f'{self.img_type}_{self.text.to_string()}'

    @classmethod
    def load(cls, path: str, languages: List[Language]):
        file_name = IO.file_name(path)
        if file_name.startswith('#'):
            _, img_type, text_str = file_name.split('_', 2)
        else:
            img_type, text_str = file_name.split('_', 1)

        img = IO.load_image(path)
        text = Text.from_string(text_str, languages)

        return cls(img, text, img_type)
