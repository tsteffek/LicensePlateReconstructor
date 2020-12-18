import glob

import numpy as np

from src.OCR.data.Fonts_and_Languages import Font, Languages

default_font_root = '../../../fonts/'


def read_fonts_for_language(language: Languages, root: str = None, ending: str = '.*tf') -> np.ndarray:
    if root is None:
        root = default_font_root
    paths = glob.glob(f'{root}/{language.name.lower()}/*{ending}')
    return np.array([Font(font_path) for font_path in paths])
