import glob
import json
import os
from os import path
from pathlib import Path
from typing import Dict, List, Any

from PIL import Image

from src.OCR.image_gen.model.Language import Language

ROOT_DIR = '../'

DATA_PATH = Path(os.path.join(ROOT_DIR, 'data'))


def create_path(*paths) -> Path:
    p = Path(os.path.join(*paths))
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_json(file_name: str, data_path: Path = DATA_PATH) -> Any:
    with open(data_path / file_name, 'r') as file:
        return json.load(file)


def read_languages(data_path: str = DATA_PATH):
    languages = find_language_names(data_path)
    fonts = {lang: read_font_paths(data_path, lang) for lang in languages}
    chars = {lang: read_chars(data_path, lang) for lang in languages}
    return [Language(language, chars[language], fonts[language]) for language in languages]


def load_languages(path: str, language_file: str = 'languages.json') -> Dict[int, Language]:
    with open(os.path.join(path, language_file)) as fh:
        languages = json.load(fh)
        return {lang['id']: Language.parse_obj(lang) for lang in languages}


def find_language_names(data_path: str):
    return [path.basename(x.rstrip('/')) for x in glob.glob(os.path.join(data_path, '*/'))]


def read_font_paths(data_path: str, language: str, name='*.*tf') -> List[str]:
    return glob.glob(path.join(data_path, language, 'fonts', name))


def get_image_paths(path: str, image_files: str) -> List[str]:
    return glob.glob(os.path.join(path, image_files), recursive=True)


def read_chars(data_path: str, language: str):
    char_path = path.join(data_path, language, 'chars.json')
    with open(char_path, 'r') as f:
        chars = json.load(f)
        return chars


def load_image(image_path: str) -> Image:
    img = Image.open(image_path)
    return img.getdata()
