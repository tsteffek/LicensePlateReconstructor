import glob
import json
import logging
import os
from os import path
from pathlib import Path
from typing import List, Any, Tuple

from PIL import Image

from src.OCR.image_gen.model.Language import Language

log = logging.getLogger(__name__)

DATA_PATH = Path(os.getcwd()) / 'img_gen_config'
try:
    move_upwards = ''
    while not DATA_PATH.exists():
        move_upwards += '../'
        DATA_PATH = Path(move_upwards) / 'img_gen_config'
except OSError:
    raise IOError('img_gen_config folder not found')


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
    log.info(f'Detected {len(languages)} languages in {data_path}: {languages}')
    fonts = {lang: read_font_paths(data_path, lang) for lang in languages}
    chars = {lang: read_chars(data_path, lang) for lang in languages}
    return [Language(language, chars[language], fonts[language]) for language in languages]


def load_languages_file(file_path: str, language_file: str = 'languages.json') -> Tuple[List, List, List]:
    json_obj = read_json(language_file, Path(file_path))
    json_obj['languages'] = [Language.parse_obj(lang) for lang in json_obj['languages']]
    return json_obj['vocab'], json_obj['languages'], json_obj['noise']


def find_language_names(data_path: str):
    return [path.basename(x.rstrip('/')) for x in glob.glob(os.path.join(data_path, '*/'))]


def read_font_paths(data_path: str, language: str, name='*.*tf') -> List[str]:
    return glob.glob(path.join(data_path, language, 'fonts', name))


def get_image_paths(*paths: str) -> List[str]:
    return glob.glob(os.path.join(*paths), recursive=True)


def read_chars(data_path: str, language: str):
    char_path = path.join(data_path, language, 'chars.json')
    with open(char_path, 'r') as f:
        chars = json.load(f)
        return chars


def load_image(image_path: str) -> Image:
    return Image.open(image_path)
