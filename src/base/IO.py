import glob
import json
import logging
import os
from os import path
from pathlib import Path
from typing import List, Any, Tuple, Union

from PIL import Image

from src.OCR.image_gen.model.Language import Language

log = logging.getLogger(__name__)


def locate_data_directory(data_dir_name='res'):
    data_path = Path(os.getcwd()) / data_dir_name
    try:
        move_upwards = ''
        while not data_path.exists():
            move_upwards += '../'
            data_path = Path(move_upwards) / data_dir_name
        return data_path
    except OSError:
        raise IOError(f'{data_dir_name} folder not found')


DATA_PATH = locate_data_directory()


def create_path(*paths) -> Path:
    p = Path(os.path.join(*paths))
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_json(file_name: str, data_path: Path = DATA_PATH) -> Any:
    with open(data_path / file_name, 'r') as file:
        return json.load(file)


def read_languages(data_path: Union[str, Path] = DATA_PATH / 'img_gen_config'):
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


def load_resource_image(image_path: str) -> Image:
    return load_image(DATA_PATH / image_path)
