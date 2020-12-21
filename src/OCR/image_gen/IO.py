import glob
import json
import os
from os import path

from src.OCR.image_gen.model.Language import Language

ROOT_DIR = '../'

DATA_PATH = os.path.join(ROOT_DIR, 'data')


def read_languages(data_path: str = DATA_PATH):
    languages = find_language_names(data_path)
    fonts = {lang: read_font_paths(data_path, lang) for lang in languages}
    chars = {lang: read_chars(data_path, lang) for lang in languages}
    return [Language(language, chars[language], fonts[language]) for language in languages]


def find_language_names(data_path: str):
    return [path.basename(x) for x in glob.glob(os.path.join(data_path, '*'))]


def read_font_paths(data_path: str, language: str, name='*.*tf'):
    return glob.glob(path.join(data_path, language, 'fonts', name))


def read_chars(data_path: str, language: str):
    char_path = path.join(data_path, language, 'chars.json')
    with open(char_path, 'r') as f:
        chars = json.load(f)
        return chars
