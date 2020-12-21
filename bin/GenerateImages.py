import argparse
import dataclasses
import json
import os
from pathlib import Path
from time import process_time

from src.OCR.image_gen.IO import read_languages
from src.OCR.image_gen.RNG import RandomTextGenerator, RandomTextImageGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of generated images', type=str)
    parser.add_argument('--width', help='Width of generated images', type=int, default=800*4)
    parser.add_argument('--height', help='Height of generated images', type=int, default=128*4)

    args = parser.parse_args()

    size = (args.width, args.height)

    languages = read_languages()
    text_rng = RandomTextGenerator(languages, size[1])
    img_rng_gen = RandomTextImageGenerator(size, text_rng)()
    root = args.save_dir
    print('writing to', root)

    idx = 0
    with open(os.path.join(root, 'languages.json'), 'w+') as fh:
        json.dump([dataclasses.asdict(lang) for lang in languages], fh)

    start = process_time()
    for text_img in img_rng_gen:
        path = Path(os.path.join(root, f'{len(text_img.text)}/{text_img.img_type}'))
        path.mkdir(parents=True, exist_ok=True)
        text_img.img.save(str(path / f'{text_img.to_string()}.jpg'))
        idx += 1
        if idx % 10 == 0:
            elapsed = process_time() - start
            print(idx, elapsed, elapsed / idx)

        if idx == 100:
            break
