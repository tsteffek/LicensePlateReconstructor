import argparse
import os
from pathlib import Path

from src.LPR.CCPD.model.Image import CCPDImage
from src.base import IO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str)
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)

    args = parser.parse_args()
    split = args.split

    path = Path(args.source)
    target_path = Path(args.target)
    split_files_path = path / 'splits'

    resize = (277, 89)

    with open(os.path.join(split_files_path / (split + '.txt'))) as fh:
        file_names = [line.rstrip() for line in fh]

    print(f'Moving {len(file_names)} for split {split}')

    log = int(len(file_names) * 0.05)
    for idx, file_name in enumerate(file_names):
        img = CCPDImage.load(path / file_name, cropped=True).img
        img = img.resize(resize)

        new_file_path = target_path / split / file_name
        IO.create_path(new_file_path.parent)
        img.save(new_file_path)
        if idx % log == 0:
            print(f'{split}: {idx}')
