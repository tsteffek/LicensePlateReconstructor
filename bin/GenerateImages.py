import argparse
import dataclasses
import json
import logging
from time import process_time

from OCR.IO import read_languages, create_path
from OCR.image_gen.RNG import RandomTextGenerator, RandomTextImageGenerator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of generated images', type=str, required=True)
    parser.add_argument('--width', help='Width of generated images', type=int, default=800 * 4)
    parser.add_argument('--height', help='Height of generated images', type=int, default=128 * 4)
    parser.add_argument('--num_train', help='Number of generated train images', type=int, default=1000)
    parser.add_argument('--num_test', help='Number of generated test images', type=int, default=0)
    parser.add_argument('--num_val', help='Number of generated val images', type=int, default=0)
    parser.add_argument('--logging', help='Logging step', type=float, default=0.1)

    args = parser.parse_args()

    size = (args.width, args.height)

    languages = read_languages()
    text_rng = RandomTextGenerator(languages, size[1])
    img_rng_gen = RandomTextImageGenerator(size, text_rng)
    root = args.save_dir

    with open(create_path(root) / 'languages.json', 'w+') as fh:
        log.info('Dumping language.json...')
        json.dump([dataclasses.asdict(lang) for lang in languages], fh)

    datasets = ['train', 'test', 'val']
    nums = [args.num_train, args.num_test, args.num_val]

    start = process_time()
    for dataset, num in zip(datasets, nums):
        dataset_path = create_path(root, dataset)
        logging_step = num * args.logging
        log.info(f'Writing {num} {dataset} images to path {dataset_path}...')
        for idx, text_img in enumerate(img_rng_gen(), start=1):
            path = create_path(dataset_path, str(len(text_img.text)), text_img.img_type)
            text_img.img.save(str(path / f'{text_img.to_string()}.jpg'))

            if idx % logging_step == 0:
                elapsed = process_time() - start
                log.info('%s: %i; seconds since start: %f, s/element: %f', dataset, idx, elapsed, elapsed / idx)

            if idx == num:
                break
