import argparse
import dataclasses
import json
import logging
from time import process_time

from src.OCR.image_gen.RNG import RandomTextImageGenerator, RandomTextGenerator
from src.base import IO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of generated images', type=str, required=True)

    parser.add_argument('--width', help='Width of generated images', type=int, default=1387)
    parser.add_argument('--height', help='Height of generated images', type=int, default=445)
    parser.add_argument('--fit_to_text', help='Images fit to text', action='store_true', default=False)

    parser.add_argument('--margin_xy', help='margin of generated images', type=int)
    parser.add_argument('--margin_x', help='Horizontal margin of generated images', type=int, default=5)
    parser.add_argument('--margin_y', help='Vertical margin of generated images', type=int, default=5)

    parser.add_argument('--fixed_width', help='Chars have a fixed width', action='store_true', default=False)
    parser.add_argument('--fixed_height', help='Chars have a fixed height', action='store_true', default=False)

    parser.add_argument('--max_chars', help='Max chars', type=int, default=8)
    parser.add_argument('--min_chars', help='Max chars', type=int, default=7)

    parser.add_argument('--max_char_width', help='Max width of chars', type=int)
    parser.add_argument('--max_char_height', help='Max height of chars', type=int)
    parser.add_argument('--min_char_height', help='Min height of chars', type=int, default=10)
    parser.add_argument('--min_char_width', help='Min width of chars', type=int, default=10)

    parser.add_argument('--num_train', help='Number of generated train images', type=int, default=1000)
    parser.add_argument('--num_test', help='Number of generated test images', type=int, default=0)
    parser.add_argument('--num_val', help='Number of generated val images', type=int, default=0)

    parser.add_argument('--logging', help='Logging step', type=float, default=0.1)

    args = parser.parse_args()

    size = (args.width, args.height)
    if isinstance(args.margin_xy, (int, float)):
        args.margin_x = args.margin_xy
        args.margin_y = args.margin_xy

    languages = IO.read_languages()
    text_rng = RandomTextGenerator(
        languages,
        min_actual_height=args.min_char_height,
        max_actual_height=args.max_char_height if args.fit_to_text or args.max_char_height else size[1],
        min_actual_width=args.min_char_width,
        max_actual_width=args.max_char_width if args.fit_to_text or args.max_char_width else args.width / args.max_chars - 5,
        min_chars=args.min_chars, max_chars=args.max_chars,
        fixed_width=args.fixed_width, fixed_height=args.fixed_height
    )
    img_rng_gen = RandomTextImageGenerator(
        size, text_rng,
        fit_to_text=args.fit_to_text, margin_x=args.margin_x, margin_y=args.margin_y
    )
    root = args.save_dir

    with open(IO.create_path(root) / 'languages.json', 'w+') as fh:
        log.info('Dumping language.json...')

        all_chars = []
        for lang in languages:
            all_chars += lang.chars

        language_json = {
            'vocab': all_chars,
            'languages': [dataclasses.asdict(lang) for lang in languages],
            'noise': IO.read_json('noise.json')
        }
        json.dump(language_json, fh)

    datasets = ['train', 'test', 'val']
    nums = [args.num_train, args.num_test, args.num_val]

    for dataset, num in zip(datasets, nums):
        if num == 0:
            continue
        dataset_path = IO.create_path(root, dataset)
        logging_step = num * args.logging
        log.info(f'Writing {num} {dataset} images to path {dataset_path}...')
        start = process_time()
        for idx, text_img in enumerate(img_rng_gen(num), start=1):
            path = IO.create_path(dataset_path, str(len(text_img.text)), text_img.img_type)
            text_img.img.save(str(path / f'#{idx}_{text_img.to_string()}.jpg'))

            if idx % logging_step == 0:
                elapsed = process_time() - start
                log.info('%s: %i; seconds since start: %f, s/element: %f', dataset, idx, elapsed, elapsed / idx)

            if idx == num:
                break
