import argparse
import multiprocessing
from multiprocessing.queues import Queue
from queue import Empty

from PIL.Image import Image

from src.base import IO


def repeat(queue: Queue, resize):
    try:
        while True:
            load_resize_save(queue.get(True, 5), resize)
    except Empty:
        return


def load_resize_save(image_file, resize):
    img: Image = IO.load_image(image_file)
    img = img.resize(resize)
    img.save(image_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--target_size', type=float)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--image_glob', type=str, default='**/*.jpg')

    args = parser.parse_args()

    images = IO.get_image_paths(args.source, args.image_glob)

    resize = IO.load_image(images[0]).size
    resize = tuple(int(args.target_size * el) for el in resize)

    multiprocessing.set_start_method('spawn')
    q = multiprocessing.Queue()
    for img_file in images:
        q.put(img_file)

    processes = [
        multiprocessing.Process(target=repeat, args=(q, resize), daemon=True) for w in range(args.worker)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()
