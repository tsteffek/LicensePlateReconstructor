import logging
import os
from typing import Tuple, List, Callable, Union

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.base import IO
from src.base.model import ImageWithText, Text

log = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, path: str,
                 load_fn: Callable[[str], ImageWithText], encode_fn: Callable[[Union[List[str], Text]], List[int]],
                 image_file_glob: str = '**/*.jpg', precision: int = 32,
                 target_size: Union[float, Tuple[int, int]] = None):
        super().__init__()
        self.path = os.path.join(path, image_file_glob)
        self.load_fn = load_fn
        self.encode_fn = encode_fn
        self.dtype = torch.float16 if precision == 16 else torch.float32

        self.images = IO.get_image_paths(self.path)
        log.warning(f'Detected {len(self.images)} images in {self.path}')

        if type(target_size) is tuple:
            self.resize = target_size
        else:  # it's for scaling
            self.resize = IO.load_image(self.images[0]).size
            if type(target_size) is float:
                self.resize = tuple(int(target_size * el) for el in self.resize)

        w, h = self.resize
        self.size = h, w

    def __getitem__(self, item) -> Tuple[Tensor, Tuple[Text, Tensor]]:
        image_path = self.images[item]
        ti = self.load_fn(image_path)
        ti.img = ti.img.resize(self.resize)
        labels = self.encode_fn(ti.text)
        return to_torch_format(ti.img, self.dtype), (ti.text, torch.tensor(labels, dtype=torch.int64))

    def __len__(self):
        return len(self.images)


def to_torch_format(img: Image, dtype: torch.dtype) -> Tensor:
    return torch.tensor(img.getdata(), dtype=dtype).reshape(*img.size, -1).T / 255
