import os
from typing import Tuple, Union

import torch

from src.OCR.GeneratedImages.model.Image import TypedImageWithText
from src.base import IO
from src.base.data import ImagesDataModule, ImageDataset


class GeneratedImagesDataModule(ImagesDataModule):
    def __init__(
            self,
            path: str,
            batch_size: int,
            multi_core: bool = True,
            cuda: bool = torch.cuda.is_available(),
            shuffle: bool = True,
            precision: int = 32,
            image_file_glob: str = '**/*.jpg',
            target_size: Union[float, Tuple[int, int]] = None,
            language_file: str = 'languages.json',
            **kwargs
    ):
        chars, self.languages, noise = IO.load_languages_file(path, language_file)
        super().__init__(path, chars, batch_size, multi_core, cuda, shuffle, precision, image_file_glob, noise,
                         target_size, **kwargs)

    def _make_dataset(self, stage):
        return ImageDataset(
            path=self.path,
            load_fn=self.load_fn,
            encode_fn=self.vocab.encode_text,
            image_file_glob=os.path.join(stage, self.image_file_glob),
            precision=self.precision,
            target_size=self.target_size
        )

    def load_fn(self, path) -> TypedImageWithText:
        return TypedImageWithText.load(path, self.languages)
