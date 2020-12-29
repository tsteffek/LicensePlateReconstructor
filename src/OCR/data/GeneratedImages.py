import os
from typing import Optional, List, Tuple, Iterable

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from src.OCR import IO
from src.OCR.data.model.Vocabulary import Vocabulary
from src.OCR.image_gen.model.Text import Text, TextImage


class GeneratedImages(Dataset):
    def __init__(self, path: str, vocab: Vocabulary, image_files: str = '**/*.jpg'):
        super().__init__()
        self.vocab = vocab
        self.path = path
        self.image_files = image_files

        self.images = IO.get_image_paths(path, image_files)

        w, h = IO.load_image(self.images[0]).size
        self.size = h, w

    def __getitem__(self, item) -> Tuple[Tensor, Tuple[Text, Tensor]]:
        image_path = self.images[item]
        text, img, _ = TextImage.load(image_path, self.vocab.languages)
        labels = self.vocab.encode_text(text)
        return to_torch_format(img), (text, torch.tensor(labels, dtype=torch.int64))

    def __len__(self):
        return len(self.images)


def to_torch_format(img: Image) -> Tensor:
    return torch.tensor(img, dtype=torch.float32).reshape(*img.size, -1).T / 255


class GeneratedImagesDataModule(pl.LightningDataModule):
    def __init__(self, path: str, batch_size, multi_core: bool = True, cuda: bool = torch.cuda.is_available(),
                 shuffle: bool = True, language_file: str = 'languages.json', noise_file: str = 'noise.json', ):
        super().__init__()
        self.path = path
        self.language_file = language_file
        self.noise_file = noise_file
        self.batch_size = batch_size
        self.cuda = cuda
        self.shuffle = shuffle
        self.multi_core = multi_core
        if multi_core:
            self.num_workers = max(os.cpu_count(), batch_size)
        else:
            self.num_workers = 0

        self.vocab = Vocabulary(IO.load_languages(path, language_file), noise=IO.read_json(noise_file))

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.size = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = GeneratedImages(self.path, self.vocab, image_files='train/**/*.jpg')
            self.val_dataset = GeneratedImages(self.path, self.vocab, image_files='val/**/*.jpg')
            self.size = self.train_dataset.size
        if stage == 'test' or stage is None:
            self.test_dataset = GeneratedImages(self.path, self.vocab, image_files='test/**/*.jpg')
            self.size = self.test_dataset.size

    @staticmethod
    def collate_fn(
            batch: List[Tuple[Tensor, Tuple[Text, Tensor]]]
    ) -> Tuple[Tensor, Tuple[Iterable[Text], Tensor, Tensor]]:
        imgs, labels = list(zip(*batch))
        texts, label_indices = list(zip(*labels))
        label_lengths = torch.tensor([len(label) for label in label_indices], dtype=torch.int64)

        return torch.stack(imgs), (np.array(texts, dtype=object), torch.cat(label_indices), label_lengths)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.cuda and self.multi_core,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.cuda and self.multi_core,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.cuda and self.multi_core,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )
