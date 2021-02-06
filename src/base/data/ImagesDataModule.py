import os
from abc import abstractmethod
from math import ceil
from typing import Optional, List, Tuple, Iterable, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from src.base.data import ImageDataset
from src.base.model import Vocabulary, Text


class ImagesDataModule(pl.LightningDataModule):
    def __init__(
            self,
            path: str,
            batch_size: int,
            chars: Iterable[str] = None,
            multi_core: bool = True,
            cuda: bool = torch.cuda.is_available(),
            shuffle: bool = True,
            precision: int = 32,
            image_file_glob: str = '**/*.jpg',
            noise: Iterable[str] = None,
            target_size: Union[float, Tuple[int, int]] = None,
            vocab: Vocabulary = None,
            **kwargs
    ):
        super().__init__()
        self.path = path
        self.vocab = vocab if vocab is not None else Vocabulary(chars, noise)

        self.image_file_glob = image_file_glob

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size

        self.cuda = cuda
        self.precision = precision
        self.multi_core = multi_core

        if multi_core:
            self.num_workers = min(os.cpu_count(), batch_size)
        else:
            self.num_workers = 0

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.size = None
        self.max_steps = None

    @abstractmethod
    def _make_dataset(self, stage) -> ImageDataset:
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self._make_dataset('train')
            self.max_steps = ceil(len(self.train_dataset) / self.batch_size)
            self.val_dataset = self._make_dataset('val')
            self.size = self.train_dataset.size
        if stage == 'test' or stage is None:
            self.test_dataset = self._make_dataset('test')
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
        return self._make_dataloader(self.train_dataset, self.shuffle)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test_dataset, False)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_dataset, False)

    def _make_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.cuda and self.multi_core,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
