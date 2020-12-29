import torch
from torch import Tensor
from torch import nn

from src.OCR.data.model.Vocabulary import Vocabulary


class Img2Seq(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: Tensor):
        B, C, H, W = x.shape
        assert H == 1, '{} should have height 1'.format(x.shape)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # width/time step, batch, channel
        return x


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.mat = torch.full((num_classes, num_classes), fill_value=0, dtype=torch.int64)

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        # # this
        # uniques, count = torch.unique(torch.stack((targets, predictions)), return_counts=True)
        # t, p = uniques.unbind()
        # self.mat[t, p] = self.mat[t, p] + count
        # # or that
        for t, p in zip(targets, predictions):
            self.mat[t, p] = self.mat[t, p] + 1

    def print(self, vocab: Vocabulary):
        chars = vocab.noisy_chars
        print('\n \t' + '\t'.join(chars))
        for idx, char in enumerate(chars):
            l = map(str, map(Tensor.item, list(self.mat[idx])))
            print(f'{char}\t' + '\t'.join(l))