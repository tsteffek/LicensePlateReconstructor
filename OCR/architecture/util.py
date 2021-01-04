import logging
from typing import Iterable, Any, List

import torch
from pytorch_lightning.metrics import Metric
from torch import Tensor
from torch import nn

log = logging.getLogger("lightning").getChild(__name__)


class Img2Seq(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: Tensor):
        B, C, H, W = x.shape
        x = x.reshape(B, -1, W)
        x = x.permute(2, 0, 1)  # width/time step, batch, channel
        return x


class ConfusionMatrix(Metric):
    matrix: Tensor

    def __init__(self, classes: List[Any]):
        super().__init__()
        self.classes = classes
        self.add_state(
            'matrix', default=torch.full((len(classes), len(classes)), fill_value=0, dtype=torch.int64),
            dist_reduce_fx=lambda x: torch.sum(x, dim=-1), persistent=True
        )

    def compute(self) -> str:
        if torch.max(self.matrix) == 0:
            return '\nConfusion Matrix: nothing registered.'

        total = self.matrix.sum()
        tp = self.matrix.diagonal().sum()
        fp = total - tp

        str_matrix = '\nConfusion Matrix:\n' \
                     f'Total: {total} | Correct: {tp} | Wrong: {fp} | Acc: {tp / total}' \
                     '\n \t' + '\t'.join(self.classes) + '\tacc\ttotal'

        row_totals = self.matrix.sum(dim=1)
        row_accs = self.matrix.diagonal() / row_totals
        for char, row, acc, total in zip(self.classes, self.matrix, row_accs, row_totals):
            str_matrix += f'\n{char}\t' + '\t'.join(tensor_to_list(row)) + \
                          f'\t{tensor_to_string(acc)}\t{tensor_to_string(total)}'

        col_totals = self.matrix.sum(dim=0)
        col_accs = self.matrix.diagonal() / col_totals
        str_matrix += '\n \t' + '\t'.join(tensor_to_list(col_accs))
        str_matrix += '\n \t' + '\t'.join(tensor_to_list(col_totals))

        return str_matrix

    def update(self, preds: Tensor, target: Tensor):
        # # this
        # uniques, count = torch.unique(torch.stack((targets, predictions)), return_counts=True)
        # t, p = uniques.unbind()
        # self.mat[t, p] = self.mat[t, p] + count
        # # or that
        for t, p in zip(target, preds):
            self.matrix[t, p] = self.matrix[t, p] + 1


def tensor_to_string(t: Tensor) -> str:
    return str(t.item())


def tensor_to_list(t: Tensor) -> Iterable[str]:
    return map(tensor_to_string, list(t))
