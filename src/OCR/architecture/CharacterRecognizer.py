import logging
from argparse import ArgumentParser
from typing import Tuple, List, Iterable, Union

import pytorch_lightning as pl
import pytorch_warmup as warmup
import torch
from guppy import hpy
from torch import nn, Tensor, optim

from src.base.model import Vocabulary, Text
from .mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from .util import Img2Seq, ConfusionMatrix

log = logging.getLogger('pytorch_lightning').getChild(__name__)


def profile():
    log.info('\n%s', hpy().heap())


class CharacterRecognizer(pl.LightningModule):
    def __init__(
            self, vocab: Vocabulary, img_size: Tuple[int, int], max_iterations: int,
            mobile_net_variant: str = 'small', width_mult: float = 1.,
            first_filter_shape: Union[Tuple[int, int], int] = 3, first_filter_stride: Union[Tuple[int, int], int] = 2,
            lstm_hidden: int = 48, lr: float = 1e-4, lr_schedule: str = None, lr_warm_up: str = None,
            ctc_reduction: str = 'mean', **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = vocab
        self.input_size = img_size

        mobile_net_kwargs = {
            'width_mult': width_mult,
            'first_filter_shape': first_filter_shape,
            'first_filter_stride': first_filter_stride
        }
        if mobile_net_variant == 'small':
            self.cnns = mobilenetv3_small(**mobile_net_kwargs)
        elif mobile_net_variant == 'large':
            self.cnns = mobilenetv3_large(**mobile_net_kwargs)

        self.img2seq = Img2Seq()

        w, _, lstm_input_dim = self.img2seq(self.cnns(self.example_input_array)).shape

        self.lstms = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden,
                             bidirectional=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(lstm_hidden * 2, len(self.vocab))
        self.output_size = torch.tensor(w, dtype=torch.int64, device=self.device)

        self.loss_func = nn.CTCLoss(reduction=ctc_reduction, zero_infinity=True)

        self.lr = lr
        self.max_steps = max_iterations
        self.lr_schedule = lr_schedule
        self.lr_warm_up = lr_warm_up

        self.accuracy_cha = pl.metrics.Accuracy(compute_on_step=False)
        self.accuracy_len = pl.metrics.Accuracy(compute_on_step=False)
        self.confusion_matrix = ConfusionMatrix(self.vocab.noisy_chars)
        self.confusion_matrix_len = ConfusionMatrix(list(map(str, range(w))))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--mobile_net_variant', type=str, default='small')
        parser.add_argument('--width_mult', type=float, default=1.)
        parser.add_argument('--first_filter_shape', type=int, nargs='+', default=3)
        parser.add_argument('--first_filter_stride', type=int, nargs='+', default=2)
        parser.add_argument('--lstm_hidden', type=int, default=48)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--lr_schedule', type=str, default=None, choices=['cosine', None])
        parser.add_argument('--lr_warm_up', type=str, default=None, choices=['linear', 'exponential', None])
        parser.add_argument('--ctc_reduction', type=str, default='mean', choices=['mean', 'sum', None])

        return parser

    @property
    def example_input_array(self, batch_size: int = 4) -> Tensor:
        return torch.randn(batch_size, 3, *self.input_size, dtype=torch.float32, device=self.device)

    @property
    def example_input_and_label(self) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return torch.randn(4, *self.input_size, 3), \
               (torch.randint(1, 2, (4, 5)), torch.full([4], 5, dtype=torch.int64, device=self.device))

    def forward(self, x: Tensor) -> Tensor:
        cnn_output = self.cnns(x)
        formatted = self.img2seq(cnn_output)
        lstm_output, _ = self.lstms(formatted)
        return self.fc(lstm_output)

    def loss(
            self, output: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        logits = nn.functional.log_softmax(output, dim=-1)
        return logits, self.loss_func(logits, targets, input_lengths, target_lengths)

    def predict(self, x: Tensor):
        output = self.forward(x)
        logits = nn.functional.log_softmax(output, dim=-1).transpose(0, 1)
        texts = self._decode_raw(logits)
        return [self.vocab.decode_text(text) for text in texts]

    def step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        # profile()
        x, (_, y, y_lengths) = batch
        x_hat = self.forward(x)
        batch_size = x_hat.shape[1]
        return self.loss(x_hat, y, self.output_size.repeat(batch_size), y_lengths)

    def training_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int) -> Tensor:
        loss = self.step(batch)[1]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int) -> Tensor:
        logits, loss = self.step(batch)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)

        self.detailed_log(logits, batch)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int):
        logits, loss = self.step(batch)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)

        self.detailed_log(logits, batch)

    def detailed_log(self, logits: Tensor, batch: Tuple[Tensor, Tuple[Tensor, Tensor]]):
        _, (_, y, y_lengths) = batch

        predictions, pred_lengths = self.cat(self._decode_raw(logits.transpose(0, 1)))

        self.accuracy_len.update(pred_lengths, y_lengths)
        self.confusion_matrix_len.update(pred_lengths, y_lengths)

        matching_pred, matching_targets = self._get_matching_length_elements(predictions, pred_lengths, y, y_lengths)

        self.accuracy_cha.update(matching_pred, matching_targets)
        self.confusion_matrix.update(matching_pred, matching_targets)

    def _decode_raw(self, logits: Tensor) -> List[Tensor]:
        arg_max_batch = logits.argmax(dim=-1)
        uniques = [torch.unique_consecutive(arg_max) for arg_max in arg_max_batch]
        return [unique[unique != self.vocab.blank_idx] for unique in uniques]

    def cat(self, arr: List[Tensor]) -> Tuple[Tensor, Tensor]:
        pred_lengths = torch.tensor([len(pred) for pred in arr], dtype=torch.int64, device=self.device)
        return torch.cat(arr), pred_lengths

    @staticmethod
    def _get_matching_length_elements(pred: Tensor, pred_lengths: Tensor, target: Tensor, target_lengths: Tensor):
        mask: Tensor = pred_lengths.__eq__(target_lengths)
        return pred[mask.repeat_interleave(pred_lengths)], \
               target[mask.repeat_interleave(target_lengths)],

    def log_epoch(self, stage: str):
        acc_len = self.accuracy_len.compute()
        acc_cha = self.accuracy_cha.compute()
        self.log(f'{stage}_acc_len_epoch', acc_len, sync_dist=True)
        self.log(f'{stage}_acc_cha_epoch', acc_cha, sync_dist=True)
        self.log(f'{stage}_accuracy', acc_len * acc_cha, sync_dist=True)
        log.info(self.confusion_matrix_len.compute())
        log.info(self.confusion_matrix.compute())

    def validation_epoch_end(self, outputs: List[Tuple[Iterable[Text], Iterable[Iterable[str]]]]) -> None:
        self.log_epoch('val')

    def test_epoch_end(self, outputs: List[Tuple[Iterable[Text], Iterable[Iterable[str]]]]) -> None:
        self.log_epoch('test')

    def configure_optimizers(self):
        if self.lr_schedule is None:
            return optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.lr_schedule == 'cosine':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

            schedules = [{
                'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_steps),
                'interval': 'step',
            }]
            if self.lr_warm_up:
                if self.lr_warm_up == 'linear':
                    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
                elif self.lr_warm_up == 'exponential':
                    warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
                else:
                    raise ValueError('lr_warm_up can only be "linear" or "exponential", but was ' + self.lr_warm_up)

                warmup_scheduler.step = warmup_scheduler.dampen
                schedules.append({
                    'scheduler': warmup_scheduler,
                    'interval': 'step'
                })

            return [optimizer], schedules

    def on_epoch_end(self):
        log.info('\n')
