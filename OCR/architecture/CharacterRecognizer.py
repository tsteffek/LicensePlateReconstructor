import logging
from argparse import ArgumentParser
from typing import Tuple, List, Iterable, Union

import pytorch_lightning as pl
import torch
from torch import nn, Tensor, optim

from OCR.architecture.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from OCR.architecture.util import Img2Seq, ConfusionMatrix
from OCR.data.model.Vocabulary import Vocabulary
from OCR.image_gen.model.Text import Text

log = logging.getLogger("lightning").getChild(__name__)


class CharacterRecognizer(pl.LightningModule):
    def __init__(
            self, vocab: Vocabulary, img_size: Tuple[int, int],
            mobile_net_variant: str = 'small', width_mult: float = 1.,
            first_filter_shape: Union[Tuple[int, int], int] = 3, first_filter_stride: Union[Tuple[int, int], int] = 2,
            lstm_hidden: int = 48, lr: float = 1e-4, **kwargs
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

        self.lstms = nn.LSTM(input_size=lstm_input_dim, hidden_size=48, bidirectional=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(lstm_hidden * 2, len(self.vocab))
        self.output_size = torch.tensor(w, dtype=torch.int64, device=self.device)

        self.loss_func = nn.CTCLoss()

        self.lr = lr

        self.test_accuracy_len = None
        self.test_accuracy = None
        self.test_confusion_matrix = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--mobile_net_variant', type=str, default='small')
        parser.add_argument('--width_mult', type=float, default=1.)
        parser.add_argument('--first_filter_shape', type=int, nargs='+', default=3)
        parser.add_argument('--first_filter_stride', type=int, nargs='+', default=2)
        parser.add_argument('--lstm_hidden', type=int, nargs='+', default=48)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser

    @property
    def example_input_array(self, batch_size: int = 4) -> Tensor:
        return torch.randn(batch_size, 3, *self.input_size, dtype=torch.float32)

    @property
    def example_input_and_label(self) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return torch.randn(4, *self.input_size, 3), \
               (torch.randint(1, 2, (4, 5)), torch.full([4], 5, dtype=torch.int64))

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

    def step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        x, (_, y, y_lengths) = batch
        x_hat = self.forward(x)
        batch_size = x_hat.shape[1]
        return self.loss(x_hat, y, self.output_size.repeat(batch_size), y_lengths)

    def training_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int) -> Tensor:
        loss = self.step(batch)[1]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int) -> Tensor:
        loss = self.step(batch)[1]
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def on_test_epoch_start(self):
        self.test_accuracy = pl.metrics.Accuracy(compute_on_step=False)
        self.test_accuracy_len = pl.metrics.Accuracy(compute_on_step=False)
        self.test_confusion_matrix = ConfusionMatrix(len(self.vocab.noisy_chars))

    def test_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int):
        _, (_, y, y_lengths) = batch
        logits, loss = self.step(batch)
        self.log('test_loss', loss, on_epoch=True)

        predictions, pred_lengths = self._decode_raw(logits.transpose(0, 1))
        self.test_accuracy_len.update(pred_lengths, y_lengths)
        matching_pred, matching_targets = self._get_matching_length_elements(predictions, pred_lengths, y, y_lengths)
        self.test_accuracy.update(matching_pred, matching_targets)
        self.test_confusion_matrix.update(matching_pred, matching_targets)

    def _decode_raw(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        arg_max_batch = logits.argmax(dim=-1)
        uniques = [torch.unique_consecutive(arg_max) for arg_max in arg_max_batch]
        unique_non_blank = [unique[unique != self.vocab.blank_idx] for unique in uniques]
        pred_lengths = torch.tensor([len(pred) for pred in unique_non_blank], dtype=torch.int64, device=self.device)
        return torch.cat(unique_non_blank), pred_lengths

    @staticmethod
    def _get_matching_length_elements(pred: Tensor, pred_lengths: Tensor, target: Tensor, target_lengths: Tensor):
        mask: Tensor = pred_lengths.__eq__(target_lengths)
        return pred[mask.repeat_interleave(pred_lengths)], target[mask.repeat_interleave(target_lengths)],

    def test_epoch_end(self, outputs: List[Tuple[Iterable[Text], Iterable[Iterable[str]]]]) -> None:
        self.log('test_acc_len_epoch', self.test_accuracy_len.compute(), on_epoch=True)
        self.log('test_acc_epoch', self.test_accuracy.compute(), on_epoch=True)
        self.test_confusion_matrix.print(self.vocab)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def on_epoch_end(self):
        log.info('\n')
