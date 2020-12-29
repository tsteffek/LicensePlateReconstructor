from typing import Tuple, List, Iterable, Any

import pytorch_lightning as pl
import torch
from torch import nn, Tensor, optim

from src.OCR.architecture.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from src.OCR.architecture.util import Img2Seq, ConfusionMatrix
from src.OCR.data.model.Vocabulary import Vocabulary
from src.OCR.image_gen.model.Text import Text


class CharacterRecognizer(pl.LightningModule):
    def __init__(
            self, vocab: Vocabulary, img_size: Tuple[int, int], mobile_net_variant: str = 'small',
            lstm_hidden: int = 48, num_classes: int = 60, lr: float = 1e-4
    ):
        super().__init__()
        self.vocab = vocab
        self.input_size = img_size
        if mobile_net_variant == 'small':
            self.cnns = mobilenetv3_small()
        elif mobile_net_variant == 'large':
            self.cnns = mobilenetv3_large()

        _, lstm_input_dim, h, w = self.cnns(self.example_input_array).shape

        self.img2seq = Img2Seq()
        self.lstms = nn.LSTM(input_size=lstm_input_dim, hidden_size=48, bidirectional=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.output_size = torch.tensor(w, dtype=torch.int64, device=self.device)

        self.loss_func = nn.CTCLoss()

        self.lr = lr

        self.test_accuracy_len = None
        self.test_accuracy = None
        self.test_confusion_matrix = None

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

    def training_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int) -> Tensor:
        x, (_, y, y_lengths) = batch
        x_hat = self.forward(x)
        _, loss = self.loss(x_hat, y, self.output_size.repeat(x_hat.shape[1]), y_lengths)
        self.log('train_loss', loss)
        return loss

    def on_test_epoch_start(self):
        self.test_accuracy = pl.metrics.Accuracy(compute_on_step=False)
        self.test_accuracy_len = pl.metrics.Accuracy(compute_on_step=False)
        self.test_confusion_matrix = ConfusionMatrix(len(self.vocab.noisy_chars))

    def test_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int):
        x, (texts, y, y_lengths) = batch
        x_hat = self.forward(x)
        batch_size = x_hat.shape[1]
        logits, loss = self.loss(x_hat, y, self.output_size.repeat(batch_size), y_lengths)
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
        print('\n')
