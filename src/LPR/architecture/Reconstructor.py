import logging
from argparse import ArgumentParser
from typing import Tuple, List

import pytorch_lightning as pl
import pytorch_warmup as warmup
import torch
from PIL.Image import Image
from torch import Tensor, optim, nn

from src.OCR.architecture import CharacterRecognizer
from .PI_REC import G_Generator

log = logging.getLogger('pytorch_lightning').getChild(__name__)


class Reconstructor(pl.LightningModule):
    def __init__(
            self, ocr_path: str,
            base_channel_number: int = 64, res_blocks: int = 6, light: bool = False,
            log_per_epoch: int = 3,
            lr: float = 1e-4, lr_schedule: str = None, lr_warm_up: str = None,
            ctc_reduction: str = 'mean', mode: str = 'PI-REC',
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        ocr = CharacterRecognizer.load_from_checkpoint(ocr_path)
        ocr.freeze()
        ocr.loss_func = nn.CTCLoss(reduction=ctc_reduction, zero_infinity=True)
        ocr.log = self.log
        self.ocr = ocr

        self.mode = mode
        # if mode == 'PI-REC':
        self.model = G_Generator(rs_blocks=res_blocks)
        # else:
        #     self.model = ResnetGenerator(
        #         input_nc=3, output_nc=3,
        #         ngf=base_channel_number, n_blocks=res_blocks,
        #         img_size=int(math.sqrt(math.prod(ocr.input_size))), light=light
        #     )
        #     self.Rho_clipper = RhoClipper(0, 1)

        self.forward = self.model.forward

        self.log_results = log_per_epoch

        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_warm_up = lr_warm_up

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ocr_path', type=str)
        parser.add_argument('--base_channel_number', type=int, default=64, help='base channel number per layer')
        parser.add_argument('--res_blocks', type=int, default=4, help='The number of resblock')
        parser.add_argument('--light', action='store_true', default=False,
                            help='[U-GAT-IT full version / U-GAT-IT light version]')
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--lr_schedule', type=str, default=None, choices=['cosine', None])
        parser.add_argument('--lr_warm_up', type=str, default=None, choices=['linear', 'exponential', None])
        parser.add_argument('--ctc_reduction', type=str, default='mean', choices=['mean', 'sum', None])

        return parser

    @property
    def example_input_array(self, batch_size: int = 4) -> Tensor:
        return torch.randn(batch_size, 3, *self.ocr.input_size, dtype=torch.float32, device=self.device)

    def loss(
            self, output: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        ocr_prediction = self.ocr(output)
        return self.ocr.loss(ocr_prediction, targets, input_lengths, target_lengths)

    def predict(self, x: Tensor) -> Tuple[Image, List[List[str]]]:
        output = self.forward(x)[0]
        return self._to_image(output), self.ocr.predict(output)

    def step(
            self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]]
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        x, y = batch
        # if self.mode == 'PI-REC':
        x_hat = self.forward(x)
        # else:
        #     x_hat, cam_logits, heatmap = self.forward(x)
        return x_hat, self.ocr.step((x_hat, y))

    def step_with_logging(
            self, stage: str, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], return_probe: bool
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, str]]:
        preds, (logits, loss) = self.step(batch)
        self.log(f'{stage}_loss', loss, on_epoch=True, sync_dist=True)

        decoded = self.ocr.decode_raw(logits)
        self.ocr.update_metrics(decoded, batch)

        if return_probe:
            return preds[0], (decoded[0], batch[0][0], batch[1][0][0])

    def training_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int) -> Tensor:
        preds, (logits, loss) = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int
                        ) -> Tuple[Tensor, Tuple[Tensor, Tensor, str]]:
        return self.step_with_logging('val', batch, batch_idx < self.log_results)

    def test_step(self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int
                  ) -> Tuple[Tensor, Tuple[Tensor, Tensor, str]]:
        return self.step_with_logging('test', batch, batch_idx < self.log_results)

    def validation_epoch_end(self, outputs: List[Tuple[Tensor, Tensor]]) -> None:
        self.log_epoch('val', outputs)

    def test_epoch_end(self, outputs: List[Tuple[Tensor, Tensor]]) -> None:
        self.log_epoch('test', outputs)

    def log_epoch(self, stage: str, outputs: List[Tuple[Tensor, Tensor]]):
        lpr_output, ocr_output = list(zip(*outputs))
        predicted_images = torch.stack(lpr_output)

        self.logger.experiment.add_images(f'{stage}_pred', predicted_images, self.global_step)
        self.ocr.logger = self.logger
        self.ocr.log_epoch(stage, ocr_output)

    def configure_optimizers(self):
        if self.lr_schedule is None:
            return optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.lr_schedule == 'cosine':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

            max_steps = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches * self.trainer.max_epochs
            schedules = [{
                'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps),
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