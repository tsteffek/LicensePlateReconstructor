import argparse
import logging
from typing import Type

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner

from src.OCR.GeneratedImages import GeneratedImagesDataModule
from src.OCR.architecture import CharacterRecognizer

log = logging.getLogger('pytorch_lightning').getChild(__name__)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--no_train', default=False, action='store_true')
    p.add_argument('--no_test', default=False, action='store_true')
    p.add_argument('--progress_bar_refresh_ratio', type=float)
    p.add_argument('--early_stopping', default=False, action='store_true')
    p = Trainer.add_argparse_args(p)
    return p


def load_or_create(model_cls: Type[LightningModule], dict_args, resume_path: str, *args):
    if resume_path:
        log.warning('Loading checkpoint from %s', resume_path)
        return model_cls.load_from_checkpoint(resume_path)
    else:
        return model_cls(*args, **dict_args)


def setup_trainer(args, max_steps) -> Trainer:
    callbacks = [
        ModelCheckpoint(
            monitor='accuracy/val',
            mode='max',
            save_top_k=10,
            save_last=True,
            verbose=True
        )
    ]
    if args.lr_schedule:
        callbacks.append(LearningRateMonitor())
    if args.early_stopping:
        callbacks.append(EarlyStopping(
            monitor='accuracy/val',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='max'
        ))

    if args.progress_bar_refresh_ratio:
        args.progress_bar_refresh_rate = int(max_steps * args.progress_bar_refresh_ratio)

    if args.log_per_epoch and args.num_sanity_val_steps:
        args.num_sanity_val_steps = args.log_per_epoch if args.log_per_epoch > args.num_sanity_val_steps \
            else args.num_sanity_val_steps
    elif args.log_per_epoch:
        args.num_sanity_val_steps = args.log_per_epoch

    return Trainer.from_argparse_args(args, callbacks=callbacks)


def auto_scale_batch_size(trainer, model, datamodule, dict_args):
    tuner = Tuner(trainer)
    new_batch_size = tuner.scale_batch_size(
        init_val=dict_args['batch_size'], mode=dict_args['auto_scale_batch_size'],
        model=model, datamodule=datamodule
    )
    model.hparams.batch_size = new_batch_size
    datamodule.batch_size = new_batch_size

    # remove the flags, so we can call tune without repeating the search
    trainer.auto_scale_batch_size = False
    args.auto_scale_batch_size = False


def tune_and_fit(trainer, model, datamodule, dict_args):
    if dict_args['auto_scale_batch_size']:
        auto_scale_batch_size(trainer=trainer, model=model, datamodule=datamodule, dict_args=dict_args)
    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = make_parser()
    parser = CharacterRecognizer.add_model_specific_args(parser)
    parser = GeneratedImagesDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    datamodule = GeneratedImagesDataModule(**dict_args)
    datamodule.setup()

    model = load_or_create(CharacterRecognizer, dict_args, dict_args['resume_from_checkpoint'], datamodule.vocab,
                           datamodule.size)

    trainer = setup_trainer(args, datamodule.max_steps)

    if not args.no_train:
        tune_and_fit(trainer, model, datamodule, dict_args)
        trainer.test()  # test best
    if not args.no_test:
        trainer.test(model=model, datamodule=datamodule)  # test latest

    log.info('Best model score: %s > %s', trainer.checkpoint_callback.best_model_score,
             trainer.checkpoint_callback.best_model_path)
