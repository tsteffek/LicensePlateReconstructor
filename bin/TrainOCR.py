import argparse
import logging

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from src.OCR.GeneratedImages import GeneratedImagesDataModule
from src.OCR.architecture import CharacterRecognizer

log = logging.getLogger('lightning').getChild(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_train', default=False, action='store_true')
    parser.add_argument('--no_test', default=False, action='store_true')
    parser.add_argument('--progress_bar_refresh_ratio', type=float)
    parser.add_argument('--early_stopping', default=False, action='store_true')
    parser = CharacterRecognizer.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = GeneratedImagesDataModule.add_argparse_args(parser)

    args = parser.parse_args()

    dict_args = vars(args)
    datamodule = GeneratedImagesDataModule(**dict_args)
    datamodule.setup()

    max_iterations = datamodule.max_steps * args.max_epochs

    if args.resume_from_checkpoint:
        log.warning('Loading checkpoint from %s', args.resume_from_checkpoint)
        model = CharacterRecognizer.load_from_checkpoint(
            args.resume_from_checkpoint, vocab=datamodule.vocab, img_size=datamodule.size,
            max_iterations=max_iterations, **dict_args
        )
    else:
        model = CharacterRecognizer(
            vocab=datamodule.vocab, img_size=datamodule.size, max_iterations=max_iterations,
            **dict_args
        )

    trainer_callbacks = [
        ModelCheckpoint(
            monitor='val_accuracy',
            mode='max',
            save_top_k=10,
            save_last=True,
            verbose=True
        )
    ]
    if args.lr_schedule:
        trainer_callbacks.append(LearningRateMonitor())
    if args.early_stopping:
        trainer_callbacks.append(EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='max'
        ))

    if args.progress_bar_refresh_ratio:
        args.progress_bar_refresh_rate = int(max_iterations * args.progress_bar_refresh_ratio)

    trainer = Trainer.from_argparse_args(args, callbacks=trainer_callbacks)
    if not args.no_train:
        trainer.tune(model=model, datamodule=datamodule)
        trainer.fit(model=model, datamodule=datamodule)
    if not args.no_test:
        trainer.test(model=model, datamodule=datamodule)

    log.info('Best model score: %s > %s', trainer_callbacks[0].best_model_score, trainer_callbacks[0].best_model_path)
