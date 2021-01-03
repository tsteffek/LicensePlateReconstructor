import argparse
import logging

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from OCR.architecture.CharacterRecognizer import CharacterRecognizer
from OCR.data.GeneratedImages import GeneratedImagesDataModule

log = logging.getLogger('lightning').getChild(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_train', default=False, action='store_true')
    parser.add_argument('--no_test', default=False, action='store_true')
    parser = CharacterRecognizer.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = GeneratedImagesDataModule.add_argparse_args(parser)

    args = parser.parse_args()

    chkpt_config = ModelCheckpoint(monitor='train_loss', save_top_k=10, save_last=True, verbose=True)
    # chkpt_config = ModelCheckpoint(save_last=True, verbose=True)

    dict_args = vars(args)
    datamodule = GeneratedImagesDataModule(**dict_args)
    datamodule.setup()
    if args.resume_from_checkpoint:
        log.warning('Loading checkpoint from %s', args.resume_from_checkpoint)
        model = CharacterRecognizer.load_from_checkpoint(
            args.resume_from_checkpoint, vocab=datamodule.vocab, img_size=datamodule.size,
            max_iterations=datamodule.max_steps * args.max_epochs
        )
    else:
        model = CharacterRecognizer(
            vocab=datamodule.vocab, img_size=datamodule.size, max_iterations=datamodule.max_steps * args.max_epochs,
            **dict_args
        )
        log.info(datamodule.max_steps * args.max_epochs)
    trainer = Trainer.from_argparse_args(args, callbacks=[chkpt_config])
    if not args.no_train:
        trainer.tune(model=model, datamodule=datamodule)
        trainer.fit(model=model, datamodule=datamodule)
    if not args.no_test:
        trainer.test(model=model, datamodule=datamodule)
