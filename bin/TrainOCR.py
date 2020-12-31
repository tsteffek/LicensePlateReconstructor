import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from OCR.architecture.CharacterRecognizer import CharacterRecognizer
from OCR.data.GeneratedImages import GeneratedImagesDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    parser = CharacterRecognizer.add_model_specific_args(parser)
    parser = GeneratedImagesDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    chkpt_config = ModelCheckpoint(monitor='train_loss', save_top_k=10, save_last=True, verbose=True)
    # chkpt_config = ModelCheckpoint(save_last=True, verbose=True)

    dict_args = vars(args)
    datamodule = GeneratedImagesDataModule(**dict_args)
    datamodule.setup()
    model = CharacterRecognizer(datamodule.vocab, datamodule.size, **dict_args)
    trainer = Trainer.from_argparse_args(args, callbacks=[chkpt_config])
    if args.train:
        trainer.tune(model, datamodule=datamodule)
        trainer.fit(model, datamodule=datamodule)
    if args.test:
        trainer.test(datamodule=datamodule)
