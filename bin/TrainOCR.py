import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.OCR.architecture.CharacterRecognizer import CharacterRecognizer
from src.OCR.data.GeneratedImages import GeneratedImagesDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Location of images', type=str)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # chkpt_config = ModelCheckpoint(monitor='train_loss', save_top_k=10, save_last=True, verbose=True)
    chkpt_config = ModelCheckpoint(save_last=True, verbose=True)

    datamodule = GeneratedImagesDataModule(args.data_dir, 8, multi_core=False)
    datamodule.setup()
    model = CharacterRecognizer(datamodule.vocab, datamodule.size)
    trainer = Trainer.from_argparse_args(args, callbacks=[chkpt_config])
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    result = trainer.test(datamodule=datamodule)
