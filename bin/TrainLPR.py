import logging

from bin.TrainOCR import make_parser, load_or_create, setup_trainer, tune_and_fit
from src.LPR.CCPD.DataModule import CCPDImagesDataModule
from src.LPR.architecture import Reconstructor

log = logging.getLogger('pytorch_lightning').getChild(__name__)

if __name__ == '__main__':
    parser = make_parser()
    parser = Reconstructor.add_model_specific_args(parser)
    parser = CCPDImagesDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    model = load_or_create(Reconstructor, dict_args, dict_args['resume_from_checkpoint'])

    h, w = model.ocr.input_size  # PyTorch wants (h, w), PILLOW expects (w, h)
    dict_args['target_size'] = w, h
    datamodule = CCPDImagesDataModule(vocab=model.ocr.vocab, **dict_args)
    datamodule.setup()

    trainer = setup_trainer(args, datamodule.max_steps)

    if not args.no_train:
        tune_and_fit(trainer, model, datamodule, dict_args)
    if not args.no_test:
        trainer.test(model=model, datamodule=datamodule)

    log.info('Best model score: %s > %s', trainer.checkpoint_callback.best_model_score,
             trainer.checkpoint_callback.best_model_path)
