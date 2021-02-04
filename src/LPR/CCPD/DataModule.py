import os

from src.LPR.CCPD.model.Image import CCPDImage
from src.base.data import ImagesDataModule, ImageDataset


class CCPDImagesDataModule(ImagesDataModule):

    def _make_dataset(self, stage):
        return ImageDataset(
            path=self.path,
            load_fn=CCPDImage.load,
            encode_fn=self.vocab.encode_strings,
            image_file_glob=os.path.join(stage, self.image_file_glob),
            precision=self.precision,
            target_size=self.target_size
        )
