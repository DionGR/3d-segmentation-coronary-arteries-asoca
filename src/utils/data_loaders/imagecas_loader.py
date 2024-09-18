from typing import List

import monai
from monai.data import (
    DataLoader,
    set_track_meta,
)
from monai.transforms import (
    Compose
)

from utils.utils import get_imagecas_json, json_reader


class ImageCASLoader:
    """ImageCAS Dataset Loader
    A class to load the ImageCAS dataset and return the training, validation, and test dataloaders.
    """

    def __init__(self,
                 batch_size: int = 1, num_workers: int = 0,
                 train_transforms=[], val_transforms=[], test_transforms=[]
                 ):

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = self.compose_transforms(train_transforms)
        self.val_transforms = self.compose_transforms(val_transforms)
        self.test_transforms = self.compose_transforms(test_transforms)

        self.train_dataloader, self.val_dataloader, self.test_dataloader = None, None, None

    def load_imagecas_train(self):
        def get_data_dict():
            dataset_dir = get_imagecas_json()
            data_dicts = json_reader(dataset_dir)

            return data_dicts

        self.data_dicts = get_data_dict()

        train_data = self.data_dicts["train"]
        val_data = self.data_dicts["validation"]

        self.train_dataset = monai.data.Dataset(data=train_data, transform=self.train_transforms)
        self.val_dataset = monai.data.Dataset(data=val_data, transform=self.val_transforms)

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size, num_workers=self.num_workers,
                                           shuffle=True, collate_fn=monai.transforms.croppad.batch.PadListDataCollate())
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.batch_size, num_workers=self.num_workers,
                                         shuffle=False, collate_fn=monai.transforms.croppad.batch.PadListDataCollate())

        set_track_meta(True)

    def load_imagecas_test(self):
        def get_data_dict():
            dataset_dir = get_imagecas_json()
            data_dicts = json_reader(dataset_dir)

            return data_dicts

        self.data_dicts = get_data_dict()

        test_data = self.data_dicts["test"]

        self.test_dataset = monai.data.Dataset(data=test_data, transform=self.test_transforms)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.batch_size, num_workers=self.num_workers,
                                          shuffle=False, collate_fn=monai.transforms.croppad.batch.PadListDataCollate())

        set_track_meta(True)

    def get_train_val_dataloaders(self):
        if self.train_dataloader is None or self.val_dataloader is None:
            self.load_imagecas_train()

        return self.train_dataloader, self.val_dataloader

    def get_test_dataloader(self):
        if self.test_dataloader is None:
            self.load_imagecas_test()

        return self.test_dataloader

    def compose_transforms(self, transforms):
        return Compose(transforms)
