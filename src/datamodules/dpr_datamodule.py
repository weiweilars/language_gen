from torch.utils.data import DataLoader
from src.datamodules.datasets.dpr_dataset import DPRDataset
from pytorch_lightning import LightningDataModule
from typing import Optional
import numpy as np
import torch
import pdb


class DPRDataModule(LightningDataModule):

    def __init__(
            self,
            **kwargs):
        
        super().__init__()

        self.train_data_params = kwargs['train_data_loader']

        self.val_data_params = kwargs['val_data_loader']

        self.train_batch_size = self.train_data_params.pop('batch_size')
        self.val_batch_size = self.val_data_params.pop('batch_size')

    def train_dataloader(self):

        dataset = DPRDataset(**self.train_data_params)
        result = DataLoader(dataset=dataset,
                            batch_size=self.train_batch_size,
                            shuffle=True)
        
        return result

    def val_dataloader(self):

        dataset = DPRDataset(**self.val_data_params)
        result = DataLoader(dataset=dataset,
                            batch_size=self.val_batch_size,
                            shuffle=False)
        return result
