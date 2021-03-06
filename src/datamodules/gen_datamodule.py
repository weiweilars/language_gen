from torch.utils.data import DataLoader, random_split
from src.datamodules.datasets.dataset import SkosaGenDataset, SkosaQADataset, LongTextDataset
from pytorch_lightning import LightningDataModule
from typing import Optional
import numpy as np
import torch
import pdb


class SkosaGenDataModule(LightningDataModule):

    def __init__(
            self,
            **kwargs):
        
        super().__init__()

        self.data_params = kwargs['data_loader']

        self.test_ratio = self.data_params.pop('test_ratio')

        self.batch_size = self.data_params.pop('batch_size')

        self.num_workers = self.data_params.pop('num_workers')

        self.seed = self.data_params.pop('seed')

        self.dataset_fn = self.data_params.pop('dataset')

    def setup(self, stage: Optional[str] = None):

        if self.dataset_fn == 'SkosaGenDataset':

            dataset = SkosaGenDataset(**self.data_params)

        elif self.dataset_fn == 'SkosaQADataset':

            dataset = SkosaQADataset(**self.data_params)
        else:
            
            dataset = LongTextDataset(**self.data_params)
            
        dataset_size = dataset.data_length
        split = int(np.floor(self.test_ratio * dataset_size))

        self.train_dataset, self.valid_dataset = random_split(dataset, [dataset_size-split, split], generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        result = DataLoader(dataset=self.train_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=True)
        
        return result

    def val_dataloader(self):
        result = DataLoader(dataset=self.valid_dataset,
                            batch_size=self.batch_size,
                            shuffle=False)
        return result
    
