from torch.utils.data import DataLoader, random_split
from src.datamodules.datasets.dataset import SkosaGenDataset
from pytorch_lightning import LightningDataModule
from typing import Optional
import numpy as np
import torch


class SkosaGenDataModule(LightningDataModule):

    def __init__(
            self,
            **kwargs):
        
        super().__init__()

        self.data_params = kwargs['data_loader']


    def setup(self, stage: Optional[str] = None):        

        dataset = SkosaGenDataset(self.data_params['data_folder'], self.data_params['model_folder'], self.data_params['max_token_size'], self.data_params['generate_size'], self.data_params['generate_new'])
             
        dataset_size = dataset.data_length
        split = int(np.floor(self.data_params['test_ratio'] * dataset_size))

        self.train_dataset, self.valid_dataset = random_split(dataset, [dataset_size-split, split], generator=torch.Generator().manual_seed(self.data_params['seed']))

    def train_dataloader(self):
        result = DataLoader(dataset=self.train_dataset,
                            batch_size=self.data_params['batch_size'],
                            num_workers=self.data_params['num_workers'],
                            shuffle=True)
        
        return result

    def val_dataloader(self):
        result = DataLoader(dataset=self.valid_dataset,
                            batch_size=self.data_params['batch_size'],
                            shuffle=False)
        return result
