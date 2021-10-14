import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytorch_lightning
import numpy as np
import torch
from torch import nn 
import pdb
from src.models.help_function import object_from_dict

from transformers import AutoModel


def dpr_loss(question_embedings, pos_embedings, neg_embedings):

    pos_similar = (question_embedings * pos_embedings).sum(-1)
    neg_similar = (question_embedings * neg_embedings).sum(-1)

    #index_select = (pos_similar < neg_similar) 
    loss = torch.log(1+ torch.exp(neg_similar-pos_similar))
    
    return loss.mean()
    
    

class DPRModel(pytorch_lightning.LightningModule):

    def __init__(self,**kwargs):
        super().__init__()

        self.save_hyperparameters()

        model_params = self.hparams['model'].copy()
        
        self.query_encoder = AutoModel.from_pretrained(model_params['query_model'])
        self.passage_encoder = AutoModel.from_pretrained(model_params['passage_model'])

        self.dropout = nn.Dropout(model_params['dropout'])

        
    def forward(self, inputs):

        question_embeddings = self.query_encoder(inputs[0], inputs[1])
        pos_embeddings = self.passage_encoder(inputs[2], inputs[3])
        neg_embeddings = self.passage_encoder(inputs[4], inputs[5])

        question_embeddings = self.dropout(question_embeddings.pooler_output)
        pos_embeddings = self.dropout(pos_embeddings.pooler_output)
        neg_embeddings = self.dropout(neg_embeddings.pooler_output)

        return question_embeddings, pos_embeddings, neg_embeddings


    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()

    def training_step(self, batch, batch_idx):

        question_embeddings, pos_embeddings, neg_embeddings = self(batch)
        
        loss = dpr_loss(question_embeddings, pos_embeddings, neg_embeddings)
        
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):

        question_embeddings, pos_embeddings, neg_embeddings = self(batch)
        
        loss = dpr_loss(question_embeddings, pos_embeddings, neg_embeddings)
        
        self.log("val_loss", loss)
        
        return loss

    
    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.query_encoder.parameters() if x.requires_grad] +
            [x for x in self.passage_encoder.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]
