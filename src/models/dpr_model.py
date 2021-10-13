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

    loss = torch.log(1+ torch.exp(neg_similar-pos_similar)).mean()

    return loss
    
    

class DPRModel(pytorch_lightning.LightningModule):

    def __init__(self,**kwargs):
        super().__init__()

        self.save_hyperparameters()

        model_params = self.hparams['model'].copy()
        
        self.query_encoder = AutoModel.from_pretrained(model_params['query_model'])
        self.passage_encoder = AutoModel.from_pretrained(model_params['passage_model'])

        
    def forward(self, question_input_id, question_mask, context_input_ids, context_masks):
        question_embeddings = self.query_encoder(question_input_id, question_mask)
        context_embeddings = self.passage_encoder(context_input_ids, context_masks)


    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()

    def training_step(self, batch, batch_idx):
        question_input_ids = batch[0]
        question_attention_mask = batch[1]
        pos_context_input_ids = batch[2]
        pos_context_attention_mask = batch[3]
        neg_context_input_ids = batch[4]
        neg_context_attention_mask = batch[5]

        question_embedings = self.query_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask).pooler_output
        pos_embedings = self.passage_encoder(input_ids=pos_context_input_ids, attention_mask=pos_context_attention_mask).pooler_output
        neg_embedings = self.passage_encoder(input_ids=neg_context_input_ids, attention_mask=neg_context_attention_mask).pooler_output

        loss = dpr_loss(question_embedings, pos_embedings, neg_embedings)
        
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):

        question_input_ids = batch[0]
        question_attention_mask = batch[1]
        pos_context_input_ids = batch[2]
        pos_context_attention_mask = batch[3]
        neg_context_input_ids = batch[4]
        neg_context_attention_mask = batch[5]

        question_embedings = self.query_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask).pooler_output
        pos_embedings = self.passage_encoder(input_ids=pos_context_input_ids, attention_mask=pos_context_attention_mask).pooler_output
        neg_embedings = self.passage_encoder(input_ids=neg_context_input_ids, attention_mask=neg_context_attention_mask).pooler_output

        loss = dpr_loss(question_embedings, pos_embedings, neg_embedings)
        
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
