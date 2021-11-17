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

from transformers import DPRContextEncoder, DPRQuestionEncoder, AutoModel


def dpr_loss(question_embedings, context_embedings, label, loss_fn):

    scores = torch.matmul(question_embedings.unsqueeze(1), torch.transpose(context_embedings, 2,1))

    q_num = question_embedings.size(0)

    scores = scores.view(q_num, -1)

    softmax_scores = nn.functional.log_softmax(scores, dim=1)

    positive_idx_per_question = torch.tensor([(i == 1).nonzero(as_tuple=False) for i in label]).to(softmax_scores.device)

    loss = loss_fn(scores, positive_idx_per_question)

    return loss
    
    

class DPRModel(pytorch_lightning.LightningModule):

    def __init__(self,**kwargs):
        super().__init__()

        self.save_hyperparameters()

        model_params = self.hparams['model'].copy()
        
        self.query_encoder = AutoModel.from_pretrained(model_params['query_model'])
        self.passage_encoder = AutoModel.from_pretrained(model_params['passage_model'])
        
        self.dropout = nn.Dropout(model_params['dropout'])

        self.loss_fn = nn.NLLLoss()

        # for parameter in self.query_encoder.parameters():
        #     parameter.requires_grad = False

        # for i, m in enumerate(self.query_encoder.encoder.layer):

        #     #Only un-freeze the last n transformer blocks
        #     if i >= model_params['freeze_layers']:
        #         for parameter in m.parameters():
        #             parameter.requires_grad = True 

        # for parameter in self.query_encoder.pooler.parameters():        
        #     parameter.requires_grad = True

            
        # for parameter in self.passage_encoder.parameters():
        #     parameter.requires_grad = False

        # for i, m in enumerate(self.passage_encoder.encoder.layer):

        #     #Only un-freeze the last n transformer blocks
        #     if i >= model_params['freeze_layers']:
        #         for parameter in m.parameters():
        #             parameter.requires_grad = True 

        # for parameter in self.passage_encoder.pooler.parameters():        
        #     parameter.requires_grad = True



        
    def forward(self, inputs):

        question_embeddings = self.query_encoder(inputs[0], inputs[1])
        question_embeddings = self.dropout(question_embeddings.pooler_output)

        b,n,d = inputs[2].shape

        context_embeddings = self.passage_encoder(inputs[2].reshape(b*n,d), inputs[3].reshape(b*n,d))
        context_embeddings = self.dropout(context_embeddings.pooler_output)


        context_embeddings = context_embeddings.reshape(b,n,context_embeddings.shape[1])

        return question_embeddings, context_embeddings


    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()

    def training_step(self, batch, batch_idx):

        question_embeddings, context_embeddings = self(batch)
        
        loss = dpr_loss(question_embeddings, context_embeddings, batch[-1], self.loss_fn)
        
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):

        
        question_embeddings, context_embeddings = self(batch)
        
        loss = dpr_loss(question_embeddings, context_embeddings, batch[-1], self.loss_fn)
        
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
