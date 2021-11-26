import pytorch_lightning
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import torch.nn as nn
import pdb

from src.models.help_function import object_from_dict

class SkosaGenModel(pytorch_lightning.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        model_params = self.hparams['model'].copy()

        
        self.model = GPT2LMHeadModel.from_pretrained(model_params['model_folder'])

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_params['model_folder'])

        self.model.resize_token_embeddings(len(self.tokenizer))

        # freeze the first 6 layers
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for i, m in enumerate(self.model.transformer.h):        
            #Only un-freeze the last n transformer blocks
            if i >= model_params['freeze_layers']:
                for parameter in m.parameters():
                    parameter.requires_grad = True 

        for parameter in self.model.transformer.ln_f.parameters():        
            parameter.requires_grad = True

        for parameter in self.model.lm_head.parameters():        
            parameter.requires_grad = True

        # self.model.lm_head = nn.Linear(in_features=768, out_features=model_params['modified_voc_len'], bias=False)

        # self.model.transformer.wte = nn.Embedding(model_params['modified_voc_len'], 768)


        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        # pdb.set_trace()


    def foward(self, x):
        output = self.model(x)
        return output

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask= batch
        
        results = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        #results = self.model(**inputs)

        loss = results.loss

        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):

        # pdb.set_trace()
        # self.model.save_pretrained(self.hparams['model']['save_model_folder'])


        input_ids, attention_mask= batch

        results = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        loss = results.loss

        self.log("val_loss", loss)
        
        return loss

    
    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]



