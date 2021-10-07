from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch
import torch.nn as nn
import os 

model_path  = "./models/"

org_model_path = os.path.join(model_path, 'skosa_model')
save_model_path = os.path.join(model_path, 'skosa_saved_model')
saved_model = "./models/epoch=8-step=14750.ckpt"

#tokenizer = GPT2Tokenizer.from_pretrained(org_model_path)

configuration = GPT2Config.from_pretrained(org_model_path)

model = GPT2LMHeadModel(configuration)

model.lm_head = nn.Linear(in_features=768, out_features=50400, bias=False)

model.transformer.wte = nn.Embedding(50400, 768)

import pdb
pdb.set_trace()

model.load_state_dict(torch.load(saved_model))

model.save_pretrained(torch.load(saved_model))
