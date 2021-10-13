import os
import glob
import pdb
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import shutil
import json
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

import re
import time





class SkosaGenDataset(Dataset):

    input_path_name = 'input'
    data_name = 'dataset.csv'
    label_path_name = 'label'
    
    def __init__(self, data_path, tokenizer_path, max_token_size=512, generate_size=100, generate_new=True, generate_result=True):


        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        self.input_path = os.path.join(data_path, self.input_path_name)
        self.max_token = 512

        if generate_new:

            df = pd.read_csv(os.path.join(data_path, self.data_name), sep=';')
            df.reset_index()

            if generate_size > 0:
                df = df.head(generate_size)
            
            new_tokens = [*set(df.targetName.to_list()),]
        
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            self.tokenizer.add_tokens(new_tokens)
        
            #self.tokenizer.save_pretrained(tokenizer_path)

            for index, row in tqdm(df.iterrows(), total=df.shape[0]):

                org_sentence = row.targetName + row.featureValue + self.tokenizer.eos_token
            
                inputs = self.tokenizer(org_sentence, return_tensors="pt", max_length=max_token_size, padding='max_length', truncation=True)

                #inputs['labels'] = self.tokenizer(label, return_tensors="pt", max_length=max_token_size, padding='max_length', truncation=True)['input_ids']
            
                with open(os.path.join(self.input_path, str(index)+'.pickle'), 'wb') as handle:
                    pickle.dump(inputs, handle)

            self.data_length = index

        else:

            list = os.listdir(self.input_path)

            self.data_length = len(list)
        

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):

        with open(os.path.join(self.input_path, str(idx)+'.pickle'), 'rb') as handle:

            inputs = pickle.load(handle)
            
        input_ids = inputs['input_ids'][0]

        attention_mask = inputs['attention_mask'][0]

        #labels = inputs['labels'][0]

        # item['labels'] = self.labels['input_ids'][idx]
        
        return input_ids, attention_mask


class SkosaQADataset(Dataset):

    input_path_name = 'input'
    data_name = 'question_answer.json'

    def __init__(self, data_folder, model_folder, max_token_size=512):


        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_folder)
        self.input_path = os.path.join(data_folder, self.input_path_name)
        self.max_token = 512

        f = open(os.path.join(data_folder, self.data_name))

        data = json.load(f,object_pairs_hook=OrderedDict)

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})

        self.tokenizer.save_pretrained(model_folder)

        self.token_result = []
        for i in data:
            input = i['category'] + self.tokenizer.sep_token + i['question'] + self.tokenizer.sep_token + i['answer'] + self.tokenizer.eos_token
            self.token_result.append(self.tokenizer(input, return_tensors="pt", max_length=max_token_size, padding='max_length', truncation=True))

        self.data_length = len(self.token_result)

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):

        input_ids = self.token_result[idx]['input_ids'][0]

        attention_mask = self.token_result[idx]['attention_mask'][0]

        return input_ids, attention_mask



        


if __name__ == '__main__':


    import torch.nn as nn
    data_path = "../../../data/skosa_data/"

    tokenizer_path = "../../../models/qa_model/"

    
    dataset=SkosaQADataset(data_path, tokenizer_path)


    data_size = 10
    for i in range(data_size):

        dataset.__getitem__(i)
