import os
import glob
import pdb
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import shutil

import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


class SkosaGenDataset(Dataset):

    input_path_name = 'input'
    data_name = 'dataset.csv'
    label_path_name = 'label'
    
    def __init__(self, data_path, tokenizer_path, max_token_size=512, generate_size=100, generate_new=True):


        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        self.input_path = os.path.join(data_path, self.input_path_name)
        self.max_token = 512
        
        # model = GPT2LMHeadModel.from_pretrained(tokenizer_path)


        # inputs_seq = self.tokenizer(["d130"], return_tensors="pt")

        # result = model.generate(inputs_seq['input_ids'], max_length=10, num_beams=100, temperature=0.2, no_repeat_ngram_size=2, num_return_sequences=5)

        # pdb.set_trace()

        # tokenizer.decode(result[4])
    
        # print(result)


        if generate_new:

            df = pd.read_csv(os.path.join(data_path, self.data_name), sep=';')
            df.reset_index()

            if generate_size > 0:
                df = df.head(generate_size)
            
            new_tokens = [*set(df.targetName.to_list()),]
        
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            self.tokenizer.add_tokens(new_tokens)
        
            self.tokenizer.save_pretrained(tokenizer_path)

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
        


if __name__ == '__main__':


    import torch.nn as nn
    data_path = "../../../data/skosa_data/dataset.csv"

    tokenizer_path = "../../../models/skosa_model/"

    
    # dataset=SkosaGenDataset(data_folder, tokenizer_path, generate_size=10)


    # data_size = 10
    # for i in range(data_size):

    #     dataset.__getitem__(i)

    #     #pdb.set_trace()
   
    # # dataset = SegInfDataset(data_folder, )

    #df = pd.read_csv(data_path, sep=';')
    #df.reset_index()

    #new_tokens = [*set(df.targetName.to_list()),]
    
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    #tokenizer.add_tokens(new_tokens)
    
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    #tokenizer.save_pretrained(tokenizer_path)

    # configuration = GPT2Config.from_pretrained(tokenizer_path)

    # pdb.set_trace()

   # model = GPT2LMHeadModel(configuration)

    inputs_seq = tokenizer(["jag"], return_tensors="pt")

    print(inputs_seq)
    # pdb.set_trace()
    
    
    model = GPT2LMHeadModel.from_pretrained(tokenizer_path)

    model.lm_head = nn.Linear(in_features=768, out_features=50300, bias=False)

    model.transformer.wte = nn.Embedding(50300, 768)

    # inputs = {key: val for key, val in inputs_seq.items()}

    # pdb.set_trace()
    
    # outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])

    
    result = model.generate(inputs_seq['input_ids'], max_length=10, num_beams=100, temperature=0.2, no_repeat_ngram_size=2, num_return_sequences=5)

    pdb.set_trace()

    tokenizer.decode(result[4])

    print(result)

    
    
    # from pathlib import Path

    # def read_imdb_split(split_dir):
    #     split_dir = Path(split_dir)
    #     texts = []
    #     labels = []
    #     for label_dir in ["pos", "neg"]:
    #         for text_file in (split_dir/label_dir).iterdir():
    #             texts.append(text_file.read_text())
    #             labels.append(0 if label_dir is "neg" else 1)

    #     return texts, labels

    # train_texts, train_labels = read_imdb_split('../../../data/aclImdb/train')
    # test_texts, test_labels = read_imdb_split('../../../data/aclImdb/test')


    # from sklearn.model_selection import train_test_split
    # train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    

    # from transformers import DistilBertTokenizerFast
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    # val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    # test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # pdb.set_trace()

    