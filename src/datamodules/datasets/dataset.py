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
import subprocess
import os

from src.datamodules.datasets.PDFReader import Clean_PDF


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

        pdb.set_trace()

        return input_ids, attention_mask


class LongTextDataset(Dataset):

    input_path_name = 'input'
    data_name = 'text.pdf'

    def __init__(self, data_folder, model_folder, max_token_size=512, remove_numeric_tables=True):


        print(model_folder)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_folder)
        #self.data_path = os.path.join(data_folder, self.data_name)
        self.input_path = os.path.join(data_folder, self.input_path_name)
        self.max_token = max_token_size
        self.remove_numeric_tables = remove_numeric_tables
            
        result = Clean_PDF(self.data_name, data_folder)

        text = result.run()

        tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})

        # self.tokenizer.save_pretrained(model_folder)

        self.examples = []
        for i in range(0, len(tokenized_text) - max_token_size + 1, max_token_size-1):
            self.examples.append(tokenized_text[i : i + max_token_size-1] + [self.tokenizer.eos_token_id])


        self.data_length = len(self.examples)

    def _read_json(self, file_path):
        f = open(file_path)
        data = json.load(f,object_pairs_hook=OrderedDict)

        clean_text = []
        for i in data:
            temp = i['content']
            clean_text.append(temp.splitlines())

        return clean_text


    def _read_pdf(self, file_path, encoding="UTF-8"):
        command = ["pdftotext", "-enc", encoding, str(file_path), "-"]
        output = subprocess.run(command, stdout=subprocess.PIPE, shell=False)  # type: ignore
        document = output.stdout.decode(errors="ignore")
        
        pages = document.split("\f")
        
        pages = pages[:-1]  # the last page in the split is always empty.
        cleaned_pages = []
        for page in pages:
            lines = page.splitlines()
            cleaned_lines = []
            for line in lines:
                words = line.split()
                digits = [word for word in words if any(i.isdigit() for i in word)]

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if self.remove_numeric_tables:
                    if (
                        words
                        and len(digits) / len(words) > 0.4
                        and not line.strip().endswith(".")
                    ):
                        continue
                cleaned_lines.append(line)

            page = "\n".join(cleaned_lines)
            cleaned_pages.append(page)
            
        return "".join(cleaned_pages[9:])
        

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):

        input_ids = torch.tensor(self.examples[idx])

        attention_mask = torch.ones(len(input_ids), dtype=torch.int64)

        return input_ids, attention_mask



        


if __name__ == '__main__':

    # data_path = "../../../data/skosa_data/"

    # tokenizer_path = "../../../models/qa_model/"

    
    # dataset=SkosaQADataset(data_path, tokenizer_path)

    data_path = "../../../data/swedish_pdf/"
    tokenizer_path = "../../../models/long_text_generator"

    dataset=LongTextDataset(data_path, tokenizer_path)


    data_size = 10
    for i in range(data_size):

        dataset.__getitem__(i)
