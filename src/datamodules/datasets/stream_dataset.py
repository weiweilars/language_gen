import json
import os
import pdb
from itertools import islice, cycle
from collections import OrderedDict
from torch.utils.data import Dataset, IterableDataset, DataLoader 
from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

class SkosaStreamDataset(IterableDataset):

    input_path_name = 'input'
    data_name = 'icf.json'

    def __init__(self, data_path, tokenizer_path, max_token_size=512):


        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
        self.input_path = os.path.join(data_path, self.data_name)
        self.max_token = max_token_size

        f = open(self.input_path)

        data = json.load(f)

        for i in data:

            pdb.set_trace()

        f.close()
        
        # if generate_size > 0:
        #     df = df.head(generate_size)
            
        #     new_tokens = [*set(df.targetName.to_list()),]
        
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        #     self.tokenizer.add_tokens(new_tokens)
        
        #     self.tokenizer.save_pretrained(tokenizer_path)

            

        #     for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        #         org_sentence = row.targetName + row.featureValue + self.tokenizer.eos_token
            
        #         inputs = self.tokenizer(org_sentence, return_tensors="pt", max_length=max_token_size, padding='max_length', truncation=True)

        #     self.data_length = index

        # else:

        #     list = os.listdir(self.input_path)

        #     self.data_length = len(list)

    # def parse_file(self, file_path):
    #     with open(file_path, 'r') as file_obj:
    #         for line in file_obj:
    #             data = json.load(line)
    #             yield from data 

    # def get_stream(self, file_path):
    #     return cycle(self.parse_file(file_path))

    
    def __iter__(self):

        # result = self.get_stream(self.input_path)
        pass



if __name__ == '__main__':

    data_path = "../../../data/skosa_data/"

    # tokenizer_path = "../../../models/test_model/"

    # iterable_dataset = SkosaStreamDataset(data_path, tokenizer_path)

    # loader = DataLoader(iterable_dataset, batch_size=5)

    # for batch in islice(loader, 8):

    #     pdb.set_trace()

    from datasets import load_dataset 

    dataset = load_dataset('wiki40b', 'en', split='train', beam_runner='DirectRunner')
    

        
        



