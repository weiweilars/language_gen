from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import json
import os
from collections import OrderedDict
import pdb
import re

data_path = "./data/skosa_data/question_answer.json"

model_path = "./models/qa_model/"

tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})

model = GPT2LMHeadModel.from_pretrained(model_path)

f = open(data_path)


data = json.load(f, object_pairs_hook=OrderedDict)

file = open('result_qa.txt', "a")

categories = []
for i in data:

    categories.append(i['category'])

categories = set(categories)

for i in categories:

    input = i + tokenizer.sep_token

    inputs_seq = tokenizer(input, return_tensors="pt")

    results = model.generate(inputs_seq['input_ids'], do_sample=True, max_length=100, top_k=30, top_p=0.95, num_return_sequences=10)

    # pdb.set_trace()

    for j in results:

        result = tokenizer.decode(j)

        #pdb.set_trace()
        result = result.replace('<|endoftext|>', '')
        result = result.replace('[SEP]', ' : ')
        result = re.sub('\[PAD\]', '', result)

        print(result)
            
        file.write(result)
        file.write('\n')
