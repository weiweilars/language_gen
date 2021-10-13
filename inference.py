from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import json
import os
from collections import OrderedDict
import pdb
import re


def predict_qa(data_path, model_path, save_path):


    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})

    model = GPT2LMHeadModel.from_pretrained(model_path)

    f = open(data_path)


    data = json.load(f, object_pairs_hook=OrderedDict)

    file = open(os.path.join(save_path, 'result_qa.txt'), "a")

    categories = []
    for i in data:

        categories.append(i['category'])

    categories = set(categories)

    for i in categories:

        input = i + tokenizer.sep_token

        inputs_seq = tokenizer(input, return_tensors="pt")

        results = model.generate(inputs_seq['input_ids'], do_sample=True, max_length=100, top_k=30, top_p=0.95, num_return_sequences=10)

        for j in results:

            result = tokenizer.decode(j)

            #pdb.set_trace()
            result = result.replace('<|endoftext|>', '')
            result = result.replace('[SEP]', ' : ')
            result = re.sub('\[PAD\]', '', result)

            #print(result)
            
            file.write(result)
            file.write('\n')

def predict_icf(data_path, model_path, save_path):


    df = pd.read_csv(data_path, sep=';')
    df.reset_index()

    new_tokens = [*set(df.targetName.to_list()),]

    new_tokens = [x for x in new_tokens if x]
    
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

    model = GPT2LMHeadModel.from_pretrained(model_path)

    file = open(os.path.join(save_path, 'result_icf.txt'), "a")

    for i in new_tokens:

        file.write(i)
        file.write('\n')

        #print(i)

        inputs_seq = tokenizer([i], return_tensors="pt")

        #results = model.generate(inputs_seq['input_ids'], max_length=20, num_beams=5, temperature=0.9, no_repeat_ngram_size=2, num_return_sequences=1, do_sample=True, repetition_penalty=0.8)

        results = model.generate(inputs_seq['input_ids'], do_sample=True, max_length=30, top_k=30, top_p=0.95, num_return_sequences=10)

        for j in results:
            
            result = tokenizer.decode(j).replace(i, '')

            result = result.replace('<|endoftext|>', '')
            result = re.sub('\[PAD\]', '', result)

            #rint(result)
            
            file.write(result)
            file.write('\n')

    


if __name__ == '__main__':


    data_path = "./data/skosa_data/question_answer.json"

    model_path = "./models/qa_model/"

    
