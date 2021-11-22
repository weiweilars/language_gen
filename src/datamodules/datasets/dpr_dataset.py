from torch.utils.data import Dataset
from transformers import AutoTokenizer  
import logging
import pdb
import os
import json
import random
import numpy as np
import torch
import pickle
from tqdm import tqdm
logger = logging.getLogger(__name__)

def read_dpr_json(file, max_samples=None, proxies=None, num_hard_negatives=10, num_positives=1, shuffle_negatives=True, shuffle_positives=False):
    """
    Reads a Dense Passage Retrieval (DPR) data file in json format and returns a list of dictionaries.
    :param file: filename of DPR data in json format
    Returns:
        list of dictionaries: List[dict]
        each dictionary: {
                    "query": str -> query_text
                    "passages": List[dictionaries] -> [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
                                {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
                                ...]
                    }
        example:
                ["query": 'who sings does he love me with reba'
                "passages" : [{'title': 'Does He Love You',
                    'text': 'Does He Love You "Does He Love You" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba\'s album "Greatest Hits Volume Two". It is one of country music\'s several songs about a love triangle. "Does He Love You" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members',
                    'label': 'positive'},
                    {'title': 'When the Nightingale Sings',
                    'text': "When the Nightingale Sings When The Nightingale Sings is a Middle English poem, author unknown, recorded in the British Library's Harley 2253 manuscript, verse 25. It is a love poem, extolling the beauty and lost love of an unknown maiden. When þe nyhtegale singes þe wodes waxen grene.<br> Lef ant gras ant blosme springes in aueryl y wene,<br> Ant love is to myn herte gon wiþ one spere so kene<br> Nyht ant day my blod hit drynkes myn herte deþ me tene. Ich have loved al þis er þat y may love namore,<br> Ich have siked moni syk lemmon for",
                    'label': 'hard_negative'}]
                ]
    """
    # get remote dataset if needed
    if not (os.path.exists(file)):
        logger.error(" Couldn't find {file} locally")

    dicts = json.load(open(file, encoding='utf-8'))

    if max_samples:
        dicts = random.sample(dicts, min(max_samples, len(dicts)))

    # convert DPR dictionary to standard dictionary
    query_json_keys = ["question", "questions", "query"]
    positive_context_json_keys = ["positive_contexts", "positive_ctxs", "positive_context", "positive_ctx"]
    hard_negative_json_keys = ["hard_negative_contexts", "hard_negative_ctxs", "hard_negative_context", "hard_negative_ctx"]
    standard_dicts = []
    for dict in dicts:
        sample = {}
        passages = []
        for key, val in dict.items():
            if key in query_json_keys:
                sample["query"] = val
            elif key in positive_context_json_keys:
                if shuffle_positives:
                    random.shuffle(val)
                for passage in val[:num_positives]:
                    passages.append({
                        "title": passage["title"],
                        "text": passage["text"],
                        "label": "positive",
                        })
            elif key in hard_negative_json_keys:
                if shuffle_negatives:
                    random.shuffle(val)
                for passage in val[:num_hard_negatives]:
                    passages.append({
                        "title": passage["title"],
                        "text": passage["text"],
                        "label": "hard_negative",
                        })
        sample["passages"] = passages
        standard_dicts.append(sample)
    return standard_dicts

class DPRDataset(Dataset):
    def __init__(
            self,
            query_tokenizer,
            passage_tokenizer,
            max_seq_len_query=256,
            max_seq_len_passage=512,
            data_dir="data/dpr_data",
            data_type="swedish_dpr_train",
            max_samples=None,
            embed_title=False,
            num_positives=1,
            num_hard_negatives=10,
            shuffle_negatives=True,
            shuffle_positives=True,
            generate_new=True):

    
        self.save_path = os.path.join(data_dir, data_type)

        

        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_passage = max_seq_len_passage
        self.num_positives = num_positives
        self.num_hard_negatives = num_hard_negatives
        self.shuffle_negatives = shuffle_negatives
        self.shuffle_positives = shuffle_positives
        self.embed_title = embed_title

        self.query_tokenizer = AutoTokenizer.from_pretrained(query_tokenizer)
        self.passage_tokenizer = AutoTokenizer.from_pretrained(passage_tokenizer)

        if generate_new:
            data_file = os.path.join(data_dir, data_type +'.json')
            data_dict = read_dpr_json(data_file)
            self.data_length = len(data_dict)
            
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            else:
                list = os.listdir(self.save_path)
                for f in list:
                    os.remove(os.path.join(self.save_path, f))
                
            self.data = self._convert_queries(data_dict)
        else:
            assert os.path.exists(self.save_path)
            list = os.listdir(self.save_path)
            self.data_length = len(list)
            

        
    # def dataset_from_dicts(self, dicts, fp):
         
    #     queries = self._convert_queries(dicts)
        

    def _convert_queries(self, dicts):

        for idx, dict in enumerate(tqdm(dicts)):

            clear_text = {}
            tokenized = {}
            features = {}

            query = self._normalize_question(dict['query'])
            query_inputs = self.query_tokenizer.encode_plus(
                        text=query,
                        max_length=self.max_seq_len_query,
                        add_special_tokens=True,
                        truncation=True,
                        truncation_strategy='longest_first',
                        padding="max_length",
                        return_token_type_ids=True,
                    )

            tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_inputs["input_ids"])

            if len(tokenized_query) == 0:
                        logger.warning(
                            "The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize")
                        return None

            clear_text['query_text'] = query
            tokenized['query_tokens'] = tokenized_query
            features['query_input_ids'] = query_inputs["input_ids"]
            features['query_segment_ids'] = query_inputs['token_type_ids']
            features['query_attention_mask'] = query_inputs['attention_mask']

   
            positive_context = list(filter(lambda x: x["label"] == "positive", dict["passages"]))
            if self.shuffle_positives:
                random.shuffle(positive_context)
            positive_context = positive_context[:self.num_positives]
            hard_negative_context = list(filter(lambda x: x["label"] == "hard_negative", dict["passages"]))
            if self.shuffle_negatives:
                random.shuffle(hard_negative_context)
            hard_negative_context = hard_negative_context[:self.num_hard_negatives]

            
            positive_ctx_titles = [passage.get("title", None) for passage in positive_context]
            positive_ctx_texts = [passage["text"] for passage in positive_context]
            hard_negative_ctx_titles = [passage.get("title", None) for passage in hard_negative_context]
            hard_negative_ctx_texts = [passage["text"] for passage in hard_negative_context]

            # all context passages and labels: 1 for positive context and 0 for hard-negative context
            ctx_label = [1] * self.num_positives + [0] * self.num_hard_negatives
            # featurize context passages
            if self.embed_title:
                # concatenate title with positive context passages + negative context passages
                all_ctx = self._combine_title_context(positive_ctx_titles, positive_ctx_texts) + \
                    self._combine_title_context(hard_negative_ctx_titles, hard_negative_ctx_texts)
            else:
                all_ctx = positive_ctx_texts + hard_negative_ctx_texts

                # assign empty string tuples if hard_negative passages less than num_hard_negatives
            all_ctx += [('', '')] * ((self.num_positives + self.num_hard_negatives) - len(all_ctx))

            ctx_inputs = self.passage_tokenizer.batch_encode_plus(all_ctx,
                                                                  add_special_tokens=True,
                                                                  truncation=True,
                                                                  padding="max_length",
                                                                  max_length=self.max_seq_len_passage,
                                                                  return_token_type_ids=True)
            ctx_segment_ids = np.zeros_like(ctx_inputs["token_type_ids"], dtype=np.int32)
            tokenized_passage = [self.passage_tokenizer.convert_ids_to_tokens(ctx) for ctx in ctx_inputs["input_ids"]]

            clear_text['passages'] = positive_context + hard_negative_context
            tokenized['passages_tokens'] = tokenized_passage
            features['passage_input_ids'] = ctx_inputs['input_ids']
            features['passage_segment_ids'] = ctx_segment_ids
            features['passage_attention_mask'] = ctx_inputs['attention_mask']
            features['label_ids'] = ctx_label

            result = {'clear_text': clear_text, 'tokenized': tokenized, 'features': features}
            
            with open(os.path.join(self.save_path, str(idx) + '.pkl'), 'wb') as handle:
                pickle.dump(result, handle)

            #self.final_data.append(result)

        
    @staticmethod
    def _combine_title_context(titles, texts):
        res = []
        for title, ctx in zip(titles, texts):
            if title is None:
                title = ""
                logger.warning(
                    "Couldn't find title although `embed_title` is set to True for DPR. Using title='' now. Related passage text: '{ctx}' ")
            res.append(tuple((title, ctx)))
        return res


    @staticmethod
    def _normalize_question(question):
        """Removes '?' from queries/questions"""
        if question[-1] == '?':
            question = question[:-1]
        return question

            
    def __len__(self):
        return self.data_length

    def __getitem__(self, index):

        with open(os.path.join(self.save_path, str(index) + '.pkl'), 'rb') as handle:
            data = pickle.load(handle)

        data = data['features']
        return torch.LongTensor(data['query_input_ids']), torch.LongTensor(data['query_attention_mask']), torch.LongTensor(data['passage_input_ids']), torch.LongTensor(data['passage_attention_mask']), torch.LongTensor(data['label_ids'])
            
            
        


if __name__ == '__main__':


    query_tok = '../../../models/bert-base-swedish-cased/'
    passage_tok = '../../../models/bert-base-swedish-cased/'
    data_dir = '../../../data/dpr_data/'

    
    dataset = DPRDataset(query_tokenizer=query_tok, passage_tokenizer=passage_tok, data_dir=data_dir)


    
    

    
