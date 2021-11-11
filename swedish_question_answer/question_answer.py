from transformers import BertTokenizer, BertForQuestionAnswering, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, AutoTokenizer
import torch
import numpy as np
from file_convert import convert_files_to_dicts
import pdb

def question_answer(question, text, model_path):

    model = BertForQuestionAnswering.from_pretrained(model_path)

    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    #tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #number of tokens in segment A (question)
    num_seg_a = sep_idx+1
    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s for segment embeddings
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
                
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    
    print("\nPredicted answer:\n{}".format(answer.capitalize()))


def answer_document(question, document, question_model, context_model, squad_model):

    document = convert_files_to_dicts(document)

    question_tokenizer = AutoTokenizer.from_pretrained(question_model)
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_model)

    question_model = DPRQuestionEncoder.from_pretrained(question_model).eval()
    context_model = DPRContextEncoder.from_pretrained(context_model).eval()

    context_encoding = [i['content'] for i in document]

    context_encoding = context_tokenizer.batch_encode_plus(context_encoding,
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           padding="max_length",
                                                           max_length = 512,
                                                           return_token_type_ids=True,
                                                           return_tensors="pt")

    question_encoding = question_tokenizer(question,
                                           add_special_tokens=True,
                                           truncation=True,
                                           padding="max_length",
                                           max_length = 512,
                                           return_token_type_ids=True,
                                           return_tensors="pt")
    with torch.no_grad():
        context_embedding = context_model(context_encoding["input_ids"]).pooler_output
        
        question_embedding = question_model(question_encoding["input_ids"]).pooler_output

        similarity_result = [np.dot(question_embedding.numpy(), i) for i in context_embedding.numpy()]


    highest_context = document[similarity_result.index(max(similarity_result))]['content']

    
    pdb.set_trace()

    question_answer(question, highest_context, squad_model)



if __name__ == '__main__':

    question_model = "../swedish_base_models/dpr-question_encoder-bert-base-multilingual/"
    context_model = "../swedish_base_models/dpr-ctx_encoder-bert-base-multilingual/"

    squad_model = '../swedish_base_models/bert-base-swedish-squad2/'

    document = '../data/swedish_pdf'

    question = 'vad är min nyårslöfte?'

    text = 'Har ni avlagt något nyårslöfte? Det är nästan så man inte ska berätta om dem meeeen jag har ett, jag ska dricka mer vatten. Oroväckande kanske beroende på hur man associerar. Men jag lovar att det bara handlar om att dricka mer vatten. Jag dricker ofta för lite vatten och känner mig trött och har nästan ont huvet frammåt kvällen. Därför ska jag tänka på att dricka mer vatten. Nån som hänger på?!'


    question_answer(question, text, squad_model)
    
    answer_document(question, document, question_model, context_model, squad_model)
    
