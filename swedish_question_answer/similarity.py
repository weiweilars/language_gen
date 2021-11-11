from transformers import AutoTokenizer, AutoModel
import torch
import pdb
import json
import pickle
import os
import numpy

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class FindQuestion():
    
    def __init__(self, data_path, generate_new=False):

        

        self.tokenizer = AutoTokenizer.from_pretrained('KBLab/sentence-bert-swedish-cased')
        self.model = AutoModel.from_pretrained('KBLab/sentence-bert-swedish-cased')

        if generate_new:
            f = open(os.path.join(data_path, 'question_answer.json'))
            data = json.load(f)
            f.close()

        
            self.questions = []
            self.answers = []
            self.question_embeddings = []

            for i in data:
                temp_question = i['question']
                self.questions.append(temp_question)
                self.answers.append(i['answer'])

                encoded_input = self.tokenizer(temp_question, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                temp_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                self.question_embeddings.append(temp_embeddings[0])

            with open(os.path.join(data_path, 'questions.pkl'), 'wb') as f:
                pickle.dump(self.questions, f)

            with open(os.path.join(data_path, 'answers.pkl'), 'wb') as f:
                pickle.dump(self.answers, f)

            with open(os.path.join(data_path, 'question_embeddings.pkl'), 'wb') as f:
                pickle.dump(self.question_embeddings, f)

        else:

            with open(os.path.join(data_path, 'questions.pkl'), 'rb') as f:
                self.questions = pickle.load(f)

            with open(os.path.join(data_path, 'answers.pkl'), 'rb') as f:
                self.answers = pickle.load(f)

            with open(os.path.join(data_path, 'question_embeddings.pkl'), 'rb') as f:
                self.question_embeddings = pickle.load(f)


        print(self.questions)
        #print(self.question_embeddings)


    def find_question(self, text):

        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])[0]

        score = numpy.array([torch.dot(sentence_embeddings, i).numpy() for i in self.question_embeddings])

        pdb.set_trace()

        


    



if __name__ == '__main__':
   
    data_path = "./data/skosa_data/"

    find_class = FindQuestion(data_path)

    find_class.find_question('Får jag behålla vårdbidraget?')
     
