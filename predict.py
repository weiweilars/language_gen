from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import pdb

''' create app.py to do following step (see example in image_segmentation)

    api input:
        model_name: one of the model in model_config.yaml
        condition: the conditional label under the model_name, if it is None, ramdom choose one 
        num_generate_data: the number of data want to generate

    According to api input: 
        get the model dictionary from config.yaml according to model_name 

        if condition is None, randomly select one from condition in the model dictionary 
        if condition is not None, check if it is one of the condition in the model dictionary 
        get the model_path 
        get open_end
        get do_sample
        get max_length
        get top_k
        get top_p 

    Pass all argument to predict to generate result 

    Format result to required format and send back 
'''     

def predict(condition, num_generate_data, model_path, do_sample=True, max_length=100, top_k=30, top_p=0.95, open_end=False):

    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

    model = GPT2LMHeadModel.from_pretrained(model_path)

    model.config.pad_token_id = model.config.eos_token_id

    input = condition + tokenizer.sep_token

    inputs_seq = tokenizer(input, return_tensors="pt")

    results = model.generate(inputs_seq['input_ids'], do_sample=do_sample, max_length=max_length, top_k=top_k, top_p=top_p, num_return_sequences=num_generate_data)

    results = [tokenizer.decode(j).replace('<|endoftext|>', '').replace('[SEP]', ' : ').replace('[PAD]', '') for j in results]

    return results 

if __name__ == '__main__':

    condition = 'FK-Funktions_nedsättning-Aktivitetsersättning och sjukersättning-Bostadstillägg'

    #condition = 'd779'

    num_generate_data = 1

    model_path = "./models/qa_model/"

    #model_path = "./models/skosa_model/"
    
    result = predict(condition, num_generate_data, model_path)

    pdb.set_trace()
    
    
    
