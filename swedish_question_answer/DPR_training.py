
from haystack.nodes import DensePassageRetriever
from haystack.utils import fetch_archive_from_http
from haystack.document_stores import InMemoryDocumentStore

## Option 1: Training DPR from Scratch

# Here are the variables to specify our training data, the models that we use to initialize DPR
# and the directory where we'll be saving the model

doc_dir = "../data/dpr_data/"

train_filename = "DPR_squad_train_v2_sv.json"
dev_filename = "DPR_squad_dev_v2_sv.json"

save_dir = "../models/dpr"

# ## Option 2: Finetuning DPR
    #
# # Here are the variables you might want to use instead of the set above
# # in order to perform pretraining
#
# doc_dir = "PATH_TO_YOUR_DATA_DIR"
# train_filename = "TRAIN_FILENAME"
# dev_filename = "DEV_FILENAME"
#
query_model = "KB/bert-base-swedish-cased"
passage_model = "KB/bert-base-swedish-cased"
#
# save_dir = "..saved_models/dpr"

## Initialize DPR model

retriever = DensePassageRetriever(
    document_store=InMemoryDocumentStore(),
    query_embedding_model=query_model,
    passage_embedding_model=passage_model,
    max_seq_len_query=64,
    max_seq_len_passage=256
)

# Start training our model and save it when it is finished

retriever.train(
    data_dir=doc_dir,
    train_filename=train_filename,
    dev_filename=dev_filename,
    test_filename=dev_filename,
    n_epochs=1,
    batch_size=16,
    grad_acc_steps=8,
    save_dir=save_dir,
    evaluate_every=3000,
    embed_title=True,
    num_positives=1,
    num_hard_negatives=1
)

    ## Loading

reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=None)


# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
