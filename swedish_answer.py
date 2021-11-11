from haystack.document_stores import FAISSDocumentStore, MilvusDocumentStore
from haystack.utils import clean_wiki_text, print_answers, launch_milvus, convert_files_to_dicts, fetch_archive_from_http
from haystack.nodes import FARMReader, DensePassageRetriever

def swedish_retrieval_via_dpr(doc_dir, question, question_model, context_model, squad_model):
    
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    dicts=convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    # Now, let's write the docs to our DB.
    document_store.write_documents(dicts)


    
    ### Retriever
    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model=question_model,
                                      passage_embedding_model=answer_model,
                                      max_seq_len_query=64,
                                      max_seq_len_passage=256,
                                      batch_size=2,
                                      use_gpu=True,
                                      embed_title=True,
                                      use_fast_tokenizers=True)

    document_store.update_embeddings(retriever)

    reader = FARMReader(model_name_or_path=squad_model, use_gpu=True)

    ### Pipeline
    from haystack.pipelines import ExtractiveQAPipeline
    pipe = ExtractiveQAPipeline(reader, retriever)

    ## Voilà! Ask a question!
    prediction = pipe.run(
        query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )

    # prediction = pipe.run(query="Who created the Dothraki vocabulary?", params={"Reader": {"top_k": 5}})
    # prediction = pipe.run(query="Who is the sister of Sansa?", params={"Reader": {"top_k": 5}})

    print_answers(prediction, details="minimal")


if __name__ == "__main__":

    doc_dir = './data/swedish_pdf/'

    question = "Vem är sjukskriven?"

    question_model = "./swedish_base_models/dpr-question_encoder-bert-base-multilingual/"
    context_model = "./swedish_base_models/dpr-ctx_encoder-bert-base-multilingual/"
    squad_model = "./swedish_base_models/bert-base-swedish-squad2/"
    
    swedish_retrieval_via_dpr(doc_dir, question, question_model, context_model, squad_model)

    
