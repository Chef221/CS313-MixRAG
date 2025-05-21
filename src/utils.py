import json
import pandas as pd

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_data(document_path, qa_path):
    documents = read_json_file(document_path)
    qa_dataset = read_json_file(qa_path)

    return documents, qa_dataset

def load_corpus(corpus_path):
    df_corpus = pd.read_pickle(corpus_path)
    return df_corpus

def filter(doc_list):
    filter_list = []
    for entry in doc_list:
        question = entry['question']
        related_docs = entry['related_doc']
        filtered_related_docs = []
        for doc in related_docs:
            uid = doc['document_uid']
            related_doc_detail = doc['related_doc_detail']
            # Filter out chunks longer than 2000 characters
            filtered_related_doc_detail = [chunk for chunk in related_doc_detail if len(chunk) <= 800]
            if filtered_related_doc_detail:
                filtered_doc = {
                    'document_uid': uid,
                    'related_doc_detail': filtered_related_doc_detail
                }
                filtered_related_docs.append(filtered_doc)
        if filtered_related_docs:
            filter_list.append({
                'question': question,
                'related_doc': filtered_related_docs
            })
    
    return filter_list