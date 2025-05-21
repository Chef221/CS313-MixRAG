import json
import numpy as np
import pandas as pd
import time
import openai
import argparse

from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from openai import AzureOpenAI
from tqdm import tqdm
from utils import *

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

client_id = ""  # Here Provide your azure client id

token_provider = get_bearer_token_provider(
    ManagedIdentityCredential(client_id=client_id), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version="2024-05-01-preview",
    azure_endpoint="https://excelllmswedencentral.openai.azure.com/",
    azure_ad_token_provider=token_provider
)


def prepare_bm25_corpus(df_chunks, save_path=None):
    """
    Prepares a tokenized corpus for BM25 retrieval.

    This function processes a list of document chunks (from either paragraphs or tables),
    tokenizes the content using word-level tokenization, and optionally saves the resulting
    DataFrame to a pickle file.

    Args:
        df_chunks (list or pd.DataFrame): A list of dictionaries or a DataFrame containing document chunks,
            each with a 'content' field.
        save_path (str, optional): Path to save the tokenized corpus as a pickle file. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing original content and its tokenized form.
    """
    bm25_documents_chunks = df_chunks
    df_bm25_documents_chunks = pd.DataFrame(bm25_documents_chunks)
    df_bm25_documents_chunks['tokenized_content'] = df_bm25_documents_chunks['content'].apply(lambda x: word_tokenize(x.lower()))
    if save_path:
        df_bm25_documents_chunks.to_pickle(save_path)
    return df_bm25_documents_chunks


def rol_col_embedding_corpus_construction(documents):
    """
    Constructs a corpus using row-and-column-level summaries for table embeddings.

    This function processes a list of documents, where each document may contain multiple tables and paragraphs.
    For each table, it generates row-and-column-level summaries (via `table_row_column_summary`) and treats each
    summary as a separate chunk. Paragraphs are concatenated into one chunk.

    Args:
        documents (list): A list of document dictionaries, each with 'tables', 'paragraphs', and 'uid'.

    Returns:
        list: A list of chunks, each with 'document_uid', 'content', and 'label'.
    """
    documents_chunks = []
    
    for document in tqdm(documents, desc="Processing..."):
        paragraphs = document.get("paragraphs", [])
        tables = document.get("tables", [])
        uid = document.get("uid", "N/A")

        for i, table in enumerate(tables):
            retries = 5
            delay = 5
            for attempt in range(retries):
                try:
                    row_col_sums = table_row_column_summary(client, table, i)
                    break
                except openai.RateLimitError:
                    if attempt < retries - 1:
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print("Rate limit exceeded. Maximum retries reached.")
                        raise
            for row_col_sum in row_col_sums:
                documents_chunks.append(
                    {
                        "document_uid": uid,
                        "content": row_col_sum,
                        "label": f"Table {i}"
                    }
                )

        documents_chunks.append(
            {
                "document_uid": uid,
                "content": "\n".join(paragraphs),
                "label": "paragraph"
            }
        )
    
    return documents_chunks


def rol_col_general_embedding_corpus_construction(documents):
    """
    Constructs a corpus using *generalized* row-and-column-level summaries for tables.

    Similar to `rol_col_embedding_corpus_construction`, but uses a more generalized table summarization
    method for each table. Paragraphs are still concatenated.

    Args:
        documents (list): A list of document dictionaries, each with 'tables', 'paragraphs', and 'uid'.

    Returns:
        list: A list of document chunks for embedding purposes.
    """

    documents_chunks = []
    
    for document in tqdm(documents, desc="Processing..."):
        paragraphs = document.get("paragraphs", [])
        tables = document.get("tables", [])
        uid = document.get("uid", "N/A")

        for i, table in enumerate(tables):
            retries = 5
            delay = 5
            for attempt in range(retries):
                try:
                    row_col_sums = table_row_column_general_summary(client, table, i)
                    break
                except openai.RateLimitError:
                    if attempt < retries - 1:
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print("Rate limit exceeded. Maximum retries reached.")
                        raise
            for row_col_sum in row_col_sums:
                documents_chunks.append(
                    {
                        "document_uid": uid,
                        "content": row_col_sum,
                        "label": f"Table {i}"
                    }
                )

        documents_chunks.append(
            {
                "document_uid": uid,
                "content": "\n".join(paragraphs),
                "label": "paragraph"
            }
        )
    
    return documents_chunks


def rol_col_para_embedding_corpus_construction(documents):
    documents_chunks = []
    for document in tqdm(documents, desc="Processing..."):
        paragraphs = document.get("paragraphs", [])
        tables = document.get("tables", [])
        uid = document.get("uid", "N/A")

        for i, table in enumerate(tables):
            retries = 5
            delay = 5
            for attempt in range(retries):
                try:
                    row_col_sums = table_row_column_summary(client, table, i)
                    break
                except (openai.RateLimitError, openai.InternalServerError):
                    if attempt < retries - 1:
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print("Rate limit exceeded. Maximum retries reached.")
                        raise
            for row_col_sum in row_col_sums:
                documents_chunks.append(
                    {
                        "document_uid": uid,
                        "content": row_col_sum,
                        "label": f"Table {i}"
                    }
                )
        
        for j, text in enumerate(paragraphs):
            documents_chunks.append(
                {
                    "document_uid": uid,
                    "content": text,
                    "label": f"paragraph {j}"
                }
            )

    return documents_chunks
    


def table_level_embedding_corpus_construction(documents):
    """
    Constructs a corpus with table-level summaries and individual paragraphs.

    This function creates a summary for each table using `table_summary` and includes each paragraph as
    a separate chunk. This is a higher-level summarization strategy compared to row-column level methods.

    Args:
        documents (list): A list of documents with keys 'tables', 'paragraphs', and 'uid'.

    Returns:
        list: A list of document chunks labeled by table or paragraph index.
    """

    documents_chunks = []
    for document in tqdm(documents, desc="Processing..."):
        paragraphs = document.get("paragraphs", [])
        tables = document.get("tables", [])
        uid = document.get("uid", "N/A")

        for i, table in enumerate(tables):
            tablesum = table_summary(client, table, i)
            documents_chunks.append(
                {
                    "document_uid": uid,
                    "content": tablesum,
                    "label": f"Table {i}"
                }
            )

        for j, paragraph in enumerate(paragraphs):
                documents_chunks.append(
                {
                    "document_uid": uid,
                    "content": paragraph,
                    "label": f"Paragraph {j}"
                }
            )
    
    return documents_chunks


def main(args):
    doc_path = args.doc_path
    corpus_type = args.corpus_type
    output_embedding_path = args.output_embedding_path
    output_bm25_path = args.output_bm25_path

    # documents = read_json_file(doc_path)[0:2]
    documents = read_json_file(doc_path)
    if corpus_type == 1:
        documents_chunks = table_level_embedding_corpus_construction(documents)
    elif corpus_type == 2:
        documents_chunks = rol_col_embedding_corpus_construction(documents)
    elif corpus_type == 3:
        documents_chunks = rol_col_para_embedding_corpus_construction(documents)
    else:
        documents_chunks = rol_col_general_embedding_corpus_construction(documents)

    df_documents_chunks = pd.DataFrame(documents_chunks)

    df_documents_chunks['ada_v2'] = df_documents_chunks["content"].apply(lambda x: get_embedding(x, client))

    df_documents_chunks.to_pickle(output_embedding_path)

    df_bm25_documents_chunks = prepare_bm25_corpus(df_documents_chunks, save_path=output_bm25_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Processing with BM25 and Embedding")

    parser.add_argument(
        '--doc_path', type=str, required=True,
        help="Path to the input JSON document file"
    )
    parser.add_argument(
        '--corpus_type', type=int, choices=[1, 2, 3, 4], required=True,
        help="Type of corpus: 1 = table level, 2 = rol_col_embedding, 3 = rol_col_para_embedding, 4 = general"
    )
    parser.add_argument(
        '--output_embedding_path', type=str, required=True,
        help="Path to save the output embedding file"
    )
    parser.add_argument(
        '--output_bm25_path', type=str, required=True,
        help="Path to save the output BM25 file"
    )

    args = parser.parse_args()

    main(args)

    print("Finished!")