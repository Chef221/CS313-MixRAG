#!/bin/bash

# Set the path to your input document JSON file
DOC_PATH="PATH/DocLibRAG/Documents.json"

# Set the corpus type (1 = table level, 2 = rol_col_embedding, 3 = rol_col_para_embedding)
CORPUS_TYPE=3

# Set the output paths for the embedding and BM25 files
OUTPUT_EMBEDDING_PATH="PATH/src/Corpus/embedding_corpus.pkl"
OUTPUT_BM25_PATH="PATH/src/Corpus//bm25_corpus.pkl"

# Run the Python script with the arguments
python corpus_construction.py \
  --doc_path "$DOC_PATH" \
  --corpus_type "$CORPUS_TYPE" \
  --output_embedding_path "$OUTPUT_EMBEDDING_PATH" \
  --output_bm25_path "$OUTPUT_BM25_PATH"
