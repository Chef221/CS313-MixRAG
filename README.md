# Mixture-of-RAG
Code Implementation of Paper: Mixture-of-RAG: Integrating Text and Tables with Language Models

### Configuration Options

#### Option 1: **Using `config.py` for Default Configuration**

The default configuration is specified in the `config.py` file. Once you have set up this configuration file, you can run the script directly without needing to provide any command-line arguments.

To run the script, simply execute:

```bash
python main.py
```

This will load the default configuration from `config.py` and execute the script accordingly.

------

#### Option 2: **Override Configuration via Command-Line Arguments**

Alternatively, you can override the default configuration by specifying command-line arguments when running the script. Use the following syntax to pass parameters:

```bash
python main.py \
  --documents_path /path/to/dev_documents.json \
  --qa_path /path/to/dev_qa.json \
  --output_dir /path/to/output_directory \
  --embedding_corpus_path /path/to/embedding_corpus.pkl \
  --bm25_corpus_path /path/to/bm25_corpus.pkl \
  --top_n_embedding 50 \
  --top_n_bm25 50 \
  --model_RECAP_qa gpt4o-retran \
  --model_LLM_reranker gpt4o-retran \
  --debugger_mode True \
  --use_calculator True
```

### Command-Line Arguments

- `--documents_path`: Path to the development documents file (required).
- `--qa_path`: Path to the development QA file (required).
- `--output_dir`: Directory to save the output files (required).
- `--embedding_corpus_path`: Path to the embedding corpus file (optional).
- `--bm25_corpus_path`: Path to the precomputed BM25 corpus file (optional).
- `--top_n_embedding`: Number of top documents to retrieve using the embedding model (optional, default is 50).
- `--top_n_bm25`: Number of top documents to retrieve using the BM25 model (optional, default is 50).
- `--model_RECAP_qa`: Model for RECAP QA (optional, default is `"gpt4o-retran"`).
- `--model_LLM_reranker`: Model for LLM reranker (optional, default is `"gpt4o-retran"`).
- `--debugger_mode`: Enable or disable debugger mode (optional, default is `True`).
- `--use_calculator: Use external Calculator or not (optional, default is `True`).

------

### Example Usage

**1. Run the script with the default `config.py` settings:**

```bash
python main.py
```

**2. Run the script with custom command-line parameters:**

```bash
python main.py \
  --documents_path /data/dev_documents.json \
  --qa_path /data/dev_qa.json \
  --output_dir /output/ \
  --embedding_corpus_path /models/embedding_corpus.pkl \
  --bm25_corpus_path /models/bm25_corpus.pkl \
  --top_n_embedding 100 \
  --top_n_bm25 100 \
  --model_RECAP_qa gpt4o-advanced \
  --model_LLM_reranker gpt4o-advanced \
  --debugger_mode False
  --use_calculator True
```

This will override the corresponding settings in `config.py` and execute the script with the specified parameters.

------

### Notes:

- **Configuration Precedence**: If both the `config.py` file and command-line arguments are used, the values provided via the command line will take precedence over those in `config.py`.
- **Defaults**: If any command-line arguments are not provided, the script will fall back to the default values in `config.py`.
