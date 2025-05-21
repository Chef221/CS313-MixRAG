# Mixture-of-RAG
Code Implementation of Paper: **Mixture-of-RAG: Integrating Text and Tables with Language Models**

## Overview

Large language models (LLMs) achieve optimal utility when their responses are grounded in external knowledge sources.
However, real-world documents, such as annual reports, scientific papers, and clinical guidelines, frequently combine extensive narrative content with complex, hierarchically structured tables. 
While existing retrieval‑augmented generation (RAG) systems effectively integrate LLMs’ generative capabilities with external retrieval-based information, their performance significantly deteriorates especially processing such heterogeneous text-table hierarchies. 
To address this limitation, we formalize the task of Heterogeneous
Document RAG, which requires joint retrieval and reasoning across textual and hierarchical tabular data. 
We propose MixRAG, a novel three‑stage framework: (i) hierarchy row-and-column-level (H-RCL) representation that preserves hierarchical structure and heterogeneous relationship, (ii) an ensemble retriever with LLM-based reranking for evidence alignment, and (iii) multi‑step reasoning decomposition via a RECAP prompt strategy. 
To bridge the gap in available data for this domain, we release a large-scale dataset t DocRAGLib, a 2k‑document corpus paired with automatically aligned text‑table summaries and gold document annotations. 
The comprehensive experiment results demonstrate that MixRAG boosts top‑1 retrieval by 46%  over strong text‑only, table‑only, and naive‑mixture baselines, establishing new state-of-the-art performance for mixed-modality document grounding.

## Project Structure

This is the structure of the MixRAG project:

```
这里应该有一个树形文件结构
```

## DocLibRAG Dataset

这里要介绍一下数据集以及Data Format

## Baseline

| Method          | Reference                                                    |
| --------------- | ------------------------------------------------------------ |
| Standard RAG    | N/A                                                          |
| LangChain       | [Langchain](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb?ref=blog.langchain.dev) |
| SelfRAG         | [SelfRAG](https://github.com/AkariAsai/self-rag)             |
| Table Retrieval | [Table Retrieval](https://aclanthology.org/2024.acl-long.148/) |

## Requirements

## Running the Code

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

---

#### Deploy Open-Source LLM

To deploy the open-source LLM server, configure and run the script `open_source_LLM\launch_llm_server.py`. Update the following parameters:

```
model_path = '<path_to_your_model>'  # specify your model path here
port_map = {
    'QWen-7B': 6671
}  # port configuration for different models
```
