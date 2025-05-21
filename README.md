# Mixture-of-RAG
Code Implementation of Paper: **Mixture-of-RAG: Integrating Text and Tables with Language Models**

## Overview

Large language models (LLMs) achieve optimal utility when their responses are grounded in external knowledge sources.
However, real-world documents, such as annual reports, scientific papers, and clinical guidelines, frequently combine extensive narrative content with complex, hierarchically structured tables. 
While existing retrieval‑augmented generation (RAG) systems effectively integrate LLMs’ generative capabilities with external retrieval-based information, their performance significantly deteriorates especially processing such heterogeneous text-table hierarchies. 
To address this limitation, we formalize the task of Heterogeneous
Document RAG, which requires joint retrieval and reasoning across textual and hierarchical tabular data. 
We propose MixRAG, a novel three‑stage framework: (i) hierarchy row-and-column-level (H-RCL) representation that preserves hierarchical structure and heterogeneous relationship, (ii) an ensemble retriever with LLM-based reranking for evidence alignment, and (iii) multi‑step reasoning decomposition via a RECAP prompt strategy. 
To bridge the gap in available data for this domain, we release a large-scale dataset DocRAGLib, a 2k‑document corpus paired with automatically aligned text‑table summaries and gold document annotations. 
The comprehensive experiment results demonstrate that MixRAG boosts top‑1 retrieval by 46%  over strong text‑only, table‑only, and naive‑mixture baselines, establishing new state-of-the-art performance for mixed-modality document grounding.

## Project Structure

This is the structure of the MixRAG project:

```text
Mixture-of-RAG/
├── README.md                           # Project documentation
├── DocLibRAG/                          # Dataset and corpus files
│   ├── scalability_dataset/            # Scalability evaluation datasets
│   │   ├── documents-280.json          
│   │   ├── documents-1000.json      
│   │   ├── documents-1500.json      
│   │   ├── documents-2178.json         
│   │   └── QA.json                     
│   ├── Documents.json                  # Document corpus
│   ├── dev_qa.json                     # Development QA pairs
│   ├── test_qa.json                    # Test QA pairs
│   └── train_qa.json                   # Training QA pairs
├── result/                             # Output results directory
│   ├── ablation_result/               
│   ├── MixRAG-result1/                
│   ├── MixRAG-result2/                
│   └── scalability_result/             
├── src/                                # Source code
│   ├── config.py                       # Configuration settings
│   ├── main.py                         # Main entry point
│   ├── utils.py                        # Utility functions
│   ├── evaluate/                       # Evaluation modules
│   │   └── evaluate.py                
│   ├── open_source_LLM_invocation/     # Open-source LLM integration
│   │   ├── inference.py               
│   │   ├── launch_llm_json_server.py   
│   │   └── launch_llm_server.py        
│   ├── preprocess/                     # Data preprocessing
│   ├── prompt/                         # Prompt templates and strategies
│   ├── qa/                             # Question answering modules
│   │   └── QA_module.py              
│   ├── reranker/                       # Reranking components
│   │   ├── bge_reranker.py          
│   │   ├── crossEncoder_reranker.py   
│   │   └── LLM_reranker.py             
│   └── retriever/                      # Retrieval components
│       └── retriever.py               
```


## DocLibRAG Dataset

To bridge the gap of heterogeneous data availability, we construct the DocRAGLib dataset based on existing relevant datasets and data crawled from the web. DocRAGLib contains 2,178 documents, combining text and table content, and includes 4,468 QA pairs designed based on the content of these documents. Existing datasets are predominantly tailored for single-document RAG tasks, limiting their applicability to scenarios requiring the processing of large-scale and complex document corpora. DocRAGLib aims to address this limitation by providing a comprehensive dataset specifically designed for multi-document RAG tasks. 

The statistical analysis of the dataset is as follows:
| **Indicator**                                              | **Value**             |
| ---------------------------------------------------------- | --------------------- |
| **Total Document Number**                                  | **2,178**             |
| &nbsp;&nbsp;Avg. Word Number per Document                  | 1,453.05              |
| &nbsp;&nbsp;Avg. Table Number per Document                 | 3.85                  |
| &nbsp;&nbsp;Documents with &gt;2 Tables                     | 1,932 (88.71%)        |
| **Total QA Pairs Number**                                  | **4,468**             |
| &nbsp;&nbsp;Train/Dev/Test                                  | 2,990/502/976         |
| **Classification by Evidence Source**                      |                       |
| &nbsp;&nbsp;Text Only                                       | 403 (9.02%)           |
| &nbsp;&nbsp;Table Only                                      | 2,269 (50.78%)        |
| &nbsp;&nbsp;Both Text and Table                             | 1,796 (40.20%)        |
| **Answer Characteristics**                                 |                       |
| &nbsp;&nbsp;Decimal Places ≥3                               | 1,532 (34.29%)        |
| &nbsp;&nbsp;Numerical Values &gt;10,000                      | 865 (19.36%)          |
| &nbsp;&nbsp;Numerical Values &gt;100,000                     | 181 (4.05%)           |


## Baseline

| Method          | Reference                                                    |
| --------------- | ------------------------------------------------------------ |
| Standard RAG    | N/A                                                          |
| LangChain       | [Langchain](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb?ref=blog.langchain.dev) |
| SelfRAG         | [SelfRAG](https://github.com/AkariAsai/self-rag)             |
| Table Retrieval | [Table Retrieval](https://aclanthology.org/2024.acl-long.148/) |

## Requirements

## Running the Code

### Generate corpus

Switch to the preprocess directory.

```bash
cd src/preprocess
```

Execute the following command to generate the H-RCL corpus:

```bash
bash preprocess_RCL.sh
```

Execute the following command to generate the General RCL corpus:
```bash
bash preprocess_general.sh
```

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
