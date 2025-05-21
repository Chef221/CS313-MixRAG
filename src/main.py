from config import CONFIG
from utils import load_data, load_corpus, filter
from openai import AzureOpenAI
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from rank_bm25 import BM25Okapi
from retriever import Retriever
from reranker import LLM_reranker,crossEncoder_reranker
from evaluate import evaluation_prediction_result
from tqdm import tqdm
from qa import QA_module
import argparse
import json
import pandas as pd



token_provider = get_bearer_token_provider(
    ManagedIdentityCredential(client_id="f6a245af-3e86-4fc8-9470-127f5024e475"), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version="2024-05-01-preview",
    azure_endpoint="https://excelllmswedencentral.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="QA and document retrieval script")
    
    # Define command-line arguments
    parser.add_argument('--documents_path', type=str, help='Path to dev documents file')
    parser.add_argument('--qa_path', type=str, help='Path to dev QA file')
    parser.add_argument('--output_dir', type=str, help='Directory to use_calculatorsave the output files')
    parser.add_argument('--embedding_corpus_path', type=str, help='Name of the embedding model to use')
    parser.add_argument('--bm25_corpus_path', type=str, help='Path to the precomputed BM25 corpus')
    parser.add_argument('--top_n_embedding', type=int, help='Number of top documents to retrieve using embedding')
    parser.add_argument('--top_n_bm25', type=int, help='Number of top documents to retrieve using BM25')
    parser.add_argument('--model_RECAP_qa', type=str, help='Model for RECAP QA', default="gpt4o-retran")
    parser.add_argument('--model_LLM_reranker', type=str, help='Model for LLM reranker', default="gpt4o-retran")
    parser.add_argument('--debugger_mode', type=bool, help='Enable debugger mode', default=True)
    parser.add_argument('--use_calculator', type=bool, help='Use external calculator or not', default=True)
    # 这里需要再添加一下parser

    # Parse the arguments
    args = parser.parse_args()
    
     # Update the configuration based on the arguments, if provided
    if args.documents_path:
        CONFIG['documents_path'] = args.documents_path
    if args.qa_path:
        CONFIG['qa_path'] = args.qa_path
    if args.output_dir:
        CONFIG['output_dir'] = args.output_dir
    if args.embedding_corpus_path:
        CONFIG['embedding_corpus_path'] = args.embedding_corpus_path
    if args.bm25_corpus_path:
        CONFIG['bm25_corpus_path'] = args.bm25_corpus_path
    if args.top_n_embedding:
        CONFIG['top_n_embedding'] = args.top_n_embedding
    if args.top_n_bm25:
        CONFIG['top_n_bm25'] = args.top_n_bm25
    if args.model_RECAP_qa:
        CONFIG['model_RECAP_qa'] = args.model_RECAP_qa
    if args.model_LLM_reranker:
        CONFIG['model_LLM_reranker'] = args.model_LLM_reranker
    if args.debugger_mode is not None:
        CONFIG['debugger_mode'] = args.debugger_mode
    if args.debugger_mode is not None:
        CONFIG['use_calculator'] = args.use_calculator
    
    return CONFIG


    # 'llama-3.1-8B': 6666,
    # 'Mistral-Nemo': 6667,
    # 'Phi-3.5-mini': 6668,
    # 'Qwen2.5-7B': 6669,
    # 'Qwen2.5-72B': 6670,
    # 'Qwen2.5-32B': 6671,
    # 'Gemma-2-9b-it': 6672,
    # 'Yi-1.5-9B':6673 ,
    
    
def main():
    config = parse_arguments()
    # Here should be added into config
    model_RECAP_qa="gpt4o-retran"
    model_LLM_reranker="gpt4o-retran"
    stage_1_output_path="intermidiate_output_file/stage1_output.json" #对应output_list的内容
    stage_1_new_output_path="intermidiate_output_file/stage1_output_less2000.json"  #对应output_list2的内容
    stage_2_output_path="ex_0115/stage2_output.json"  #问题和Top 1 Doc的uid一一对应
    stage_3_output_evaluate_path="ex_0115/stage3_evaluate_output.json"  #对RECAP回答进行评估后的结果
    
    stage_3_output_path=CONFIG['output_dir']  #RECAP回答的结果
    top_n_embedding = CONFIG['top_n_embedding']
    top_n_bm25 = CONFIG['top_n_bm25']
    debugger_mode = CONFIG['debugger_mode']

    print("Configuration loaded:")
    print(json.dumps(config, indent=4)) 

    documents, qa_dataset = load_data(config['documents_path'], config['qa_path'])
    df_embedding_doc, df_bm25_doc = load_corpus(config["embedding_corpus_path"]), load_corpus(config["bm25_corpus_path"])

    # our_retriever = Retriever(
    #     documents=documents,
    #     df_embedding=df_embedding_doc,
    #     df_bm25=df_bm25_doc,
    #     client=client
    # )
    
    # count = 0
    # middle_output_file = []
    # output_list = []  # Stores documents with top 3N chunks
    # all_questions_related_document_num_list=[]
    
    # # total_question_num=100
    total_question_num=len(qa_dataset)
    
    # #TODO：当仅用embedding进行检索时
    # # 由于每个文档用ensemble取的top n文档的数目都是不确定的，因此需要读取all_questions_related_document_num_list文件，获取每个文档用ensemble方法取的top n，这是一个一维列表[90,88,91,...]
    # # with open("intermidiate_output_file/all_questions_related_document_num_list.json", "w") as f:
    # #     all_questions_related_document_num_list=json.load(f)
    # for i, entry in enumerate(tqdm(qa_dataset[0:total_question_num], desc="Processing QA...")):
    #     question = entry.get('question', '')
    #     document_id = entry.get('document_uid', '')
    #     # print(f"Processing question {i+1}: {question}")

    #     sub_questions = [question]

    #     df_search_doc_embedding, df_search_chunks_embedding = our_retriever.embedding_retriever(sub_questions, top_n_embedding=top_n_embedding)
    #     # df_search_doc_embedding, df_search_chunks_embedding = our_retriever.embedding_retriever(sub_questions, top_n_embedding=all_questions_related_document_num_list[i])
        
    #     df_search_doc_bm25, df_search_chunks_bm25 = our_retriever.bm25_retriever(sub_questions, top_n_bm25=top_n_bm25)

    #     related_document_uids_embedding = df_search_doc_embedding["document_uid"].tolist()
    #     related_document_uids_bm25 = df_search_doc_bm25["document_uid"].tolist()

    #     related_document_uids, retrieval_documents,related_document_num = our_retriever.ensembel_docs(related_document_uids_embedding, related_document_uids_bm25)
    #     all_questions_related_document_num_list.append(related_document_num)   #用ensemble检索的top n文档的数目
    #     with open("intermidiate_output_file/all_questions_related_document_num_list.json", "w") as f:
    #         json.dump(all_questions_related_document_num_list, f, indent=4)
    
    #     for re_doc in retrieval_documents:
    #         temp = {
    #             "uid": re_doc["uid"],
    #             "paragraphs": re_doc["paragraphs"],
    #             # "table_description": re_doc["table_description"],
    #             "tables": re_doc["tables"],
    #             "qa": entry
    #         }
    #         middle_output_file.append(temp)

    #     # -------------------------
    #     # Collect the chunks from both methods
    #     df_all_chunks = pd.concat([df_search_chunks_embedding, df_search_chunks_bm25], ignore_index=True)
    #     df_all_chunks = df_all_chunks.drop_duplicates(subset=['document_uid', 'content'])

    #     # For each document UID, collect the chunks
    #     related_documents = []
        
    #     for uid in related_document_uids:
    #         each_related_documents = {}
    #         # Get all chunks for this uid from df_all_chunks
    #         chunks = df_all_chunks[df_all_chunks['document_uid'] == uid]['content'].tolist()
    #         each_related_documents['document_uid'] = uid
    #         each_related_documents['related_doc_detail'] = chunks
    #         related_documents.append(each_related_documents)
        
    #     question_output = {
    #         "question": question,
    #         "related_doc": related_documents
    #     }
    #     output_list.append(question_output)

    #     if document_id in related_document_uids:
    #         count += 1
    #         # print("Document found in retrieved documents.")
    #     # else:
    #     #     print("Document not found in retrieved documents.")
    
    # hit_rate1= count / total_question_num
    # print(f"Matched: {count}/{total_question_num} = {hit_rate1:.5f}")

    # # Step 1: Write the original output file if debugger_mode is True
    # if debugger_mode:
    #     with open(stage_1_output_path, "w") as f:
    #         json.dump(output_list, f, indent=4)

    # # with open(stage_1_output_path, 'r', encoding='utf-8') as f:
    # #      output_list = json.load(f)
    # # Filter 'related_doc_detail'
    # filter_doc_list = filter(output_list)
    # if debugger_mode:
    #     with open(stage_1_new_output_path, "w") as f:
    #         json.dump(filter_doc_list, f, indent=4)
    with open(stage_1_new_output_path, 'r', encoding='utf-8') as f:
        filter_doc_list = json.load(f)

    #利用LLM进行top 1的选择
    rerank_stage=LLM_reranker(filter_doc_list,model_LLM_reranker,qa_dataset,total_question_num,client)
    #通过run_llm_rerank Compute the top 1 hit rate，
    qa_input = rerank_stage.run_llm_rerank() #处理后的问题与top 1 Doc uid的结果存到stage_2_output_path
    # rerank_stage=crossEncoder_reranker(output_list2,stage_2_output_path,qa_dataset,client)
    # rerank_stage.run_crossEncoder_reranker() 
    rerank_stage.hit_rate_evaluate(qa_input)
    if debugger_mode:
        with open(stage_2_output_path, "w") as f:
            json.dump(qa_input, f, indent=4)

    # QA
    QA_stage=QA_module(config['documents_path'], config['qa_path'], config['use_calculator'],model_RECAP_qa, client)
    stage3_output=QA_stage.run_QA(qa_input)   #这一步骤可以得到RECAP输出的结果
    if debugger_mode:
        with open(stage_3_output_path, "w") as f:
            json.dump(stage3_output, f, indent=4)
    
    # Evaluate
    pred_ans = []
    gold_ans = []
    # global true_or_flase_list
    # true_or_flase_list=[]
    
    # 读取 RECAP 的输出 文件
    with open(stage_3_output_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # 遍历每个字典，提取 predict_answer 和 true_answer
        for item in data:
            if 'predict_answer' in item:
                pred_ans.append(str(item['predict_answer']))
            if 'true_answer' in item:
                gold_ans.append(str(item['true_answer']))
    EM, f1, true_or_flase=evaluation_prediction_result(pred_ans,gold_ans)
    count_true=0
    for i in true_or_flase:
        if i=="True":
            count_true+=1
    final_EM = count_true / total_question_num
    print("回答正确的问题数量："+str(count_true))
    print("总问题的数量："+str(total_question_num))
    print("Exact Match Score:"+str(final_EM))
    
    # 1. 读取原始 JSON 文件
    with open(stage_3_output_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    if len(data2) != len(true_or_flase):
        raise ValueError("JSON 文件的列表长度与 true_or_false 列表的长度不一致。")
    # 4. 遍历并添加 'judge' 属性
    for item, judge_value in zip(data2, true_or_flase):
        item['judge'] = judge_value
    # 5. 将结果保存到新的 JSON 文件
    with open(stage_3_output_evaluate_path, 'w', encoding='utf-8') as f:
        json.dump(data2, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
