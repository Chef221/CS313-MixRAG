import json
import openai
import re
import time
from tqdm import tqdm
import ast  # For safe parsing of LLM response
import requests

port_map = {
    'llama-3.1-8B': 6666,
    'Mistral-Nemo': 6667,
    'Phi-3.5-mini': 6668,
    'Qwen2.5-7B': 6669,
    'Qwen2.5-72B': 6670,
    'Qwen2.5-32B': 6671,
    'Gemma-2-9b-it': 6672,
    'Yi-1.5-9B':6673
}

def get_llm_response(model, prompt, temperature=0):
    port = port_map[model]
    url = f"http://localhost:{port}/generate"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "temperature": temperature,
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result.get('response', 'No response key in JSON')
    else:
        return f"Error: {response.status_code}, {response.text}"

class LLM_reranker:
    def __init__(self,output_list2,model_LLM_reranker, qa_dataset,total_question_num,client):
        """
        初始化RECAP_QA类，放一下dev_doc_path（数据集文档部分）和dev_qa_path（数据集的问题部分）。
        """
        self.output_list2 = output_list2   #第一阶段的带的
        self.model_LLM_reranker = model_LLM_reranker   #reranker用的LLM模型
        self.qa_dataset = qa_dataset   #数据集的QA部分（数据集分为QA部分和Document部分）
        self.total_question_num = total_question_num
        self.client=client
        
    def run_llm_rerank(self):
        qa_input = []    #qa_input即本阶段处理后的问题与top 1 Doc uid的结果，作为下阶段QA的输入
        for idx, entry in enumerate(tqdm(self.output_list2, desc="Processing LLM for top 1 documents")):
            question = entry['question']
            related_docs = entry['related_doc']

            with open("/home/TableSense/hadong/ssd2/table_rag/code/prompt/LLM_rerank_to_top1.md", 'r', encoding="utf-8") as prompt_file:
                prompt_template = prompt_file.read()
            prompt_top1 = prompt_template.replace("{{question}}", question)
            prompt_top1 +=f"\n"
            for doc in related_docs:
                if len(doc['related_doc_detail']) > 0:   #先把只有paragraphs chunk的doc过滤掉
                    uid = doc['document_uid']
                    related_doc_detail = doc['related_doc_detail']
                    # Concatenate the chunks
                    doc_content = ' '.join(related_doc_detail)
                    # Include the uid and the content in the prompt
                    prompt_top1 += f"Document UID: {uid}\nContent: {doc_content}\n\n"

            # - retries: 重试次数
            # - delay: 每次重试之间的延迟时间（秒）
            if self.model_LLM_reranker in ["gpt4o-retran","gpt-35-turbo-rag","gpt4-0613","gpt-4o-mini"]:
                retries=5
                delay=5
                # 初始化重试计数
                for attempt in range(retries):
                    try:
                        # 发送API请求
                        response = self.client.chat.completions.create(
                            model= self.model_LLM_reranker,  # model = "deployment_name" gpt35-turbo-16k-0613  gpt4o-retran"   gpt-35-turbo-rag
                            messages=[
                                {"role": "system", "content": "You are a professional document analyst tasked with identifying the most relevant document from a set."},
                                {"role": "user", "content": prompt_top1}
                            ]
                        )
                        # 处理响应
                        answer = response.choices[0].message.content.strip()
                        break
                    except openai.RateLimitError:
                        # 处理速率限制错误
                        if attempt < retries - 1:  # 如果还有重试机会
                            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                            time.sleep(delay)  # 等待指定时间后重试
                            delay *= 2         # 延长延迟时间（指数回退）
                        else:
                            print("Rate limit exceeded. Maximum retries reached.")
                            raise 
            #使用开源LLM('llama': 6666,'mistral': 6667, 'phi': 6668,)
            else:
                answer=get_llm_response(self.model_LLM_reranker, prompt_top1).strip()                
            print("####answer:")
            print(answer)
            # Parse the response to extract the list of UIDs
            try:
                related_top1 = ast.literal_eval(answer)
                if isinstance(related_top1, list):
                    # Ensure all items are strings
                    related_top1 = [str(uid) for uid in related_top1]
                    if (len(related_top1) == 0) :
                        related_top1 = [doc['document_uid'] for doc in related_docs[:1]]
                    # if len(related_top1) > 10:
                    #     related_top1=related_top1[:10]
                else:
                    print(f"Unexpected response format for question: {question}. Using first 5 UIDs.")
                    # Use the first UID from related_docs
                    related_top1 = [doc['document_uid'] for doc in related_docs[:1]]
            except:
                print(f"Failed to parse LLM response for question: {question}. Using first 5 UIDs.")
                # Use the first UID from related_docs 如果答案解析错误就用相关文档的第一个作为top1
                related_top1 = [doc['document_uid'] for doc in related_docs[:1]]

            qa_input.append({
                'question': question,
                'related_top1': related_top1
            })

        return qa_input

        
    def hit_rate_evaluate(self, qa_input):
        count_top1 = 0
        for idx, entry in enumerate(qa_input):
            question = entry['question']
            related_top1 = entry['related_top1']
            document_id = self.qa_dataset[idx]['document_uid']
            if document_id in related_top1:
                count_top1 += 1

        hit_rate_top1 = count_top1 / self.total_question_num
        print(f"Top 1 Hit Rate: {count_top1}/{self.total_question_num} = {hit_rate_top1:.5f}")

        

