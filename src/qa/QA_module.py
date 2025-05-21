import json
import openai
import re
import time
import ast  # For safe parsing of LLM response
import operator
import requests

port_map = {
    'llama-3.1-8B': 6666,
    'Mistral-Nemo': 6667,
    'Phi-3.5-mini': 6668,
    'Qwen2.5-7B': 6669,
    'Qwen2.5-72B': 6670,
    'Qwen2.5-32B': 6671,
    'Gemma-2-9b-it': 6672,
    'Yi-1.5-9B':6673,
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
qa_count=0
# Load necessary data
#dev_doc_path = "/home/TableSense/hadong/ssd2/table_rag/data/dev_documents.json"
# dev_qa_path = "/home/TableSense/hadong/ssd2/table_rag/data/new_dev_qa.json"   #new_dev_qa_no_million3
#dev_qa_path = '/home/TableSense/hadong/ssd2/table_rag/data/new_dev_qa_all_num.json'   #现在仅保留答案为数字的问题

class QA_module:
    def __init__(self, dev_doc_path, dev_qa_path, use_calculator, model_RECAP_qa, client):
        """
        初始化RECAP_QA类，放一下dev_doc_path（数据集文档部分）和dev_qa_path（数据集的问题部分）。
        """
        self.dev_doc_path=dev_doc_path
        self.dev_qa_path=dev_qa_path
        self.client = client
        self.model_RECAP_qa=model_RECAP_qa               #QA阶段使用的LLM
        self.use_calculator=use_calculator
        
    def call_oneshot_RECAP_prompt(self, information1,information2,question, model_RECAP_qa):
        info1,info2,info3=information1,information2,question
        with open("prompt/qa_RECAP.md", 'r', encoding="utf-8") as prompt_file:
        # with open("prompt/qa_CoT.md", 'r', encoding="utf-8") as prompt_file:      #qa_RECAP   qa_our_prompt
            prompt_template = prompt_file.read()
        oneshot_RECAP_prompt = prompt_template.replace("{{info1}}", str(info1)).replace("{{info2}}", str(info2)).replace("{{info3}}", str(info3))
        # - retries: 重试次数
        # - delay: 每次重试之间的延迟时间（秒）
        retries=5
        delay=10
        
        if self.model_RECAP_qa in ["gpt4o-retran","gpt-35-turbo-rag","gpt4-0613","gpt-4o-mini"]:
            # 初始化重试计数
            for attempt in range(retries):
                try:
                    # 发送API请求
                    response = self.client.chat.completions.create(
                        model= model_RECAP_qa,  # model = "deployment_name"
                        messages=[
                            {"role": "system", "content": "You are a document analysis AI designed to extract and synthesize information from documents to answer specific inquiries."},
                            {"role": "user", "content": oneshot_RECAP_prompt}
                        ]
                    )
                    # 处理响应
                    full_answer = response.choices[0].message.content
                    matches1 = re.findall(r'<(.*?)>', full_answer)
                    matches2 = re.findall(r'##(.*?)##', full_answer)
                    res1 = matches1[-1] if matches1 else ""
                    res2 = matches2[-1] if matches2 else ""

                    # 计算公式结果
                    if res2:
                        result = self.safe_eval(res2)
                        formula_cal_res = round(result, 5) if isinstance(result, (int, float)) else ""
                    else:
                        formula_cal_res = ""

                    final_ans = res1

                    # 更新计数并返回结果
                    global qa_count
                    qa_count += 1
                    print(qa_count)  # 监控函数运行次数
                    return full_answer, final_ans, formula_cal_res, res2

                except openai.RateLimitError:
                    # 处理速率限制错误
                    if attempt < retries - 1:  # 如果还有重试机会
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        time.sleep(delay)  # 等待指定时间后重试 
                        delay *= 2  # 可选择延长延迟时间（指数回退）
                    else:
                        print("Rate limit exceeded. Maximum retries reached.")
                        raise  # 超出重试次数后重新抛出异常                  
        else:
            answer=get_llm_response(self.model_RECAP_qa, oneshot_RECAP_prompt).strip()
            # 处理响应
            full_answer = answer
            matches1 = re.findall(r'<(.*?)>', full_answer)
            matches2 = re.findall(r'##(.*?)##', full_answer)
            res1 = matches1[-1] if matches1 else ""
            res2 = matches2[-1] if matches2 else ""

            # 计算公式结果
            if res2:
                result = self.safe_eval(res2)
                formula_cal_res = round(result, 5) if isinstance(result, (int, float)) else ""
            else:
                formula_cal_res = ""

            final_ans = res1

            # 更新计数并返回结果
            qa_count += 1
            print(qa_count)  # 监控函数运行次数
            return full_answer, final_ans, formula_cal_res, res2

    def safe_eval(self, expr):
        """
        Safely evaluate a mathematical expression provided as a string.
        Only supports basic arithmetic operations.
        """
        # Define the allowed operators
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def eval_node(node):
            if isinstance(node, ast.Num):  # For Python < 3.8
                return node.n
            elif isinstance(node, ast.Constant):  # For Python 3.8+
                if isinstance(node.value, (int, float)):
                    return node.value
                else:
                    raise TypeError(f"Unsupported constant type: {type(node.value)}")
            elif isinstance(node, ast.BinOp):  # Binary operations
                if type(node.op) in allowed_operators:
                    left = eval_node(node.left)
                    right = eval_node(node.right)
                    return allowed_operators[type(node.op)](left, right)
                else:
                    raise ValueError(f"Unsupported operator: {type(node.op)}")
            elif isinstance(node, ast.UnaryOp):  # Unary operations
                if type(node.op) in allowed_operators:
                    operand = eval_node(node.operand)
                    return allowed_operators[type(node.op)](operand)
                else:
                    raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            else:
                raise TypeError(f"Unsupported expression: {type(node)}")

        try:
            # Parse the expression
            node = ast.parse(expr, mode='eval').body
            # Evaluate the parsed expression
            result = eval_node(node)
            return result
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return ""
        
    # Filter the paragraphs and tables as per evidence
    def filter_paragraphs(self, paragraphs, text_evidence_info):
        filtered_paragraphs = [paragraphs[i] for i in text_evidence_info if 0 <= i < len(paragraphs)]
        return filtered_paragraphs, len(filtered_paragraphs)


    def filter_tables(self, tables, table_evidence_info):
        unique_table_indices = set()
        for info in table_evidence_info:
            parts = info.split('-')
            if len(parts) == 3:
                try:
                    table_index = int(parts[0])
                    if 0 <= table_index < len(tables):
                        unique_table_indices.add(table_index)
                except ValueError:
                    continue
        filtered_tables = [tables[i] for i in sorted(unique_table_indices)]
        return filtered_tables, len(filtered_tables)


    # Function to load the dev_documents.json and new_dev_qa.json data
    def load_documents_and_qa(self):
        with open(self.dev_doc_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        with open(self.dev_qa_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        return documents, qa_data

    def find_document_by_uid(documents, uid):
        for doc in documents:
            if doc.get('uid') == uid:
                return doc
        return None
        
    # Main logic to perform the QA
    def process_qa_input(self, qa_input, documents, qa_data , model_RECAP_qa):
        current_data = []
        for item in qa_input:
            question = item['question']
            related_uid = item['related_top1'][0]  # Assuming there's only one related UID per question

            # Match the question in qa_data
            matching_qa_entry = next((entry for entry in qa_data if entry['question'] == question), None)

            if matching_qa_entry is None:
                print(f"No matching entry found for question: {question}")
                continue

            # Now retrieve the corresponding document using the related_uid
            document = next((doc for doc in documents if doc['uid'] == related_uid), None)

            if document is None:
                print(f"No document found for UID: {related_uid}")
                compare = {
                    "question": question,
                    "direct_answer": "No document found",
                    "formula_ans":"",
                    "formula": "",
                    "predict_answer": "",
                    "true_answer": matching_qa_entry['answer']
                }
                
            else:
                # Extract paragraphs and tables from the matched document
                paragraphs = document.get('paragraphs', [])
                tables = document.get('tables', [])

                # Proceed to call the QA function using the retrieved information
                full_answer, direct_ans, formula_ans, formula = self.call_oneshot_RECAP_prompt(paragraphs, tables, question, model_RECAP_qa)
                
                # Process and save the results
                if self.use_calculator== True:
                    compare = {
                        "question": question,
                        "direct_answer": direct_ans,
                        "formula_ans": formula_ans,
                        "formula": formula,
                        "predict_answer": formula_ans if formula_ans else direct_ans,
                        "true_answer": matching_qa_entry['answer']
                    }
                else:
                    # 如果没有calculator，则使用下面的
                    compare = {
                        "question": question,
                        "direct_answer": direct_ans,
                        "formula_ans": formula_ans,
                        "formula": formula,
                        "predict_answer": direct_ans,
                        "true_answer": matching_qa_entry['answer']
                    }     

                # Append new result
            current_data.append(compare)

            print(f"Processed question: {question}")
                    # Read the current file and append the new result
        # with open(stage_3_output_path, "w") as f:
        #     json.dump(current_data, f, indent=4)
        return current_data
            
            


    # Function to execute the QA process
    def run_QA(self,qa_input):
        documents, qa_data = self.load_documents_and_qa()

        qa_input = qa_input
        
        stage_3_output=self.process_qa_input(qa_input, documents, qa_data, self.model_RECAP_qa)
        return stage_3_output