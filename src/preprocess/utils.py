import pandas as pd
import numpy as np
import tiktoken
import openai
import ast
import time
import json

from bs4 import BeautifulSoup

EMBED_TOKEN_LIMIT = 8192


def extract_left_header(table_html):
    """
    提取表格的left header
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    left_header = []

    # 遍历表格的每一行，提取第一列内容
    for row in table.find_all('tr'):
        first_cell = row.find(['th', 'td'])
        if first_cell:
            # 提取文本内容
            cell_text = first_cell.get_text(strip=True)
            if len(cell_text) != 0:
                left_header.append(cell_text)
    
    return left_header


def extract_top_header(table_html):
    """
    提取表格的top header。处理层次表头，直到行中所有单元格均不包含合并单元格为止。
    如果行中没有任何合并单元格，则保留该行并停止提取。
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    top_header = []

    # 遍历表格的每一行
    for row in table.find_all('tr'):
        row_header = []
        merge_found = False  # 标志位，用于判断是否存在合并单元格
        
        # 检查每个单元格是否包含 rowspan 或 colspan 属性
        for cell in row.find_all(['th', 'td']):
            # 提取单元格内容
            cell_text = cell.get_text(strip=True)
            if len(cell_text) != 0:
                row_header.append(cell_text)
            
            # 检查是否存在合并单元格
            if cell.has_attr('rowspan') or cell.has_attr('colspan'):
                merge_found = True
        
        # 将当前行添加到 top_header 中
        top_header.append(row_header)
        
        # 如果没有发现任何合并单元格，则停止提取
        if not merge_found:
            break

    return top_header


def extract_relational_left_header(table_html):
    """
    提取表格的左侧表头的第一层，即第一列内容。
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    left_header = []

    # 遍历表格的每一行，提取第一列内容
    for row in table.find_all('tr'):
        # 只提取第一列内容（第一个单元格）
        first_cell = row.find(['th', 'td'])
        if first_cell:
            # 提取文本内容
            cell_text = first_cell.get_text(strip=True)
            if len(cell_text) != 0:
                left_header.append(cell_text)
    
    return left_header


def extract_relational_top_header(table_html):
    """
    提取表格的最下层表头（即没有合并单元格的行）。
    一旦发现某一行中不存在合并单元格，则停止提取并返回该行。
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    top_header = []

    # 遍历表格的每一行
    for row in table.find_all('tr'):
        row_header = []
        merge_found = False  # 标志位，用于判断是否存在合并单元格
        
        # 提取当前行的所有单元格内容
        for cell in row.find_all(['th', 'td']):
            # 提取文本内容
            cell_text = cell.get_text(strip=True)
            if len(cell_text) != 0:
                row_header.append(cell_text)
            
            # 不再检测 rowspan 和 colspan，直接提取内容
            if cell.has_attr('rowspan') or cell.has_attr('colspan'):
                merge_found = True
        
        # 如果没有发现合并单元格，则停止提取并返回当前行
        if not merge_found:
            top_header = row_header
            break

    return top_header


def table_row_column_general_summary(client, table_html, table_id, retry_count=5):
    """
    根据表行列之间的关系，对表的进行 row_col 进行 summary。
    如果 ast.literal_eval(output) 出错，则重新运行该函数，最多重试 3 次。
    """
    with open("../prompt/row_col_general_summary.md", 'r', encoding="utf-8") as prompt_file:
        prompt_template = prompt_file.read()
    
    left_header = extract_relational_left_header(table_html)
    top_header = extract_relational_top_header(table_html)

    prompt = prompt_template.replace("{{Table Information}}", table_html)
    prompt = prompt_template.replace("{{Left_Header}}", str(left_header))
    prompt = prompt_template.replace("{{Top_Header}}", str(top_header))

    response = client.chat.completions.create(
        model="gpt4o-retran",  
        messages=[
            {"role": "system", "content": "You are a professional table reader, more attention on table headers."},
            {"role": "user", "content": prompt}
        ]
    )

    output = response.choices[0].message.content.strip()
    
    try:
        output = ast.literal_eval(output)
    except (SyntaxError, ValueError, openai.InternalServerError):
        if retry_count > 0:
            print("Error parsing output, retrying...")
            return table_row_column_general_summary(client, table_html, table_id, retry_count - 1)
        else:
            return []

    return output

def table_row_column_summary(client, table_html, table_id, retry_count=5):
    """
    根据表行列之间的关系，对表的进行 row_col 进行 summary。
    如果 ast.literal_eval(output) 出错，则重新运行该函数，最多重试 3 次。
    """
    with open("../prompt/row_col_summary.md", 'r', encoding="utf-8") as prompt_file:
        prompt_template = prompt_file.read()
    
    hi_left_header = extract_left_header(table_html)
    hi_top_header = extract_top_header(table_html)

    prompt = prompt_template.replace("{{Table Information}}", table_html)
    prompt = prompt_template.replace("{{Left_Header}}", hi_left_header)
    prompt = prompt_template.replace("{{Top_Header}}", hi_top_header)

    response = client.chat.completions.create(
        model="gpt4o-retran",  
        messages=[
            {"role": "system", "content": "You are a professional table reader, more attention on table headers."},
            {"role": "user", "content": prompt}
        ]
    )

    output = response.choices[0].message.content.strip()
    
    try:
        output = ast.literal_eval(output)
    except (SyntaxError, ValueError, openai.InternalServerError):
        if retry_count > 0:
            print("Error parsing output, retrying...")
            return table_row_column_summary(client, table_html, table_id, retry_count - 1)
        else:
            raise ValueError("Failed to parse output after multiple retries.")

    return output


def table_summary(client, table_html, table_id):
    """
    根据表与表之间的关系，对多个表进行共同的summary，主要关注表头
    """
    left_headers = extract_left_header(table_html)
    top_headers = extract_top_header(table_html)

    prompt = f"""
    Summarize the information in a hierarchical table named Table {table_id} with the following headers:
        Top_headers: {top_headers},
        Left_headers: {left_headers}.
    Please give me a one-sentence summary. Only output this one-sentence summary.
    """

    response = client.chat.completions.create(
        model="gpt4o-retran",  # 请替换为您实际使用的模型名称
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
    )

    structure_summary = response.choices[0].message.content.strip()

    return structure_summary


def get_embedding(text, client, max_retries=5):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) > EMBED_TOKEN_LIMIT:
        max_tokens = EMBED_TOKEN_LIMIT - 200
        tokens = tokens[-max_tokens:]
        truncated_text = encoding.decode(tokens)
    else:
        truncated_text = text
    
    retries = 0
    retry_delay = 5
    while retries < max_retries:
        try:
            print("Embedding_text:", [truncated_text])
            text_embedding = client.embeddings.create(input=[truncated_text], model="text-embedding-3-rag").data[0].embedding
            return text_embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print("Failed to generate embedding after multiple attempts.")
                
                return client.embeddings.create(input=["None. Failed to generated embedding after multiple attempts."], model="text-embedding-3-rag").data[0].embedding
    return text_embedding


# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
