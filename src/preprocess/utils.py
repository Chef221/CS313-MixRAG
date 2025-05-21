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
    """Extract the left (first column) header from an HTML table."""
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    left_header = []

    for row in table.find_all('tr'):
        first_cell = row.find(['th', 'td'])
        if first_cell:
            cell_text = first_cell.get_text(strip=True)
            if len(cell_text) != 0:
                left_header.append(cell_text)
    
    return left_header


def extract_top_header(table_html):
    """Extract the top header rows until reaching a row without merged cells."""
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    top_header = []
    for row in table.find_all('tr'):
        row_header = []
        merge_found = False
        for cell in row.find_all(['th', 'td']):
            cell_text = cell.get_text(strip=True)
            if cell_text:
                row_header.append(cell_text)
            if cell.has_attr('rowspan') or cell.has_attr('colspan'):
                merge_found = True
        top_header.append(row_header)
        if not merge_found:
            break
    return top_header

def extract_relational_left_header(table_html):
    """Extract only the first column from each row as the relational left header."""
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    left_header = []
    for row in table.find_all('tr'):
        first_cell = row.find(['th', 'td'])
        if first_cell:
            cell_text = first_cell.get_text(strip=True)
            if cell_text:
                left_header.append(cell_text)
    return left_header


def extract_relational_top_header(table_html):
    """Extract the lowest-level header row (no merged cells)."""
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    for row in table.find_all('tr'):
        row_header = []
        merge_found = False
        for cell in row.find_all(['th', 'td']):
            cell_text = cell.get_text(strip=True)
            if cell_text:
                row_header.append(cell_text)
            if cell.has_attr('rowspan') or cell.has_attr('colspan'):
                merge_found = True
        if not merge_found:
            return row_header
    return []


def table_row_column_general_summary(client, table_html, table_id, retry_count=5):
    """Generate row-column summaries for general tables using LLM."""
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
    """Generate row-column summaries for hierarchical tables using LLM."""
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
    """Generate a single-sentence summary for a table using its headers."""
    left_headers = extract_left_header(table_html)
    top_headers = extract_top_header(table_html)

    prompt = f"""
    Summarize the information in a hierarchical table named Table {table_id} with the following headers:
        Top_headers: {top_headers},
        Left_headers: {left_headers}.
    Please give me a one-sentence summary. Only output this one-sentence summary.
    """

    response = client.chat.completions.create(
        model="gpt4o-retran",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
    )

    structure_summary = response.choices[0].message.content.strip()

    return structure_summary


def get_embedding(text, client, max_retries=5):
    """Generate text embedding using OpenAI embedding API with fallback and truncation."""
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

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
