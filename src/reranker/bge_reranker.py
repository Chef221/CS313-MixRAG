import json
import openai
import re
import time
from tqdm import tqdm
import ast  # For safe parsing of LLM response
from sentence_transformers import SentenceTransformer, CrossEncoder, util

def get_bge_embedding(text, model):
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding

class bge_reranker:
    def __init__(self, output_list2,stage_2_output_path,qa_dataset,client):
        self.output_list2 = output_list2
        self.qa_dataset = qa_dataset
        self.stage_2_output_path = stage_2_output_path
        self.client=client
        
    def run_bge_reranker(self):
        bge_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        bge_model.max_seq_length = 512

        qa_input = []
        for idx, entry in enumerate(tqdm(self.output_list2, desc="Processing BGE Reranker for top 1 document")):
            question = entry['question']
            related_docs = entry['related_doc']

            # Compute BGE embedding for the question
            question_embedding = get_bge_embedding(question, bge_model)

            doc_embeddings = []
            doc_uids = []

            for doc in related_docs:
                uid = doc['document_uid']
                related_doc_detail = doc['related_doc_detail']
                # Concatenate the chunks
                doc_content = ' '.join(related_doc_detail)
                # Compute embedding for the document content
                doc_embedding = get_bge_embedding(doc_content, bge_model)
                doc_embeddings.append(doc_embedding)
                doc_uids.append(uid)

            # Compute cosine similarities between question and documents
            similarities = util.dot_score(question_embedding, doc_embeddings)[0].cpu().tolist()

            # Pair UIDs with similarities
            uid_similarity_pairs = list(zip(doc_uids, similarities))

            # Sort the documents based on similarity scores
            uid_similarity_pairs.sort(key=lambda x: x[1], reverse=True)

            # Select the top 1 document
            related_top1 = [uid_similarity_pairs[0][0]]

            qa_input.append({
                'question': question,
                'related_top1': related_top1
            })

        with open(self.stage_2_output_path, "w") as f:
            json.dump(qa_input, f, indent=4)

        # Step 3: Compute the hit rate
        count_top1 = 0
        for idx, entry in enumerate(qa_input):
            question = entry['question']
            related_top1 = entry['related_top1']
            document_id = self.qa_dataset[idx]['document_uid']
            if document_id in related_top1:
                count_top1 += 1

        hit_rate_top1 = count_top1 / len(qa_input)
        print(f"Top 1 Hit Rate: {count_top1}/{len(qa_input)} = {hit_rate_top1:.5f}")