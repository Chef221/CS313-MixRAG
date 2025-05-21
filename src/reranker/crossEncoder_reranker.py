import json
import openai
import re
import time
from tqdm import tqdm
import ast  # For safe parsing of LLM response
from sentence_transformers import SentenceTransformer, CrossEncoder, util

class crossEncoder_reranker:
    def __init__(self, output_list2,stage_2_output_path,qa_dataset,client):
        self.output_list2 = output_list2
        self.qa_dataset = qa_dataset
        self.stage_2_output_path = stage_2_output_path
        self.client=client
        
    def run_crossEncoder_reranker(self):
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        qa_input = []
        for idx, entry in enumerate(tqdm(self.output_list2, desc="Processing Cross-Encoder Reranker for top 1 document")):
            question = entry['question']
            related_docs = entry['related_doc']

            pairs = []
            doc_uids = []

            for doc in related_docs:
                uid = doc['document_uid']
                related_doc_detail = doc['related_doc_detail']
                # Concatenate the chunks
                doc_content = ' '.join(related_doc_detail)
                pairs.append((question, doc_content))
                doc_uids.append(uid)

            # Compute relevance scores using Cross-Encoder
            scores = cross_encoder_model.predict(pairs)

            # Pair UIDs with scores
            uid_score_pairs = list(zip(doc_uids, scores))

            # Sort the documents based on scores
            uid_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Select the top 1 document
            related_top1 = [uid_score_pairs[0][0],uid_score_pairs[1][0],uid_score_pairs[2][0],uid_score_pairs[3][0],uid_score_pairs[4][0]]
            # related_top1 = [uid_score_pairs[0][0]]
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