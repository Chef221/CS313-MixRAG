import pandas as pd

from rank_bm25 import BM25Okapi
from .retriever_utils import *
from nltk.tokenize import word_tokenize


class Retriever:
    def __init__(self, documents, df_embedding, df_bm25, client):
        """
        初始化Retriever类，可以选择传入预先计算好的bm25索引。
        """
        self.documents = documents
        self.df_embedding = df_embedding
        self.df_bm25 = df_bm25

        self.bm25 = BM25Okapi(self.df_bm25['tokenized_content'].tolist())

        self.uid_mapping_doc = {}
        for doc in documents:
            self.uid_mapping_doc[doc["uid"]] = doc
        
        self.client = client


    def embedding_retriever(self, questions, top_n_embedding):
        """
            Here param: questions is a list of subquestions.
        """
        all_search_doc_embedding = []
        all_search_chunks_embedding = []
        for sub_question in questions:
            df_search_chunks_embedding, df_search_doc_embedding = self._search_docs_by_embedding(
                query=sub_question,
                top_n_embedding=top_n_embedding
            )
            all_search_doc_embedding.append(df_search_doc_embedding)
            all_search_chunks_embedding.append(df_search_chunks_embedding)
        
        if len(all_search_doc_embedding) > 1:
            df_search_doc_embedding = pd.concat(all_search_doc_embedding, ignore_index=True)
            df_search_doc_embedding = df_search_doc_embedding.sort_values("similarities", ascending=False)
            df_search_doc_embedding = df_search_doc_embedding.groupby('document_uid', group_keys=False).apply(adjust_similarity)
            df_search_doc_embedding = df_search_doc_embedding.sort_values("similarities", ascending=False).head(top_n_embedding)
        else:
            df_search_doc_embedding = all_search_doc_embedding[0]
        
        if len(all_search_chunks_embedding) > 1:
            df_search_chunks_embedding = pd.concat(all_search_chunks_embedding, ignore_index=True)
        else:
            df_search_chunks_embedding = all_search_chunks_embedding[0]

        return df_search_doc_embedding, df_search_chunks_embedding


    def _search_docs_by_embedding(self, query, top_n_embedding):
        query_embedding = get_embedding(query, self.client)
        df = self.df_embedding
        
        df['similarities'] = df.ada_v2.apply(lambda x: cosine_similarity(x, query_embedding))
    
        df_sorted = df.sort_values("similarities", ascending=False).head(top_n_embedding * 3)
        df_sorted_chunks = df.sort_values("similarities", ascending=False).head(top_n_embedding * 3)
        df_adjusted = df_sorted.groupby('document_uid', group_keys=False).apply(adjust_similarity)
        res = df_adjusted.sort_values("similarities", ascending=False).head(top_n_embedding)
        
        # Return both the top_n*3 chunks and top_n documents
        return df_sorted_chunks, res


    def bm25_retriever(self, questions, top_n_bm25):
        all_search_doc_bm25 = []
        all_search_chunks_bm25 = []

        for sub_question in questions:
            df_search_chunks_bm25, df_search_doc_bm25 = self._search_docs_bm25(
                query=sub_question,
                top_n_bm25=top_n_bm25,
            )
            all_search_doc_bm25.append(df_search_doc_bm25)
            all_search_chunks_bm25.append(df_search_chunks_bm25)
        
        if len(all_search_doc_bm25) > 1:
            df_search_doc_bm25 = pd.concat(all_search_doc_bm25, ignore_index=True)
            df_search_doc_bm25 = df_search_doc_bm25.sort_values("bm25_scores", ascending=False)
            df_search_doc_bm25 = df_search_doc_bm25.groupby('document_uid', group_keys=False).apply(adjust_similarity_bm25)
            df_search_doc_bm25 = df_search_doc_bm25.sort_values("bm25_scores", ascending=False).head(top_n_bm25)
        else:
            df_search_doc_bm25 = all_search_doc_bm25[0]

        if len(all_search_chunks_bm25) > 1:
            df_search_chunks_bm25 = pd.concat(all_search_chunks_bm25, ignore_index=True)
        else:
            df_search_chunks_bm25 = all_search_chunks_bm25[0]
        
        return df_search_doc_bm25, df_search_chunks_bm25
    

    def _search_docs_bm25(self, query, top_n_bm25):
        tokenized_query = word_tokenize(query.lower())

        doc_scores = self.bm25.get_scores(tokenized_query)

        df = self.df_bm25

        df['bm25_scores'] = doc_scores
        df_sorted = df.sort_values("bm25_scores", ascending=False).head(top_n_bm25 * 3)
        df_sorted_chunks = df.sort_values("bm25_scores", ascending=False).head(top_n_bm25 * 3)
        df_adjusted = df_sorted.groupby('document_uid', group_keys=False).apply(adjust_similarity_bm25)
        res = df_adjusted.sort_values("bm25_scores", ascending=False).head(top_n_bm25)

        return df_sorted_chunks, res

    
    def ensembel_docs(self, related_document_uids_embedding, related_document_uids_bm25):
        if(len(related_document_uids_bm25)==0):
            related_document_uids = related_document_uids_embedding
            seen = set()
            related_document_uids = [x for x in related_document_uids if not (x in seen or seen.add(x))]
            related_document_num=len(related_document_uids)   #去重后的top N的数目（e.g. top 100-> top 90）
            print(f"Related documents: {related_document_uids}")

            retrieval_documents = [self.uid_mapping_doc[uid] for uid in related_document_uids]
            
            return related_document_uids, retrieval_documents, related_document_num
        else:
            related_document_uids = related_document_uids_embedding + related_document_uids_bm25
            seen = set()
            related_document_uids = [x for x in related_document_uids if not (x in seen or seen.add(x))]
            related_document_num=len(related_document_uids)   #去重后的top N的数目（e.g. top 100-> top 90）
            print(f"Related documents: {related_document_uids}")

            retrieval_documents = [self.uid_mapping_doc[uid] for uid in related_document_uids]
            
            return related_document_uids, retrieval_documents, related_document_num
        