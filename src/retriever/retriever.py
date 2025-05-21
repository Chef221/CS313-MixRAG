from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import pandas as pd
from utils import get_embedding, cosine_similarity, adjust_similarity, adjust_similarity_bm25  # 假设这些函数在 utils 中定义

class Retriever:
    def __init__(self, documents, df_embedding, df_bm25, client):
        """Initialize Retriever with documents, embeddings, and BM25 index."""
        self.documents = documents
        self.df_embedding = df_embedding
        self.df_bm25 = df_bm25
        self.bm25 = BM25Okapi(self.df_bm25['tokenized_content'].tolist())
        self.uid_mapping_doc = {doc["uid"]: doc for doc in documents}
        self.client = client

    def embedding_retriever(self, questions, top_n_embedding):
        """Retrieve relevant documents and chunks using embedding similarity."""
        all_doc, all_chunk = [], []
        for q in questions:
            chunk, doc = self._search_docs_by_embedding(q, top_n_embedding)
            all_doc.append(doc)
            all_chunk.append(chunk)

        doc_result = pd.concat(all_doc, ignore_index=True) if len(all_doc) > 1 else all_doc[0]
        chunk_result = pd.concat(all_chunk, ignore_index=True) if len(all_chunk) > 1 else all_chunk[0]

        if len(all_doc) > 1:
            doc_result = doc_result.sort_values("similarities", ascending=False)
            doc_result = doc_result.groupby('document_uid', group_keys=False).apply(adjust_similarity)
            doc_result = doc_result.sort_values("similarities", ascending=False).head(top_n_embedding)

        return doc_result, chunk_result

    def _search_docs_by_embedding(self, query, top_n_embedding):
        """Internal method to search top-N documents and chunks by embedding similarity."""
        query_emb = get_embedding(query, self.client)
        df = self.df_embedding.copy()
        df['similarities'] = df['ada_v2'].apply(lambda x: cosine_similarity(x, query_emb))

        top_chunks = df.sort_values("similarities", ascending=False).head(top_n_embedding * 3)
        top_docs = top_chunks.groupby('document_uid', group_keys=False).apply(adjust_similarity)
        top_docs = top_docs.sort_values("similarities", ascending=False).head(top_n_embedding)

        return top_chunks, top_docs

    def bm25_retriever(self, questions, top_n_bm25):
        """Retrieve relevant documents and chunks using BM25 scores."""
        all_doc, all_chunk = [], []
        for q in questions:
            chunk, doc = self._search_docs_bm25(q, top_n_bm25)
            all_doc.append(doc)
            all_chunk.append(chunk)

        doc_result = pd.concat(all_doc, ignore_index=True) if len(all_doc) > 1 else all_doc[0]
        chunk_result = pd.concat(all_chunk, ignore_index=True) if len(all_chunk) > 1 else all_chunk[0]

        if len(all_doc) > 1:
            doc_result = doc_result.sort_values("bm25_scores", ascending=False)
            doc_result = doc_result.groupby('document_uid', group_keys=False).apply(adjust_similarity_bm25)
            doc_result = doc_result.sort_values("bm25_scores", ascending=False).head(top_n_bm25)

        return doc_result, chunk_result

    def _search_docs_bm25(self, query, top_n_bm25):
        """Internal method to search top-N documents and chunks by BM25 score."""
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)

        df = self.df_bm25.copy()
        df['bm25_scores'] = scores

        top_chunks = df.sort_values("bm25_scores", ascending=False).head(top_n_bm25 * 3)
        top_docs = top_chunks.groupby('document_uid', group_keys=False).apply(adjust_similarity_bm25)
        top_docs = top_docs.sort_values("bm25_scores", ascending=False).head(top_n_bm25)

        return top_chunks, top_docs

    def ensembel_docs(self, embedding_uids, bm25_uids):
        """Ensemble uids from embedding and BM25 retrievals (with deduplication)."""
        if not bm25_uids:
            merged_uids = list(dict.fromkeys(embedding_uids))
        else:
            merged_uids = list(dict.fromkeys(embedding_uids + bm25_uids))

        merged_docs = [self.uid_mapping_doc[uid] for uid in merged_uids]
        print(f"Related documents: {merged_uids}")
        return merged_uids, merged_docs, len(merged_uids)
