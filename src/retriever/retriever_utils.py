import numpy as np
import tiktoken

EMBED_TOKEN_LIMIT = 8192

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def adjust_similarity(group, bios=0.05):
    group = group.sort_values('similarities', ascending=False).reset_index(drop=True)
    
    if len(group) > 1:
        group.loc[0, 'similarities'] += (group['similarities'][1:] * bios).sum()
    return group.head(1)

def adjust_similarity_bm25(group):
    group = group.sort_values('bm25_scores', ascending=False).reset_index(drop=True)
    if len(group) > 1:
        group.loc[0, 'bm25_scores'] += (group['bm25_scores'][1:] * 0).sum()
    return group.head(1)

def get_embedding(text, client):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) > EMBED_TOKEN_LIMIT:
        max_tokens = EMBED_TOKEN_LIMIT - 200
        tokens = tokens[-max_tokens:]
        truncated_text = encoding.decode(tokens)
    else:
        truncated_text = text

    # text-embedding-3-rag
    # text-embedding-ada-002
    text_embedding = client.embeddings.create(input=[truncated_text], model="text-embedding-3-rag").data[0].embedding
    return text_embedding