from __future__ import annotations
from typing import List, Tuple
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    step = max(1, chunk_size - chunk_overlap)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start += step
    return chunks

class SimpleVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.texts: List[str] = []
        self.index = None
        self.embeddings = None

    def build(self, texts: List[str]):
        self.texts = texts
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = emb.astype('float32')
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings)

    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        D, I = self.index.search(q_emb, top_k)
        out = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            out.append((self.texts[idx], float(score)))
        return out
