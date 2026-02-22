import faiss
import numpy as np
import os
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, dim: int, index_path: str):
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)
        if os.path.exists(index_path):
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)

    def build_index(self, embeddings: np.ndarray):
        logger.info("Building FAISS index...")
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)

    def query(self, query_vec: np.ndarray, top_k: int = 5) -> List[int]:
        D, I = self.index.search(query_vec, top_k)
        return I[0]