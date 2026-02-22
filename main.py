import yaml
import numpy as np
from src.extractor import PDFExtractor
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract PDFs
extractor = PDFExtractor(config["pdf_folder"])
texts = extractor.extract_all()

# Chunking
chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]

chunks = []
for text in texts:
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

# Embeddings
embedder = Embedder(config["embedding_model"])
embeddings = embedder.embed(chunks)
embeddings_np = np.array(embeddings).astype("float32")

# FAISS Index
retriever = Retriever(dim=embeddings_np.shape[1], index_path=config["faiss_index_path"])
if retriever.index.ntotal == 0:
    retriever.build_index(embeddings_np)

# LLM Generator
generator = Generator(config["llm_model"])

# User Query
query = input("Enter your question: ")
query_embedding = embedder.embed([query])
indices = retriever.query(np.array(query_embedding).astype("float32"), top_k=5)
context = " ".join([chunks[i] for i in indices])
answer = generator.generate_answer(context)
print(f"\nAnswer:\n{answer}")