# LangChain PDF QA with Ollama & FAISS

_A simple but powerful Retrieval-Augmented Generation (RAG) pipeline that answers questions from PDF documents using vector search & large language models._

---

##  Features

Extract text from multiple PDF files  
Embed documents with Sentence Transformers  
Vector similarity search using FAISS  
Answer user queries with LLaMA 3 via **Ollama**  
Easy setup and run locally  
Ideal for knowledge-grounded QA workflows

---

## Architecture Overview

The pipeline follows these core steps:

1. **PDF Extraction** – Read and parse text from PDFs.  
2. **Chunking & Embeddings** – Break text into chunks and compute embeddings.  
3. **Vector Database** – Store embeddings in FAISS for fast retrieval.  
4. **Query Handling** – Retrieve relevant passages.  
5. **Generation** – Produce answers with LLaMA 3 via Ollama.

---

## Requirements

- Python 3.8+  
- Ollama installed locally  
- LLaMA 3 model pulled via Ollama  
- `pip` for Python dependencies

---

## Installation

1. Clone repository:
   ```bash
   git clone https://github.com/Diphiya/Rag_pipeline.git
   cd Rag_pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Ollama:
   ```bash
   # Follow instructions at https://ollama.com
   ollama pull llama3
   ```

---

## Usage

1. Put PDF files inside the `./pdf` folder  
2. Start the script:
   ```bash
   python rag_pipeline.py
   ```

3. Ask your questions:
   - The script will retrieve relevant text chunks, then generate an answer using the LLM

---

## Example

```bash
>>> python rag_pipeline.py
Enter your query: What is the main topic of training material?
Answer: “The dataset covers AI fundamentals and retrieval systems..."
```

---

## Configuration

You can configure the pipeline by adjusting settings like:

- Embedding model
- Chunk size & overlap
- Vector database path

(_Add details if config files exist_)

---


