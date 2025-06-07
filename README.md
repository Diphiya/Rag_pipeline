# LangChain PDF QA with Ollama & FAISS

This project extracts text from PDF documents, vectorizes it using Sentence Transformers, stores it in a FAISS vector store, and answers queries using an LLM (LLaMA 3 via Ollama).

## requirements
- Python 3.8 or higher
- LangChain
- Ollama installed locally
- pip to install Python packages

## installation

```bash
git clone https://github.com/Diphiya/Rag_pipeline.git
cd Rag_pipeline
pip install -r requirements.txt

## usage
## Place your PDF files inside the ./pdf folder.

## Make sure Ollama is running and llama3 is available:

ollama pull llama3

## Run the script:

python rag_pipeline.py