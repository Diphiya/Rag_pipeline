import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from typing import List

# --- Step 1: Load PDFs and split into documents ---
def load_pdf_texts(pdf_folder: str) -> List[str]:
    docs = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(pdf_folder, file))
            docs.extend(loader.load())
    return docs


def main():
    pdf_folder = "./pdf"
    print("Loading PDFs...")
    docs = load_pdf_texts(pdf_folder)
    print(f"Loaded {len(docs)} documents")

    # --- Step 2: Embedding of  documents ---
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- Step 3: Build FAISS vector store ---
    vectorstore = FAISS.from_documents(docs, embeddings)

    # --- Step 4: Used Ollama's llama 3 ---
    llm = Ollama(model="llama3")  

    # --- Step 5: Created a custom prompt template enforcing German answers ---
    prompt_template = """
Bitte beantworte die folgende Frage basierend auf dem Kontext unten. Antworte ausschließlich auf Deutsch.

Kontext:
{context}

Frage: {question}

Antwort:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # --- Step 6: Build QA chain with custom prompt ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # --- Queries ---
    queries = [
        "Wie hoch ist die Grundzulage?",
        "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?",
        "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?"
    
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = qa_chain.invoke(query)  
        answer = result['result']
        source_docs = result['source_documents']

        # --- Extract and print cited page numbers --- 
        cited_pages = set()
        for doc in source_docs:
            page = doc.metadata.get("page") or doc.metadata.get("pagenumber") or "unbekannt"
            cited_pages.add(str(page))

        print("Answer:", answer)
        print(f"Zitierte Seiten: {', '.join(sorted(cited_pages))}")

if __name__ == "__main__":
    main()
