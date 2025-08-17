import os
import streamlit as st
from app.rag.retriever import SimpleVectorStore, chunk_text
from app.rag.llm import load_config, generate_grounded_answer

st.set_page_config(page_title="Job Application Assistant (RAG)", layout="wide")
st.title("💼 Job Application Assistant (RAG) — Fraunhofer MVP")
st.caption("Paste your CV + job post → grounded drafts you can edit & download. No login required.")

with st.sidebar:
    st.header("Settings")
    cfg = load_config("app/rag/config.yaml")
    cfg.backend = st.selectbox("LLM backend", ["ollama","openai","none"], index=0 if cfg.backend=="ollama" else (1 if cfg.backend=="openai" else 2))
    cfg.default_model = st.text_input("Model name", cfg.default_model)
    top_k = st.slider("Top-K passages", 1, 10, 5)
    chunk_size = st.slider("Chunk size", 200, 1200, 800, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 120, step=10)
    st.markdown("---")
    if cfg.backend == "ollama":
        st.caption("Ollama: run `ollama serve` and pull a model, e.g., `ollama pull llama3.1`.")
    elif cfg.backend == "openai":
        st.caption("Set OPENAI_API_KEY in your environment to use OpenAI.")

st.subheader("1) Provide Inputs")
col1, col2 = st.columns(2)
with col1:
    cv_text = st.text_area("CV / experience (plain text)", height=220, placeholder="Skills, projects, experience, education…")
with col2:
    job_text = st.text_area("Job description", height=220, placeholder="Role, responsibilities, requirements…")

uploaded = st.file_uploader("Optional: extra context (txt/markdown)", accept_multiple_files=True)
extra_texts = []
if uploaded:
    for uf in uploaded:
        try:
            extra_texts.append(uf.read().decode("utf-8", errors="ignore"))
        except Exception:
            pass

st.subheader("2) RAG Generation")
task = st.selectbox("Task", ["Tailored bullet points","Cover letter draft","Answer screening questions"], index=0)
query = st.text_input("Prompt (optional)", "")

if st.button("Run RAG 🧠", type="primary"):
    base_corpus = []
    if cv_text.strip():
        base_corpus += chunk_text(cv_text, chunk_size, chunk_overlap)
    if job_text.strip():
        base_corpus += chunk_text(job_text, chunk_size, chunk_overlap)
    for t in extra_texts:
        base_corpus += chunk_text(t, chunk_size, chunk_overlap)

    if not base_corpus:
        st.warning("Please provide at least CV or Job Description text.")
    else:
        store = SimpleVectorStore()
        with st.spinner("Embedding & indexing…"):
            store.build(base_corpus)
        q = query if query.strip() else "Draft content tailored to this job."
        results = store.similarity_search(q, top_k=top_k)
        retrieved = [r[0] for r in results]
        if not retrieved:
            st.error("No passages retrieved. Try different inputs.")
        else:
            with st.spinner("Generating draft…"):
                answer = generate_grounded_answer(load_config("app/rag/config.yaml"), q, retrieved, task)
            st.success("Draft generated. Edit below:")
            edited = st.text_area("Editable result (markdown)", value=answer, height=380)
            st.download_button("Download .md", data=edited.encode("utf-8"), file_name="result.md")
            st.download_button("Download .txt", data=edited.encode("utf-8"), file_name="result.txt")

st.markdown("---")
st.caption("MVP for Fraunhofer IEM GenAI-Incubator • Streamlit + FAISS + Sentence-Transformers + Ollama/OpenAI")
