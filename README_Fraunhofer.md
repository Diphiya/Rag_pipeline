# Fraunhofer IEM GenAI-Incubator — Submission Package

**Project:** Job Application Assistant (RAG) — MVP  
**Value:** Tailors bullet points, cover letter drafts, and screening answers from a user's CV and a job description. No login required; results editable & downloadable.

## Quick Run (Local, with Ollama)
```bash
pip install -r requirements.txt
ollama pull llama3.1   # or mistral / qwen2.5
ollama serve           # starts at http://localhost:11434
streamlit run app/streamlit_app.py
```
Open http://localhost:8501, choose **LLM backend: ollama**, set **Model name** (e.g., `llama3.1`).

## Deploy a Shareable Link
- **Streamlit Community Cloud**: push this folder to GitHub → deploy `app/streamlit_app.py`. (If using OpenAI optionally, add `OPENAI_API_KEY` in Secrets.)
- **Docker**:
```bash
docker build -t job-app-assistant .
docker run -p 8501:8501 --env OLLAMA_BASE_URL=http://host.docker.internal:11434 job-app-assistant
```

## Files
- `app/streamlit_app.py` — no-login UI
- `app/rag/retriever.py` — chunking + FAISS retrieval
- `app/rag/llm.py` — LLM adapters (Ollama by default; OpenAI optional)
- `app/rag/config.yaml` — settings
- `docs/C4-Model.pdf` — one-page C4 diagram
- `requirements.txt`, `Dockerfile`, `Makefile`, `run.sh`, `.env.example`

## Notes for Reviewers
- Focused, tangible MVP with clear user value and editing capability.
- RAG on user-provided text only; no data stored server-side in this MVP.
