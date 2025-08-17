# Deploy Guide

## Streamlit Community Cloud
1. Push this folder to a public GitHub repository (e.g., `Rag_pipeline`).
2. Go to https://streamlit.io/cloud and select the repo.
3. Set **Main file path** to `app/streamlit_app.py`.
4. (Optional) Add **Secrets**: `OPENAI_API_KEY` if you plan to use OpenAI.
5. Deploy — you’ll get a public URL (no login for users).

> Note: If you rely solely on Ollama, Streamlit Cloud won’t run an Ollama server. For public demo, either:
> - switch backend to `openai` for the cloud demo, **or**
> - deploy to a server where you can run Ollama (VM, Docker + Ollama).

## Docker (Local / VM)
```bash
docker build -t job-app-assistant .
docker run -p 8501:8501 --env OLLAMA_BASE_URL=http://host.docker.internal:11434 job-app-assistant
```
- Make sure `ollama serve` is running on the host machine (or adjust `OLLAMA_BASE_URL` to the reachable URL).

## Plain Python (Local)
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
