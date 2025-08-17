# Install & Dev Setup

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ollama (no GPT required)
ollama pull llama3.1
ollama serve
# Optional env
export OLLAMA_BASE_URL=http://localhost:11434
# Run
streamlit run app/streamlit_app.py
```
