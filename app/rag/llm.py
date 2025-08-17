from __future__ import annotations
from typing import List
import os, yaml
from pydantic import BaseModel

class LLMConfig(BaseModel):
    backend: str = "ollama"      # ollama | openai | none
    default_model: str = "llama3.1"
    max_input_tokens: int = 8000

def load_config(path: str = "app/rag/config.yaml") -> LLMConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return LLMConfig(**raw)

def call_llm_ollama(system: str, user: str, model: str) -> str:
    import requests
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{base.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")
        return data.get("response", "")
    return str(data)

def call_llm_openai(system: str, user: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def offline_template(prompt: str) -> str:
    return f"[OFFLINE DRAFT]\n\n{prompt}\n\n(Add LLM settings to enable real responses.)"

def generate_grounded_answer(config: LLMConfig, query: str, retrieved_chunks: List[str], task: str) -> str:
    context = "\n\n".join([f"[CHUNK {i+1}]\n{c}" for i, c in enumerate(retrieved_chunks)])
    system = (
        "You are a helpful assistant for job applications.\n"
        "Use ONLY the provided [CHUNK] context. If insufficient, say so and ask for more details.\n"
        "Keep responses concise and professional."
    )
    user = (
        f"TASK: {task}\n\n"
        f"QUERY: {query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Respond with clean markdown."
    )

    try:
        if config.backend == "ollama":
            return call_llm_ollama(system, user, config.default_model)
        if config.backend == "openai" and os.getenv("OPENAI_API_KEY"):
            return call_llm_openai(system, user, config.default_model)
    except Exception as e:
        return f"LLM error: {e}\n\n" + offline_template(user)
    return offline_template(user)
