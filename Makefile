.PHONY: run dev docker

run:
	streamlit run app/streamlit_app.py

dev:
	OLLAMA_BASE_URL=http://localhost:11434 streamlit run app/streamlit_app.py

docker:
	docker build -t job-app-assistant . && docker run -p 8501:8501 --env OLLAMA_BASE_URL=http://host.docker.internal:11434 job-app-assistant
