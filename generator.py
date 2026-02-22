import logging
from typing import List
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_answer(self, prompt: str) -> str:
        """
        Using Ollama CLI to query LLaMA3 locally.
        Make sure `ollama` CLI is installed.
        """
        try:
            result = subprocess.run(
                ["ollama", "query", self.model_name, prompt],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(result.stderr)
                return "Error generating answer"
            return result.stdout.strip()
        except FileNotFoundError:
            logger.error("Ollama CLI not found. Please install Ollama.")
            return "Ollama not installed"