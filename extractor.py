import os
from typing import List
import PyPDF2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder

    def extract_text(self, filename: str) -> str:
        file_path = os.path.join(self.pdf_folder, filename)
        logger.info(f"Extracting text from {file_path}")
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except FileNotFoundError:
            logger.error(f"File {filename} not found")
        return text

    def extract_all(self) -> List[str]:
        texts = []
        for file in os.listdir(self.pdf_folder):
            if file.lower().endswith(".pdf"):
                texts.append(self.extract_text(file))
        return texts