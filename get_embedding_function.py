# requirements:
#   pip install -U langchain-ollama langchain-community

from typing import List, Union
from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
import ollama

# Wrap Ollama so it handles both single texts and batches
class SafeOllamaEmbeddings(Embeddings):
    """
    A drop-in replacement that delegates to Ollama’s /api/embeddings
    endpoint and transparently handles batch or single inputs.
    """
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.client = ollama(model=model, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ollama server already supports list input; just pass it through
        return self.client.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        # Single-string helper (LangChain calls this sometimes)
        return self.client.embed([text])[0]

def get_embedding_function():
    """
    Returns an Embeddings object that LangChain can batch-call safely.
    """
    return OllamaEmbeddings(model="nomic-embed-text")
