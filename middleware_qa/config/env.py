import os
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

class Settings:
    def __init__(self):
        load_dotenv()
        raw_key = os.getenv("DEEPSEEK_API_KEY", "")
        k = raw_key.strip()
        if k.startswith("DEEPSEEK_API_KEY="):
            k = k.split("DEEPSEEK_API_KEY=", 1)[1].strip()
        self.DEEPSEEK_API_KEY = k
        self.MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))
        self.MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048"))
        rb = os.getenv("RAG_BACKEND", "none").lower()
        self.RAG_BACKEND = rb if rb in ("none", "chroma") else "none"
        self.CHROMA_DIR = os.getenv("CHROMA_DIR", "chromadb_data")
        self.CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "middleware_docs")
        self.RAG_RESOURCE_DIR = os.getenv("RAG_RESOURCE_DIR", "RagResource")
        self.RAG_TOPK = int(os.getenv("RAG_TOPK", "3"))
        self.RAG_MERGE_N = int(os.getenv("RAG_MERGE_N", "3"))
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.HF_HOME = os.getenv("HF_HOME", "")
        self.HF_OFFLINE = os.getenv("HF_OFFLINE", "1")
        self.RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
        self.RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
        self.RAG_CLEAN_REGEX = os.getenv("RAG_CLEAN_REGEX", "")
        self.RAG_PDF_MAX_PAGES = int(os.getenv("RAG_PDF_MAX_PAGES", "0"))
        self.MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.db")
        self.MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "12"))
        self.CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", ".data/checkpoints.sqlite")
        self.OBS_ENABLE = os.getenv("OBS_ENABLE", "1")
        self.OBS_DIR = os.getenv("OBS_DIR", ".obs")
        self.EVAL_DIR = os.getenv("EVAL_DIR", ".obs")
        self.RAG_MIN_SIM = float(os.getenv("RAG_MIN_SIM", "0.65"))

def get_settings() -> Settings:
    return Settings()
