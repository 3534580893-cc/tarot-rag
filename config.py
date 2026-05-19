"""
塔罗牌 RAG 知识库 — 全局配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 路径 ──────────────────────────────────────
BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ── DeepSeek ──────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# LLM 模型
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")

# Embedding 模型（DeepSeek 或保留 DashScope）
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "deepseek")  # deepseek 或 dashscope
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "deepseek-embeddings")
EMBEDDING_DIMENSION = 1024

# ── 阿里云 DashScope（备选 Embedding）──────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# ── RAG 参数 ──────────────────────────────────
CHUNK_SIZE = 600          # 每个文本块的字符数
CHUNK_OVERLAP = 100       # 块之间的重叠字符数
TOP_K = 5                 # 每次检索返回的相关文档数
SCORE_THRESHOLD = 0.3     # 相关性阈值（0-1）

# ── ChromaDB ──────────────────────────────────
CHROMA_COLLECTION = "tarot_knowledge"

# ── FastAPI ───────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "tarot-rag-secret-change-me")

# ── Admin UI ──────────────────────────────────
ADMIN_PORT = int(os.getenv("ADMIN_PORT", "8501"))
