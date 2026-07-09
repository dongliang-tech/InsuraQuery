"""
配置模块 — 从环境变量 / .env 读取敏感信息，参数常量集中管理

读取顺序：
  1. 进程已有的环境变量（优先级最高，便于 docker / CI 注入）
  2. 项目根目录的 .env 文件（开发用）
"""
import os
from pathlib import Path

# ─── 加载 .env（如果安装了 python-dotenv）───────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # 没装 dotenv 时退化为纯依赖环境变量


def _env(name: str, default: str = "") -> str:
    """读取环境变量，去掉首尾空白。"""
    return os.getenv(name, default).strip()


# ─── Elasticsearch ─────────────────────────────────────────
ES_HOST = _env("ES_HOST", "https://localhost:9200")
ES_USER = _env("ES_USER", "elastic")
ES_PASS = _env("ES_PASS")                       # 从 .env 读取
ES_INDEX_INPUT = _env("ES_INDEX", "knowledge_base")

INDEX_NAME = ES_INDEX_INPUT                     # 主索引：分块 + 向量
OLD_INDEX_NAME = "insurance_docs"                # 旧纯文本索引（仅 es_insurance_search.py 使用）

# ─── Jina Embeddings ────────────────────────────────────────
JINA_API_KEY = _env("JINA_API_KEY")
JINA_MODEL = _env("JINA_MODEL", "jina-embeddings-v3")
JINA_EMBED_DIMS = int(_env("JINA_EMBED_DIMS", "1024"))

# ─── Tavily Web Search ──────────────────────────────────────
TAVILY_API_KEY = _env("TAVILY_API_KEY")

# ─── 分块参数 ────────────────────────────────────────────────
CHUNK_SIZE = 512        # 单 chunk 字符数（中文保险条款按字符近似 token）
CHUNK_OVERLAP = 128     # chunk 间重叠字符数

# ─── 文档目录 ────────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

# ─── 并行问答参数 ────────────────────────────────────────────
PARALLEL_CHUNK_GROUP_SIZE = 3       # 每组 chunk 数（历史参数，当前大上下文模式不再分组）
PARALLEL_MAX_WORKERS = 8            # 并行线程数
PARALLEL_SEARCH_K = 15              # 并行阶段取 chunks
HYBRID_SEARCH_K = 10                # 普通搜索最终返回条数
HYBRID_NUM_CANDIDATES = 100         # 向量搜索候选数
RRF_RANK_CONSTANT = 60              # RRF 排名常数
RRF_WINDOW_SIZE = 100               # RRF 窗口大小
BM25_TITLE_BOOST = 3.0              # title 字段 BM25 权重
SCORE_THRESHOLD = 0.1               # Tavily fallback 阈值（保留参数，当前路由基于 LLM 判断）

# ─── LLM 路由 ────────────────────────────────────────────────
LLM_API_BASE = _env("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY = _env("LLM_API_KEY")
LLM_MODEL = _env("LLM_MODEL", "deepseek-v4-pro")           # 问答主模型
LLM_MODEL_FAST = _env("LLM_MODEL_FAST", "deepseek-v4-flash")  # 大上下文场景的快速模型

# Route 系统提示词 — 判断本地知识是否足够回答
ROUTE_SYSTEM_PROMPT = """你是一个路由判断专家。你的任务是根据用户问题和检索到的本地知识库内容，判断本地知识是否足够回答用户的问题。

判断标准：
1. 如果本地知识库内容**明确包含**了问题的答案或相关信息 → 输出 "local"
2. 如果本地知识库内容**完全不相关**或**信息不足** → 输出 "web"
3. 如果本地知识库内容**部分相关但不够完整** → 输出 "web"

你必须只输出 "local" 或 "web"，不要输出其他任何内容。"""
