"""
混合搜索模块 — BM25 + Vector (手动 RRF) + Route 路由

流程：
  用户提问 → BM25 + Vector 混合搜索本地
    → LLM 判断本地知识是否足够
    → 足够: 只用本地结果
    → 不足: 再调用 Tavily 补充
"""
import json

import httpx
import requests

from config import (
    ES_HOST,
    ES_PASS,
    ES_USER,
    HYBRID_NUM_CANDIDATES,
    HYBRID_SEARCH_K,
    INDEX_NAME,
    JINA_API_KEY,
    JINA_EMBED_DIMS,
    JINA_MODEL,
    RRF_RANK_CONSTANT,
    SCORE_THRESHOLD,
    TAVILY_API_KEY,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    ROUTE_SYSTEM_PROMPT,
)
from elasticsearch import Elasticsearch


# ─── Jina 客户端 ────────────────────────────────────────────
from functools import lru_cache

JINA_HEADERS = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json",
}

# ES 连接池（复用）
_es_instance = None

_embed_cache = {}


def jina_embed(texts: list[str], task: str = "retrieval.query", max_retries: int = 2) -> list[list[float]]:
    """获取 Jina Embedding（带缓存 + 重试机制）"""
    cache_key = "||".join(texts) + "::" + task
    if cache_key in _embed_cache:
        return _embed_cache[cache_key]

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers=JINA_HEADERS,
                json={"model": JINA_MODEL, "input": texts, "task": task, "dimensions": JINA_EMBED_DIMS},
                timeout=60,
            )
            resp.raise_for_status()
            result = [d["embedding"] for d in resp.json()["data"]]
            _embed_cache[cache_key] = result
            return result
        except Exception as e:
            if attempt < max_retries:
                import time
                wait = 2 ** attempt
                print(f"⚠️ Jina Embedding 重试 {attempt+1}/{max_retries}: {e}，等待 {wait}s")
                time.sleep(wait)
            else:
                print(f"⚠️ Jina Embedding 最终失败 ({max_retries+1} 次): {e}")
                raise


# ─── ES 客户端（复用连接池）─────────────────────────────────

def get_es() -> Elasticsearch:
    global _es_instance
    if _es_instance is None:
        _es_instance = Elasticsearch(
            ES_HOST,
            basic_auth=(ES_USER, ES_PASS),
            verify_certs=False,
            ssl_show_warn=False,
        )
    return _es_instance


# ─── Tavily 搜索 ────────────────────────────────────────────

def tavily_search(query: str, max_results: int = 3) -> list[dict]:
    """通过 Tavily API 搜索网络"""
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
        },
        verify=False,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r["title"],
            "content": r["content"],
            "url": r["url"],
            "score": r.get("score", 0.5),
            "source": "tavily",
            "chunk_id": -1,
            "doc_id": r["url"],
        }
        for r in data.get("results", [])
    ]


# ─── Route 路由 ────────────────────────────────────────────

def route_decision(query: str, local_context: str) -> str:
    """让 LLM 判断本地知识是否足够回答用户问题

    Returns:
        "local" — 本地知识足够
        "web"   — 需要网络搜索补充
    """
    try:
        resp = httpx.post(
            f"{LLM_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": ROUTE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"""用户问题: {query}

本地知识库检索到的内容:
{local_context}"""},
                ],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=30,
            verify=False,
        )
        resp.raise_for_status()
        decision = resp.json()["choices"][0]["message"]["content"].strip().lower()
        # 只取 local 或 web
        if "local" in decision:
            return "local"
        return "web"
    except Exception as e:
        print(f"⚠️ Route 判断失败: {e}，默认走 local")
        return "local"


# ─── 本地搜索 (BM25 + Vector) ─────────────────────────────

def search_local(query: str, k: int = None) -> list[dict]:
    """仅在本地 ES 搜索（BM25 + Vector + 手动 RRF）"""
    es = get_es()
    k = k or HYBRID_SEARCH_K

    # 1. 向量化查询
    try:
        q_emb = jina_embed([query])[0]
    except Exception as e:
        print(f"⚠️ Jina Embedding 失败: {e}")
        q_emb = None

    # --- 2a. BM25 搜索 ---
    bm25_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^3", "content"],
                "type": "best_fields",
            }
        },
        "size": k * 2,
        "_source": {"excludes": ["embedding"]},
    }
    bm25_result = es.search(index=INDEX_NAME, body=bm25_body)

    # --- 2b. Vector 搜索 ---
    knn_results_raw = []
    if q_emb:
        knn_body = {
            "knn": {
                "field": "embedding",
                "query_vector": q_emb,
                "k": k,
                "num_candidates": HYBRID_NUM_CANDIDATES,
            },
            "size": k * 2,
            "_source": {"excludes": ["embedding"]},
        }
        knn_result = es.search(index=INDEX_NAME, body=knn_body)
        knn_results_raw = knn_result["hits"]["hits"]

    # --- 2c. 手动 RRF 融合 ---
    bm25_hits = bm25_result["hits"]["hits"]
    ranked = {}

    def _rrf_score(rank: int, const: float = 60.0) -> float:
        return 1.0 / (const + rank + 1)

    for rank, hit in enumerate(bm25_hits):
        src = hit["_source"]
        key = f"{src.get('doc_id','')}:{src.get('chunk_id','')}"
        ranked.setdefault(key, {"doc": src, "score": 0})
        ranked[key]["score"] += _rrf_score(rank)

    for rank, hit in enumerate(knn_results_raw):
        src = hit["_source"]
        key = f"{src.get('doc_id','')}:{src.get('chunk_id','')}"
        ranked.setdefault(key, {"doc": src, "score": 0})
        ranked[key]["score"] += _rrf_score(rank)

    sorted_items = sorted(ranked.items(), key=lambda x: x[1]["score"], reverse=True)

    results = []
    for key, item in sorted_items[:k]:
        src = item["doc"]
        results.append({
            "title": src.get("title", ""),
            "content": src.get("content", ""),
            "doc_id": src.get("doc_id", ""),
            "chunk_id": src.get("chunk_id", -1),
            "filename": src.get("filename", ""),
            "score": item["score"],
            "source": "local",
        })
    return results


# ─── 主搜索入口（带路由） ─────────────────────────────────

def hybrid_search(query: str, k: int = None) -> dict:
    """混合搜索主入口（带 Route 路由）

    流程：
      1. 始终先搜索本地 ES (BM25 + Vector + RRF)
      2. 用 LLM 判断本地知识是否足够回答
      3. 不够时才调用 Tavily

    Returns:
        dict: {
            "local": [...],   # ES 本地搜索结果
            "web": [...],     # Tavily 搜索结果 (可能为空)
            "has_web": bool,  # 是否使用了 Tavily
            "route": str,     # "local" 或 "web" 路由决策
        }
    """
    k = k or HYBRID_SEARCH_K

    # 1. 始终先搜索本地
    local_results = search_local(query, k)

    # 2. 构造本地上下文给 Route 判断
    local_context_parts = []
    for r in local_results[:3]:
        local_context_parts.append(f"[{r['title']}]\n{r['content'][:300]}")
    local_context = "\n\n".join(local_context_parts)

    # 3. Route 判断
    route = route_decision(query, local_context)
    print(f"🔀 Route 决策: {route}")

    # 4. 根据路由决定是否调 Tavily
    web_results = []
    has_web = False

    if route == "web":
        try:
            web_results = tavily_search(query)
            has_web = True
        except Exception as e:
            print(f"⚠️ Tavily 搜索失败: {e}")

    return {
        "local": local_results,
        "web": web_results,
        "has_web": has_web,
        "route": route,
    }


def format_results_for_llm(search_result: dict, max_chars: int = 6000) -> str:
    """将搜索结果格式化为 LLM 能消费的上下文"""
    parts = []

    if search_result["local"]:
        parts.append("## 📄 本地文档检索结果\n")
        for i, r in enumerate(search_result["local"][:6], 1):
            parts.append(f"### [{i}] {r['title']}\n")
            if r.get("filename"):
                parts.append(f"> 来源: {r['filename']}  (chunk {r['chunk_id']})\n")
            content = r["content"]
            if len(content) > 800:
                content = content[:800] + "..."
            parts.append(f"{content}\n")

    if search_result["web"]:
        parts.append("\n## 🌐 网络搜索结果\n")
        for i, r in enumerate(search_result["web"], 1):
            parts.append(f"### [Web {i}] {r['title']}\n")
            parts.append(f"> 链接: {r['url']}\n")
            content = r["content"]
            if len(content) > 800:
                content = content[:800] + "..."
            parts.append(f"{content}\n")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...(截断)"

    return text
