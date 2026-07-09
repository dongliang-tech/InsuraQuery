"""
文档索引模块 — 读取 docs/ → 分块 → Jina Embedding → ES 入库
"""

import glob
import json
import os
import sys
import time

import requests

from chunck_strategy import chunk_documents
from config import (
    DOCS_DIR,
    ES_HOST,
    ES_PASS,
    ES_USER,
    INDEX_NAME,
    JINA_API_KEY,
    JINA_EMBED_DIMS,
    JINA_MODEL,
)
from elasticsearch import Elasticsearch


# ─── ES 连接 ────────────────────────────────────────────────

def get_es() -> Elasticsearch:
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False,
        ssl_show_warn=False,
    )
    if not es.ping():
        raise RuntimeError(f"❌ 无法连接 ES: {ES_HOST}")
    print(f"✅ ES 已连接  (v{es.info()['version']['number']})")
    return es


# ─── Jina Embedding ─────────────────────────────────────────

JINA_HEADERS = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json",
}


def get_embeddings(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    """调用 Jina API 批量获取 Embedding

    Args:
        texts: 文本列表
        task: retrieval.passage (入库) 或 retrieval.query (搜索)

    Returns:
        list[list[float]]: 每个文本对应的向量
    """
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY 未设置！请在 .env 或环境变量中配置 JINA_API_KEY")

    # Jina 每次最多 100 条，分批
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=JINA_HEADERS,
            json={
                "model": JINA_MODEL,
                "input": batch,
                "task": task,
                "dimensions": JINA_EMBED_DIMS,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Jina API 错误 ({resp.status_code}): {resp.text}")

        data = resp.json()
        for d in data["data"]:
            all_embeddings.append(d["embedding"])

        print(f"   ↪ 已向量化 {len(all_embeddings)}/{len(texts)} 条", end="\r")

    print()
    return all_embeddings


# ─── 文档读取 ────────────────────────────────────────────────

def get_title_from_filename(filename: str) -> str:
    """从文件名提取语义标题"""
    name = filename.replace(".txt", "").replace(".pdf", "")
    if name[0].isdigit() and "-" in name:
        name = name.split("-", 1)[1]
    return name


def read_all_docs() -> list[dict]:
    """扫描 docs/ 下所有 .txt/.pdf，返回 [{text, title, doc_id, filename}]"""
    files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))) + \
            sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))

    if not files:
        raise FileNotFoundError(f"❌ {DOCS_DIR} 下没有 .txt 或 .pdf 文件")

    docs = []
    for fp in files:
        fn = os.path.basename(fp)
        ext = os.path.splitext(fn)[1].lower()
        title = get_title_from_filename(fn)

        if ext == ".txt":
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".pdf":
            # 复用已有的 pdf 提取逻辑（或使用 pdfplumber）
            try:
                import pdfplumber
                pdf = pdfplumber.open(fp)
                pages = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t and t.strip():
                        pages.append(t.strip())
                pdf.close()
                text = "\n".join(pages) if pages else ""
            except ImportError:
                print(f"  ⚠️ pdfplumber 未安装，跳过 {fn}")
                continue
        else:
            continue

        if not text.strip():
            print(f"  ⚠️ 空内容跳过: {fn}")
            continue

        docs.append({
            "text": text,
            "title": title,
            "doc_id": fn,
            "filename": fn,
            "file_type": ext.replace(".", ""),
        })
        print(f"  📄 {fn:55s} {len(text):>7} chars")

    print(f"   共 {len(docs)} 篇文档")
    return docs


# ─── ES 索引管理 ────────────────────────────────────────────

def create_index_if_not_exists(es: Elasticsearch):
    """如果 knowledge_base 索引不存在则创建"""
    if es.indices.exists(index=INDEX_NAME):
        print(f"📂 索引 '{INDEX_NAME}' 已存在，跳过创建")
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "doc_id":    {"type": "keyword"},
                "filename":  {"type": "keyword"},
                "title":     {"type": "text"},
                "chunk_id":  {"type": "integer"},
                "content":   {"type": "text"},
                "file_type": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": JINA_EMBED_DIMS,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }

    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"✅ 索引 '{INDEX_NAME}' 已创建 (dims={JINA_EMBED_DIMS})")


# ─── 批量写入 ES ────────────────────────────────────────────

def index_chunks(es: Elasticsearch, chunks: list[dict], embeddings: list[list[float]]):
    """将分块+向量写入 ES，每批 50 条"""
    batch_size = 50
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size]

        bulk_body = ""
        for chunk, emb in zip(batch_chunks, batch_embs):
            action = {"index": {"_index": INDEX_NAME}}
            doc = {**chunk, "embedding": emb}
            bulk_body += json.dumps(action, ensure_ascii=False) + "\n"
            bulk_body += json.dumps(doc, ensure_ascii=False) + "\n"

        resp = es.bulk(body=bulk_body, refresh=True)
        if resp.get("errors"):
            print(f"\n⚠️ 写入错误: {resp['items'][0]['index'].get('error', 'unknown')}")

        print(f"   ↪ 已写入 {min(i + batch_size, total)}/{total}", end="\r")

    print()
    print(f"✅ 全部 {total} 条 chunk 已写入 '{INDEX_NAME}'")


# ─── 主流程 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Jina Embedding → ES 文档索引")
    print("=" * 60)

    # 1. 连接 ES
    es = get_es()

    # 2. 创建索引
    create_index_if_not_exists(es)

    # 3. 读取文档
    print("\n📖 读取文档 ...")
    docs = read_all_docs()

    # 4. 分块
    print(f"\n✂️  分块 (chunk_size={512}, overlap={128}) ...")
    chunks = chunk_documents(docs)
    print(f"   → {len(chunks)} 个 chunk")

    # 5. 向量化
    print(f"\n🧠  Jina Embedding ...")
    texts = [c["content"] for c in chunks]
    embeddings = get_embeddings(texts, task="retrieval.passage")

    # 6. 写入 ES
    print(f"\n💾 写入 ES ...")
    index_chunks(es, chunks, embeddings)

    # 7. 统计
    es.indices.refresh(index=INDEX_NAME)
    count = es.count(index=INDEX_NAME)["count"]
    print(f"\n🎉 完成！索引 '{INDEX_NAME}' 中共 {count} 条文档")


if __name__ == "__main__":
    main()
