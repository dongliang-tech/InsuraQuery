"""
文档分块模块

使用 RecursiveCharacterTextSplitter 对中文保险文档进行语义分割。
默认 chunk_size=512 字符, overlap=128 字符, 按段落→句子→标点递归分割。
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _get_splitter(chunk_size: int = None, chunk_overlap: int = None):
    """创建 RecursiveCharacterTextSplitter 实例"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
        # 优先按自然段落/句子边界分割，适合中文保险条款
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,             # 用字符数近似 token 数
        is_separator_regex=False,
    )


def chunk_document(text: str, title: str = "", doc_id: str = "",
                   chunk_size: int = None, chunk_overlap: int = None) -> list[dict]:
    """将单篇文档分割为多个 chunk

    Args:
        text: 原始文档内容
        title: 文档标题
        doc_id: 文档唯一标识
        chunk_size: 可选，覆盖默认值
        chunk_overlap: 可选，覆盖默认值

    Returns:
        list[dict]: 每个 chunk 包含 title, content, chunk_id, doc_id
    """
    splitter = _get_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)

    return [
        {
            "doc_id": doc_id,
            "title": title,
            "content": chunk,
            "chunk_id": i,
        }
        for i, chunk in enumerate(chunks)
    ]


def chunk_documents(docs: list[dict], chunk_size: int = None, chunk_overlap: int = None) -> list[dict]:
    """批量分块

    Args:
        docs: 每项包含 text, title, doc_id
        chunk_size: 可选覆盖
        chunk_overlap: 可选覆盖

    Returns:
        list[dict]: 所有 chunk 的列表
    """
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(
            text=doc["text"],
            title=doc.get("title", ""),
            doc_id=doc.get("doc_id", ""),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks
