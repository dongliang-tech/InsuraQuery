"""
会话和收藏管理模块

使用 JSON 文件持久化存储：
- `.chat_data/history.json` — 所有对话记录
- `.chat_data/favorites.json` — 收藏的对话
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime

# ─── 数据存储目录 ─────────────────────────────────────────
DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".chat_data"
HISTORY_FILE = DATA_DIR / "history.json"
FAVORITES_FILE = DATA_DIR / "favorites.json"

DATA_DIR.mkdir(exist_ok=True)


# ─── 工具函数 ─────────────────────────────────────────────

def _load_json(path: Path) -> list:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_json(path: Path, data: list):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + str(int(time.time() * 1000000) % 1000000)


# ─── 历史记录管理 ─────────────────────────────────────────

def save_history(question: str, answer: str, route: str = "local", has_web: bool = False, search_result: dict = None) -> str:
    """保存一条对话记录，返回 session_id"""
    records = _load_json(HISTORY_FILE)
    session_id = _generate_id()
    records.append({
        "id": session_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "route": route,
        "has_web": has_web,
        "search": {
            "local": [{"title": r["title"], "content": r["content"][:200], "score": r["score"]}
                      for r in (search_result.get("local", []) if search_result else [])[:3]],
            "web": [{"title": r.get("title", ""), "url": r.get("url", "")}
                    for r in (search_result.get("web", []) if search_result else [])[:3]],
        } if search_result else None,
    })
    _save_json(HISTORY_FILE, records)
    return session_id


def save_history_session(messages: list, route: str = "local", has_web: bool = False) -> str:
    """保存一整段多轮对话为一个 session（共用一个 ID）。

    messages: Chatbot 格式，每项 {role: 'user'|'assistant', content: str}
    会把第一个用户问题作为 question 摘要、所有 assistant 回复拼起来作为 answer。
    content 可能是 str 或 [{'text': str, ...}] 列表
    """
    if not messages:
        return ""

    def _extract_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if isinstance(content, dict):
            return content.get("text", "")
        return str(content)

    # 消息体标准化处理
    cleaned = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else None
        raw = m.get("content", "") if isinstance(m, dict) else str(m)
        content = _extract_text(raw)

        if role == "assistant":
            stripped = content.strip()
            # 跳过中间状态行
            if stripped.startswith(("🔍", "🚀")):
                continue
            # 跳过只有 header（"> 📚...耗时..." / "> 🌐...耗时..."）的半截回答
            if stripped.startswith(">") and "耗时" in stripped and "\n\n" in stripped:
                head, _, rest = stripped.partition("\n\n")
                if not rest.strip():
                    continue
        # 标准化 content 为纯字符串
        normalized = dict(m) if isinstance(m, dict) else {"role": role, "content": content}
        normalized["content"] = _extract_text(raw)
        cleaned.append(normalized)

    if not cleaned:
        return ""

    records = _load_json(HISTORY_FILE)
    session_id = _generate_id()

    first_q = ""
    flat_a = []
    for m in cleaned:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            if not first_q:
                first_q = content
        elif role == "assistant":
            flat_a.append(content)

    records.append({
        "id": session_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": first_q,
        "answer": "\n\n---\n\n".join(flat_a) if flat_a else "",
        "messages": cleaned,  # 过滤后的多轮对话
        "route": route,
        "has_web": has_web,
        "search": None,
    })
    _save_json(HISTORY_FILE, records)
    return session_id


def get_history(page: int = 1, page_size: int = 20) -> list:
    """分页获取历史记录（最近的在前面）"""
    records = _load_json(HISTORY_FILE)
    records.reverse()
    start = (page - 1) * page_size
    return records[start:start + page_size]


def get_session(session_id: str) -> dict | None:
    """获取单条会话详情"""
    records = _load_json(HISTORY_FILE)
    for r in records:
        if r["id"] == session_id:
            return r
    # 也在收藏中查找
    favs = _load_json(FAVORITES_FILE)
    for r in favs:
        if r["id"] == session_id:
            return r
    return None


def delete_history(session_id: str) -> bool:
    """删除一条历史记录"""
    records = _load_json(HISTORY_FILE)
    new_records = [r for r in records if r["id"] != session_id]
    if len(new_records) == len(records):
        return False
    _save_json(HISTORY_FILE, new_records)
    # 同时从收藏中删除
    favs = _load_json(FAVORITES_FILE)
    _save_json(FAVORITES_FILE, [r for r in favs if r["id"] != session_id])
    return True


def clear_all_history():
    """清空所有历史（不影响收藏）"""
    _save_json(HISTORY_FILE, [])


def count_history() -> int:
    return len(_load_json(HISTORY_FILE))


# ─── 收藏管理 ─────────────────────────────────────────────

def add_favorite(session_id: str) -> bool:
    """收藏一条对话，返回是否成功"""
    records = _load_json(HISTORY_FILE)
    favs = _load_json(FAVORITES_FILE)
    # 检查是否已收藏
    existing_ids = {r["id"] for r in favs}
    if session_id in existing_ids:
        return False  # 已经收藏过了
    for r in records:
        if r["id"] == session_id:
            favs.append(r)
            _save_json(FAVORITES_FILE, favs)
            return True
    return False


def remove_favorite(session_id: str) -> bool:
    """取消收藏"""
    favs = _load_json(FAVORITES_FILE)
    new_favs = [r for r in favs if r["id"] != session_id]
    if len(new_favs) == len(favs):
        return False
    _save_json(FAVORITES_FILE, new_favs)
    return True


def get_favorites(page: int = 1, page_size: int = 20) -> list:
    """分页获取收藏列表"""
    favs = _load_json(FAVORITES_FILE)
    favs.reverse()
    start = (page - 1) * page_size
    return favs[start:start + page_size]


def count_favorites() -> int:
    return len(_load_json(FAVORITES_FILE))
