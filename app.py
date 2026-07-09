"""
完整 UI 文件 — 参考知乎直答布局
"""
import json
import os
import gradio as gr
import httpx

from search import hybrid_search, format_results_for_llm
from chat_manager import (
    save_history, save_history_session, get_history, get_session, delete_history, clear_all_history, count_history,
    add_favorite, remove_favorite, get_favorites, count_favorites,
)
from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL

_current_answer = ""
_current_question = ""
_current_search_result = None
_current_route = "local"
_cancel_flag = False  # 全局取消标志，stop 按钮触发
_latest_msgs = []      # 最近一次 respond 的完整对话（用于 unload 时自动保存）


def ask_llm_stream(query: str, context: str):
    """流式生成回答"""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    global _cancel_flag

    system_prompt = """你是一个专业的保险条款问答助手。请根据提供的参考文档回答用户问题。
要求：
1. 仅基于提供的参考文档和网络搜索结果回答
2. 如果参考文档不足以回答，明确说明
3. 引用来源（文档标题 / 网页链接）
4. 用中文回答，专业、简洁、准确
5. 如果有网络搜索结果，标注为 🌐 网络来源"""
    try:
        with httpx.Client(
            verify=False,
            timeout=httpx.Timeout(300.0, connect=15.0, read=300.0, write=30.0, pool=10.0),
            http2=False,
            trust_env=False,
        ) as client:
            with client.stream("POST", f"{LLM_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
                json={"model": LLM_MODEL, "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"## 参考文档\n\n{context}\n\n## 用户问题\n\n{query}"}
                ], "max_tokens": 4096, "temperature": 0.3, "stream": True}) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if _cancel_flag:
                        # 主动关闭连接，让服务器立即停止生成
                        resp.close()
                        break
                    if not line or line.startswith(":"): continue
                    if line.startswith("data: "):
                        ds = line[6:].strip()
                        if ds == "[DONE]": break
                        try:
                            chunk = json.loads(ds)
                            for c in chunk.get("choices", []):
                                d = c.get("delta", {})
                                if d.get("content"): yield d["content"]
                        except json.JSONDecodeError: continue
    except Exception as e:
        if not _cancel_flag:
            yield f"❌ LLM 流式调用失败: {e}"


def respond(message, history):
    """聊天回调"""
    import copy, time

    global _current_answer, _current_question, _current_search_result, _current_route, _cancel_flag, _latest_msgs

    _cancel_flag = False  # 每次新请求重置取消标志

    # 深拷贝 history，避免与上一轮 chatbot 显示的 dict 共享引用
    history = copy.deepcopy(history) if history else []

    text = message.get("text", "") if isinstance(message, dict) else str(message)
    if not text.strip():
        yield [{"role": "assistant", "content": "请输入问题"}]
        return

    _current_question = text
    is_complex = len(text) > 20 or any(kw in text for kw in ["比较", "区别", "各", "分别", "所有", "总结", "汇总"])
    start_time = time.time()

    # 1. 用户消息先加入历史（每条都是新建的 dict，不共享引用）
    msgs = [dict(m) for m in history] + [{"role": "user", "content": text}]
    yield msgs

    # 立即更新快照（即使只说了话还没回答，unload 也能保存）
    _latest_msgs = copy.deepcopy(msgs)

    if is_complex:
        # 并行问答
        msgs = msgs + [{"role": "assistant", "content": "🔍 检测到复杂问题，启动并行搜索..."}]
        yield msgs
        from parallel_qa import parallel_qa
        result = parallel_qa(text)
        route = "local"
        _current_route = route
        elapsed = time.time() - start_time
        header = f"> 🚀 并行问答 *(处理 {result['chunks_used']} 个片段, 耗时 {elapsed:.1f}s)*\n\n"
        accumulated = header
        msgs = msgs[:-1] + [{"role": "assistant", "content": accumulated}]
        yield msgs
        for token in result["final_answer"]:
            accumulated += token
            msgs = msgs[:-1] + [{"role": "assistant", "content": accumulated}]
            _latest_msgs = copy.deepcopy(msgs)
            yield msgs
        _current_answer = accumulated
        _current_search_result = None
    else:
        # 普通搜索
        msgs = msgs + [{"role": "assistant", "content": "🔍 正在搜索..."}]
        yield msgs
        search_result = hybrid_search(text)
        _current_search_result = search_result
        context = format_results_for_llm(search_result)
        route = search_result.get("route", "local")
        _current_route = route
        elapsed = time.time() - start_time
        tag = "🌐 网络补充" if route == "web" else "📚 本地知识库"
        header = f"> {tag} *(耗时 {elapsed:.1f}s)*\n\n"
        accumulated = header
        msgs = msgs[:-1] + [{"role": "assistant", "content": accumulated}]
        yield msgs
        for token in ask_llm_stream(text, context):
            accumulated += token
            msgs = msgs[:-1] + [{"role": "assistant", "content": accumulated}]
            _latest_msgs = copy.deepcopy(msgs)
            yield msgs
        if search_result["has_web"]:
            refs = [f"- 🌐 [{r['title']}]({r['url']})" for r in search_result["web"]]
            accumulated += "\n\n---\n**🌐 来源:**\n" + "\n".join(refs)
            msgs = msgs[:-1] + [{"role": "assistant", "content": accumulated}]
            _latest_msgs = copy.deepcopy(msgs)
            yield msgs
        _current_answer = accumulated

    # 缓存最新对话快照（正常结束 / 异常 / cancel 都写入），便于 unload 时自动保存
    _latest_msgs = copy.deepcopy(msgs) if msgs else []

    # 不再自动保存到历史记录；改为在「✨ 新建聊天」时由 on_new_chat 统一保存为一个 session
    # save_history(_current_question, _current_answer, _current_route,
    #              _current_search_result.get("has_web", False) if _current_search_result else False,
    #              _current_search_result)


# ─── 构建 UI ─────────────────────────────────────────────

def create_app():
    with gr.Blocks(title="保险智答 🤖") as demo:

        # ── 顶部全局标题 ──
        gr.Markdown(
            "# 保险智答  <span style=\"font-size:13px;font-weight:400;color:#94a3b8;letter-spacing:1.5px\">AI × 保险知识库</span>",
            elem_classes="chat-title",
        )

        with gr.Tab("💬 智能问答"):
            EXAMPLES = [
                "工伤保险和雇主险有什么区别？",
                "雇主责任险的赔偿范围是什么？",
                "商业综合责任保险是什么？",
                "比较所有保险产品的保障范围区别",
            ]

            # ── 状态：是否正在回答 ──
            is_responding = gr.State(False)

            # ── 新建对话按钮（将在 chat-card 右上角外面，用 JS 移过去） ──
            new_chat_btn = gr.Button("+ 新建对话", size="sm", elem_classes="new-chat-btn", variant="secondary")

            # ── 合并容器：Chatbot + 输入区放在同一卡片内 ──
            with gr.Column(elem_classes="chat-card"):
                chatbot = gr.Chatbot(
                    height=560,
                    show_label=False,
                    avatar_images=(None, None),
                    render_markdown=True,
                    elem_classes="chat-card-bot",
                )
                with gr.Row(elem_classes="chat-card-input"):
                    main_msg = gr.Textbox(
                        show_label=False,
                        placeholder="输入你的问题",
                        lines=2,
                        max_lines=6,
                        container=False,
                        elem_classes="chat-card-text",
                    )
                    send_btn = gr.Button("↑", elem_classes="chat-card-send", size="sm", variant="primary", visible=True)
                    stop_btn = gr.Button("■", elem_classes="chat-card-send chat-card-send-stop", size="sm", variant="stop", visible=False)

            # 推荐问题（放在聊天卡片下方）
            gr.Markdown("##### 推荐问题")
            with gr.Row(elem_classes="examples-row"):
                example_buttons = []
                for ex in EXAMPLES:
                    b = gr.Button(ex, elem_classes="example-chip", size="sm", variant="secondary")
                    example_buttons.append(b)

            def respond_wrapper(message, history):
                history = list(history) if history else []
                for msg_list in respond(message, history):
                    yield msg_list

            def on_send(text, history, responding):
                text = text.strip()
                if not text or responding:
                    return gr.update(), gr.update(), gr.update(), responding, history
                # 第一个 yield：禁用输入框、隐藏发送按钮、显示停止按钮
                yield (
                    gr.Textbox(value="", interactive=False, placeholder="正在回答中…"),
                    gr.Button(visible=False),                       # send_btn 隐藏
                    gr.Button(visible=True),                        # stop_btn 显示
                    True,
                    history,
                )
                result_gen = respond_wrapper({"text": text}, list(history))
                full_history = list(history)
                for msg_list in result_gen:
                    full_history = msg_list
                    yield gr.update(), gr.update(), gr.update(), True, msg_list
                # 结束：恢复输入框、显示发送按钮、隐藏停止按钮
                yield (
                    gr.Textbox(interactive=True, placeholder="输入你的问题"),
                    gr.Button("↑", elem_classes="chat-card-send", variant="primary", visible=True),
                    gr.Button("■", elem_classes="chat-card-send chat-card-send-stop", variant="stop", visible=False),
                    False,
                    full_history,
                )

            def on_stop(responding):
                """停止按钮：设置取消标志 + 恢复输入框 + 切回发送按钮"""
                global _cancel_flag
                _cancel_flag = True
                return (
                    gr.Textbox(interactive=True, placeholder="输入你的问题"),
                    gr.Button("↑", elem_classes="chat-card-send", variant="primary", visible=True),
                    gr.Button("■", elem_classes="chat-card-send chat-card-send-stop", variant="stop", visible=False),
                    False,
                    gr.update(),
                )

            # 发送按钮 click → on_send
            send_evt = send_btn.click(
                on_send, [main_msg, chatbot, is_responding],
                [main_msg, send_btn, stop_btn, is_responding, chatbot],
            )
            # 回车发送
            submit_evt = main_msg.submit(
                on_send, [main_msg, chatbot, is_responding],
                [main_msg, send_btn, stop_btn, is_responding, chatbot],
                cancels=[send_evt],
            )
            # 停止按钮 click → on_stop + 取消正在运行的请求
            stop_btn.click(
                fn=on_stop,
                inputs=[is_responding],
                outputs=[main_msg, send_btn, stop_btn, is_responding, chatbot],
                cancels=[send_evt, submit_evt],
            )

            # 新建聊天：保存 + 清空 + 刷新历史 Tab
            def on_new_chat(history, responding):
                global _latest_msgs
                print(f"[new_chat] responding={responding}, history={history}", flush=True)
                if responding:
                    return gr.update(), gr.update(), responding, gr.update()
                if history and len(history) >= 2:
                    try:
                        sid = save_history_session(list(history), _current_route, False)
                        print(f"[new_chat] saved session: {sid}", flush=True)
                    except Exception as e:
                        print(f"[new_chat] save failed: {e}", flush=True)
                else:
                    if _latest_msgs and len(_latest_msgs) >= 2:
                        try:
                            sid = save_history_session(list(_latest_msgs), _current_route, False)
                            print(f"[new_chat] saved from _latest_msgs: {sid}", flush=True)
                        except Exception as e:
                            print(f"[new_chat] fallback save failed: {e}", flush=True)
                _latest_msgs = []
                return "", gr.update(), responding, []

            # 占位：实际 click 事件在 history_df 创建后绑定（需要刷新历史 Tab）
            # 这里只定义 on_new_chat（已在上方），不在此处绑定

            # 浏览器关闭/刷新时自动保存当前对话（用模块级缓存 _latest_msgs）
            def _on_unload():
                global _latest_msgs
                msgs = _latest_msgs
                print(f"[unload] triggered, _latest_msgs len={len(msgs) if msgs else 0}", flush=True)
                if msgs and len(msgs) >= 2:
                    try:
                        sid = save_history_session(list(msgs), _current_route, False)
                        print(f"[unload] saved session: {sid}", flush=True)
                        _latest_msgs = []  # 保存后清空，防止下次启动时再被保存
                    except Exception as e:
                        print(f"[unload] save failed: {e}", flush=True)
                return None
            demo.unload(fn=_on_unload)

            # 推荐问题：每个按钮 click 时同样走 on_send
            for b in example_buttons:
                b.click(
                    fn=lambda text, responding: text if not responding else gr.update(),
                    inputs=[b, is_responding],
                    outputs=[main_msg],
                ).then(
                    on_send, [main_msg, chatbot, is_responding],
                    [main_msg, send_btn, stop_btn, is_responding, chatbot],
                )

        with gr.Tab("🕘 历史记录"):
            def _flatten_text(v):
                if isinstance(v, str): return v
                if isinstance(v, list):
                    return " ".join(p.get("text","") if isinstance(p, dict) else str(p) for p in v)
                if isinstance(v, dict): return v.get("text","")
                return str(v or "")
            def _short(a, n=120):
                s = _flatten_text(a)
                return (s[:n]).replace("\n"," ") + ("..." if len(s) > n else "")

            # ─── 搜索栏 ───
            with gr.Row(elem_classes="search-bar-row"):
                search_id = gr.Textbox(placeholder="ID 搜索（模糊）", scale=1, min_width=120, container=False, elem_classes="search-field")
                search_q = gr.Textbox(placeholder="问题搜索（模糊）", scale=2, min_width=160, container=False, elem_classes="search-field")
                search_date_start = gr.DateTime(type="string", include_time=False, show_label=False, scale=1, min_width=130, elem_classes="search-field search-date")
                search_date_end = gr.DateTime(type="string", include_time=False, show_label=False, scale=1, min_width=130, elem_classes="search-field search-date")
                search_btn = gr.Button("🔍 搜索", variant="primary", size="sm", elem_classes="auto-width-btn")
                search_clear_btn = gr.Button("✕ 清除", variant="secondary", size="sm", elem_classes="auto-width-btn")

            with gr.Row():
                refresh_h = gr.Button("↺ 刷新", variant="secondary", size="sm")
                fav_h_btn = gr.Button("☆ 收藏选中", variant="secondary", size="sm")
                clear_h   = gr.Button("✕ 清空所有", variant="stop", size="sm")
                del_h_btn = gr.Button("✕ 删除选中", variant="stop", size="sm")
                status_h  = gr.Markdown("")
            history_df = gr.Dataframe(
                headers=["", "ID", "时间", "问题", "回答摘要", "路由"],
                datatype=["bool", "str", "str", "str", "str", "str"],
                interactive=[True, False, False, False, False, False],
                column_widths=["5%", "22%", "16%", "21%", "26%", "10%"],
                wrap=False,
                elem_classes="history-table",
                row_count=15,
                max_height=480,
            )
            def load_h(pid=None, pq=None, pds=None, pde=None):
                r = get_history()
                # 客户端过滤：综合搜索 ID + 问题 + 时间范围
                if pid:
                    r = [rr for rr in r if pid.lower() in rr.get("id","").lower()]
                if pq:
                    r = [rr for rr in r if pq.lower() in _flatten_text(rr.get("question","")).lower()]
                if pds:
                    r = [rr for rr in r if rr.get("timestamp","") >= pds]
                if pde:
                    r = [rr for rr in r if rr.get("timestamp","")[:10] <= pde]
                rows = [[
                    False,
                    rr.get("id",""),
                    rr.get("timestamp",""),
                    _flatten_text(rr.get("question",""))[:80],
                    _short(rr.get("answer","")),
                    "🌐" if rr.get("has_web") else "📚",
                ] for rr in r]
                status = f"共 {len(r)} 条记录{'（已筛选）' if any([pid,pq,pds,pde]) else ''}"
                return rows, status
            refresh_h.click(lambda: load_h(), outputs=[history_df, status_h])
            search_btn.click(
                fn=load_h,
                inputs=[search_id, search_q, search_date_start, search_date_end],
                outputs=[history_df, status_h],
            )
            def _search_clear():
                rows, status = load_h()
                return "", "", None, None, rows, status
            search_clear_btn.click(
                fn=_search_clear,
                inputs=None,
                outputs=[search_id, search_q, search_date_start, search_date_end, history_df, status_h],
            )
            def do_clear_h():
                n = count_history()
                clear_all_history()
                return [], f"✅ 已清空（{n} 条）"
            clear_h.click(fn=do_clear_h, outputs=[history_df, status_h])
            def do_del_selected_h(data):
                # data 可能是 pandas DataFrame、dict 或 list
                rows = []
                if data is None:
                    return gr.update(), "⚠️ 表格数据异常"
                if hasattr(data, "values"):  # pandas DataFrame
                    rows = list(data.values.tolist())
                elif isinstance(data, dict) and "data" in data:
                    rows = data["data"]
                elif isinstance(data, list):
                    rows = data
                if not rows:
                    return gr.update(), "⚠️ 未勾选任何记录"
                sids = [str(row[1]) for row in rows if row and row[0]]
                if not sids:
                    return gr.update(), "⚠️ 未勾选任何记录"
                for sid in sids:
                    delete_history(sid)
                r = get_history()
                new_rows = [[False, rr.get("id",""), rr.get("timestamp",""), _flatten_text(rr.get("question",""))[:80], _short(rr.get("answer","")), "🌐" if rr.get("has_web") else "📚"] for rr in r]
                return new_rows, f"✅ 已删除 {len(sids)} 条 | 剩余 {len(r)} 条"
            del_h_btn.click(fn=do_del_selected_h, inputs=[history_df], outputs=[history_df, status_h])
            def do_fav_selected_h(data):
                rows = []
                if data is None: return gr.update(), "⚠️ 表格数据异常"
                if hasattr(data, "values"): rows = list(data.values.tolist())
                elif isinstance(data, dict) and "data" in data: rows = data["data"]
                elif isinstance(data, list): rows = data
                if not rows: return gr.update(), "⚠️ 未勾选任何记录"
                sids = [str(row[1]) for row in rows if row and row[0]]
                if not sids: return gr.update(), "⚠️ 未勾选任何记录"
                ok = 0
                for sid in sids:
                    sess = get_session(sid)
                    if sess:
                        add_favorite(sid)
                        ok += 1
                r = get_history()
                new_rows = [[False, rr.get("id",""), rr.get("timestamp",""), _flatten_text(rr.get("question",""))[:80], _short(rr.get("answer","")), "🌐" if rr.get("has_web") else "📚"] for rr in r]
                return new_rows, f"✅ 已收藏 {ok} 条 | 历史共 {len(r)} 条"
            fav_h_btn.click(fn=do_fav_selected_h, inputs=[history_df], outputs=[history_df, status_h])
            # 新建聊天后自动刷新历史 Tab
            new_chat_btn.click(
                fn=on_new_chat,
                inputs=[chatbot, is_responding],
                outputs=[main_msg, main_msg, is_responding, chatbot],
            ).then(
                fn=load_h,
                inputs=None,
                outputs=[history_df, status_h],
            )
            demo.load(load_h, outputs=[history_df, status_h])

        with gr.Tab("☆ 收藏"):
            with gr.Row():
                refresh_f = gr.Button("↺ 刷新", variant="secondary", size="sm")
                del_f_btn = gr.Button("✕ 删除选中", variant="stop", size="sm")
                status_f  = gr.Markdown("")
            fav_df = gr.Dataframe(
                headers=["", "ID", "时间", "问题", "回答摘要", "路由"],
                datatype=["bool", "str", "str", "str", "str", "str"],
                interactive=[True, False, False, False, False, False],
                column_widths=["5%", "22%", "16%", "21%", "26%", "10%"],
                wrap=True,
                elem_classes="history-table",
                row_count=15,
                max_height=560,
            )
            def load_f():
                r = get_favorites()
                rows = [[
                    False,
                    rr.get("id",""),
                    rr.get("timestamp",""),
                    _flatten_text(rr.get("question",""))[:80],
                    _short(rr.get("answer",""), 80),
                    "🌐" if rr.get("has_web") else "📚",
                ] for rr in r]
                return rows, f"共 {len(r)} 条收藏"
            refresh_f.click(load_f, outputs=[fav_df, status_f])
            def do_del_selected_f(data):
                rows = []
                if data is None:
                    return gr.update(), "⚠️ 表格数据异常"
                if hasattr(data, "values"):
                    rows = list(data.values.tolist())
                elif isinstance(data, dict) and "data" in data:
                    rows = data["data"]
                elif isinstance(data, list):
                    rows = data
                if not rows:
                    return gr.update(), "⚠️ 未勾选任何记录"
                sids = [str(row[1]) for row in rows if row and row[0]]
                if not sids:
                    return gr.update(), "⚠️ 未勾选任何记录"
                for sid in sids:
                    remove_favorite(sid)
                r = get_favorites()
                new_rows = [[False, rr.get("id",""), rr.get("timestamp",""), _flatten_text(rr.get("question",""))[:80], _short(rr.get("answer","")), "🌐" if rr.get("has_web") else "📚"] for rr in r]
                return new_rows, f"✅ 已删除 {len(sids)} 条 | 剩余 {len(r)} 条"
            del_f_btn.click(fn=do_del_selected_f, inputs=[fav_df], outputs=[fav_df, status_f])
            demo.load(load_f, outputs=[fav_df, status_f])

        with gr.Tab("⚙ 详情管理"):
            with gr.Row():
                sid = gr.Textbox(placeholder="输入完整 ID 后点击查看", scale=4, min_width=300)
                view_btn = gr.Button("◎ 查看", variant="primary", size="sm")
                fav_btn  = gr.Button("☆ 收藏", variant="secondary", size="sm")
                del_btn  = gr.Button("✕ 删除", variant="stop", size="sm")
            detail = gr.Markdown()

            def view(s):
                if not s.strip(): return "⚠️ 请输入 ID"
                sess = get_session(s.strip())
                if not sess: return "❌ 未找到该对话"
                route_info = "🌐 网络补充" if sess.get("has_web") else "📚 本地知识库"
                msgs = sess.get("messages") or []
                if msgs:
                    parts = []
                    for m in msgs:
                        role = m.get("role") if isinstance(m, dict) else ""
                        content = m.get("content","") if isinstance(m, dict) else str(m)
                        # 跳过中间状态行
                        if role != "user" and content.strip().startswith(("🔍", "🚀", "📚")):
                            continue
                        icon = "🙋" if role == "user" else ""
                        label = f"{icon} **{content}**" if role == "user" else content
                        parts.append(label)
                    detail = "\n\n".join(parts)
                else:
                    detail = f"**问题:** {_flatten_text(sess.get('question',''))}\n\n---\n**回答:**\n{_flatten_text(sess.get('answer',''))}"
                return f"**时间:** {sess.get('timestamp','')}  **路由:** {route_info}\n\n---\n{detail}"

            def fav(s):
                return "✅ 已收藏" if add_favorite(s.strip()) else "❌ 失败"

            def dele(s):
                return "✅ 已删除" if delete_history(s.strip()) else "❌ 失败"

            view_btn.click(view, sid, detail)
            fav_btn.click(fav, sid, detail)
            del_btn.click(dele, sid, detail)

    return demo


# ─── 启动 ──────────────────────────────────────────────────
def check_index():
    from elasticsearch import Elasticsearch
    from config import ES_HOST, ES_USER, ES_PASS, INDEX_NAME
    es = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS), verify_certs=False, ssl_show_warn=False)
    if not es.indices.exists(index=INDEX_NAME):
        print(f"⚠️  索引 '{INDEX_NAME}' 不存在，请先运行: python3 indexer.py")
        return False
    print(f"📂  索引 '{INDEX_NAME}' 中有 {es.count(index=INDEX_NAME)['count']} 条文档")
    return True


# ─── CSS：合并卡片样式 ────────────────────────────────────
CSS = """
/* ── 标题 ── */
.chat-title { text-align: center !important; margin: 8px 0 8px !important; padding-top: 4px !important; }
.chat-title h1 {
  font-size: 26px !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
  background: linear-gradient(90deg, #2563eb 0%, #6366f1 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
  display: inline-flex !important;
  align-items: baseline !important;
  gap: 8px !important;
  padding: 0 4px !important;
  margin: 2px 0 !important;
  line-height: 1.5 !important;
}
.chat-title h1 span {
  font-size: 13px !important;
  font-weight: 400 !important;
  color: #94a3b8 !important;
  letter-spacing: 1.5px !important;
  -webkit-text-fill-color: #94a3b8 !important;
  background: none !important;
  vertical-align: middle !important;
}

/* ── 整体卡片 ── */
.chat-card {
  border: 1px solid #e5e7eb !important;
  border-radius: 14px !important;
  overflow: hidden;
  background: #fff !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
  transition: box-shadow .2s ease;
  position: relative !important;
}
.chat-card:focus-within {
  box-shadow: 0 6px 28px rgba(51,112,255,0.12) !important;
  border-color: #c7d2fe !important;
}

/* ── 上方 Chatbot 区 ── */
.chat-card-bot {
  border: none !important;
  border-radius: 0 !important;
  margin: 0 !important;
  background: transparent !important;
}

/* ── 下方输入区 ── */
.chat-card-input {
  margin: 0 !important;
  padding: 10px 14px !important;
  border-top: 1px solid #f0f0f3 !important;
  background: #fafbfc !important;
  align-items: flex-end !important;
  gap: 0 !important;
}

/* ── 输入框 ── */
.chat-card-text { border: none !important; box-shadow: none !important; background: transparent !important; flex: 1 !important; }
.chat-card-text textarea {
  border: none !important;
  box-shadow: none !important;
  padding: 8px 10px !important;
  background: #fff !important;
  border-radius: 10px !important;
  font-size: 14px !important;
  line-height: 1.6 !important;
  color: #1f2937 !important;
  resize: none !important;
  transition: background .2s ease, box-shadow .2s ease !important;
}
.chat-card-text textarea::placeholder { color: #9ca3af !important; font-size: 14px !important; }
.chat-card-text textarea:focus {
  background: #fff !important;
  box-shadow: 0 0 0 2px rgba(51,112,255,0.15) !important;
  outline: none !important;
}

/* ── 发送按钮 ── */
.chat-card-send {
  min-width: 42px !important;
  max-width: 42px !important;
  height: 42px !important;
  border-radius: 12px !important;
  padding: 0 !important;
  font-size: 20px !important;
  font-weight: 400 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin-left: 10px !important;
  flex-shrink: 0 !important;
  border: none !important;
  background: linear-gradient(135deg, #2563eb 0%, #6366f1 100%) !important;
  color: #fff !important;
  box-shadow: 0 3px 12px rgba(99,102,241,0.25) !important;
  transition: all .2s ease !important;
  position: relative !important;
  overflow: hidden !important;
}
.chat-card-send::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 60%);
  pointer-events: none;
}
.chat-card-send:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
  opacity: 1 !important;
}
.chat-card-send:active {
  transform: scale(0.95) !important;
  box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
}
.chat-card-send .svelte-c3ai4c {  /* gradio button inner text */
  position: relative;
  z-index: 1;
  transform: rotate(-45deg);
  display: inline-block;
  line-height: 1;
}

/* ── 「停止」状态：红底 + 不旋转 ── */
.chat-card-send-stop {
  background: linear-gradient(135deg, #ef4444 0%, #f97316 100%) !important;
  box-shadow: 0 3px 12px rgba(239,68,68,0.30) !important;
}
.chat-card-send-stop:hover {
  box-shadow: 0 6px 20px rgba(239,68,68,0.40) !important;
}
.chat-card-send-stop .svelte-c3ai4c {
  transform: none !important;
  font-size: 16px !important;
}

/* ── 输入框禁用时的视觉 ── */
.chat-card-text textarea[disabled] {
  background: #f3f4f6 !important;
  color: #9ca3af !important;
  cursor: not-allowed !important;
}

/* ── 历史记录 / 收藏 表格 ── */
.history-table { border-radius: 10px !important; overflow: hidden; }
.history-table table { border-collapse: separate !important; border-spacing: 0 !important; }
.history-table thead th {
  background: #f8fafc !important;
  color: #475569 !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  text-align: left !important;
  border-bottom: 1px solid #e2e8f0 !important;
  padding: 10px 12px !important;
}
.history-table tbody td {
  font-size: 13px !important;
  color: #334155 !important;
  padding: 8px 12px !important;
  border-bottom: 1px solid #f1f5f9 !important;
  vertical-align: top !important;
  word-break: break-word !important;
}
/* ID 列：等宽字体 + 不换行 + 可复制 */
.history-table tbody td:first-child {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace !important;
  font-size: 12px !important;
  color: #1e40af !important;
  white-space: nowrap !important;
  user-select: all !important;
  cursor: text !important;
}
.history-table thead th:first-child { width: 24% !important; }
.history-table tbody tr:hover td { background: #f8fafc !important; }

/* ── 历史/收藏 Checkbox 选中列表 ── */
.history-checkbox { margin-bottom: 8px !important; }
.history-checkbox .wrap { max-height: 80px; overflow-y: auto; display: flex; flex-wrap: wrap; gap: 4px; padding: 4px; }
.history-checkbox label { font-size: 12px !important; padding: 2px 6px !important; border-radius: 6px !important; border: 1px solid #e2e8f0 !important; background: #f8fafc !important; }

/* ── 顶部全局标题 ── */
.chat-title { margin-bottom: 6px !important; }

/* ── 新建对话按钮（覆盖掉 .new-chat-btn-row 的旧定义） ── */
button.new-chat-btn {
  position: relative !important;
  z-index: 10 !important;
  display: inline-flex !important;
  border: 1px solid #e2e8f0 !important;
  background: #fff !important;
  color: #64748b !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  padding: 4px 12px !important;
  height: auto !important;
  width: auto !important;
  min-width: 0 !important;
  flex: 0 0 auto !important;
  transition: all .15s ease !important;
  line-height: 1.4 !important;
  cursor: pointer !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
  margin-left: auto !important;
  margin-bottom: 4px !important;
}
button.new-chat-btn:hover {
  border-color: #2563eb !important;
  background: #eff6ff !important;
  color: #2563eb !important;
}
/* ── Tab 导航 ── */
.tabs { margin-bottom: 6px !important; position: relative !important; }
.tab-nav { display: flex !important; align-items: stretch !important; }
.tab-nav button {
  background: transparent !important;
  border: none !important;
  color: #64748b !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  padding: 8px 16px !important;
  border-radius: 8px 8px 0 0 !important;
  margin-right: 2px !important;
  transition: all .15s ease !important;
  position: relative !important;
}
.tab-nav button:hover {
  background: #f1f5f9 !important;
  color: #1e293b !important;
}
.tab-nav button.selected {
  background: #fff !important;
  color: #2563eb !important;
  font-weight: 600 !important;
  box-shadow: inset 0 -2px 0 #2563eb !important;
}

/* 隐藏 DateTime 的 label-content（"Date" 文字行） */
.search-date .label-content { display: none !important; height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important; }

/* ── 搜索栏整行：各输入框挨着放，间隔 6px ── */
.search-bar-row { gap: 6px !important; }

/* ── 推荐问题按钮 ── */
.examples-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.example-chip {
  border: 1px solid #e2e8f0 !important;
  background: #f8fafc !important;
  color: #475569 !important;
  font-size: 12px !important;
  border-radius: 20px !important;
  padding: 4px 14px !important;
  height: auto !important;
  width: auto !important;
  min-width: 0 !important;
  max-width: none !important;
  flex: 0 0 auto !important;
  display: inline-flex !important;
  transition: all .15s ease !important;
}
.example-chip:hover {
  border-color: #2563eb !important;
  background: #eff6ff !important;
  color: #2563eb !important;
}

/* ── 各 Tab 内操作按钮栏 ── */
button[variant="secondary"] {
  font-size: 13px !important;
  border-radius: 8px !important;
  border: 1px solid #e2e8f0 !important;
  background: #fff !important;
  color: #475569 !important;
  transition: all .15s ease !important;
}
button[variant="secondary"]:hover {
  border-color: #93c5fd !important;
  background: #f8fafc !important;
  color: #1e40af !important;
}
button[variant="stop"] {
  font-size: 13px !important;
  border-radius: 8px !important;
  background: #fef2f2 !important;
  color: #dc2626 !important;
  border-color: #fecaca !important;
  transition: all .15s ease !important;
}
button[variant="stop"]:hover {
  background: #fee2e2 !important;
  color: #b91c1c !important;
}
button[variant="primary"] {
  border-radius: 8px !important;
  font-size: 13px !important;
  transition: all .15s ease !important;
}

/* ── 输入框内发送栏 ── */
.chat-card-input { gap: 4px !important; }

/* ── 搜索按钮自适应宽度 ── */
.auto-width-btn { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; max-width: none !important; display: inline-flex !important; align-self: stretch !important; }

/* ── 搜索栏输入框（与按钮等高） ── */
.search-field { align-self: stretch !important; padding: 0 !important; margin: 0 !important; height: 38px !important; min-height: 38px !important; border: none !important; background: transparent !important; box-shadow: none !important; }
.search-field > label,
.search-field label.svelte-1hguek3 {
  display: contents !important;
}
.search-field .wrap.hide { display: none !important; }
.search-field span.sr-only { display: none !important; }
.search-field textarea, .search-field input, .search-field .wrap, .search-field .svelte-1t1myr {
  height: 38px !important;
  min-height: 38px !important;
  font-size: 13px !important;
  border-radius: 8px !important;
  border: 1px solid #e2e8f0 !important;
  padding: 6px 10px !important;
  box-sizing: border-box !important;
  width: 100% !important;
}
.search-field .wrap,
.search-field .form,
.search-field > div,
.search-field .svelte-1t1myr,
.search-field .svelte-1cllns0,
.search-field .svelte-4k8s31,
.search-field .svelte-wuhayf,
.search-field.block,
.search-field.svelte-1ih1fbe {
  padding: 0 !important;
  margin: 0 !important;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  min-height: 38px !important;
}
.search-field,
.search-field.contain,
.search-field > .svelte-1ih1fbe,
.search-field > .block-container {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
.auto-width-btn { height: 38px !important; min-height: 38px !important; padding: 0 14px !important; }
.search-date .fp-date-bound { display: none !important; }
.search-date { padding: 0 !important; gap: 0 !important; margin: 0 !important; align-self: stretch !important; }
.search-date > label { display: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }
.search-date > div, .search-date > .container, .search-date .svelte-1flm9jx,
.search-date .gradio-container, .search-date .gradio-container > *,
.search-date .svelte-1plpy97, .search-date .block {
  padding: 0 !important;
  gap: 0 !important;
  margin: 0 !important;
  min-height: 38px !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.search-date input {
  height: 38px !important;
  min-height: 38px !important;
  font-size: 13px !important;
  border-radius: 8px !important;
  border: 1px solid #e2e8f0 !important;
  padding: 6px 10px !important;
  box-sizing: border-box !important;
  width: 100% !important;
  background: #fff !important;
}

/* flatpickr 日历 */
.fp-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 9998; }
.fp-calendar {
  position: absolute; z-index: 9999;
  background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
  box-shadow: 0 12px 32px rgba(0,0,0,0.12); padding: 10px;
  font-size: 13px; width: 240px; font-family: -apple-system, sans-serif;
}
.fp-calendar .fp-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.fp-calendar .fp-nav { display: flex; gap: 6px; }
.fp-calendar .fp-nav button {
  border: 1px solid #e2e8f0; background: #fff; color: #475569;
  border-radius: 6px; padding: 2px 8px; cursor: pointer; font-size: 13px;
}
.fp-calendar .fp-nav button:hover { background: #eff6ff; color: #2563eb; border-color: #2563eb; }
.fp-calendar select { border: 1px solid #e2e8f0; border-radius: 6px; padding: 2px 4px; font-size: 13px; }
.fp-calendar .fp-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; }
.fp-calendar .fp-dow { text-align: center; color: #94a3b8; font-size: 11px; padding: 4px 0; }
.fp-calendar .fp-day {
  text-align: center; padding: 5px 0; cursor: pointer; border-radius: 6px;
  color: #1f2937; font-size: 12px;
}
.fp-calendar .fp-day:hover { background: #eff6ff; color: #2563eb; }
.fp-calendar .fp-day.other { color: #cbd5e1; }
.fp-calendar .fp-day.today { font-weight: 700; box-shadow: inset 0 0 0 1px #2563eb; }
.fp-calendar .fp-day.selected { background: #2563eb; color: #fff; }
"""


def main():
    print("=" * 60)
    print("  保险文档智能问答系统")
    print("=" * 60)
    from config import JINA_API_KEY
    print(f"{'✅' if JINA_API_KEY else '⚠️'} JINA_API_KEY {'已设置' if JINA_API_KEY else '未设置'}")
    check_index()
    app = create_app()
    print("\n🚀 启动服务: http://127.0.0.1:7860")
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True, theme=gr.themes.Soft(), css=CSS, js="""
function() {
    setTimeout(function() {
        var newBtn = document.querySelector('button.new-chat-btn');
        var card = document.querySelector('.chat-card');
        if (newBtn && card && card.parentNode) {
            // 用一个 right-aligned wrapper 包住按钮，插到 chat-card 上方
            var wrap = document.createElement('div');
            wrap.style.textAlign = 'right';
            wrap.style.margin = '0 4px 4px 0';
            wrap.appendChild(newBtn);
            card.parentNode.insertBefore(wrap, card);
        }
    }, 800);

    // （日期选择器改用 gr.DateTime 原生组件，无需 JS 注入）
    // 给两个日期输入框加 placeholder
    setTimeout(function() {
        document.querySelectorAll('.search-date input').forEach(function(inp, i){
            inp.placeholder = (i === 0) ? '起始日期' : '结束日期';
        });
    }, 1500);

    // 删掉搜索栏中的空白条（容器 label/info 行、外层 fill 边框等）
    function stripWhiteBars() {
        // DateTime 的 label 容器、外层 margin-fill 产生的白条
        document.querySelectorAll('.search-date, .search-field').forEach(function(el) {
            // 删除没内容的空标签（不删有文字的，否则会删除搜索框本身）
            el.querySelectorAll('label, .info-wrap, .info, .s-r-select').forEach(function(n) {
                if (!n.textContent || !n.textContent.trim()) n.remove();
            });
            // 把所有"没内容、没input、仅占位的 div"清掉
            el.querySelectorAll(':scope > div').forEach(function(d) {
                if (!d.querySelector('input, textarea, button')) {
                    if (!d.textContent || !d.textContent.trim()) d.remove();
                }
            });
        });
    }
    setTimeout(stripWhiteBars, 800);
    setTimeout(stripWhiteBars, 2000);
    // 每次切换到历史 Tab 后再清理一次（Gradio 重新 hydrate）
    var obs = new MutationObserver(function(){ stripWhiteBars(); });
    obs.observe(document.body, {childList:true, subtree:true});
}
""")


if __name__ == "__main__":
    main()