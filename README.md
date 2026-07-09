# InsuraQuery 开源说明

> ⚠️ 本项目目前不托管开源语料文档。
> 要运行索引流程，请在项目根目录下创建 `docs/` 文件夹，放入你的 `.txt` 或 `.pdf` 文件。
>
> 文件名命名规范（举例）：
> ```
> docs/
> ├── 01-雇主责任险条款.txt
> ├── 02-工伤保险条例.txt
> └── 03-商业综合责任险.txt
> ```
> 文件名开头如为 `序号-标题` 格式，系统会自动提取标题（如 `02-工伤保险条例.txt` →「工伤保险条例」）。

---

# InsuraQuery — AI 保险文档智能问答系统

**保险智答** 是一个基于 **Elasticsearch** + **向量检索** + **大语言模型** 的保险条款智能问答系统。它能够：

- 📥 将保险文档（TXT/PDF）分块后使用 **Jina Embeddings** 向量化，存入 Elasticsearch
- 🔍 混合检索（BM25 + 向量相似度 + 手动 RRF）召回最相关的文档片段
- 🧠 通过 LLM 路由判断本地知识是否足够回答，不足时自动通过 **Tavily** 搜索互联网补充
- 💬 支持流式对话、收藏历史记录、推荐问题快捷入口
- ⚡ 检测到复杂（多文档对比/总结类）问题时自动切换为大上下文并行问答模式，一次返回完整结果

## 系统架构

```
用户输入
   │
   ▼
┌─────────────────────────────┐
│  问题长度 / 关键词检测       │  ←> 20 字 或含"比较/区别/所有"等
│                             │
│  简单:  普通搜索             │    复杂:  并行问答
│   ┌───────────────┐         │     ┌────────────────┐
│   │ BM25+Vec → RRF│         │     │ search_local()  │
│   │ route_decision│         │     │ 取 TOP-15 chunks │
│   │ LLM 判断路由  │         │     │ 大 context 一次  │
│   ├─ local: 仅本地│         │     │ LLM 回答 (fast) │
│   ├─ web: +Tavily│         │     └────────────────┘
│   └───────┬───────┘         │
└───────────┼─────────────────┘
            ▼
        LLM 流式回答
        (Gradio UI)
```

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| **UI** | Gradio 6.x | 响应式 Web UI，聊天 + 历史 + 收藏 Tab 布局 |
| **搜索引擎** | Elasticsearch 8.x | 混合索引：`text` (BM25) + `dense_vector` (cosine) |
| **向量模型** | Jina Embeddings v3 | `jina-embeddings-v3`, 1024 维，`retrieval.passage` / `retrieval.query` |
| **LLM 推理** | OpenAI 兼容 API | 可对接任意 /v1/chat/completions 接口（DeepSeek / OpenAI / vLLM 等） |
| **网络搜索** | Tavily API | 仅当本地知识不足时自动触发，需在 `.env` 中配置 API Key |
| **分块策略** | LangChain RecursiveCharacterTextSplitter | `chunk_size=512`, `overlap=128`，按段 → 句 → 标点递归分割 |
| **检索融合** | 手动 RRF (Reciprocal Rank Fusion) | BM25 + Vector 各自排名按 `1/(k+rank+1)` 融合，`k=60` |
| **PDF 处理** | pdfplumber | 常规 PDF 文本提取（可选扫描件 OCR 用 pypdfium2 + RapidOCR） |

## 快速开始

### 前置条件

- Python 3.10+
- Elasticsearch 8.x（推荐通过 Docker 启动，见下方）
- 互联网连接（调用 Jina Embedding / LLM API / Tavily API 需要）

### 1. 克隆并安装依赖

```bash
git clone <your-repo-url>
cd InsuraQuery
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 启动 Elasticsearch（Docker）

```bash
docker compose up -d
```

启动后会自动创建一个单节点 ES 实例，默认密码为 `changeme_elastic_password`（在 `docker-compose.yml` 的 `ELASTIC_PASSWORD` 环境变量中设置）。

> 你可以修改 docker-compose.yml 中的密码，然后在 `.env` 中填写对应的 `ES_PASS`。

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的密钥：

| 变量 | 必填 | 获取方式 |
|---|---|---|
| `ES_PASS` | ✅ | docker-compose.yml 中设置的密码 |
| `JINA_API_KEY` | ✅ | [jina.ai](https://jina.ai/embeddings/) 注册获取 |
| `TAVILY_API_KEY` | 可选 | [tavily.com](https://tavily.com/) 注册获取 |
| `LLM_API_BASE` | ✅ | API 服务地址，默认兼容 OpenAI 格式 |
| `LLM_API_KEY` | ✅ | API 密钥 |
| `LLM_MODEL` | ✅ | 模型名称（默认 deepseek-v4-pro） |

### 4. 放入文档并建立索引

```bash
mkdir -p docs
# 复制你的 .txt / .pdf 文件到 docs/ 目录下

python3 indexer.py
```

索引器会：
1. 扫描 `docs/` 下所有 `.txt` 和 `.pdf` 文件
2. 调用 Jina Embeddings API 向量化
3. 创建 ES 索引 `knowledge_base` 并写入分块 + 向量

### 5. 启动 Web UI

```bash
python3 app.py
```

浏览器访问 **http://127.0.0.1:7860**

## 项目结构

```
InsuraQuery/
├── app.py                  # Gradio UI 主文件，含聊天 / 历史 / 收藏 Tab
├── config.py               # 配置模块（环境变量 + 参数常量）
├── indexer.py              # 文档索引：扫描 docs/ → 分块 → Jina Embed → ES 入库
├── search.py               # 混合搜索：BM25 + Vector (RRF) + Route 路由 + Tavily
├── parallel_qa.py          # 并行问答：大上下文模式，一次 LLM 回答
├── chunck_strategy.py      # 文本分块策略（RecursiveCharacterTextSplitter）
├── chat_manager.py         # 会话 / 收藏持久化（JSON 文件）
├── es_insurance_search.py  # [历史遗留] 纯文本 ES 索引导航脚本（含扫描件 OCR）
├── docker-compose.yml      # 一键启动本地 ES 8.x
├── .env.example            # 环境变量模板（复制为 .env 后填入密钥）
├── .gitignore              # Git 忽略规则
├── requirements.txt        # Python 依赖
└── docs/                   # 文档目录（用户自行放入 .txt / .pdf）
```

## 核心功能详解

### 1. 混合搜索（Hybrid Search）

```
用户查询
   ├── Jina Embeddings → 向量查询
   ├── BM25 (title^3 + content) → 全文查询
   └── 手动 RRF 融合两个排名 → 最终 TOP-K
```

- 使用 Jina Embeddings v3 的 `retrieval.query` 任务模式生成查询向量
- 标题字段在 BM25 中加权 3 倍，提升文档标题命中权重
- RRF 常数 `k=60` 平衡两种排序方法

### 2. 智能路由（Route）

本地搜索完毕后，将检索结果摘要提交给 LLM，由 LLM 判断：
- **local**：本地知识足以完整回答，直接使用本地结果
- **web**：本地知识不足，自动调用 Tavily API 搜索互联网补充

路由判断使用 `temperature=0` 确保决策的确定性。

### 3. 复杂问题检测

自动检测用户问题是否复杂（长度 > 20 字符或包含「比较」「区别」「总结」等关键词）：

| 模式 | 检索策略 | LLM 调用 | 适用场景 |
|---|---|---|---|
| **普通搜索** | BM25 + Vector + RRF (K=10) | 流式生成 | 单点事实查询 |
| **并行问答** | search_local (K=15) + 大 context | 一次非流式 (fast model) | 多文档对比/总结 |

### 4. 会话管理

- 每次问答自动保存到 `.chat_data/history.json`
- 支持收藏 / 取消收藏 / 删除 / 清空
- 浏览器关闭时自动保存当前对话
- 支持多轮对话 session 存储

## API 密钥获取地址

| 服务 | 注册地址 | 用途 |
|---|---|---|
| Jina Embeddings | https://jina.ai/embeddings/ | 文本向量化（免费额度 1M tokens） |
| Tavily Search | https://tavily.com/ | 网络搜索补充（免费每月 1000 次） |
| Elasticsearch | — | 本地部署（Docker）免费开源 |
| LLM (DeepSeek) | https://platform.deepseek.com/ | 问答推理（按量计费） |

## 许可

MIT License — 详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 或 Pull Request！
