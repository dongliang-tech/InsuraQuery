# InsuraQuery — AI 保险文档智能问答系统

**保险智答** 是一个基于 **Elasticsearch** + **向量检索** + **大语言模型** 的保险条款智能问答系统。它能够：

- 📥 将保险文档（TXT/PDF）分块后使用 **Jina Embeddings** 向量化，存入 Elasticsearch
- 🔍 混合检索（BM25 + 向量相似度 + RRF 融合）召回最相关的文档片段
- 🧠 通过 LLM 路由判断本地知识是否足够回答，不足时自动通过 **Tavily** 搜索互联网补充
- 💬 支持流式对话、收藏历史记录、推荐问题快捷入口
- ⚡ 检测到复杂（多文档对比/总结类）问题时自动切换为大上下文并行问答模式，一次返回完整结果

## 系统架构

```
用户输入
   │
   ▼
┌─────────────────────────────┐
│  问题复杂度检测              │  ← 长度 > 20 或含"比较/区别/所有"等关键词
│                             │
│  简单:  普通搜索             │     复杂:  并行问答
│   ┌───────────────┐         │     ┌────────────────┐
│   │ BM25+Vec → RRF│         │     │ search_local(K) │
│   │ route_decision│         │     │ 取 TOP-15 chunks │
│   │ LLM 判断路由  │         │     │ 大 context 一次  │
│   ├─ local: 仅本地│         │     │ LLM 回答         │
│   ├─ web: +Tavily│         │     └────────────────┘
│   └───────┬───────┘         │
└───────────┼─────────────────┘
            ▼
        LLM 流式回答
        (Gradio Web UI)
```

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| **UI** | Gradio 6.x | 响应式 Web UI，聊天 + 历史 + 收藏 Tab 布局 |
| **搜索引擎** | Elasticsearch 8.x | 混合索引：`text` (BM25) + `dense_vector` (cosine) |
| **向量模型** | Jina Embeddings v3 | `jina-embeddings-v3`, 1024 维, `retrieval.passage` / `retrieval.query` |
| **LLM 推理** | OpenAI 兼容 API | 可对接任意 `/v1/chat/completions` 接口（DeepSeek / OpenAI / vLLM 等） |
| **网络搜索** | Tavily API | 仅当本地知识不足时自动触发，需在 `.env` 中配置 API Key |
| **分块策略** | LangChain RecursiveCharacterTextSplitter | `chunk_size=512`, `overlap=128`, 按 `\n\n` → `\n` → 标点递归分割 |
| **检索融合** | RRF (Reciprocal Rank Fusion) | BM25 + Vector 各自排名按 `1/(k+rank+1)` 融合, `k=60` |
| **PDF 处理** | pdfplumber | 常规 PDF 文本提取 |

## 功能详解

### 1. 混合搜索（Hybrid Search）

本地搜索同时使用两种检索方式，通过 RRF 融合排名：

```
用户查询
   ├── BM25 全文检索 → title^3 + content
   ├── Vector 向量检索 → Jina Embedding (1024维)
   └── RRF 融合 → 最终 TOP-K
```

- 标题字段在 BM25 中加权 3 倍，提升文档标题命中权重
- RRF 常数 `k=60` 平衡两种排序方法
- 向量搜索使用 `num_candidates=100` 保证召回率

### 2. 智能路由（Route）

本地搜索完毕后，将检索结果摘要提交给 LLM，由 LLM 判断：

| 路由结果 | 触发条件 | 行为 |
|---|---|---|
| **local** | 本地知识库明确包含答案 | 直接使用本地结果回答 |
| **web** | 本地内容不相关或不完整 | 调用 Tavily API 搜索互联网补充 |

路由判断使用 `temperature=0` 确保决策的确定性。

### 3. 复杂问题检测

自动检测用户问题是否复杂，使用两种模式：

| 模式 | 触发条件 | 检索策略 | LLM 调用 | 适用场景 |
|---|---|---|---|---|
| **普通搜索** | 简单问题 | BM25 + Vector + RRF (K=10) | 流式生成 | 单点事实查询 |
| **并行问答** | 长度 > 20 或含比较/区别/总结等关键词 | 搜索 TOP-15 chunks | 大 context 一次非流式 (fast model) | 多文档对比/总结 |

复杂问题检测关键词：`比较`、`区别`、`各`、`分别`、`所有`、`总结`、`汇总`

### 4. 会话管理

- 每次问答自动保存到 `.chat_data/history.json`
- 多轮对话自动聚合为一个 session 保存
- 支持收藏 / 取消收藏 / 删除 / 清空
- 浏览器关闭/刷新时自动保存当前对话
- 历史记录支持按 ID、问题内容、日期范围筛选

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
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. 启动 Elasticsearch

请自行部署 Elasticsearch 8.x，确保服务可访问。推荐配置：

- **地址**: `https://localhost:9200`
- **用户**: `elastic`
- **密码**: 在部署时设置的密码

部署方式（任选其一）：
- **Docker**: `docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "ELASTIC_PASSWORD=your_password" docker.elastic.co/elasticsearch/elasticsearch:8.19.0`
- **官方安装**: 参考 [Elasticsearch 官方文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的密钥（详见 [API 密钥获取](#api-密钥获取)）：

| 变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `ES_HOST` | ✅ | `https://localhost:9200` | ES 地址 |
| `ES_USER` | ✅ | `elastic` | ES 用户名 |
| `ES_PASS` | ✅ | — | ES 密码 |
| `ES_INDEX` | 可选 | `knowledge_base` | ES 索引名 |
| `JINA_API_KEY` | ✅ | — | Jina Embeddings API Key |
| `JINA_MODEL` | 可选 | `jina-embeddings-v3` | 向量模型名 |
| `JINA_EMBED_DIMS` | 可选 | `1024` | 向量维度 |
| `TAVILY_API_KEY` | 可选 | — | Tavily 网络搜索 API Key |
| `LLM_API_BASE` | ✅ | `https://api.openai.com/v1` | LLM API 地址 |
| `LLM_API_KEY` | ✅ | — | LLM API 密钥 |
| `LLM_MODEL` | ✅ | `deepseek-v4-pro` | 问答主模型 |
| `LLM_MODEL_FAST` | 可选 | `deepseek-v4-flash` | 大上下文场景的快速模型 |

### 4. 放入文档并建立索引

```bash
mkdir -p docs
# 复制你的 .txt / .pdf 文件到 docs/ 目录下

python3 indexer.py
```

索引器执行流程：
1. 扫描 `docs/` 下所有 `.txt` 和 `.pdf` 文件
2. 按 `\n\n` → `\n` → 标点递归分割为 chunk（`chunk_size=512`, `overlap=128`）
3. 调用 Jina Embeddings API 批量向量化（每批 100 条）
4. 创建 ES 索引 `knowledge_base`（如果不存在）并批量写入 chunks + 向量（每批 50 条）

> **文件名规范**：建议使用 `序号-标题` 格式（如 `01-雇主责任险条款.txt`），系统会自动提取标题。

### 5. 启动 Web UI

```bash
python3 app.py
```

浏览器访问 **http://127.0.0.1:7860**

首次启动时会自动检查 ES 索引是否存在，如果不存在会提示先运行 `indexer.py`。

## 项目结构

```
InsuraQuery/
├── app.py                  # Gradio UI 主文件（聊天 / 历史 / 收藏 / 详情管理）
├── chunk_strategy.py       # 文本分块策略（RecursiveCharacterTextSplitter）
├── chat_manager.py         # 会话和收藏持久化（JSON 文件读写）
├── config.py               # 配置模块（环境变量 + 参数常量）
├── indexer.py              # 文档索引器（扫描 docs/ → 分块 → 向量化 → ES 入库）
├── parallel_qa.py          # 并行问答模块（大上下文模式，一次 LLM 回答）
├── search.py               # 混合搜索模块（BM25 + Vector + RRF + Route + Tavily）
├── docker-compose.yml      # 一键启动本地 ES 8.x
├── .env.example            # 环境变量模板（复制为 .env 后填入密钥）
├── .gitignore              # Git 忽略规则
├── requirements.txt        # Python 依赖项
└── docs/                   # 文档目录（用户自行放入 .txt / .pdf）
```

## API 密钥获取

| 服务 | 注册地址 | 用途 |
|---|---|---|
| Jina Embeddings | https://jina.ai/embeddings/ | 文本向量化（免费额度 1M tokens） |
| Tavily Search | https://tavily.com/ | 网络搜索补充（免费每月 1000 次） |
| Elasticsearch | — | 本地 Docker 部署（免费开源） |
| LLM (DeepSeek) | https://platform.deepseek.com/ | 问答推理（按量计费） |

## 许可

MIT License — 详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 或 Pull Request！