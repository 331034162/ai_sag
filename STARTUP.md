# ai_sag — 启动与使用指南

基于 SAG 论文（SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges）实现的轻量级多跳 RAG 系统。本文档涵盖环境准备、配置、启动、接口调用全流程。

---

## 前置说明：目录结构

clone 项目后，你的目录结构如下（仓库名不一定叫 `ai_sag_git`，按你实际 clone 的目录名为准）：

```
your-repo/                  ← Git 仓库根目录，所有启动命令在此目录执行
└── ai_sag/                 ← Python 包（含 __init__.py，即项目根目录）
    ├── .env.example        ← 环境变量模板（提交到 git）
    ├── .env                ← 环境变量（从 .env.example 复制，不提交 git）
    ├── api.py              ← API 服务入口
    ├── web.py              ← Web UI 入口
    ├── requirements.txt
    ├── base/               ← 配置、日志、数据库基类
    ├── retrieval/          ← SAG 检索 & 问答引擎
    ├── ingest/             ← 入库流水线
    ├── embeddings/         ← 向量化模型
    ├── splitter/           ← 文档切分
    ├── extractor/          ← 事件 & 实体抽取
    ├── vector_store/       ← 向量库封装（Chroma）
    ├── storage/            ← MySQL 持久化
    ├── llm/                ← LLM 调用封装
    ├── loader/             ← 文件加载（md/txt/docx/pdf/xlsx）
    ├── doc_parser/         ← 文档解析（OCR、版式分析等）
    ├── static/             ← Web 前端页面
    └── logs/               ← 日志输出目录（自动创建）
```

> **重要**：`ai_sag/` 目录本身即是 Python 包，所有 `python -m ai_sag.xxx` 命令必须在上层 `your-repo/` 目录下执行。下文用 `your-repo/` 代指仓库根目录，`ai_sag/` 代指项目根目录。

---

## 一、环境准备

### 1.1 Python 环境

要求 Python 3.10+，建议使用 conda 创建独立环境：

```bash
conda create -n ai_sag python=3.10
conda activate ai_sag
```

### 1.2 安装依赖

在 `ai_sag/`（即项目根目录，含 `requirements.txt` 的目录）下执行：

```bash
pip install -r requirements.txt
```

核心依赖一览：

| 类别 | 包 | 用途 |
|------|-----|------|
| 框架 | `llama-index-core` 等 | RAG 框架底座 |
| Web | `fastapi` + `uvicorn` | HTTP 服务 |
| 向量库 | `chromadb` | 向量存储与检索 |
| 数据库 | `pymysql` | MySQL 异步连接 |
| 日志 | `loguru` | 控制台 + 文件双输出 |
| 配置 | `python-dotenv` | 从 `.env` 加载环境变量 |
| 上传 | `python-multipart` | 文件上传解析 |

### 1.3 MySQL 数据库

本地需要运行一个 MySQL 实例，并创建数据库：

```sql
CREATE DATABASE IF NOT EXISTS sag DEFAULT CHARSET utf8mb4;
```

表结构由 `MysqlStore.ensure_schema()` 在首次启动 API 时自动创建，无需手动建表。

### 1.4 Embedding 模型

需要在本地下载好 Embedding 模型。支持两种后端：

| 后端 | 模型 | 下载渠道 |
|------|------|---------|
| `bge`（推荐） | BAAI/bge-small-zh-v1.5 | HuggingFace / ModelScope |
| `qwen3` | Qwen/Qwen3-Embedding-0.6B | HuggingFace / ModelScope |

下载后记下模型所在目录的**绝对路径**，后续填写到 `.env` 中。

---

## 二、配置（.env）

所有配置集中在 `ai_sag/.env` 文件中。首次使用时，**从模板文件复制一份再编辑**即可：

```bash
# Linux / Mac
cp .env.example .env

# Windows
copy .env.example .env
```

> **说明**：`config.py` 会从自身所在目录逐级向上搜索 `.env`，无论你把仓库 clone 到什么路径都能自动找到，无需修改任何代码。

配置分为两部分：必填的基础配置（`SAG_` 前缀）、可选的高级配置（`AISAG_` 前缀，不填则使用默认值）。

### 2.1 必填配置 — MySQL / Embedding / LLM

以下 4 类配置必须填写，否则服务无法启动：

```ini
# ============================================================
# 1. MySQL 连接
# ============================================================
SAG_MYSQL_HOST=localhost          # MySQL 地址
SAG_MYSQL_PORT=3306               # MySQL 端口
SAG_MYSQL_DATABASE=sag            # 数据库名（需手动创建）
SAG_MYSQL_USER=root               # 用户名
SAG_MYSQL_PASSWORD=your_password  # 密码 ← 改成你自己的

# ============================================================
# 2. Embedding 模型
# ============================================================
# 后端选择：bge（推荐，轻量稳定） 或 qwen3
SAG_EMBEDDING_BACKEND=bge
SAG_EMBEDDING_DEVICE=cpu          # cpu 或 cuda
# bge 模型本地路径（如果使用 bge 后端）← 改成你自己的路径
SAG_BGE_MODEL_PATH=/path/to/bge-small-zh-v1.5
# qwen3 模型本地路径（如果使用 qwen3 后端）
SAG_EMBEDDING_MODEL_PATH=/path/to/Qwen3-Embedding-0.6B

# ============================================================
# 3. LLM 大模型（OpenAI 兼容协议）
# ============================================================
SAG_LLM_BASE_URL=https://api.deepseek.com   # API 地址
SAG_LLM_MODEL=deepseek-chat                 # 模型名称
SAG_LLM_API_KEY=sk-your-api-key-here        # API Key ← 改成你自己的
SAG_LLM_TIMEOUT=120                         # 请求超时（秒）
SAG_LLM_MAX_RETRIES=3                       # 最大重试次数
```

> **LLM 兼容性**：只要遵循 OpenAI 兼容协议的 API 都可以使用（DeepSeek / OpenAI / 本地 vLLM / Ollama 等），修改 `SAG_LLM_BASE_URL` 和 `SAG_LLM_MODEL` 即可切换。

### 2.2 可选配置 — ai_sag 高级参数

以下所有配置项均有合理默认值，**初次使用无需做任何修改**。当你需要调优检索效果或入库性能时，再按需取消注释并调整：

```ini
# ---- 组件后端 ----
# AISAG_LLM_BACKEND=deepseek              # deepseek | openai_like | openai（默认 deepseek）
# AISAG_VECTOR_STORE_BACKEND=chroma       # 向量库后端（默认 chroma）
# AISAG_CHROMA_PATH=./.chroma             # 向量库目录（默认 项目根/.chroma）

# ---- MySQL 连接池 ----
# AISAG_MYSQL_POOL_SIZE=10                # 连接池大小（默认 10）
# AISAG_MYSQL_MAX_OVERFLOW=5              # 最大溢出连接数（默认 5）
# AISAG_MYSQL_POOL_RECYCLE=3600           # 连接回收秒数（默认 3600）

# ---- 文档解析 ----
# AISAG_DOC_OCR_BACKEND=rapidocr          # OCR 引擎：rapidocr | paddleocr
# AISAG_DOC_OCR_IMAGES=true               # 是否 OCR 提取图片文字
# AISAG_UPLOAD_TMP_DIR=                   # 上传临时目录，默认用系统 TEMP

# ---- 文档切分 ----
# AISAG_SPLITTER_MODE=semantic            # semantic | auto | markdown | sentence
# AISAG_CHUNK_SIZE=8192                   # chunk 大小（默认 8192）
# AISAG_CHUNK_OVERLAP=800                 # chunk 重叠（默认 800）
# AISAG_BREAKPOINT_PERCENTILE=95          # 语义断点阈值（默认 95）

# ---- 检索参数（详见第九章）----
# AISAG_FUSION=concat                     # concat | supplement
# AISAG_SIMILARITY_THRESHOLD=0.4
# AISAG_ENTITY_EXPAND_THRESHOLD=0.3
# AISAG_MAX_HOPS=2
# AISAG_SUB_STRATEGY=hopllm               # multi | hopllm
# AISAG_ENTITY_FRONTIER_BUDGET=100
# AISAG_MAX_EVENTS=100
# AISAG_RERANK_TOP_K=5
# AISAG_RERANK_CANDIDATE_LIMIT=100
# AISAG_MAX_SECTIONS=5
# AISAG_REWRITE_MAX_ROUNDS=5

# ---- 入库流程 ----
# AISAG_EXTRACT_PARALLEL=false            # 是否并行抽取事件
# AISAG_INGEST_CONCURRENCY=2              # 并发入库上限
# AISAG_RECONCILE_INTERVAL=300            # 向量-数据库对账间隔（秒），0=禁用

# ---- 日志 ----
# AISAG_LOG_LEVEL=INFO                    # DEBUG | INFO | WARNING | ERROR
# AISAG_LOG_DIR=./logs
# AISAG_LOG_ROTATION=500 MB
# AISAG_LOG_RETENTION=30 days
# AISAG_LOG_COLORIZE=false
```

---

## 三、启动服务

ai_sag 需要启动**两个进程**：API（后端）+ Web UI（前端），建议各开一个终端。

> **在哪执行？** 所有 `python -m ai_sag.xxx` 命令都在 `your-repo/`（仓库根目录，即 `ai_sag/` 的上级目录）执行，**不是**在 `ai_sag/` 目录下执行。Python 需要从上级目录识别 `ai_sag` 这个包。

### 3.1 启动 API 服务（终端 1）

```bash
# 1. 进入仓库根目录
cd your-repo

# 2. 激活 conda 环境
conda activate ai_sag

# 3. 启动（默认监听 0.0.0.0:8777）
python -m ai_sag.api

# 或指定端口 + 热重载
python -m ai_sag.api --host 0.0.0.0 --port 8777 --reload
```

启动成功后：
- API 文档（Swagger）：http://localhost:8777/docs
- 健康检查：http://localhost:8777/api/health

### 3.2 启动 Web UI（终端 2）

```bash
cd your-repo
conda activate ai_sag

# 默认 0.0.0.0:8080，自动指向 http://localhost:8777
python -m ai_sag.web

# 指定端口 + 自定义 API 地址
python -m ai_sag.web --port 8080 --api http://localhost:8777
```

浏览器打开 **http://localhost:8080** 即可使用。

### 3.3 端口一览

| 服务 | 默认端口 | 启动命令 |
|------|---------|---------|
| API（后端） | 8777 | `python -m ai_sag.api` |
| Web UI（前端） | 8080 | `python -m ai_sag.web` |

---

## 四、API 接口一览

### 4.1 文档管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents` | 上传文档（文件：md / txt / docx / pdf） |
| POST | `/api/documents/text` | 上传文档（纯文本 JSON） |
| GET | `/api/documents` | 文档列表 `?keyword=&limit=&offset=` |
| GET | `/api/documents/search` | 全文搜索 `?q=&source_ids=&limit=&context_size=` |
| GET | `/api/documents/{id}` | 文档详情（含原文 + 统计） |
| GET | `/api/documents/{id}/download` | 下载原文（.md） |
| PATCH | `/api/documents/{id}` | 更新元信息 |
| PUT | `/api/documents/{id}` | 更新文档内容（重建） |
| DELETE | `/api/documents/{id}` | 删除文档 |

### 4.2 检索与问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/search` | SAG 检索（返回切片 + trace） |
| POST | `/api/ask` | 问答（答案 + 切片 + trace） |
| POST | `/api/chat` | 多轮对话 |
| POST | `/api/chat/stream` | 多轮对话（SSE 流式） |

请求体示例：
```json
{
  "query": "数脉科技跟哪个公司签了合同？",
  "source_ids": null,
  "fusion": "concat"
}
```

### 4.3 系统接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/stats` | 统计信息 |
| GET | `/api/health` | 健康检查 |

---

## 五、接口调用示例

```bash
# 上传文件
curl -X POST http://localhost:8777/api/documents -F "file=@合同公告.md"

# 上传纯文本
curl -X POST http://localhost:8777/api/documents/text \
  -H "Content-Type: application/json" \
  -d '{"title":"测试文档","content":"文档正文内容"}'

# 文档列表
curl "http://localhost:8777/api/documents?keyword=合同&limit=20&offset=0"

# 全文搜索
curl "http://localhost:8777/api/documents/search?q=数脉科技&limit=20&context_size=80"

# 问答
curl -X POST http://localhost:8777/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"数脉科技跟哪个公司签了合同？","fusion":"concat"}'

# 删除文档
curl -X DELETE http://localhost:8777/api/documents/{source_id}
```

---

## 六、前端对接示例

```javascript
const API_BASE = "http://localhost:8777";

// 上传文件
const form = new FormData();
form.append("file", fileInput.files[0]);
const { source_id } = await fetch(`${API_BASE}/api/documents`, {
  method: "POST", body: form
}).then(r => r.json());

// 问答
const { answer, sections, trace } = await fetch(`${API_BASE}/api/ask`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "数脉科技跟谁签了合同？", fusion: "concat" })
}).then(r => r.json());
```

---

## 七、组件可配置项

所有核心组件均支持通过 `.env` 切换，无需改代码：

| 组件 | 配置项 | 可选值 |
|------|--------|--------|
| Embedding | `SAG_EMBEDDING_BACKEND` | `bge` / `qwen3` |
| LLM | `AISAG_LLM_BACKEND` | `deepseek` / `openai_like` / `openai` |
| 向量库 | `AISAG_VECTOR_STORE_BACKEND` | `chroma` |
| 切分器 | `AISAG_SPLITTER_MODE` | `semantic` / `auto` / `markdown` / `sentence` |
| OCR | `AISAG_DOC_OCR_BACKEND` | `rapidocr` / `paddleocr` |

---

## 八、日志

基于 loguru，控制台 + 文件双输出，支持按大小轮转与 gz 压缩。

- 日志文件：`ai_sag/logs/ai_sag_YYYY-MM-DD.log`
- 默认轮转大小：500 MB
- 默认保留时长：30 天
- 每条 API 请求自动注入 `trace_id`，方便链路追踪

通过 `AISAG_LOG_LEVEL` 调整日志级别（DEBUG / INFO / WARNING / ERROR）。

---

## 九、检索参数详解

以下参数对应 SAG 论文 3.3 / 3.4 节的检索流程，均可在 `.env` 中覆盖：

| 参数 | 默认值 | 作用 | 说明 |
|------|--------|------|------|
| `AISAG_SIMILARITY_THRESHOLD` | 0.4 | 事件召回 + 粗排 + 基线 chunk 召回 | 低于此阈值的向量结果被过滤 |
| `AISAG_ENTITY_EXPAND_THRESHOLD` | 0.3 | 实体向量扩展 | 默认偏宽松；严格场景（数据量大、噪声多）建议上调到 0.9 |
| `AISAG_MAX_HOPS` | 2 | 动态超边 BFS 扩展跳数 | 越大召回越全，但耗时增加 |
| `AISAG_SUB_STRATEGY` | hopllm | 多跳扩展策略 | `multi` = 固定跳数；`hopllm` = 每跳由 LLM 判断是否继续 |
| `AISAG_ENTITY_FRONTIER_BUDGET` | 100 | BFS 每跳边界实体数上限 | 控制多跳扩展的宽度 |
| `AISAG_MAX_EVENTS` | 100 | 粗排候选事件数 (Kcand) | 候选池大小 |
| `AISAG_COARSE_THRESHOLD` | 0 | 粗排相似度裁剪 | 0 = 关闭裁剪，仅排序截断 |
| `AISAG_RERANK_TOP_K` | 5 | LLM 重排事件数 (Kevent) | 送入精排后保留的事件数 |
| `AISAG_RERANK_CANDIDATE_LIMIT` | 100 | 重排最大候选数 | 送入 LLM 的候选事件上限 |
| `AISAG_MAX_SECTIONS` | 5 | 单路输出 chunk 数 | 结构化路径和语义路径各取的数量 |
| `AISAG_FUSION` | concat | 双路融合策略 | `concat` = 两路拼接；`supplement` = 事件为主、向量补足 |
| `AISAG_REWRITE_MAX_ROUNDS` | 5 | 多轮查询重写 | 保留最近 N 轮对话用于指代消解 |

### 调优建议

| 症状 | 调整方向 |
|------|---------|
| 召回太少 | 降低 `SIMILARITY_THRESHOLD`（如 0.3）、增加 `MAX_HOPS`、降低 `ENTITY_EXPAND_THRESHOLD` |
| 召回噪声多 | 提高 `ENTITY_EXPAND_THRESHOLD`（如 0.9）、减少 `MAX_HOPS` |
| 上下文不够 | 增大 `RERANK_TOP_K`、`MAX_SECTIONS` |
| BFS 太慢 | 减小 `ENTITY_FRONTIER_BUDGET`（如 50）、使用 `SUB_STRATEGY=hopllm` |

---

## 十、常见问题

### Q1：`ModuleNotFoundError: No module named 'ai_sag'`

**原因**：在 `ai_sag/` 目录内执行了 `python -m ai_sag.api`。

**正确做法**：回到上级目录再启动：

```bash
cd your-repo          # 不是 ai_sag/ 目录！
python -m ai_sag.api
```

### Q2：缺少第三方库

```bash
pip install loguru python-docx pypdf
# 或一次性安装全部依赖
pip install -r requirements.txt
```

### Q3：MySQL 连接失败

检查 `.env` 中以下配置是否正确，且 MySQL 服务已启动：
- `SAG_MYSQL_HOST` / `SAG_MYSQL_PORT`
- `SAG_MYSQL_USER` / `SAG_MYSQL_PASSWORD`
- `SAG_MYSQL_DATABASE`（需手动创建）

### Q4：Embedding 模型加载失败

确认 `.env` 中 `SAG_BGE_MODEL_PATH`（或 `SAG_EMBEDDING_MODEL_PATH`）指向的目录存在，且包含模型文件（`config.json`、`pytorch_model.bin` 等）。

从 ModelScope 下载示例：
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/bge-small-zh-v1.5')"
```

### Q5：LLM API 返回错误

- 检查 `SAG_LLM_API_KEY` 是否正确
- 检查 `SAG_LLM_BASE_URL` 是否可达
- 如使用代理，确保设置了 `HTTPS_PROXY` / `HTTP_PROXY` 环境变量
