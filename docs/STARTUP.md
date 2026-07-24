# ai_sag — 启动与使用指南

基于 SAG 论文（SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges）实现的轻量级多跳 RAG 系统。本文档涵盖环境准备、配置、启动、接口调用全流程。

> 详细流程文档：
> - 离线入库阶段：[入库流程.md](./入库流程.md)
> - 在线检索/问答阶段：[检索流程.md](./检索流程.md)
> - 存储后端替换：[存储后端替换指南.md](./存储后端替换指南.md)

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
    ├── requirements.txt        ← 依赖清单（CPU 完整版）
    ├── requirements-gpu.txt    ← 依赖清单（GPU 完整版，自包含，不叠加 CPU 版）
    ├── base/               ← 配置、日志、数据库基类
    ├── retrieval/          ← SAG 检索 & 问答引擎
    ├── ingest/             ← 入库流水线
    ├── embeddings/         ← 向量化模型
    ├── splitter/           ← 文档切分
    ├── extractor/          ← 事件 & 实体抽取
    ├── vector_store/       ← 向量库封装（PGVector 推荐 / Chroma/FAISS/Milvus）
    ├── storage/            ← 关系库持久化（PostgreSQL 推荐 / MySQL）
    ├── llm/                ← LLM 调用封装
    ├── loader/             ← 文件加载（md/txt/docx/pdf/xlsx/csv/图片）
    ├── doc_parser/         ← 文档解析（OCR、版式分析等）
    ├── static/             ← Web 前端页面
    ├── tmp/                ← 上传/解析临时目录（自动创建）
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
# CPU 环境
pip install -r requirements.txt

# GPU 环境（已自包含全部依赖，无需叠加 CPU 版）
pip install -r requirements-gpu.txt
```

> **⚠️ 重要**：CPU 版与 GPU 版二选一，**不要叠加安装**。GPU 版 `requirements-gpu.txt` 已包含全部依赖（含 CPU 版的全部 + GPU 加速版本），叠加安装会导致 `torch` / `paddlepaddle` / `onnxruntime` 被反复卸载重装。
>
> **GPU 版额外要求**：系统已安装 NVIDIA 驱动 + CUDA Toolkit（推荐 12.4），`requirements-gpu.txt` 已配置 PyTorch 官方 CUDA 源和 PaddlePaddle 官方源。

核心依赖一览：

| 类别 | 包 | 用途 |
|------|-----|------|
| 框架 | `llama-index-core` 等 | RAG 框架底座 |
| Web | `fastapi` + `uvicorn` | HTTP 服务 |
| 向量库 | `llama-index-vector-stores-postgres`（推荐）等 | 向量存储与检索（5 套集合） |
| 数据库 | `asyncpg` + `aiomysql` | PostgreSQL（推荐）/ MySQL 异步连接 |
| 日志 | `loguru` | 控制台 + 文件双输出 |
| 配置 | `python-dotenv` | 从 `.env` 加载环境变量 |
| 上传 | `python-multipart` + `aiofiles` | 文件上传解析 |
| 文档解析 | `pypdf` + `python-docx` + `openpyxl` | PDF/Word/Excel 解析 |
| OCR | `rapidocr-onnxruntime`（默认）/ `paddleocr` | 图片文字识别（PDF/Word 内嵌图片 + 纯图片文件） |

### 1.3 数据库

**推荐方案：PostgreSQL + pgvector**（一套数据库同时承担关系库 + 向量库，无需安装多个中间件）

```sql
-- 1. 创建数据库
CREATE DATABASE sag;

-- 2. 连接到 sag 数据库后，安装 pgvector 扩展（向量库后端用 pgvector 时必装）
\c sag
CREATE EXTENSION IF NOT EXISTS vector;

-- 验证扩展已安装
\dx vector
```

**备选方案 A：MySQL**（仅作关系库，向量库需另选 Chroma/FAISS/Milvus）
```sql
CREATE DATABASE IF NOT EXISTS sag DEFAULT CHARSET utf8mb4;
```

**备选方案 B：PostgreSQL + 其他向量库**（关系库用 PG，向量库用 Chroma/FAISS/Milvus，不装 pgvector 扩展）

表结构由 `ensure_schema()` 在首次启动 API 时自动创建（PG 用 `schema_pg.sql`，MySQL 用 `schema.sql`），向量表由 PGVector 自动创建（使用 pgvector 后端时），均无需手动建表。

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

> **说明**：`config.py` 会从自身所在目录逐级向上搜索 `.env`，无论你把仓库 clone 到什么路径都能自动找到，无需修改任何代码。找不到时会给出明确提示，不会静默失败。

配置分为两部分：必填的基础配置（`SAG_` 前缀）、可选的高级配置（`AISAG_` 前缀，不填则使用默认值）。

### 2.1 必填配置 — 数据库 / 向量库 / Embedding / LLM

以下 4 类配置必须填写，否则服务无法启动（缺失时启动报错并列出缺失项）：

```ini
# ============================================================
# 1. 关系数据库（PostgreSQL 推荐 / MySQL 二选一）
# ============================================================
AISAG_DB_BACKEND=postgresql       # postgresql（推荐）| mysql

# PostgreSQL（推荐：AISAG_DB_BACKEND=postgresql 时填写）
SAG_PG_HOST=localhost
SAG_PG_PORT=5432
SAG_PG_DATABASE=sag               # 需手动创建，并安装 pgvector 扩展
SAG_PG_USER=postgres
SAG_PG_PASSWORD=your_password

# MySQL（备选：AISAG_DB_BACKEND=mysql 时填写）
# SAG_MYSQL_HOST=localhost
# SAG_MYSQL_PORT=3306
# SAG_MYSQL_DATABASE=sag
# SAG_MYSQL_USER=root
# SAG_MYSQL_PASSWORD=your_password

# ============================================================
# 2. 向量库后端
# ============================================================
# pgvector（推荐，与 PG 共用）| chroma | faiss | milvus
AISAG_VECTOR_STORE_BACKEND=pgvector
# PGVector 连接串（使用 pgvector 后端时必填，通常与关系库同一个 PG 实例）
AISAG_PG_CONNECTION_STRING=postgresql://postgres:your_password@localhost:5432/sag
# 向量维度（必须与 embedding 模型一致：bge=512, qwen3=1024）
AISAG_VECTOR_STORE_DIM=512

# ============================================================
# 3. Embedding 模型
# ============================================================
# 后端选择：bge（默认，CLS pooling，轻量稳定） 或 qwen3（last_token pooling + 查询指令前缀）
SAG_EMBEDDING_BACKEND=bge
# 模型本地路径（两种后端共用此变量，切后端时同步修改路径）
#   bge:   下载 bge-small-zh-v1.5（modelscope download --model AI-ModelScope/bge-small-zh-v1.5）
#   qwen3: 下载 Qwen3-Embedding-0.6B（modelscope download --model Qwen/Qwen3-Embedding-0.6B）
SAG_EMBEDDING_MODEL_PATH=/path/to/bge-small-zh-v1.5

# ============================================================
# 3. LLM 大模型（多场景配置：配合 llm_profiles.yaml）
# ============================================================
# 步骤 1：在 llm_profiles.yaml 中定义 profile（参考 llm_profiles.yaml.example）
# 步骤 2：在 .env 中按场景选择 profile 并配置参数

# 答案生成 → qwen3.6-27b 开启思考
SAG_LLM_PROFILE_ANSWER_LLM_NAME=qwen_thinking
SAG_LLM_PROFILE_ANSWER_ADDITIONAL_KWARGS={"temperature": 0.7, "max_tokens": 12288}
SAG_LLM_PROFILE_ANSWER_EXTRA_BODY={"enable_thinking": true}

# 结构化场景（不能用思考）→ DeepSeek
SAG_LLM_PROFILE_GENRE_CLASSIFY_LLM_NAME=deepseek_chat
SAG_LLM_PROFILE_EVENT_EXTRACT_LLM_NAME=deepseek_chat
SAG_LLM_PROFILE_QUERY_REWRITE_LLM_NAME=deepseek_chat
SAG_LLM_PROFILE_ENTITY_EXTRACT_LLM_NAME=deepseek_chat
SAG_LLM_PROFILE_RERANK_LLM_NAME=deepseek_chat
```

> **LLM 兼容性**：只要遵循 OpenAI 兼容协议的 API 都可以使用（DeepSeek / OpenAI / 本地 vLLM / Ollama 等）。
> 在 `llm_profiles.yaml` 中定义 profile，在 `.env` 中按场景选择 profile 即可切换。
> 后端由 factory 按 `profile.model` 自动判断（OpenAI 官方模型走 openai，其他走 openai_like），无需也无法手动配置。
> 每个场景必须显式配置 `_LLM_NAME`，不存在 DEFAULT 回退机制。

### 2.2 可选配置 — ai_sag 高级参数

以下所有配置项均有合理默认值，**初次使用无需做任何修改**。当你需要调优检索效果或入库性能时，再按需取消注释并调整：

```ini
# ---- 组件后端 ----
# LLM 后端由 factory 按 profile.model 自动判断，无需配置
# AISAG_DB_BACKEND=postgresql             # 关系库后端（默认 postgresql）| mysql
# AISAG_VECTOR_STORE_BACKEND=pgvector     # 向量库后端（默认 pgvector）| chroma | faiss | milvus
# AISAG_CHROMA_PATH=./.chroma             # Chroma 目录（默认 项目根/.chroma）
# AISAG_FAISS_PATH=./.faiss               # FAISS 目录（默认 项目根/.faiss）

# ---- 关系库连接池 ----
# PostgreSQL 连接池（AISAG_DB_BACKEND=postgresql 时生效）
# AISAG_PG_POOL_SIZE=10                   # 连接池大小（默认 10）
# AISAG_PG_MAX_OVERFLOW=5                 # 最大溢出连接数（默认 5）
# AISAG_PG_POOL_TIMEOUT=30                # 获取连接超时秒数（默认 30）
# AISAG_PG_POOL_RECYCLE=3600              # 连接回收秒数（默认 3600）
# MySQL 连接池（AISAG_DB_BACKEND=mysql 时生效）
# AISAG_MYSQL_POOL_SIZE=10                # 连接池大小（默认 10）
# AISAG_MYSQL_MAX_OVERFLOW=5              # 最大溢出连接数（默认 5）
# AISAG_MYSQL_POOL_TIMEOUT=30             # 获取连接超时秒数（默认 30）
# AISAG_MYSQL_POOL_RECYCLE=3600           # 连接回收秒数（默认 3600，防 8 小时断连）

# ---- 文档解析 ----
# AISAG_DOC_OCR_BACKEND=rapidocr          # OCR 引擎：rapidocr（轻量默认）| paddleocr（高精度）
# AISAG_DOC_OCR_IMAGES=true               # 是否 OCR 提取 PDF/Word 中图片文字（默认 true）
# AISAG_PDF_MARKDOWN_MODE=direct          # PDF 转 MD 模式：direct（默认，快）| pymupdf4llm（表格准）
# AISAG_UPLOAD_TMP_DIR=./tmp              # 上传临时目录（默认 项目根/tmp）

# ---- 文档切分（默认 semantic 语义切分）----
# AISAG_SPLITTER_MODE=semantic            # semantic | auto | markdown | sentence
# AISAG_CHUNK_SIZE=8192                   # chunk 大小（默认 8192，语义完整优先）
# AISAG_CHUNK_OVERLAP=800                 # chunk 重叠（默认 800，防事件被切断）
# AISAG_BREAKPOINT_PERCENTILE=95          # 语义断点阈值（默认 95，仅差异最大 5% 处断句）

# ---- 检索参数（详见第九章 + 检索流程.md）----
# 核心阈值与开关
# AISAG_SIMILARITY_THRESHOLD=0.4          # 事件召回/粗排/chunk 召回相似度阈值（默认 0.4）
# AISAG_ENTITY_EXPAND_ENABLED=true        # 实体向量扩展开关（默认 true，false=仅精确匹配）
# AISAG_ENTITY_EXPAND_THRESHOLD=0.5       # 实体向量扩展阈值（默认 0.5，独立于事件召回）
# AISAG_ENTITY_EXPAND_TOPK=10             # 每实体向量召回近邻数（默认 10）
# AISAG_MAX_HOPS=2                        # BFS 最大跳数（默认 2）
# AISAG_SUB_STRATEGY=hopllm               # multi=固定跳数 | hopllm=动态停止（默认）
# AISAG_HOP_RELEVANCE_THRESHOLD=0.3       # hopllm 停止阈值：新跳最佳分<此值终止（默认 0.3）
# AISAG_HOP_EVENT_SOFT_THRESHOLD=0.0      # BFS 扩展事件软过滤阈值（默认 0=禁用，>0 则低于此分不进候选池）
# AISAG_HOP_SEED_TOPK=8                   # hopllm 每跳保留种子事件数（默认 8）
# AISAG_FUSION=concat                     # concat=双路拼接 | supplement=事件为主向量补足
# AISAG_SEED_RECALL=mixed                 # 种子事件向量召回：title | summary | mixed（默认双路）
#
# BFS 边界实体预算与过滤
# AISAG_ENTITY_FRONTIER_BUDGET=100        # 每跳边界实体数上限（默认 100）
# AISAG_ENTITY_FRONTIER_FILTER=true       # 是否启用边界实体相关性筛选（默认 true）
# AISAG_ENTITY_FRONTIER_QUERY_WEIGHT=0.6  # 综合分中 query 相似度权重 α（默认 0.6）
#
# 实体度数离群过滤（枢纽实体抑制）
# AISAG_ENTITY_DEGREE_METHOD=otsu         # 离群检测法：otsu（默认）| percentile | mad | tukey | none
# AISAG_ENTITY_DEGREE_ABS_MAX=50          # 度数绝对硬上限（默认 50，0=关闭）
# AISAG_ENTITY_DEGREE_PERCENTILE=95       # percentile 法分位点（默认 P95）
# AISAG_ENTITY_DEGREE_OUTLIER_K=3.0       # mad/tukey 法离群倍数 k（默认 3.0，越大越保守）
# AISAG_ENTITY_DEGREE_MIN_BATCH=10        # BFS 边界实体最小触发 batch（默认 10，小样本跳过）
#
# 种子实体过滤（复用 BFS 度数过滤参数，更保守）
# AISAG_SEED_ENTITY_BUDGET=200            # 种子实体总量硬上限（默认 200，防御性，几乎不触发）
# 注：种子阶段离群检测复用 AISAG_ENTITY_DEGREE_MIN_BATCH（默认 10）和 AISAG_ENTITY_DEGREE_METHOD
#     过度过滤保护下限为硬编码 max(min_batch, 5)，不可配置
#
# 粗排/精排
# AISAG_MAX_EVENTS=100                    # 粗排候选事件数 Kcand（默认 100）
# AISAG_COARSE_THRESHOLD=0                # 粗排相似度裁剪（默认 0=仅排序截断）
# AISAG_RERANK_TOP_K=5                    # LLM 重排保留事件数 Kevent（默认 5）
# AISAG_RERANK_CANDIDATE_LIMIT=100        # 送入 LLM 重排最大候选数（默认 100）
# AISAG_RERANK_WEIGHT_STRONG=0.7          # 实体关联权重强信号阈值（默认 0.7）
# AISAG_RERANK_WEIGHT_WEAK=0.4            # 实体关联权重弱信号阈值（默认 0.4）
# AISAG_RERANK_EVENT_DECAY=0.6            # 事件深度轮次衰减基数（默认 0.6，depth=0→1.0, depth=1→0.6）
# AISAG_MAX_SECTIONS=5                    # 单路输出 chunk 数（默认 5）
# AISAG_REWRITE_MAX_ROUNDS=5              # 查询重写保留对话轮数（默认 5）

# ---- 入库流程（详见入库流程.md）----
# AISAG_EXTRACT_PARALLEL=false            # 事件抽取是否并行（默认 false，并发提速需开 true）
# AISAG_EXTRACT_PARALLEL_WORKERS=4        # 并行抽取 worker 数（默认 4）
# AISAG_EXTRACT_MAX_RETRIES=2             # 单 chunk 抽取最大重试次数（默认 2）
# AISAG_INGEST_CONCURRENCY=2              # 并发入库文档数上限（默认 2，防 rate limit/OOM）
# AISAG_TITLE_MAX_CHARS=100               # 事件标题字数上限（默认 100）
# AISAG_SUMMARY_MAX_CHARS=500             # 事件摘要字数上限（默认 500）
# AISAG_EMBED_SUMMARY=true                # 是否给事件摘要生成向量（默认 true）
# AISAG_GENRE_DETECT=true                 # 是否启用文档级文体识别（默认 true，驱动抽取规则）

# ---- 日志 ----
# AISAG_LOG_LEVEL=INFO                    # DEBUG | INFO | WARNING | ERROR
# AISAG_LOG_DIR=./logs
# AISAG_LOG_ROTATION=500 MB
# AISAG_LOG_RETENTION=30 days
# AISAG_LOG_COLORIZE=false                # 控制台彩色输出（容器环境建议关）
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
- 首次启动自动执行：关系库建表（PG 用 `schema_pg.sql`，MySQL 用 `schema.sql`）→ 向量表（pgvector 后端时自动建 `data_sag_*` 5 张表）

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
| POST | `/api/documents` | 上传文档（文件：md / txt / docx / pdf / xlsx / csv / 图片） |
| POST | `/api/documents/text` | 上传文档（纯文本 JSON） |
| GET | `/api/documents` | 文档列表 `?keyword=&limit=&offset=` |
| GET | `/api/documents/search` | 全文搜索 `?q=&source_ids=&limit=&context_size=` |
| GET | `/api/documents/{id}` | 文档详情（含原文 + 统计） |
| GET | `/api/documents/{id}/download` | 下载原文（.md） |
| PATCH | `/api/documents/{id}` | 更新元信息 |
| PUT | `/api/documents/{id}` | 更新文档内容（重建索引） |
| DELETE | `/api/documents/{id}` | 删除文档（DB 软删 + 向量硬删） |

### 4.2 检索与问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/search` | SAG 检索（返回切片 + trace） |
| POST | `/api/ask` | 单轮问答（答案 + 切片 + trace） |
| POST | `/api/chat` | 多轮对话 |
| POST | `/api/chat/stream` | 多轮对话（SSE 流式输出） |

请求体示例：
```json
{
  "query": "数脉科技跟哪个公司签了合同？",
  "source_ids": null,
  "fusion": "concat",
  "seed_recall": "mixed"
}
```

### 4.3 系统接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/stats` | 统计信息（文档/事件/实体数量） |
| GET | `/api/health` | 健康检查 |

---

## 五、接口调用示例

```bash
# 上传文件
curl -X POST http://localhost:8777/api/documents -F "file=@合同公告.md"

# 上传 PDF/Word 并指定 OCR 引擎与开关（ocr_images: true/false, ocr_backend: rapidocr/paddleocr）
curl -X POST http://localhost:8777/api/documents \
  -F "file=@scan.pdf" -F "ocr_images=true" -F "ocr_backend=paddleocr"

# 上传图片（自动走 OCR）
curl -X POST http://localhost:8777/api/documents -F "file=@receipt.jpg" -F "ocr_backend=rapidocr"

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
| 关系数据库 | `AISAG_DB_BACKEND` + `SAG_PG_*` / `SAG_MYSQL_*` | `postgresql`（推荐）/ `mysql` |
| 向量库 | `AISAG_VECTOR_STORE_BACKEND` | `pgvector`（推荐，与 PG 共用）/ `chroma` / `faiss` / `milvus` |
| Embedding | `SAG_EMBEDDING_BACKEND` + `SAG_EMBEDDING_MODEL_PATH` | `bge`（默认）/ `qwen3` |
| LLM | 无（按 profile.model 自动判断） | `openai_like` / `openai` |
| 切分器 | `AISAG_SPLITTER_MODE` | `semantic`（默认）/ `auto` / `markdown` / `sentence` |
| OCR | `AISAG_DOC_OCR_BACKEND` | `rapidocr`（默认）/ `paddleocr` |
| BFS 子策略 | `AISAG_SUB_STRATEGY` | `hopllm`（默认动态停止）/ `multi`（固定跳数） |
| 种子召回 | `AISAG_SEED_RECALL` | `mixed`（默认双路）/ `title` / `summary` |
| 度数过滤 | `AISAG_ENTITY_DEGREE_METHOD` | `otsu`（默认）/ `percentile` / `mad` / `tukey` / `none` |

---

## 八、日志

基于 loguru，控制台 + 文件双输出，支持按大小轮转与 gz 压缩。

- 日志文件：`ai_sag/logs/ai_sag_YYYY-MM-DD.log`
- 默认轮转大小：500 MB
- 默认保留时长：30 天
- 每条 API 请求自动注入 `trace_id`，方便链路追踪
- 检索 trace 日志包含：实体抽取/扩展、种子事件召回、BFS 每跳过滤统计、精排得分等全链路信息

通过 `AISAG_LOG_LEVEL` 调整日志级别（DEBUG / INFO / WARNING / ERROR）。

---

## 九、检索参数详解

以下参数对应 SAG 论文 3.3 / 3.4 节及本项目扩展实现，均可在 `.env` 中覆盖。完整流程说明见 [检索流程.md](./检索流程.md)。

### 9.1 核心召回参数

| 参数 | 默认值 | 作用 | 说明 |
|------|--------|------|------|
| `AISAG_SIMILARITY_THRESHOLD` | 0.4 | 事件召回/粗排/chunk 召回阈值 | 低于此阈值的向量结果被过滤 |
| `AISAG_ENTITY_EXPAND_ENABLED` | true | 实体向量扩展总开关 | false=仅 SQL 精确匹配，true=补充同义词近邻 |
| `AISAG_ENTITY_EXPAND_THRESHOLD` | 0.5 | 实体向量扩展相似度阈值 | 独立于事件召回阈值，默认偏严格防漂移 |
| `AISAG_ENTITY_EXPAND_TOPK` | 20 | 每实体向量近邻数 | 配合 seed_entity_min_batch=15 确保统计量稳定 |
| `AISAG_SEED_RECALL` | mixed | 种子事件向量召回策略 | title=仅标题 / summary=仅摘要 / mixed=双路并发合并 |
| `AISAG_FUSION` | concat | 双路融合策略 | concat=两路拼接；supplement=事件为主、向量补足 |

### 9.2 BFS 多跳扩展参数

| 参数 | 默认值 | 作用 | 说明 |
|------|--------|------|------|
| `AISAG_MAX_HOPS` | 2 | BFS 最大跳数 | 越大召回越全，但耗时线性增加 |
| `AISAG_SUB_STRATEGY` | hopllm | 多跳扩展子策略 | multi=固定跳数；hopllm=每跳相似度动态停止 |
| `AISAG_HOP_RELEVANCE_THRESHOLD` | 0.15 | hopllm 动态停止阈值 | 新跳最佳内容相似度<此值终止扩展，安全阀 |
| `AISAG_HOP_SEED_TOPK` | 8 | hopllm 每跳保留种子数 | 对齐旧版 topK 剪枝，防止爆炸 |
| `AISAG_ENTITY_FRONTIER_BUDGET` | 100 | 每跳边界实体数上限 | 控制多跳扩展宽度（论文 Section 4.4） |
| `AISAG_ENTITY_FRONTIER_FILTER` | true | 边界实体相关性筛选 | 综合分=α·IDF + (1-α)·query相似度，抑制枢纽桥接 |
| `AISAG_ENTITY_FRONTIER_QUERY_WEIGHT` | 0.6 | query 相似度权重 α | 越大越偏语义相关，越小越偏低频优先 |

### 9.3 实体度数过滤参数（枢纽实体抑制）

| 参数 | 默认值 | 作用 | 适用方法 |
|------|--------|------|---------|
| `AISAG_ENTITY_DEGREE_METHOD` | otsu | 离群检测算法 | otsu（默认，数据驱动自动阈值）/ percentile / mad / tukey / none |
| `AISAG_ENTITY_DEGREE_ABS_MAX` | 50 | 度数绝对硬上限 | 所有方法共用兜底，度数>50 的实体直接剔除 |
| `AISAG_ENTITY_DEGREE_PERCENTILE` | 95 | percentile 法分位点 | 仅 percentile 法生效，剔除最高 5% 枢纽 |
| `AISAG_ENTITY_DEGREE_OUTLIER_K` | 3.0 | mad/tukey 离群倍数 | k 越大越保守，k≥5 近似关闭 |
| `AISAG_ENTITY_DEGREE_MIN_BATCH` | 10 | BFS 边界最小触发 batch | 实体数<10 跳过统计检测（小样本不稳）；种子阶段复用此参数 |
| `AISAG_SEED_ENTITY_BUDGET` | 200 | 种子实体总量硬上限 | 精确匹配+向量扩展去重后上限，防御性，几乎不触发 |

> **注**：种子实体阶段不再有独立配置项——离群检测复用 `AISAG_ENTITY_DEGREE_METHOD` / `AISAG_ENTITY_DEGREE_MIN_BATCH`；过度过滤保护下限为硬编码 `max(min_batch, 5)`，不可配置。

### 9.4 排序与输出参数

| 参数 | 默认值 | 作用 | 说明 |
|------|--------|------|------|
| `AISAG_MAX_EVENTS` | 100 | 粗排候选事件数 Kcand | 候选池大小 |
| `AISAG_COARSE_THRESHOLD` | 0 | 粗排相似度裁剪 | 0=关闭，仅排序截断 |
| `AISAG_RERANK_TOP_K` | 5 | LLM 重排保留事件数 Kevent | 送入精排后保留的事件数 |
| `AISAG_RERANK_CANDIDATE_LIMIT` | 100 | 重排最大候选数 | 送入 LLM 的候选事件上限（对齐论文 Top-100） |
| `AISAG_RERANK_WEIGHT_STRONG` | 0.7 | 实体关联强信号阈值 | weight≥0.7 为核心关联 |
| `AISAG_RERANK_WEIGHT_WEAK` | 0.4 | 实体关联弱信号阈值 | weight<0.4 为背景引用 |
| `AISAG_RERANK_EVENT_DECAY` | 0.6 | 事件深度轮次衰减 | depth=0→1.0, depth=1→0.6, depth=2→0.36，跳越远可信度越低 |
| `AISAG_MAX_SECTIONS` | 5 | 单路输出 chunk 数 | 结构化路径和语义路径各取的数量 |
| `AISAG_REWRITE_MAX_ROUNDS` | 5 | 查询重写对话轮次 | 保留最近 N 轮用于指代消解 |

### 9.5 入库参数（详见入库流程.md）

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `AISAG_EXTRACT_PARALLEL` | false | 事件抽取并行开关 |
| `AISAG_EXTRACT_PARALLEL_WORKERS` | 4 | 并行抽取 worker 数 |
| `AISAG_INGEST_CONCURRENCY` | 2 | 并发入库文档上限 |
| `AISAG_GENRE_DETECT` | true | 文档级文体识别开关 |
| `AISAG_EMBED_SUMMARY` | true | 事件摘要向量化开关 |

### 9.6 调优建议

| 症状 | 调整方向 |
|------|---------|
| 召回太少/漏检 | ① 开 `ENTITY_EXPAND_ENABLED=true`；② 降低 `SIMILARITY_THRESHOLD`（如 0.3）；③ 增加 `MAX_HOPS=3`；④ 降低 `ENTITY_EXPAND_THRESHOLD`（如 0.4）；⑤ 使用 `SEED_RECALL=mixed` |
| 召回噪声多/不相关 | ① 关 `ENTITY_EXPAND_ENABLED=false`（仅精确匹配最干净）；② 提高 `ENTITY_EXPAND_THRESHOLD`（如 0.7~0.8）；③ 减少 `ENTITY_FRONTIER_BUDGET`（如 50）；④ `ENTITY_DEGREE_METHOD=otsu` 保持开启 |
| 枢纽实体污染（"2024年""全行"等高频词桥接无关事件） | ① 降低 `ENTITY_DEGREE_ABS_MAX`（如 30）；② `ENTITY_DEGREE_METHOD=mad` + `OUTLIER_K=2.0`（更激进）；③ 确保 `ENTITY_FRONTIER_FILTER=true` |
| BFS 扩展太慢 | ① 减小 `ENTITY_FRONTIER_BUDGET`（如 50）；② 使用 `SUB_STRATEGY=hopllm`（动态提前停止）；③ 降低 `HOP_SEED_TOPK`（如 5） |
| 上下文不够/答案不全 | 增大 `RERANK_TOP_K`（如 8）、`MAX_SECTIONS`（如 8） |
| 入库太慢 | ① 开 `EXTRACT_PARALLEL=true`；② 增大 `INGEST_CONCURRENCY`（如 3~4）；③ 关 `GENRE_DETECT=false`（省一次 LLM 调用/文档） |
| 种子实体被过度过滤/断链 | ① 增大 `ENTITY_DEGREE_MIN_BATCH`（如 20，小样本不检测）；② `ENTITY_DEGREE_METHOD=none`（仅 abs_max 兜底）；③ 提高 `ENTITY_DEGREE_ABS_MAX`（如 100） |

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
pip install loguru python-docx pypdf openpyxl rapidocr-onnxruntime
# 或一次性安装全部依赖
pip install -r requirements.txt
```

### Q3：关系库连接失败

检查 `.env` 中以下配置是否正确，且对应数据库服务已启动：

**PostgreSQL（推荐）：**
- `SAG_PG_HOST` / `SAG_PG_PORT`
- `SAG_PG_USER` / `SAG_PG_PASSWORD`
- `SAG_PG_DATABASE`（需手动 `CREATE DATABASE sag;`，表结构自动建）
- 如使用 pgvector 后端，确认已安装扩展：`CREATE EXTENSION IF NOT EXISTS vector;`

**MySQL：**
- `SAG_MYSQL_HOST` / `SAG_MYSQL_PORT`
- `SAG_MYSQL_USER` / `SAG_MYSQL_PASSWORD`
- `SAG_MYSQL_DATABASE`（需手动创建，表结构自动建）

### Q4：Embedding 模型加载失败

确认 `.env` 中 `SAG_EMBEDDING_MODEL_PATH` 指向的目录存在，且包含模型文件（`config.json`、`pytorch_model.bin` / `model.safetensors` 等）。

从 ModelScope 下载示例：
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/bge-small-zh-v1.5')"
```

> **`The size of tensor a (xxx) must match the size of tensor b (512)` 报错**：
> 原因是 `bge-small-zh-v1.5` 位置编码硬上限为 512，输入 chunk 超长。
> 已修复：bge 后端会自动读取 `max_position_embeddings` 并把 `max_length` 压缩到模型上限。
> 若 chunk 大量超过 512 tokens，建议调小 `AISAG_CHUNK_SIZE` 或切换到 `qwen3` 后端（支持 8192）。

### Q5：LLM API 返回错误

- 检查 `llm_profiles.yaml` 中 profile 的 `api_key` / `base_url` 是否正确（直接写在 yaml 中，无需环境变量）
- 检查 `llm_profiles.yaml` 中 profile 的 `base_url` 是否可达
- 检查 `.env` 中每个场景的 `SAG_LLM_PROFILE_<SCENE>_LLM_NAME` 是否在 yaml 中存在
- 如使用代理，确保设置了 `HTTPS_PROXY` / `HTTP_PROXY` 环境变量
- 429 错误（rate limit）：降低 `INGEST_CONCURRENCY=1` 串行入库

### Q6：向量库与关系库数据不一致

入库采用"先写关系库、后写向量库、失败回滚"的原子性保障：
- 正常完成：两侧数据一致
- 向量写入失败：自动回滚关系库（删除刚写入的数据 + 孤儿实体向量）
- 进程崩溃（关系库已提交，向量库未写完）：标记 `vector_synced=False`，重启服务后系统自动检测并清理残留数据

如需手动清理，可通过 API 删除对应文档后重新上传。

### Q7：向量库存放在哪里？

不同后端的存储位置不同，均通过 `.env` 配置：
- **PGVector（推荐）**：与关系库共存于同一个 PostgreSQL，由 `AISAG_PG_CONNECTION_STRING` 指定。向量表名 `data_sag_*`（5 张表），可通过 psql 直接查询
- ChromaDB：`ai_sag/.chroma/`（`AISAG_CHROMA_PATH`）
- FAISS：`ai_sag/.faiss/`（`AISAG_FAISS_PATH`），每个 collection 写 `{name}.index` + `{name}.meta.json`
- Milvus：远程服务，由 `AISAG_MILVUS_URI` 指定

向量库统一管理 5 个 collection：
- `chunks`：原始文本 chunk 向量（Path B 语义兜底）
- `event_titles`：事件标题向量
- `event_contents`：事件原文 chunk 向量（BFS 内容相似度计算）
- `event_summaries`：事件摘要向量（受 `EMBED_SUMMARY` 开关控制）
- `entities`：实体名向量（Path A 实体扩展）