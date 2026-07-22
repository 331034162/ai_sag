# ai_sag — 基于 SAG 的轻量级多跳 RAG 系统

基于论文 **SAG（SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges）** 实现的知识库检索增强生成系统。专为跨文档、跨片段、多跳实体关联的场景设计。

---

## 目录

- [核心思路](#核心思路)
- [SAG vs 传统 RAG vs GraphRAG](#sag-vs-传统-rag-vs-graphrag)
- [系统架构](#系统架构)
- [功能特性](#功能特性)
- [访问方式](#访问方式)
- [快速开始](#快速开始)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [API 概览](#api-概览)
- [适用场景](#适用场景)
- [调优参考](#调优参考)
- [文档索引](#文档索引)

---

## 核心思路

> **离线不建图，在线动态织网。**

| 阶段 | 做了什么 | 不做什么 |
|------|---------|---------|
| **离线（入库）** | 文档切分 → 事件抽取 → 实体识别 → 建事件-实体多对多索引（关系库） + 向量化（向量库，推荐 pgvector 与关系库合一） | 不建知识图谱、不预计算关联路径 |
| **在线（查询）** | 双路种子召回 → SQL Join 动态超边 BFS 扩展 → 粗排 → LLM 精排 → 生成回答 | 不提前固定图结构，路径用完即弃 |

---

## SAG vs 传统 RAG vs GraphRAG

### 三种方案对比

| 维度 | 传统 RAG | GraphRAG | **SAG（本项目）** |
|------|---------|----------|-------------------|
| **核心原理** | 向量相似度检索 → LLM 生成 | 离线构建实体关系图谱 → 图遍历检索 | SQL Join 动态超边 → 在线织网 → LLM 生成 |
| **离线成本** | 低（仅切片 + 向量化） | **高**（实体抽取 + 关系抽取 + 图谱构建 + 存储） | 中（切片 + 事件抽取 + 实体识别，不建图） |
| **更新代价** | 低（增量向量化） | **高**（需重建或增量更新图谱） | 低（增量入库，无需维护图结构） |
| **多跳能力** | ❌ 弱（靠 LLM 分步推理，易遗漏） | ✅ 强（图遍历天然多跳） | ✅ 强（SQL Join BFS 扩展，可控可观测） |
| **跨文档关联** | ❌ 弱（各文档向量独立） | ✅ 强（实体为纽带跨文档联通） | ✅ 强（event_entities 表动态跨文档 Join） |
| **语义理解** | ✅ 强（向量相似度） | 中（依赖实体抽取质量） | ✅ 强（双路融合：向量 + 结构化） |
| **可解释性** | ❌ 弱（黑盒向量距离） | ✅ 强（图路径清晰可视） | ✅ 强（trace 记录完整 SQL Join 链路） |
| **查询灵活性** | 高（任意自然语言） | 低（受限于预定义的关系类型） | 高（自然语言驱动，实体类型仅做召回） |
| **资源占用** | 低（仅向量库） | **高**（图数据库 + 向量库） | 中（关系库 + 向量库，推荐 PostgreSQL + pgvector 合一部署） |
| **Schema 依赖** | 无 | **有**（需预定义实体类型和关系类型） | 弱（实体类型用于辅助分类，不依赖固定关系） |

### SAG 的定位

| 优势 | 劣势 |
|------|------|
| **兼具两者的优点**：有 GraphRAG 的多跳能力 + 传统 RAG 的灵活性 | 查询时 SQL Join 开销随跳数线性增长（需调优 BFS 预算） |
| **零静态图维护**：不存图谱，只存事件-实体关联表 | 对超大规模知识库（亿级事件），SQL Join 性能需关注 |
| **更新即生效**：入库操作就是表 INSERT，无重建成本 | 需同时维护关系库 + 向量库两个存储（推荐合一部署：PostgreSQL + pgvector） |
| **全链路可追踪**：trace 记录每一步 SQL，查问题比图遍历更直观 | 实体抽取质量直接影响多跳召回（但比 GraphRAG 容错更好） |
| **Schema 弱依赖**：实体类型仅用于辅助分类，不约束关系 | 极端复杂推理（5 跳以上）效率不如预计算图谱 |

### 一句话总结

> SAG = 传统 RAG 的灵活性 + GraphRAG 的多跳能力 − 知识图谱的维护成本

---

## 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────┐
│                    接入层                         │
│   Web UI (:8080)  │  Swagger API (:8777/docs)    │
│   对话交互界面      │  API 调试 & 系统配置管理       │
└────────┬──────────────────┬─────────────────────┘
         │                  │
┌────────┴──────────────────┴─────────────────────┐
│                  API 服务层 (:8777)               │
│  文档管理  │  检索问答  │  多轮对话  │  系统管理    │
└────────┬──────────────────┬─────────────────────┘
         │                  │
┌────────┴──────────────────┴─────────────────────┐
│                   业务内核层                       │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │  入库流水线 │  │  SAG 检索器 │  │  问答引擎     │  │
│  │ Ingest    │  │  Retriever │  │  QA Engine   │  │
│  └─────┬─────┘  └─────┬─────┘  └──────┬───────┘  │
│        │              │               │          │
│  ┌─────┴──────────────┴───────────────┴───────┐  │
│  │         基础能力层                           │  │
│  │  Embedding │ LLM  │ Splitter │ Extractor   │  │
│  │  Loader │ DocParser │ Cleaner │ Logger      │  │
│  └──────────────────┬─────────────────────────┘  │
└─────────────────────┼────────────────────────────┘
                      │
┌─────────────────────┴────────────────────────────┐
│                   存储层（均可替换）                   │
│    MySQL / PostgreSQL    │   ChromaDB / FAISS / Milvus / PGVector   │
│   events / entities /   │  事件向量 / 实体向量 /    │
│   chunks / sources /    │  chunk 向量 / 标题向量    │
│   event_entities        │                          │
│                         │  推荐 PGVector：与关系库  │
│                         │  共用一个 PostgreSQL      │
└───────────────────────────────────────────────────┘
```

### 查询链路详解

```
         ┌──────────────────────────────────────────┐
         │              查询："A 公司签了哪些合同？"    │
         └──────────────────┬───────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │     双路种子召回（粗筛）     │
              ├───────────────────────────┤
              │  A 路：SQL 实体关联查询      │
              │  B 路：事件标题向量相似匹配   │
              └─────────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
              │   SQL 动态超边 BFS 扩展     │
              │   通过 event_entities 表    │
              │   Join 发现多跳关联路径     │
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │   粗排 + LLM 精排            │
              │  向量相似度粗排 → LLM 重排    │
              │  枢纽实体抑制 + 融合去重截断  │
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │       LLM 生成回答          │
              │   基于 Top-K 上下文生成     │
              │   支持 SSE 流式输出          │
              └───────────────────────────┘
```

### 数据模型

```
Document (source)
  └── Chunk ── 1:1 ── Event
                          │
                    ┌─────┴─────┐
                    │           │
                 Entity₁     Entity₂
                  (通过 event_entities 多对多关联)
```

- **Chunk ↔ Event**：1:1 映射，一个语义块对应一个完整事件，不拆三元组
- **Event ↔ Entity**：多对多，通过 `event_entities` 关联表动态织网
- **11 类实体**：人物 / 机构 / 地点 / 时间 / 产品 / 主题 / 动作 / 指标 / 文件 / 法规 / 其他

---

## 功能特性

### 文档上传与管理

- 多格式上传：PDF / Word / Excel / Markdown / 纯文本
- 文档 CRUD：列表浏览、详情查看、内容更新、下载原文、删除（含关联清理）
- 全文检索：关键词命中 + 上下文片段高亮
- 归档管理：支持文档归档/取消归档，按状态筛选

### SAG 检索

- **双路融合召回**：结构化路径（SQL 实体关联）+ 语义路径（事件标题/摘要向量匹配），支持 `concat` / `supplement` 融合策略
- **实体向量扩展**：用 LLM 抽取的实体名 embedding 去向量库找近邻，补充同义词/别名（可配置开关 `AISAG_ENTITY_EXPAND_ENABLED`）
- **BFS 多跳扩展**：通过 `event_entities` 关联表动态发现跨文档实体路径，支持 `hopllm`（动态停止）和 `multi`（固定跳数）两种策略
- **枢纽实体抑制**：OTSU / percentile / MAD 离群检测自动过滤高频桥接实体（如"众邦银行"），抑制噪声扩散
- **种子实体保护**：独立于 BFS 的保守过滤参数，防止种子实体过度剔除导致断链
- **LLM 精排**：大模型对粗排结果二次排序，支持实体关联权重（强/弱信号）和事件深度衰减
- **全链路可追溯**：每次检索返回完整 `trace`，记录每一步的输入输出

### 对话问答

- 单轮问答 + 多轮对话（查询重写 + 指代消解）
- SSE 流式输出，Markdown 渲染
- 答案附带引用来源，可追溯每个论断的出处

### 系统运维

- `.env` 统一配置，重启即生效；`config.py` 自动发现 `.env`，无需关心 clone 路径
- LLM / Embedding / OCR / 切分器均可通过配置切换后端
- Swagger 文档（`:8777/docs`）提供完整 API 调试界面

### 可靠性

- **全链路异步**：PostgreSQL（asyncpg）/ MySQL（aiomysql）+ Embedding + LLM 全异步并发
- **跨库一致性**：关系库 ↔ 向量库自动对账，失败回滚
- **可观测性**：trace_id 注入 + 每步结果日志，便于排查检索效果

---

## 访问方式

系统提供两种访问入口，各司其职：

| 入口 | 地址 | 用途 |
|------|------|------|
| **Web UI** | `http://localhost:8080` | 对话交互界面，上传文档、检索问答、多轮对话一站完成 |
| **Swagger API** | `http://localhost:8777/docs` | API 调试界面，查看所有接口定义，在线测试请求响应 |
| **API 根路径** | `http://localhost:8777` | 直接访问根路径可进行对话和系统配置管理 |

### Web UI — 知识库控制台

浏览器打开 `http://localhost:8080`，进入「ai_sag 知识库控制台」：

- **左侧面板**：文档管理区 — 上传文件 / 纯文本，浏览文档列表，搜索、查看、下载、删除文档
- **右侧面板**：对话交互区 — 输入问题，选择检索策略，查看答案、引用片段、检索追踪链
- **底部区域**：多轮对话窗口 — 持续追问，系统自动处理指代消解和上下文关联

### Swagger API — 开发调试

浏览器打开 `http://localhost:8777/docs`，可交互式调试所有 15 个 API 端点，包括文档管理、检索、问答、多轮对话等。接口详情及调用示例见下方 [API 概览](#api-概览)。

---

## 快速开始

### 1. 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | ≥ 3.10 | 推荐 3.10 ~ 3.12 |
| 关系数据库 | PostgreSQL ≥ 13（推荐）/ MySQL ≥ 8.0 | 关系数据存储 |
| 向量数据库 | pgvector（推荐，与 PG 共用）/ ChromaDB / FAISS / Milvus | 向量检索 |
| pgvector 扩展 | ≥ 0.5.0 | PostgreSQL 向量扩展（使用 PGVector 后端时必装） |
| 磁盘空间 | ≥ 2 GB | 用于 Embedding 模型下载 |
| 内存 | ≥ 8 GB | 运行本地 Embedding + LLM 推理 |

> **推荐部署**：PostgreSQL + pgvector（一个数据库同时承担关系库 + 向量库，无需安装多个中间件，部署运维最简单）

### 2. 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd your-repo          # 进入仓库根目录（ai_sag_git/）

# 创建虚拟环境
conda create -n ai_sag python=3.10
conda activate ai_sag

# 安装依赖（二选一，不要叠加安装）
pip install -r ai_sag/requirements.txt        # CPU 环境
pip install -r ai_sag/requirements-gpu.txt    # GPU 环境（已自包含全部依赖）
```

> **⚠️ 重要：CPU 版与 GPU 版二选一，不要叠加安装**
>
> - **CPU 环境**：只装 `requirements.txt`
> - **GPU 环境**：只装 `requirements-gpu.txt`（**已自包含全部依赖**，无需先装 CPU 版）
>
> 叠加安装会导致 `torch` / `paddlepaddle` / `onnxruntime` 被反复卸载重装，浪费带宽且可能产生版本冲突。
>
> **GPU 版额外要求**：
> - 系统已安装 NVIDIA 驱动 + CUDA Toolkit（推荐 12.4）
> - `requirements-gpu.txt` 已配置 PyTorch 官方 CUDA 源和 PaddlePaddle 官方源，直接 `pip install` 即可
>
> **如何选择**：
> - 不确定或仅测试 → CPU 版（轻量、易装）
> - 有 NVIDIA GPU 且需大批量入库 / OCR → GPU 版（Embedding 和 OCR 都能显著加速）

### 3. 准备数据库

**推荐方案：PostgreSQL + pgvector**（一套数据库同时承担关系库 + 向量库，无需额外中间件）

```sql
-- 1. 创建数据库
CREATE DATABASE sag;

-- 2. 连接到 sag 数据库后，安装 pgvector 扩展
\c sag
CREATE EXTENSION IF NOT EXISTS vector;
```

**备选方案 A：MySQL**（仅作关系库，向量库需另选 Chroma/FAISS/Milvus）
```sql
CREATE DATABASE IF NOT EXISTS sag DEFAULT CHARSET utf8mb4;
```

**备选方案 B：PostgreSQL + 其他向量库**（关系库用 PG，向量库用 Chroma/FAISS/Milvus）

表结构会在首次启动 API 时**自动创建**（MySQL 用 `schema.sql`，PG 用 `schema_pg.sql`），向量表由 PGVector 自动创建，均无需手动建表。

### 4. 配置

```bash
cd ai_sag

# 复制配置模板
cp .env.example .env    # Linux / Mac
copy .env.example .env  # Windows

# 编辑 .env，填入关键配置
```

必填配置项（其余配置均有合理默认值，初次使用无需修改）：

| 变量 | 说明 | 示例 |
|------|------|------|
| `AISAG_DB_BACKEND` | 关系库后端（`postgresql` 推荐 / `mysql`） | `postgresql` |
| `SAG_PG_PASSWORD` | PostgreSQL 密码（使用 PG 时必填） | `your_password` |
| `SAG_MYSQL_PASSWORD` | MySQL 密码（使用 MySQL 时必填） | `your_password` |
| `AISAG_VECTOR_STORE_BACKEND` | 向量库后端（`pgvector` 推荐 / `chroma` / `faiss` / `milvus`） | `pgvector` |
| `AISAG_PG_CONNECTION_STRING` | PGVector 连接串（使用 pgvector 时必填） | `postgresql://user:pwd@host:port/db` |
| `SAG_LLM_PROFILE_<场景>_LLM_NAME` | 各场景选用的 profile 名（6 个场景必填） | `deepseek_chat` |
| `SAG_EMBEDDING_BACKEND` | Embedding 后端（`bge` / `qwen3`） | `bge` |
| `SAG_EMBEDDING_MODEL_PATH` | Embedding 模型本地路径（两种后端共用） | `/models/bge-small-zh-v1.5` |

> **推荐配置**：`AISAG_DB_BACKEND=postgresql` + `AISAG_VECTOR_STORE_BACKEND=pgvector`，两者共用一个 PostgreSQL 实例，部署最简单。

> **LLM 连接信息**：`api_key` / `base_url` / `model` 等直接写在 `llm_profiles.yaml` 中（参考 `llm_profiles.yaml.example` 创建，文件已在 `.gitignore` 中）。后端由 factory 按 `profile.model` 自动判断，无需配置。

> **提示**：
> 1. `config.py` 自动从自身目录逐级向上搜索 `.env`，无论仓库 clone 到哪个路径，无需修改代码。
> 2. `.env.example` 列出了全部 **48 个可配置项**，每个参数都有默认值注释，按需取消注释调整即可。
> 3. 实体向量扩展默认开启（`AISAG_ENTITY_EXPAND_ENABLED=true`），若召回噪声多可手动设为 `false` 退化为精确匹配。

### 5. 一键启动（推荐）

项目提供了 `setup` 脚本，自动完成虚拟环境创建、依赖安装、服务启动：

```bash
cd ai_sag/scripts

# 默认 CPU 模式（创建 .venv → pip install requirements.txt → 启动 API + Web UI）
./setup.sh                     # Linux / macOS
setup.bat                      # Windows

# GPU 模式（创建 .venv → pip install requirements-gpu.txt → 启动 API + Web UI）
# 注意：GPU 模式只装 requirements-gpu.txt，不会叠加 CPU 版
./setup.sh gpu

# 仅安装依赖
./setup.sh install

# 仅检查环境（Python 版本、.env 配置、关系库连接）
./setup.sh check

# 仅启动服务（跳过安装）
./setup.sh start
```

脚本流程：检查 Python 3.10+ → 创建虚拟环境 → 安装依赖（CPU/GPU 二选一，不叠加）→ 首次运行时从 `.env.example` 复制配置模板 → 启动 `api.py` (8777) + `web.py` (8080)。

### 6. 手动启动

需要两个终端分别启动 API 和 Web UI。**注意必须在仓库根目录（`ai_sag_git/`）执行，不是在 `ai_sag/` 子目录！**

#### 6.1 启动 API 服务（终端 1）

```bash
# 1. 进入仓库根目录
cd your-repo

# 2. 激活 conda 环境
conda activate ai_sag

# 3. 启动（默认监听 0.0.0.0:8777，首次启动自动建表）
python -m ai_sag.api

# 或指定端口 + 热重载（开发时推荐）
python -m ai_sag.api --host 0.0.0.0 --port 8777 --reload
```

启动成功后访问 **http://localhost:8777/docs** 查看 Swagger API 文档。

#### 6.2 启动 Web UI（终端 2）

```bash
# 1. 进入仓库根目录
cd your-repo

# 2. 激活 conda 环境
conda activate ai_sag

# 3. 启动 Web UI（默认 0.0.0.0:8080）
python -m ai_sag.web --port 8080 --api http://localhost:8777
```

启动成功后：
- 浏览器打开 **http://localhost:8080** → Web 知识库控制台
- 浏览器打开 **http://localhost:8777/docs** → Swagger API 文档
- 访问 **http://localhost:8777** → 根路径对话与系统配置管理

> 详细启动说明、完整 API 接口、检索参数调优、常见问题等请参阅 **[docs/STARTUP.md](./docs/STARTUP.md)**。

---

## 技术栈

### 分层一览

| 层级 | 技术选型 | 版本要求 |
|------|---------|---------|
| **RAG 框架** | LlamaIndex（切分/Embedding 底层） | ≥ 0.12.0 |
| **Web 服务** | FastAPI + Uvicorn | — |
| **向量数据库** | PGVector（推荐）/ ChromaDB / FAISS / Milvus | pgvector ≥ 0.5.0 |
| **关系数据库** | PostgreSQL + asyncpg（推荐）/ MySQL + aiomysql | PG ≥ 13 / MySQL ≥ 8.0 |
| **LLM 后端** | DeepSeek / OpenAI / vLLM / Ollama | OpenAI 兼容协议 |
| **Embedding** | BAAI/bge-small-zh-v1.5 / Qwen3-Embedding-0.6B | 本地推理 |
| **文档解析** | PyMuPDF + python-docx + openpyxl + PaddleOCR / RapidOCR | — |
| **文本切分** | 自研 ChunkSplitter（语义/Markdown/代码等多模式） | 底层依赖 LlamaIndex |
| **异步运行时** | asyncio（全链路异步 I/O） | Python 3.10+ |
| **日志** | loguru | 控制台 + 文件双输出 |
| **配置管理** | python-dotenv + dataclasses | Pydantic 仅用于 API 请求/响应模型 |
| **前端** | 原生 HTML/CSS/JS + marked.js | 无构建依赖 |

### 组件可切换

所有环境变量均以 `AISAG_` 前缀开头：

| 组件 | 可选后端 | 配置方式（`.env` 中） |
|------|---------|----------------------|
| 关系数据库 | `postgresql`（推荐）/ `mysql` | `AISAG_DB_BACKEND` + `SAG_PG_*` / `SAG_MYSQL_*` |
| 向量库 | `pgvector`（推荐，与 PG 共用）/ `chroma` / `faiss` / `milvus` | `AISAG_VECTOR_STORE_BACKEND` |
| Embedding | `bge`（默认，CLS pooling）/ `qwen3`（last_token pooling + 查询指令前缀） | `SAG_EMBEDDING_BACKEND` + `SAG_EMBEDDING_MODEL_PATH` |
| LLM | `openai_like` / `openai`（按 profile.model 自动判断） | 无需配置 |
| OCR 引擎 | `rapidocr`（默认，轻量）/ `paddleocr`（高精度） | `AISAG_DOC_OCR_BACKEND` |
| OCR 开关 | `true`（默认，OCR图片）/ `false`（不OCR，入库更快） | `AISAG_DOC_OCR_IMAGES` |
| PDF Markdown 模式 | `direct`（默认，快）/ `pymupdf4llm`（表格更准） | `AISAG_PDF_MARKDOWN_MODE` |
| 切分器 | `semantic`（默认）/ `auto` / `markdown` / `sentence` / `token` / `code` | `AISAG_SPLITTER_MODE` |
| 切分语言（code 模式） | `python` / `java` / `go` / 等 | `AISAG_SPLITTER_LANGUAGE` |
| 实体扩展开关 | `true`（默认）/ `false`（仅精确匹配） | `AISAG_ENTITY_EXPAND_ENABLED` |
| BFS 扩展策略 | `hopllm`（默认，动态停止）/ `multi`（固定跳数） | `AISAG_SUB_STRATEGY` |
| 双路融合策略 | `concat`（默认拼接）/ `supplement`（事件为主补足） | API 请求参数 `fusion_strategy` |

> **存储后端**：关系库（MySQL/PG）和向量库（Chroma/FAISS/Milvus/PGVector）均已实现完整支持，**通过 `.env` 配置即可切换，无需改代码**。详细步骤见 **[docs/存储后端替换指南.md](./docs/存储后端替换指南.md)**。

---

## 项目结构

```
your-repo/
└── ai_sag/
    ├── api.py                  ← FastAPI 后端（端口 8777，15 个端点）
    ├── web.py                  ← Web UI 服务（端口 8080）
    ├── requirements.txt        ← 依赖清单（CPU 完整版）
    ├── requirements-gpu.txt    ← 依赖清单（GPU 完整版，自包含，不叠加 CPU 版）
    ├── .env.example            ← 环境变量模板
    ├── .env                    ← 实际配置（不提交 git）
    ├── README.md               ← 项目总览（本文档）
    │
    ├── docs/                   ← 技术文档
    │   ├── STARTUP.md          ← 启动与使用指南
    │   ├── 入库流程.md          ← 入库流程中文详解
    │   ├── 检索流程.md          ← 检索全链路详解（双路召回→BFS→精排→生成）
    │   └── 存储后端替换指南.md    ← MySQL/PG + Chroma/FAISS/Milvus/PGVector 切换指南
    │
    ├── base/                   ← 基础设施：配置、日志、数据模型、提示词
    ├── retrieval/              ← SAG 检索器（核心）+ 问答引擎
    ├── ingest/                 ← 入库流水线编排
    ├── embeddings/             ← BGE / Qwen3 向量化
    ├── splitter/               ← 文档语义切分
    ├── extractor/              ← 事件 & 实体抽取
    ├── vector_store/           ← 向量库封装（PGVector 推荐 / Chroma/FAISS/Milvus，5 个 collection）
    ├── storage/                ← 关系库持久化（PostgreSQL 推荐 / MySQL，6 张表，全异步）
    ├── llm/                    ← DeepSeek / OpenAI 兼容 LLM 封装
    ├── loader/                 ← 文件加载（md/txt/docx/pdf/xlsx）
    ├── doc_parser/             ← 文档深度解析（OCR、版式分析等）
    ├── cleaner/                ← 文本清洗
    ├── static/                 ← Web 前端页面
    └── scripts/                ← 一键部署脚本（setup.sh / setup.bat）
```

---

## API 概览

### 文档管理（9 个端点）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/documents` | 上传文件（md/txt/docx/pdf/xlsx） |
| `POST` | `/api/documents/text` | 上传纯文本 JSON |
| `GET` | `/api/documents` | 文档列表（关键词/分页/归档筛选） |
| `GET` | `/api/documents/search` | 全文搜索（含上下文片段） |
| `GET` | `/api/documents/{id}` | 文档详情（含 chunk/事件统计） |
| `GET` | `/api/documents/{id}/download` | 下载原文（.md） |
| `PATCH` | `/api/documents/{id}` | 更新文档元信息（标题/描述） |
| `PUT` | `/api/documents/{id}` | 更新文档内容（重建索引） |
| `DELETE` | `/api/documents/{id}` | 删除文档（含事件+孤儿实体清理） |

### 检索与问答（4 个端点）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/search` | SAG 检索（切片 + trace 审计链） |
| `POST` | `/api/ask` | 单轮问答（答案 + 引用来源） |
| `POST` | `/api/chat` | 多轮对话 |
| `POST` | `/api/chat/stream` | 多轮对话（SSE 流式） |

### 系统管理（2 个端点）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/stats` | 统计信息（文档数/归档数/chunk 数） |
| `GET` | `/api/health` | 健康检查（含数据库连接检测） |

### 调用示例

```bash
# 上传文档
curl -X POST http://localhost:8777/api/documents -F "file=@合同公告.md"

# 问答
curl -X POST http://localhost:8777/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"数脉科技跟哪个公司签了合同？"}'

# SSE 流式对话
curl -N -X POST http://localhost:8777/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"详细说一下合同金额条款"}'
```

---

## 适用场景

| 场景 | 说明 | 为什么适合 SAG |
|------|------|---------------|
| **合同分析** | 跨文档关联甲方/乙方/签署日期/金额等实体 | 多实体交叉关联，SQL Join 动态织网 |
| **供应链溯源** | 多跳追踪上下游企业与产品关系 | 实体间多层关联，BFS 逐跳扩展 |
| **人物关系追踪** | 跨新闻报道发现人物-机构-事件关联 | 多文档实体共指，动态发现隐蔽关系 |
| **法规合规检索** | 法规条文 ↔ 适用场景 ↔ 处罚案例 | 条文、场景、案例三类实体交叉检索 |
| **金融研报分析** | 公司-指标-行业-事件的交叉检索 | 多维度实体关联，超越关键词匹配 |

**不适合**：单轮简单事实查询（传统 RAG 即可）；需要复杂逻辑推理但不涉及实体关联的问答。

---

## 调优参考

检索效果不理想时，可调整 `ai_sag/.env` 中的关键参数（所有参数均以 `AISAG_` 为前缀）：

| 问题 | 调整方向 | 相关参数 |
|------|---------|---------|
| 召回太少 | ↓ 相似度阈值、↑ 最大跳数、↓ 实体扩展阈值 | `AISAG_SIMILARITY_THRESHOLD`、`AISAG_MAX_HOPS`、`AISAG_ENTITY_EXPAND_THRESHOLD` |
| 召回噪声多 | ↑ 实体扩展阈值、↓ 最大跳数、开启枢纽实体抑制 | `AISAG_ENTITY_EXPAND_THRESHOLD`、`AISAG_MAX_HOPS`、`AISAG_ENTITY_FRONTIER_FILTER=true` |
| 上下文不够 | ↑ 重排 Top-K、↑ 最大片段数、↑ 粗排候选 | `AISAG_RERANK_TOP_K`、`AISAG_MAX_SECTIONS`、`AISAG_MAX_EVENTS` |
| BFS 太慢 | ↓ 前沿预算、启用 hopllm 动态停止策略、↓ 每跳种子数 | `AISAG_ENTITY_FRONTIER_BUDGET`、`AISAG_SUB_STRATEGY=hopllm`、`AISAG_HOP_SEED_TOPK` |
| 枢纽实体噪声多（如"众邦银行"类高频实体污染） | 启用 OTSU 离群过滤、降低度数绝对上限 | `AISAG_ENTITY_DEGREE_METHOD=otsu`、`AISAG_ENTITY_DEGREE_ABS_MAX=30~50` |
| 切分太碎 | ↑ chunk 大小、↓ 重叠、↓ 断点分位阈值 | `AISAG_CHUNK_SIZE`、`AISAG_CHUNK_OVERLAP`、`AISAG_BREAKPOINT_PERCENTILE` |
| 入库太快但抽取质量差 | 开启文体识别、关闭摘要向量生成不影响效果仅省算力 | `AISAG_GENRE_DETECT=true`、`AISAG_EMBED_SUMMARY=true` |

> 完整参数说明与默认值见 `.env.example` 注释，详细调优指南见 **[docs/STARTUP.md § 九、检索参数详解](./docs/STARTUP.md)**。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| **[STARTUP.md](./docs/STARTUP.md)** | 启动与使用指南：环境准备、配置、启动、API 调用示例、检索参数调优、常见问题 |
| **[入库流程.md](./docs/入库流程.md)** | 入库流程中文详解：概要流程 → 流程细节 → 实现逻辑，三个层次逐级深入 |
| **[检索流程.md](./docs/检索流程.md)** | 用户问答检索全链路：双路召回 → BFS 扩展 → 粗排精排 → 生成回答，含枢纽实体过滤、hopllm动态停止等最新功能 |
| **[存储后端替换指南.md](./docs/存储后端替换指南.md)** | MySQL ↔ PostgreSQL / Chroma ↔ FAISS/Milvus/PGVector 的切换步骤、配置说明、SQL 语法适配 |

---

## 许可证

内部项目，仅供团队使用。