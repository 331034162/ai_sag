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
- [存储后端替换指南](#存储后端替换指南)
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
| **离线（入库）** | 文档切分 → 事件抽取 → 实体识别 → 建事件-实体多对多索引（MySQL） + 向量化（ChromaDB） | 不建知识图谱、不预计算关联路径 |
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
| **资源占用** | 低（仅向量库） | **高**（图数据库 + 向量库） | 中（关系库 + 向量库，默认 MySQL + ChromaDB，均可替换） |
| **Schema 依赖** | 无 | **有**（需预定义实体类型和关系类型） | 弱（实体类型用于辅助分类，不依赖固定关系） |

### 各自的优势与劣势

#### 传统 RAG

| 优势 | 劣势 |
|------|------|
| 实现简单，组件成熟，生态丰富 | 无法处理多跳关联（"A 签了哪些合同？"需多步跳转） |
| 离线成本低，更新快 | 跨文档实体关联几乎为零（各文档向量孤岛） |
| 无 Schema 约束，通用性强 | 答案来源不透明，难以追溯推理链 |
| 增量入库零摩擦 | 对关键词不敏感的语义查询可能漏召回 |

#### GraphRAG

| 优势 | 劣势 |
|------|------|
| 多跳推理能力天然强（图遍历） | **建图成本极高**（实体对齐、关系抽取、消歧、存储） |
| 实体关系路径清晰可解释 | Schema 需提前定义，场景适配差 |
| 一次建图，多次高效查询 | 更新代价大（新增文档需重建/增量更新图谱） |
| 结构化关联查询效率高 | 对非实体类语义查询支持弱（如"摘要概括""情感分析"） |

#### SAG（本项目）

| 优势 | 劣势 |
|------|------|
| **兼具两者的优点**：有 GraphRAG 的多跳能力 + 传统 RAG 的灵活性 | 查询时 SQL Join 开销随跳数线性增长（需调优 BFS 预算） |
| **零静态图维护**：不存图谱，只存事件-实体关联表 | 对超大规模知识库（亿级事件），SQL Join 性能需关注 |
| **更新即生效**：入库操作就是表 INSERT，无重建成本 | 需同时维护关系库 + 向量库两个存储（但可通过接口替换为其他后端） |
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
│    MySQL / PostgreSQL    │   ChromaDB / Milvus / ...   │
│   events / entities /   │  事件向量 / 实体向量 /    │
│   chunks / sources /    │  chunk 向量 / 标题向量    │
│   event_entities        │                          │
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
              │     粗排 + LLM 精排         │
              │   BM25 → Reranker LLM     │
              │   融合排序，去重截断         │
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

### 📄 文档管理

- 多格式上传：PDF / Word / Excel / Markdown / 纯文本
- 文档列表：分页浏览 + 关键词搜索 + 归档筛选
- 文档操作：查看详情、更新内容、下载原文、删除（含关联清理）
- 全文检索：关键词命中 + 上下文片段高亮

### 🔍 智能检索

- **双路融合召回**：结构化路径（SQL 实体关联） + 语义路径（事件标题向量匹配）
- 支持 `concat`（并集）和 `supplement`（补集）两种融合策略
- **BFS 多跳扩展**：通过 `event_entities` 关联表动态发现跨文档实体路径
- **LLM 精排**：用大模型对粗排结果二次排序，提升精准度
- **全链路可追溯**：每次检索返回完整 `trace`，记录每一步的输入输出

### 💬 对话问答

- 单轮问答：基于检索到的上下文直接生成答案
- 多轮对话：查询重写 + 指代消解，支持上下文连续追问
- SSE 流式输出：实时打字效果，降低等待焦虑
- Markdown 渲染：答案中的代码块、表格、列表自动美化

### ⚙️ 系统配置

- `.env` 统一配置：所有参数集中在环境变量文件，重启即生效
- 配置自发现：`config.py` 从自身目录逐级向上搜索 `.env`，无需关心 clone 路径
- 组件可插拔：LLM / Embedding / OCR / 切分器均可通过配置切换后端
- Swagger 文档：`http://localhost:8777/docs` 提供完整 API 调试界面

### 🛡️ 可靠性

- **全链路异步**：MySQL（aiomysql）+ Embedding + LLM，高效并发；存储后端可替换
- **跨库一致性保障**：MySQL ↔ ChromaDB 自动对账，失败回滚；可扩展至其他数据库组合
- **全链路可观测**：trace_id 注入 + 每步结果记录，便于排查检索效果

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

浏览器打开 `http://localhost:8777/docs`，可交互式调试所有 15 个 API 端点，包括文档管理、检索、问答、多轮对话等。

### 命令行调用

```bash
# 上传文档
curl -X POST http://localhost:8777/api/documents -F "file=@合同公告.md"

# 单轮问答
curl -X POST http://localhost:8777/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"数脉科技跟哪个公司签了合同？"}'

# SSE 流式对话
curl -X POST http://localhost:8777/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"这份合同的主要条款有哪些？"}'
```

---

## 快速开始

### 1. 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | ≥ 3.10 | 推荐 3.10 ~ 3.12 |
| MySQL | ≥ 8.0 | 关系数据存储 |
| 磁盘空间 | ≥ 2 GB | 用于 Embedding 模型下载 |
| 内存 | ≥ 8 GB | 运行本地 Embedding + LLM 推理 |

### 2. 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd your-repo          # 进入仓库根目录（ai_sag_git/）

# 创建虚拟环境
conda create -n ai_sag python=3.10
conda activate ai_sag

# 安装依赖
pip install -r ai_sag/requirements.txt
```

### 3. 准备 MySQL

```sql
CREATE DATABASE IF NOT EXISTS sag DEFAULT CHARSET utf8mb4;
```

表结构会在首次启动 API 时**自动创建**，无需手动建表。

### 4. 配置

```bash
cd ai_sag

# 复制配置模板
cp .env.example .env    # Linux / Mac
copy .env.example .env  # Windows

# 编辑 .env，填入关键配置
```

必填配置项：

| 变量 | 说明 | 示例 |
|------|------|------|
| `SAG_MYSQL_PASSWORD` | MySQL 密码 | `your_password` |
| `SAG_LLM_API_KEY` | LLM API Key | `sk-xxxxxxxx` |
| `SAG_BGE_MODEL_PATH` | BGE 模型本地路径 | `/models/bge-small-zh-v1.5` |

> **提示**：`config.py` 自动从自身目录逐级向上搜索 `.env`，无论仓库 clone 到哪个路径，无需修改代码。

### 5. 启动

需要两个终端分别启动 API 和 Web UI：

```bash
# 确保在仓库根目录（ai_sag_git/），不是 ai_sag/ 子目录！

# 终端 1：API 服务（端口 8777）
cd your-repo
python -m ai_sag.api

# 终端 2：Web UI（端口 8080）
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
| **RAG 框架** | LlamaIndex | ≥ 0.12.0 |
| **Web 服务** | FastAPI + Uvicorn | — |
| **向量数据库** | ChromaDB（可替换为 Milvus / Qdrant 等） | ≥ 1.0.0 |
| **关系数据库** | MySQL + aiomysql（可替换为 PostgreSQL 等） | MySQL ≥ 8.0 |
| **LLM 后端** | DeepSeek / OpenAI / vLLM / Ollama | OpenAI 兼容协议 |
| **Embedding** | BAAI/bge-small-zh-v1.5 / Qwen3-Embedding-0.6B | 本地推理 |
| **文档解析** | PyMuPDF + python-docx + openpyxl + PaddleOCR / RapidOCR | — |
| **文本切分** | LlamaIndex SentenceSplitter / TokenTextSplitter | — |
| **异步运行时** | asyncio（全链路异步 I/O） | Python 3.10+ |
| **日志** | loguru | 控制台 + 文件双输出 |
| **配置管理** | python-dotenv + Pydantic v2 | — |
| **前端** | 原生 HTML/CSS/JS + marked.js | 无构建依赖 |

### 组件可切换

| 组件 | 可选后端 | 配置方式 |
|------|---------|---------|
| Embedding | `bge` / `qwen3` | `.env` 中 `EMBEDDING_BACKEND` |
| LLM | `deepseek` / `openai_like` / `openai` | `.env` 中 `LLM_BACKEND` |
| OCR | `rapidocr` / `paddleocr` | `.env` 中 `OCR_BACKEND` |
| 切分器 | `semantic` / `auto` / `markdown` / `sentence` | `.env` 中 `SPLIT_MODE` |
| 融合策略 | `concat` / `supplement` | API 请求参数 `fusion_strategy` |
| 向量库 | ChromaDB（默认），可扩展 Milvus / Qdrant | `.env` 中 `VECTOR_STORE_BACKEND` |
| 关系库 | MySQL（默认），可扩展 PostgreSQL | `.env` 中 `MYSQL_*` 连接配置 |

> **注意**：关系库和向量库的替换目前需要**二次开发**（非配置即可切换）。详情见下方 [存储后端替换指南](#存储后端替换指南)。

### 存储后端替换指南

当前默认使用 MySQL + ChromaDB，替换为其他数据库需进行二次开发。两部分的替换成本差异较大：

#### 向量库替换（ChromaDB → Milvus / Qdrant）— 成本较低

向量库已有完善的 `BaseVectorStore` 抽象基类 + `create_vector_store()` 工厂模式，所有调用方依赖抽象接口，仅需新增实现类：

| 序号 | 文件 | 修改内容 | 工作量 |
|------|------|---------|--------|
| 1 | `vector_store/milvus_store.py` | **新建** — 实现 `BaseVectorStore` 的 9 个抽象方法（add / query / delete / get_embeddings 等） | 高 |
| 2 | `vector_store/factory.py` | **修改** — 注册新后端分支，取消 `NotImplementedError` | 极低 |
| 3 | `vector_store/__init__.py` | **修改** — 导出新实现类 | 极低 |
| 4 | `base/config.py` | **修改** — `VectorStoreConfig` 增加连接参数（host / port / collection） | 低 |

> `ingest/pipeline.py`、`retrieval/sag_retriever.py`、`retrieval/qa_engine.py` 等 **3 个调用方文件无需修改**，因为它们只依赖 `BaseVectorStore` 抽象接口。

#### 关系库替换（MySQL → PostgreSQL）— 成本较高

关系库目前**没有抽象接口层**，所有调用方直接依赖 `MysqlStore` 具体类。替换需先建立抽象层，再实现新后端：

| 序号 | 文件 | 修改内容 | 工作量 |
|------|------|---------|--------|
| 1 | `storage/base.py` | **新建** — 定义 `BaseRelationalStore` 抽象基类，声明 CRUD 方法签名 | 中 |
| 2 | `storage/postgres_store.py` | **新建** — 使用 `asyncpg` 实现 `BaseRelationalStore`，适配 PostgreSQL | **高** |
| 3 | `storage/schema_pg.sql` | **新建** — PostgreSQL 版 DDL 建表语句 | 低 |
| 4 | `storage/factory.py` | **新建** — `create_relational_store(cfg)` 工厂函数 | 低 |
| 5 | `storage/__init__.py` | **修改** — 导出工厂和新实现 | 低 |
| 6 | `storage/mysql_store.py` | **修改** — 让 `MysqlStore` 继承 `BaseRelationalStore` | 低 |
| 7 | `base/config.py` | **修改** — 改造 `MysqlConfig` 或新增 `PostgresConfig` | 中 |
| 8 | `ingest/pipeline.py` | **修改** — `from ..storage import MysqlStore` 改用工厂创建 | 低 |
| 9 | `retrieval/sag_retriever.py` | **修改** — 类型注解 `MysqlStore` → `BaseRelationalStore` | 低 |
| 10 | `retrieval/qa_engine.py` | **修改** — import + 类型注解 + 实例化改用工厂 | 低 |

**MySQL 特有语法需适配**（共约 8 处）：

| MySQL 语法 | 出现位置 | PostgreSQL 等价写法 |
|-----------|---------|-------------------|
| `ON DUPLICATE KEY UPDATE` | `mysql_store.py` 4 处（upsert 逻辑） | `INSERT ... ON CONFLICT ... DO UPDATE` |
| `INSERT IGNORE INTO` | `mysql_store.py` 2 处（去重插入） | `INSERT ... ON CONFLICT DO NOTHING` |
| `JSON_SET()` / `JSON_EXTRACT()` | `mysql_store.py` 2 处（metadata 操作） | `jsonb_set()` / `->>` 操作符 |
| `ON UPDATE CURRENT_TIMESTAMP` | `schema.sql` 2 处（自动更新时间戳） | 需用触发器实现 |

---

## 项目结构

```
your-repo/
└── ai_sag/
    ├── api.py                  ← FastAPI 后端（端口 8777，15 个端点）
    ├── web.py                  ← Web UI 服务（端口 8080）
    ├── requirements.txt        ← 依赖清单
    ├── requirements-gpu.txt    ← GPU 版依赖
    ├── .env.example            ← 环境变量模板
    ├── .env                    ← 实际配置（不提交 git）
    ├── README.md               ← 项目总览（本文档）
    │
    ├── docs/                   ← 技术文档
    │   ├── STARTUP.md          ← 启动与使用指南
    │   ├── FLOW.md             ← 全流程详解（离线建库 + 在线检索）
    │   ├── INGEST_FLOW.md      ← 入库流程（英文版）
    │   ├── 入库流程.md          ← 入库流程中文详解
    │   ├── 问答流程.md          ← 用户问答处理全链路
    │   ├── 问答对话参数配置.md   ← 问答链路所有参数汇总
    │   ├── 多轮对话历史处理.md   ← 多轮对话历史流转与实现
    │   ├── 内容字段抽取与作用详解.md ← 字段抽取逻辑
    │   ├── 实体字段抽取与作用详解.md ← 实体字段抽取逻辑
    │   └── 多跳查询性能隐患.md   ← BFS 性能分析与优化
    │
    ├── base/                   ← 基础设施：配置、日志、数据模型、提示词
    ├── retrieval/              ← SAG 检索器（核心）+ 问答引擎
    ├── ingest/                 ← 入库流水线编排
    ├── embeddings/             ← BGE / Qwen3 向量化
    ├── splitter/               ← 文档语义切分
    ├── extractor/              ← 事件 & 实体抽取
    ├── vector_store/           ← ChromaDB 封装（4 个 collection）
    ├── storage/                ← MySQL 持久化（6 张表，全异步）
    ├── llm/                    ← DeepSeek / OpenAI 兼容 LLM 封装
    ├── loader/                 ← 文件加载（md/txt/docx/pdf/xlsx）
    ├── doc_parser/             ← 文档深度解析（OCR、版式分析等）
    ├── cleaner/                ← 文本清洗
    └── static/                 ← Web 前端页面
```

---

## API 概览

### 文档管理（8 个端点）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/documents` | 上传文件（md/txt/docx/pdf/xlsx） |
| `POST` | `/api/documents/text` | 上传纯文本 JSON |
| `GET` | `/api/documents` | 文档列表（关键词/分页/归档筛选） |
| `GET` | `/api/documents/search` | 全文搜索（含上下文片段） |
| `GET` | `/api/documents/{id}` | 文档详情（含 chunk/事件统计） |
| `GET` | `/api/documents/{id}/download` | 下载原文（.md） |
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

检索效果不理想时，可调整 `ai_sag/.env` 中的关键参数：

| 问题 | 调整方向 | 相关参数 |
|------|---------|---------|
| 召回太少 | ↓ 相似度阈值、↑ 最大跳数、↓ 实体扩展阈值 | `SIMILARITY_THRESHOLD`、`MAX_HOPS`、`ENTITY_EXPAND_THRESHOLD` |
| 召回噪声多 | ↑ 实体扩展阈值、↓ 最大跳数 | `ENTITY_EXPAND_THRESHOLD`、`MAX_HOPS` |
| 上下文不够 | ↑ 重排 Top-K、↑ 最大片段数 | `RERANK_TOP_K`、`MAX_SECTIONS` |
| BFS 太慢 | ↓ 前沿预算、启用 hopllm 子策略 | `ENTITY_FRONTIER_BUDGET`、`SUB_STRATEGY=hopllm` |
| 切分太碎 | ↑ chunk 大小、↓ 重叠 | `CHUNK_SIZE`、`CHUNK_OVERLAP` |

> 完整参数说明见 **[docs/STARTUP.md § 九、检索参数详解](./docs/STARTUP.md)**。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| **[STARTUP.md](./docs/STARTUP.md)** | 启动与使用指南：环境准备、配置、启动、API 调用示例、检索参数调优、常见问题 |
| **[FLOW.md](./docs/FLOW.md)** | 全流程详解（英文），覆盖离线建库和在线检索两个阶段的每一步及对应代码位置 |
| **[INGEST_FLOW.md](./docs/INGEST_FLOW.md)** | 文档离线入库全流程（英文）：loader → cleaner → splitter → extractor → 持久化 |
| **[入库流程.md](./docs/入库流程.md)** | 入库流程中文详解：概要流程 → 流程细节 → 实现逻辑，三个层次逐级深入 |
| **[问答流程.md](./docs/问答流程.md)** | 用户问答处理全链路：双路召回 → BFS 扩展 → 粗排精排 → 生成回答 |
| **[问答对话参数配置.md](./docs/问答对话参数配置.md)** | 问答链路所有参数汇总：环境变量、检索参数、对话参数、LLM 参数、Prompt 模板 |
| **[多轮对话历史处理.md](./docs/多轮对话历史处理.md)** | 多轮对话中"历史"在各环节的作用、流转、实现逻辑（含流程图、实例分析） |
| **[内容字段抽取与作用详解.md](./docs/内容字段抽取与作用详解.md)** | Events / Chunks / Documents 三类内容的字段抽取逻辑及在问答中的作用 |
| **[实体字段抽取与作用详解.md](./docs/实体字段抽取与作用详解.md)** | Entities / Event_Entities 字段抽取逻辑及在问答中的作用 |
| **[多跳查询性能隐患.md](./docs/多跳查询性能隐患.md)** | SAG BFS 多跳扩展的 SQL 性能分析：索引命中、潜在瓶颈、优化方案 |

---

## 许可证

内部项目，仅供团队使用。
