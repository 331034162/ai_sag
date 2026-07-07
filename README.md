# ai_sag — 基于 SAG 的轻量级多跳 RAG 系统

基于论文 **SAG（SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges）** 实现的知识库检索增强生成系统。专为需要跨文档、跨片段、多跳实体关联的场景设计。

---

## 核心思路

> **离线不建图，在线动态织网。**

| 阶段 | 做了什么 | 不做什么 |
|------|---------|---------|
| **离线（入库）** | 文档切分 → 事件抽取 → 实体识别 → 建事件-实体多对多索引（MySQL） + 向量化（ChromaDB） | 不建知识图谱、不预计算关联路径 |
| **在线（查询）** | 双路种子召回 → SQL Join 动态超边 BFS 扩展 → 粗排 → LLM 精排 → 生成回答 | 不提前固定图结构，路径用完即弃 |

### 为什么不是知识图谱？

传统 KG-RAG 离线构建图结构，面临"建图成本高、模式难预定义、更新困难"的问题。SAG 把关联关系推迟到查询时动态发现：用 SQL Join 临时织网，查询结束即丢弃，零静态图维护成本。

---

## 架构概览

```
         ┌──────────────────────────────────────────┐
         │              查询："A 公司签了哪些合同？"    │
         └──────────────────┬───────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │     双路种子召回（粗筛）     │
              ├───────────────────────────┤
              │  A 路：SQL 实体关联         │
              │  B 路：事件标题向量匹配      │
              └─────────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
              │   SQL 动态超边 BFS 扩展     │
              │   （多跳实体关联路径发现）    │
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │     粗排 + LLM 精排         │
              │   （融合排序，去重截断）      │
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │       LLM 生成回答          │
              └───────────────────────────┘
```

---

## 功能特性

- **动态多跳关联**：查询时 SQL Join 发现跨文档的实体关联路径（支持 BFS 多跳扩展）
- **双路融合检索**：结构化路径（SQL） + 语义路径（向量），`concat` / `supplement` 两种融合策略
- **Chunk ↔ Event 1:1 映射**：保留完整语义，不拆三元组
- **11 类实体识别**：人物/机构/地点/时间/产品/主题/动作/指标/文件/法规/其他
- **全链路异步**：MySQL（aiomysql）+ Embedding + LLM，高效并发
- **跨库一致性保障**：MySQL ↔ ChromaDB 自动对账，失败回滚
- **全链路可观测**：trace_id 注入 + 每步结果记录，便于排查检索效果
- **多格式文档支持**：PDF / Word / Excel / Markdown / 纯文本，含 OCR 图片文字提取
- **LLM 可插拔**：DeepSeek / OpenAI / vLLM / Ollama 等任意 OpenAI 兼容协议
- **多轮对话**：查询重写 + 指代消解 + SSE 流式输出

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- MySQL 8.0+
- 本地 Embedding 模型（BAAI/bge-small-zh-v1.5 或 Qwen3-Embedding-0.6B）

### 2. 安装

```bash
# 创建虚拟环境
conda create -n ai_sag python=3.10
conda activate ai_sag

# 安装依赖
cd your-repo/ai_sag
pip install -r requirements.txt
```

### 3. 准备 MySQL

```sql
CREATE DATABASE IF NOT EXISTS sag DEFAULT CHARSET utf8mb4;
```

表结构会在首次启动 API 时自动创建。

### 4. 配置

```bash
# 从模板复制配置文件
cp .env.example .env        # Linux / Mac
copy .env.example .env      # Windows

# 编辑 .env，填入你的 MySQL 密码、LLM API Key、模型路径
```

必填配置项：

| 变量 | 说明 |
|------|------|
| `SAG_MYSQL_PASSWORD` | MySQL 密码 |
| `SAG_LLM_API_KEY` | LLM API Key |
| `SAG_BGE_MODEL_PATH` | BGE 模型本地路径（使用 bge 后端时必填） |

> `config.py` 自动从自身目录逐级向上搜索 `.env`，无论仓库 clone 到哪个路径都能找到，无需修改任何代码。

### 5. 启动

需要启动两个进程（各开一个终端）：

```bash
# 终端 1：API 服务（端口 8777）
cd your-repo
python -m ai_sag.api

# 终端 2：Web UI（端口 8080）
python -m ai_sag.web
```

浏览器打开 **http://localhost:8080**，或访问 Swagger 文档 **http://localhost:8777/docs**。

> 详细启动说明、API 接口、调优参数等请参阅 **[STARTUP.md](./STARTUP.md)**。

---

## 项目结构

```
ai_sag/
├── api.py                ← FastAPI 后端（端口 8777）
├── web.py                ← Web UI（端口 8080）
├── requirements.txt      ← 依赖清单
├── .env.example          ← 环境变量模板
├── README.md             ← 项目总览（本文档）
├── STARTUP.md            ← 启动与使用指南
├── FLOW.md               ← 全流程详解（离线建库 + 在线检索）
├── INGEST_FLOW.md        ← 文档离线入库全流程（英文版）
├── 入库流程.md            ← 入库流程中文详解
├── 问答流程.md            ← 用户问答处理全链路
├── 问答对话参数配置.md     ← 问答链路所有参数汇总
├── 多轮对话历史处理.md     ← 多轮对话历史流转与实现
├── 内容字段抽取与作用详解.md ← Events / Chunks / Documents 字段抽取逻辑
├── 实体字段抽取与作用详解.md ← Entities / Event_Entities 字段抽取逻辑
├── 多跳查询性能隐患.md     ← BFS 扩展 SQL 性能分析与优化建议
│
├── base/                 ← 基础设施：配置、日志、数据模型、提示词
├── retrieval/            ← SAG 检索器（核心）+ 问答引擎
├── ingest/               ← 入库流水线编排
├── embeddings/           ← BGE / Qwen3 向量化
├── splitter/             ← 文档语义切分
├── extractor/            ← 事件 & 实体抽取
├── vector_store/         ← ChromaDB 封装（4 个 collection）
├── storage/              ← MySQL 持久化（6 张表，全异步）
├── llm/                  ← DeepSeek / OpenAI 兼容 LLM 封装
├── loader/               ← 文件加载（md/txt/docx/pdf/xlsx）
├── doc_parser/           ← 文档深度解析（OCR、版式分析等）
├── cleaner/              ← 文本清洗
└── static/               ← Web 前端页面
```

---

## 文档索引

根目录下各 Markdown 文档的用途说明：

| 文档 | 内容 |
|------|------|
| **[STARTUP.md](./STARTUP.md)** | 启动与使用指南：环境准备、配置、启动、API 调用示例、检索参数调优、常见问题 |
| **[FLOW.md](./FLOW.md)** | 全流程详解（英文），覆盖离线建库和在线检索两个阶段的每一步及对应代码位置 |
| **[INGEST_FLOW.md](./INGEST_FLOW.md)** | 文档离线入库全流程（英文）：loader → cleaner → splitter → extractor → 持久化 |
| **[入库流程.md](./入库流程.md)** | 入库流程中文详解：概要流程 → 流程细节 → 实现逻辑，三个层次逐级深入 |
| **[问答流程.md](./问答流程.md)** | 用户问答处理全链路：双路召回 → BFS 扩展 → 粗排精排 → 生成回答 |
| **[问答对话参数配置.md](./问答对话参数配置.md)** | 问答链路所有参数汇总：环境变量、检索参数、对话参数、LLM 参数、Prompt 模板 |
| **[多轮对话历史处理.md](./多轮对话历史处理.md)** | 多轮对话中"历史"在各环节的作用、流转、实现逻辑（含流程图、实例分析）|
| **[内容字段抽取与作用详解.md](./内容字段抽取与作用详解.md)** | Events / Chunks / Documents 三类内容的字段抽取逻辑及在问答中的作用 |
| **[实体字段抽取与作用详解.md](./实体字段抽取与作用详解.md)** | Entities / Event_Entities 字段抽取逻辑及在问答中的作用 |
| **[多跳查询性能隐患.md](./多跳查询性能隐患.md)** | SAG BFS 多跳扩展的 SQL 性能分析：索引命中、潜在瓶颈、优化方案 |

---

## API 概览

### 文档管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/documents` | 上传文件 |
| `POST` | `/api/documents/text` | 上传纯文本 |
| `GET` | `/api/documents` | 文档列表（分页+搜索）|
| `GET` | `/api/documents/{id}` | 文档详情 |
| `PUT` | `/api/documents/{id}` | 更新文档内容 |
| `DELETE` | `/api/documents/{id}` | 删除文档 |

### 检索与问答

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/search` | SAG 检索（切片 + trace）|
| `POST` | `/api/ask` | 问答（答案 + 切片 + trace）|
| `POST` | `/api/chat` | 多轮对话 |
| `POST` | `/api/chat/stream` | 多轮对话 SSE 流式 |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/stats` | 统计信息 |
| `GET` | `/api/health` | 健康检查 |

示例：

```bash
# 上传文档
curl -X POST http://localhost:8777/api/documents -F "file=@合同公告.md"

# 问答
curl -X POST http://localhost:8777/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"数脉科技跟哪个公司签了合同？"}'
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| RAG 框架 | LlamaIndex ≥0.12.0 |
| Web 服务 | FastAPI + Uvicorn |
| 向量数据库 | ChromaDB ≥1.0.0 |
| 关系数据库 | MySQL + aiomysql（全异步）|
| LLM | DeepSeek / OpenAI / vLLM / Ollama（OpenAI 兼容协议）|
| Embedding | BAAI/bge-small-zh-v1.5 / Qwen3-Embedding-0.6B（本地推理）|
| 文档解析 | PyMuPDF + python-docx + openpyxl + PaddleOCR/RapidOCR |
| 日志 | loguru（控制台+文件双输出，trace_id 全链路追踪）|
| 配置 | python-dotenv + Pydantic v2 |

---

## 适用场景

- **合同分析**：跨文档关联甲方/乙方/签署日期/金额等实体
- **供应链溯源**：多跳追踪上下游企业与产品关系
- **人物关系追踪**：跨新闻报道发现人物-机构-事件关联
- **法规合规检索**：法规条文 ↔ 适用场景 ↔ 处罚案例 的关联查询
- **金融研报分析**：公司-指标-行业-事件的交叉检索

不适合：单轮简单事实查询（传统 RAG 即可）；需要复杂推理但不涉及实体关联的问答。

---

## 调优参考

检索效果不理想时，可调整 `.env` 中的关键参数：

| 问题 | 调整方向 |
|------|---------|
| 召回太少 | ↓ `SIMILARITY_THRESHOLD`、↑ `MAX_HOPS`、↓ `ENTITY_EXPAND_THRESHOLD` |
| 召回噪声多 | ↑ `ENTITY_EXPAND_THRESHOLD`、↓ `MAX_HOPS` |
| 上下文不够 | ↑ `RERANK_TOP_K`、↑ `MAX_SECTIONS` |
| BFS 太慢 | ↓ `ENTITY_FRONTIER_BUDGET`、使用 `SUB_STRATEGY=hopllm` |

完整参数说明见 **[STARTUP.md § 九、检索参数详解](./STARTUP.md#九检索参数详解)**。

---

## 许可证

内部项目，仅供团队使用。
