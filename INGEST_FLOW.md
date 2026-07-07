# ai_sag 文档离线入库全流程

## 一、整体架构

ai_sag 入库流程严格遵循 SAG 论文的离线建库阶段：**不建静态图谱，只建事件-实体多对多索引**。

```
原始文档 → 加载 → 清洗 → 切分 → LLM抽取(事件+实体) → 混合存储(MySQL + 向量库)
```

### 核心原则

- **Chunk ↔ Event 严格 1:1**：一个 chunk 抽一个融合事件，不拆三元组
- **实体只做索引锚点**：不存储完整语义，通过 `event_entities` 关联表串联跨文档事件
- **实体全局去重**：按 `(type, normalized_name)` 跨文档共享
- **全链路异步**：MySQL 用 aiomysql 连接池，LLM 用 LlamaIndex 原生异步，Embedder/VectorStore 用异步接口

---

## 二、入库流程详解

### 步骤 0：初始化

| 项 | 说明 |
|----|------|
| 位置 | `ingest/pipeline.py` `IngestPipeline.__init__` + `init()` |
| 作用 | 创建组件实例、建表、启动对账 |
| 关键 | `__init__` 只创建对象（同步），`await init()` 做异步建表+对账启动 |

初始化时序：
```
IngestPipeline.__init__(cfg)
  ├─ loader = DocumentLoader.default()      # 文档加载器
  ├─ cleaner = TextCleaner()                 # 文本清洗器
  ├─ splitter = create_splitter(cfg)         # 文档切分器
  ├─ embedder = create_embedder(cfg)         # Embedding 模型
  ├─ llm = create_llm(cfg)                   # LLM（DeepSeek/OpenAI）
  ├─ extractor = EventExtractor(llm)         # 事件抽取器
  ├─ db = MysqlStore(...)                    # MySQL 异步连接池
  ├─ vectors = create_vector_store(cfg)      # 向量库（ChromaDB）
  └─ _reconcile_task = None                  # 对账任务占位

await pipeline.init()
  ├─ db.ensure_schema()                      # 建表（如不存在）
  ├─ _reconcile()                            # 启动时对账一次
  └─ _start_reconcile_loop()                 # 启动定时对账后台任务
```

### 步骤 1：文档加载

| 项 | 说明 |
|----|------|
| 位置 | `loader/` `DocumentLoader` |
| 入口 | `ingest_file(path)` / `ingest_text(title, content)` |
| 输入 | 文件路径（.md/.txt/.docx/.pdf）或纯文本 |
| 输出 | `LoadedDocument`（含 title + content + file_type + metadata） |
| 关键 | 根据扩展名路由到不同 Reader，统一返回结构 |

### 步骤 2：文本清洗

| 项 | 说明 |
|----|------|
| 位置 | `cleaner/` `TextCleaner` |
| 作用 | 去除多余空白、规范化标点、剔除乱码 |
| 输出 | 清洗后的 `LoadedDocument` |

### 步骤 3：文档切分

| 项 | 说明 |
|----|------|
| 位置 | `splitter/` `ChunkSplitter` |
| 模式 | `auto`（默认）：md→MarkdownNodeParser，其余→SentenceSplitter |
| 参数 | `chunk_size=512`，`chunk_overlap=100` |
| 输出 | `list[Chunk]`（每个含 id/document_id/source_id/rank_index/heading/content） |
| 关键 | auto 模式按文档类型自适应；Markdown 两级切分（先按标题，超长再按句子） |

切分配置：
```
AISAG_SPLITTER_MODE=auto          # auto/markdown/sentence/token/code
AISAG_CHUNK_SIZE=512
AISAG_CHUNK_OVERLAP=100
```

### 步骤 4：LLM 抽取事件 + 实体（SAG 核心）

| 项 | 说明 |
|----|------|
| 位置 | `extractor/event_extractor.py` `EventExtractor.extract_batch` |
| 原则 | **强制单事件**：每个 chunk 融合为一个综合顶层事件 |
| 实体类型 | 11 类：time/location/person/organization/group/topic/work/product/action/metric/label |
| 跨片段关联 | 顺序模式传 `previous_context`（前一个 chunk 的 summary），解析代词指代 |
| 输出 | 每个 chunk 对应一个 `ExtractedEvent`（含 title/summary/content/entities） |

抽取模式：
```
AISAG_EXTRACT_PARALLEL=false     # 默认串行（保 previous_context 连贯）
                                  # true 时并发抽取（4线程，不传 previous_context）
```

串行模式（默认，质量优先）：
```
chunk1 → LLM → event1(summary1)
                              ↓ previous_context=summary1
chunk2 → LLM → event2(summary2)
                              ↓ previous_context=summary2
chunk3 → LLM → event3
```

并行模式（速度优先）：
```
chunk1 ─┐
chunk2 ─┼→ ThreadPoolExecutor(4) → LLM 并发 → event1/event2/event3/event4
chunk3 ─┤    （不传 previous_context，用 doc_title 全局上下文）
chunk4 ─┘
```

LLM 抽取容错：
- LLM 失败时直接抛出 `ExtractionError`，不再有 `_fallback` 静默降级
- 由 `extract_batch` 统一处理异常（重试或记录）

### 步骤 5：持久化（三库同步写入）

| 项 | 说明 |
|----|------|
| 位置 | `ingest/pipeline.py` `_persist` |
| 策略 | 先 MySQL 事务写入 → 再向量库写入 → 成功后标记 synced |
| 容错 | 向量库写入失败 → 回滚 MySQL（补偿删除） |

#### 5.1 生成向量

```
chunk_embs = embedder.aembed_texts([c.content for c in chunks])     # chunk 内容向量
```

#### 5.2 MySQL 事务写入（原子）

```
db.persist_source()  # 一个事务内写入 6 张表
  ├─ aisag_sources       (文档元信息，vector_synced=False)
  ├─ aisag_documents     (原文)
  ├─ aisag_chunks        (切片，N 条)
  ├─ aisag_events        (事件，N 条，与 chunk 1:1)
  ├─ aisag_entities      (实体，全局去重 INSERT ON DUPLICATE KEY)
  └─ aisag_event_entities(事件-实体关联，存 role 角色信息)
```

实体去重机制：
```
实体按 (entity_type, normalized_name) 全局去重
  ├─ normalized_name = 去空格 + lower
  ├─ UNIQUE KEY uq_aisag_entity (entity_type, normalized_name)
  ├─ INSERT ON DUPLICATE KEY UPDATE name=VALUES(name)
  └─ description 不覆盖（保留首次抽取的属性）
```

#### 5.3 向量库写入

```
vectors.aadd_chunks(items, source_id)           # chunks collection
vectors.aadd_events(items, source_id)           # event_titles collection
vectors.aadd_event_contents(items, source_id)   # event_contents collection
vectors.aadd_entities(items)                     # entities collection（不绑 source_id）
```

向量生成（批量 embed）：
```
chunk 向量    = embedder.aembed_texts([c.content for c in chunks])
标题向量      = embedder.aembed_texts([e.title for e in events])
内容向量      = embedder.aembed_texts([e.content for e in events])
实体向量      = embedder.aembed_texts([entity_names])  # 批量，用 name_lookup 查找表
```

#### 5.4 标记同步完成

```
db.mark_vector_synced(source_id, True)   # metadata.vector_synced = 1
```

#### 5.5 失败回滚

```
向量库写入失败时：
  ├─ db.delete_by_source(source_id)              # 补偿删除 MySQL 数据
  ├─ vectors.adelete_by_source(source_id)        # 清理已写入的向量
  └─ vectors.adelete_entities_by_ids(orphan_ids) # 清理孤儿实体向量
```

---

## 三、跨库一致性保障

### 核心机制：vector_synced 标记 + 定时对账

```
入库流程：
  MySQL 写入（synced=False）→ 向量库写入 → 标记 synced=True
  ├─ 正常完成：synced=True，数据一致
  ├─ 向量库写失败：补偿删除 MySQL，抛异常
  └─ 进程崩溃（MySQL 已提交，向量库未写完）：synced=False 残留

对账任务（每 300 秒）：
  发现 synced=False 的 source → 清理 MySQL + 向量库残留
```

### 对账三类检查

| 类型 | 检查方式 | 覆盖场景 |
|------|----------|----------|
| 第一类 | MySQL 中 `vector_synced=False` 的 source | 入库崩溃、删除中崩溃 |
| 第二类 | 向量库有 source_id 但 MySQL 无 | 删除时 source 物理删但向量库删失败 |
| 第三类 | 向量库有 entity_id 但 MySQL 无 | 孤儿实体向量残留 |

对账配置：
```
AISAG_RECONCILE_INTERVAL=300     # 对账间隔秒数（0 禁用定时对账）
```

---

## 四、删除流程

### 删除顺序：先向量库后 MySQL

```
delete_source(source_id)
  │
  ├─ 步骤1：标记删除中
  │   db.mark_vector_synced(source_id, False)
  │
  ├─ 步骤2：向量库删除（检索立即不可见）
  │   vectors.adelete_by_source(source_id)
  │   → 删 chunks / event_titles / event_contents
  │   → 失败不中断，靠对账兜底
  │
  ├─ 步骤3：MySQL 事务删除（页面才消失）
  │   db.delete_by_source(source_id)
  │   → event_entities (DELETE)
  │   → events (UPDATE deleted_at，软删除保留审计)
  │   → chunks (DELETE)
  │   → documents (DELETE)
  │   → sources (DELETE)
  │   → 孤儿 entities (DELETE，NOT EXISTS 引用)
  │   → 返回孤儿实体 id 列表
  │
  └─ 步骤4：孤儿实体向量删除
      vectors.adelete_entities_by_ids(orphan_entity_ids)
      → 失败不中断，靠对账第三类兜底
```

### 用户体验

| 时间点 | 检索（依赖向量库） | 页面（依赖 MySQL） |
|--------|-------------------|-------------------|
| 删除前 | ✅ 能搜到 | ✅ 能看到 |
| 步骤2后 | ❌ 搜不到了 | ✅ 还能看到 |
| 步骤3后 | ❌ 搜不到 | ❌ 看不到了 |

### 重建文档（先入新后删旧）

```
rebuild_document(source_id)
  ├─ 1. 用新 source_id 入库（旧数据不受影响）
  ├─ 2. 入库成功后删除旧数据
  └─ 3. 入库失败时旧数据仍在，不影响使用
```

---

## 五、数据模型

### MySQL 表结构

```
aisag_sources          文档元信息（id, name, metadata.vector_synced）
    │
    ├─ 1:1 → aisag_documents    原文（id, source_id, title, content）
    │
    ├─ 1:N → aisag_chunks       切片（id, source_id, document_id, rank_index, heading, content）
    │            │
    │            └─ 1:1 → aisag_events   事件（id, source_id, chunk_id, title, summary, content, deleted_at）
    │                          │
    │                          └─ M:N → aisag_entities    实体（id, entity_type, name, normalized_name, description）
    │                                     │
    │                                     └─ UNIQUE KEY (entity_type, normalized_name) 全局去重
    │
    └─ aisag_event_entities    关联表（event_id, entity_id, weight, description=role）
```

### 向量库 Collection

| Collection | 内容 | 绑定 source_id | 用途 |
|------------|------|---------------|------|
| `chunks` | chunk 内容向量 | ✅ | 基线向量召回（ChunkB） |
| `event_titles` | 事件标题向量 | ✅ | 事件标题向量召回 |
| `event_contents` | 事件内容向量 | ✅ | 粗排（余弦相似度） |
| `entities` | 实体名向量 | ❌（跨 source 共享） | 实体向量扩展召回 |

---

## 六、配置参数

| 环境变量 | 默认值 | 作用 |
|----------|--------|------|
| `AISAG_SPLITTER_MODE` | auto | 切分模式 |
| `AISAG_CHUNK_SIZE` | 512 | 切片大小 |
| `AISAG_CHUNK_OVERLAP` | 100 | 切片重叠 |
| `AISAG_EXTRACT_PARALLEL` | false | 事件抽取是否并行 |
| `AISAG_RECONCILE_INTERVAL` | 300 | 对账间隔秒数 |
| `AISAG_MYSQL_POOL_SIZE` | 10 | MySQL 连接池大小 |
| `AISAG_MYSQL_MAX_OVERFLOW` | 5 | 连接池溢出上限 |
| `AISAG_MYSQL_POOL_TIMEOUT` | 30 | 获取连接超时 |
| `AISAG_MYSQL_POOL_RECYCLE` | 3600 | 连接回收周期 |

---

## 七、异步架构

```
api.py (FastAPI async 端点)
  │  await
  └─→ IngestPipeline (async)
        ├─→ MysqlStore.* (async, aiomysql 连接池)        ← 原生异步
        ├─→ embedder.aembed_texts()                       ← 异步接口（to_thread 包装）
        ├─→ vectors.aadd_chunks() / aadd_events()         ← 异步接口（to_thread 包装）
        └─→ extractor.extract_batch()                     ← to_thread 包装（含 ThreadPoolExecutor）
```

| 组件 | 异步方式 | 原因 |
|------|----------|------|
| MySQL | aiomysql 原生异步 | 网络 IO |
| Embedder | `asyncio.to_thread` 包装 | CPU 密集（本地模型推理） |
| VectorStore | `asyncio.to_thread` 包装 | 本地磁盘 IO（ChromaDB 无异步 API） |
| LLM（入库抽取） | `asyncio.to_thread` 包装 | LlamaIndex 同步 + 内部 ThreadPoolExecutor |
