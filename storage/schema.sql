-- ai_sag 表结构（与 sag 模块独立，前缀 aisag_ 避免冲突）

CREATE TABLE IF NOT EXISTS aisag_sources (
  id              VARCHAR(36) NOT NULL PRIMARY KEY,
  name            VARCHAR(128) NOT NULL,
  description     TEXT,
  md5             VARCHAR(32) NOT NULL DEFAULT '',
  vector_synced   TINYINT NOT NULL DEFAULT 0 COMMENT '向量库同步状态：0=未同步/删除中，1=已同步',
  archived_at     DATETIME NULL,
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_aisag_source_name_md5 (name, md5),
  INDEX idx_aisag_source_vector_synced (vector_synced)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS aisag_documents (
  id          VARCHAR(36) NOT NULL PRIMARY KEY,
  source_id   VARCHAR(36) NOT NULL,
  title       VARCHAR(1024) NOT NULL,
  content     MEDIUMTEXT,
  status      VARCHAR(32) NOT NULL DEFAULT 'COMPLETED',
  created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_aisag_doc_source (source_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS aisag_chunks (
  id            VARCHAR(36) NOT NULL PRIMARY KEY,
  source_id     VARCHAR(36) NOT NULL,
  document_id   VARCHAR(36) NOT NULL,
  rank_index    INT NOT NULL DEFAULT 0,
  heading       VARCHAR(512),
  content       MEDIUMTEXT NOT NULL,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_aisag_chunk_source (source_id),
  INDEX idx_aisag_chunk_doc (document_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS aisag_events (
  id            VARCHAR(36) NOT NULL PRIMARY KEY,
  source_id     VARCHAR(36) NOT NULL,
  document_id   VARCHAR(36) NOT NULL,
  chunk_id      VARCHAR(36) NOT NULL,
  rank_index    INT NOT NULL DEFAULT 0,
  title         VARCHAR(1024) NOT NULL,
  summary       TEXT,
  content       MEDIUMTEXT NOT NULL,
  deleted_at    DATETIME NULL,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_aisag_event_source (source_id),
  INDEX idx_aisag_event_chunk (chunk_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS aisag_entities (
  id               VARCHAR(36) NOT NULL PRIMARY KEY,
  entity_type      VARCHAR(64) NOT NULL,
  name             VARCHAR(512) NOT NULL,
  normalized_name  VARCHAR(512) NOT NULL,
  description      TEXT,
  created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_aisag_entity (entity_type, normalized_name),
  INDEX idx_aisag_entity_norm (normalized_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS aisag_event_entities (
  id           VARCHAR(36) NOT NULL PRIMARY KEY,
  event_id     VARCHAR(36) NOT NULL,
  entity_id    VARCHAR(36) NOT NULL,
  weight       FLOAT NOT NULL DEFAULT 1.0,
  description  TEXT,
  created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_aisag_ee (event_id, entity_id),
  -- 按 entity_id 查（取实体关联的事件，BFS 高频调用），覆盖索引避免回表取 event_id
  -- uq_aisag_ee (event_id, entity_id) 覆盖按 event_id 查的方向，无需单列索引
  INDEX idx_aisag_ee_entity_event (entity_id, event_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ==================== FAISS 映射表 ====================
-- 设计目的：把 FAISS 后端专用的 faiss_hash ↔ UUID/source_id/document_id 映射
--          从业务表剥离到独立表，便于未来把 FAISS 单独拆为微服务。
-- 业务表（aisag_chunks/events/entities）保持后端无关，无 FAISS 污染。
-- 仅 FAISS 后端会写入这些映射表；其他后端（chroma/milvus/pgvector）不写入。
-- 查询流程：FAISS search 返回 hash → JOIN 映射表回查 UUID/source_id → 拿业务字段。
-- 一致性：映射表与业务表同事务写入，单一真相源仍是 MySQL。

CREATE TABLE IF NOT EXISTS faiss_chunks_map (
  faiss_hash  BIGINT NOT NULL PRIMARY KEY COMMENT 'blake2b(uuid) → int64，作为 FAISS IndexIDMap2 的 id',
  uuid        VARCHAR(36) NOT NULL COMMENT '对应 aisag_chunks.id',
  source_id   VARCHAR(36) NOT NULL,
  document_id VARCHAR(36) NOT NULL,
  UNIQUE KEY uq_faiss_chunks_uuid (uuid),
  INDEX idx_faiss_chunks_source (source_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS faiss_events_map (
  faiss_hash  BIGINT NOT NULL PRIMARY KEY,
  uuid        VARCHAR(36) NOT NULL COMMENT '对应 aisag_events.id',
  source_id   VARCHAR(36) NOT NULL,
  document_id VARCHAR(36) NOT NULL,
  UNIQUE KEY uq_faiss_events_uuid (uuid),
  INDEX idx_faiss_events_source (source_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS faiss_entities_map (
  faiss_hash BIGINT NOT NULL PRIMARY KEY,
  uuid       VARCHAR(36) NOT NULL COMMENT '对应 aisag_entities.id',
  UNIQUE KEY uq_faiss_entities_uuid (uuid)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;