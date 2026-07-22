-- ai_sag PostgreSQL 表结构（前缀 aisag_ 避免冲突）
-- MySQL → PostgreSQL 迁移要点：
--   MEDIUMTEXT → TEXT
--   TINYINT → SMALLINT
--   DATETIME → TIMESTAMP
--   ENGINE/CHARSET 移除
--   ON DUPLICATE KEY UPDATE → ON CONFLICT ... DO UPDATE
--   INSERT IGNORE → INSERT ... ON CONFLICT DO NOTHING

-- 自动更新 updated_at 的触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ==================== 业务表 ====================

CREATE TABLE IF NOT EXISTS aisag_sources (
    id              VARCHAR(36) NOT NULL PRIMARY KEY,
    name            VARCHAR(128) NOT NULL,
    description     TEXT,
    md5             VARCHAR(32) NOT NULL DEFAULT '',
    vector_synced   SMALLINT NOT NULL DEFAULT 0,
    archived_at     TIMESTAMP NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_aisag_source_name_md5 UNIQUE (name, md5)
);
CREATE INDEX IF NOT EXISTS idx_aisag_source_vector_synced ON aisag_sources (vector_synced);
-- 自动更新 updated_at
DROP TRIGGER IF EXISTS trg_aisag_sources_updated_at ON aisag_sources;
CREATE TRIGGER trg_aisag_sources_updated_at
    BEFORE UPDATE ON aisag_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE IF NOT EXISTS aisag_documents (
    id          VARCHAR(36) NOT NULL PRIMARY KEY,
    source_id   VARCHAR(36) NOT NULL,
    title       VARCHAR(1024) NOT NULL,
    content     TEXT,
    status      VARCHAR(32) NOT NULL DEFAULT 'COMPLETED',
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_aisag_doc_source ON aisag_documents (source_id);

CREATE TABLE IF NOT EXISTS aisag_chunks (
    id            VARCHAR(36) NOT NULL PRIMARY KEY,
    source_id     VARCHAR(36) NOT NULL,
    document_id   VARCHAR(36) NOT NULL,
    rank_index    INT NOT NULL DEFAULT 0,
    heading       VARCHAR(512),
    content       TEXT NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_aisag_chunk_source ON aisag_chunks (source_id);
CREATE INDEX IF NOT EXISTS idx_aisag_chunk_doc ON aisag_chunks (document_id);

CREATE TABLE IF NOT EXISTS aisag_events (
    id            VARCHAR(36) NOT NULL PRIMARY KEY,
    source_id     VARCHAR(36) NOT NULL,
    document_id   VARCHAR(36) NOT NULL,
    chunk_id      VARCHAR(36) NOT NULL,
    rank_index    INT NOT NULL DEFAULT 0,
    title         VARCHAR(1024) NOT NULL,
    summary       TEXT,
    content       TEXT NOT NULL,
    deleted_at    TIMESTAMP NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_aisag_event_source ON aisag_events (source_id);
CREATE INDEX IF NOT EXISTS idx_aisag_event_chunk ON aisag_events (chunk_id);

CREATE TABLE IF NOT EXISTS aisag_entities (
    id               VARCHAR(36) NOT NULL PRIMARY KEY,
    entity_type      VARCHAR(64) NOT NULL,
    name             VARCHAR(512) NOT NULL,
    normalized_name  VARCHAR(512) NOT NULL,
    description      TEXT,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_aisag_entity UNIQUE (entity_type, normalized_name)
);
CREATE INDEX IF NOT EXISTS idx_aisag_entity_norm ON aisag_entities (normalized_name);
DROP TRIGGER IF EXISTS trg_aisag_entities_updated_at ON aisag_entities;
CREATE TRIGGER trg_aisag_entities_updated_at
    BEFORE UPDATE ON aisag_entities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE IF NOT EXISTS aisag_event_entities (
    id           VARCHAR(36) NOT NULL PRIMARY KEY,
    event_id     VARCHAR(36) NOT NULL,
    entity_id    VARCHAR(36) NOT NULL,
    weight       FLOAT NOT NULL DEFAULT 1.0,
    description  TEXT,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_aisag_ee UNIQUE (event_id, entity_id)
);
-- 按 entity_id 查（取实体关联的事件，BFS 高频调用），覆盖索引避免回表取 event_id
-- uq_aisag_ee (event_id, entity_id) 覆盖按 event_id 查的方向，无需单列索引
CREATE INDEX IF NOT EXISTS idx_aisag_ee_entity_event ON aisag_event_entities (entity_id, event_id);

-- ==================== FAISS 映射表 ====================
-- 设计目的：把 FAISS 后端专用的 faiss_hash ↔ UUID/source_id/document_id 映射
--          从业务表剥离到独立表，便于未来把 FAISS 单独拆为微服务。
-- 仅 FAISS 后端会写入这些映射表；其他后端（chroma/milvus/pgvector）不写入。
-- 查询流程：FAISS search 返回 hash → JOIN 映射表回查 UUID/source_id → 拿业务字段。

CREATE TABLE IF NOT EXISTS faiss_chunks_map (
    faiss_hash  BIGINT NOT NULL PRIMARY KEY,
    uuid        VARCHAR(36) NOT NULL,
    source_id   VARCHAR(36) NOT NULL,
    document_id VARCHAR(36) NOT NULL,
    CONSTRAINT uq_faiss_chunks_uuid UNIQUE (uuid)
);
CREATE INDEX IF NOT EXISTS idx_faiss_chunks_source ON faiss_chunks_map (source_id);

CREATE TABLE IF NOT EXISTS faiss_events_map (
    faiss_hash  BIGINT NOT NULL PRIMARY KEY,
    uuid        VARCHAR(36) NOT NULL,
    source_id   VARCHAR(36) NOT NULL,
    document_id VARCHAR(36) NOT NULL,
    CONSTRAINT uq_faiss_events_uuid UNIQUE (uuid)
);
CREATE INDEX IF NOT EXISTS idx_faiss_events_source ON faiss_events_map (source_id);

CREATE TABLE IF NOT EXISTS faiss_entities_map (
    faiss_hash BIGINT NOT NULL PRIMARY KEY,
    uuid       VARCHAR(36) NOT NULL,
    CONSTRAINT uq_faiss_entities_uuid UNIQUE (uuid)
);
