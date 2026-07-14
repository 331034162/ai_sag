-- ai_sag 表结构（与 sag 模块独立，前缀 aisag_ 避免冲突）

CREATE TABLE IF NOT EXISTS aisag_sources (
  id           VARCHAR(36) NOT NULL PRIMARY KEY,
  name         VARCHAR(128) NOT NULL,
  description  TEXT,
  md5          VARCHAR(32) NOT NULL DEFAULT '',
  metadata     JSON,
  archived_at  DATETIME NULL,
  created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_aisag_source_name_md5 (name, md5)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS aisag_documents (
  id          VARCHAR(36) NOT NULL PRIMARY KEY,
  source_id   VARCHAR(36) NOT NULL,
  title       VARCHAR(1024) NOT NULL,
  content     MEDIUMTEXT,
  status      VARCHAR(32) NOT NULL DEFAULT 'COMPLETED',
  metadata    JSON,
  created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_aisag_doc_source (source_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS aisag_chunks (
  id           VARCHAR(36) NOT NULL PRIMARY KEY,
  source_id    VARCHAR(36) NOT NULL,
  document_id  VARCHAR(36) NOT NULL,
  rank_index   INT NOT NULL DEFAULT 0,
  heading      VARCHAR(512),
  content      MEDIUMTEXT NOT NULL,
  metadata     JSON,
  created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_aisag_chunk_source (source_id),
  INDEX idx_aisag_chunk_doc (document_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS aisag_events (
  id           VARCHAR(36) NOT NULL PRIMARY KEY,
  source_id    VARCHAR(36) NOT NULL,
  document_id  VARCHAR(36) NOT NULL,
  chunk_id     VARCHAR(36) NOT NULL,
  rank_index   INT NOT NULL DEFAULT 0,
  title        VARCHAR(1024) NOT NULL,
  summary      TEXT,
  content      MEDIUMTEXT NOT NULL,
  metadata     JSON,
  deleted_at   DATETIME NULL,
  created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
  INDEX idx_aisag_ee_entity (entity_id),
  INDEX idx_aisag_ee_event (event_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;