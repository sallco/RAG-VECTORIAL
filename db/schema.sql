CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faqs (
    id TEXT PRIMARY KEY,
    categoria TEXT NOT NULL,
    pregunta TEXT NOT NULL,
    respuesta TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(384) NOT NULL,
    CONSTRAINT faqs_id_no_vacio CHECK (btrim(id) <> ''),
    CONSTRAINT faqs_categoria_no_vacia CHECK (btrim(categoria) <> ''),
    CONSTRAINT faqs_pregunta_no_vacia CHECK (btrim(pregunta) <> ''),
    CONSTRAINT faqs_respuesta_no_vacia CHECK (btrim(respuesta) <> '')
);

CREATE INDEX IF NOT EXISTS faqs_embedding_hnsw_idx
    ON faqs USING hnsw (embedding vector_cosine_ops);
