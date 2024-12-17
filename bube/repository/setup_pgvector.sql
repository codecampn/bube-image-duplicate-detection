CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS feex_embeddings
(
    filename  TEXT NOT NULL UNIQUE,
    embedding VECTOR(2048)
);

DROP FUNCTION get_neighbours(search_embedding VECTOR(2048), max_distance FLOAT, n_neighbours INTEGER);
CREATE OR REPLACE FUNCTION get_neighbours(search_embedding VECTOR(2048),
                                          max_distance FLOAT DEFAULT NULL,
                                          n_neighbours INTEGER DEFAULT NULL)
    RETURNS TABLE
            (
                filename  TEXT,
                embedding VECTOR(2048),
                distance  FLOAT
            )
    LANGUAGE 'plpgsql'
    PARALLEL SAFE
    COST 200
AS
$$
BEGIN
    RETURN QUERY
        SELECT a.filename,
               a.embedding,
               a.embedding <-> search_embedding AS distance
        FROM feex_embeddings a
        WHERE (max_distance IS NULL OR a.embedding <-> search_embedding <= max_distance)
        ORDER BY distance ASC
        LIMIT COALESCE(n_neighbours, 100);
END
$$;