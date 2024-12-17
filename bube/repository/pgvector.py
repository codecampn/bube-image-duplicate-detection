import ast
import atexit
import logging
from importlib import resources as impresources
from typing import Any, Optional

import psycopg2
from psycopg2 import sql as pgsql
from psycopg2.extras import execute_values

from ..config import config
from ..models import ImageEmbedding, ImageEmbeddingNeighbour
from .vector_db_repository import VectorDBRepository


class PgVector(VectorDBRepository):
    """Repository class for the pgVector database.

    This DB can be used to connect to a remote pgVector database and store and retrieve image embeddings.
    To configure the connection, the config file can be used or environment variables.
    """

    _connection: Any
    _table_name: pgsql.Identifier
    _logger: logging.Logger

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Connecting to pgVector database on {config.PGVECTOR_DB_HOST}:{config.PGVECTOR_DB_PORT}")
        self._table_name = pgsql.Identifier(config.PGVECTOR_DB_TABLE_NAME)
        self._connection = psycopg2.connect(
            host=config.PGVECTOR_DB_HOST,
            port=config.PGVECTOR_DB_PORT,
            user=config.PGVECTOR_DB_USER,
            password=config.PGVECTOR_DB_PWD.get_secret_value(),
            dbname=config.PGVECTOR_DB_DATABASE_NAME,
            sslmode="require" if config.PGVECTOR_DB_HTTP_SSL else "disable",
        )
        self._setup_database()
        atexit.register(self.close)

    def _setup_database(self) -> None:
        self._logger.info("Setting up pgVector database.")
        with self._connection.cursor() as cursor:
            with impresources.open_text("bube.repository", "setup_pgvector.sql") as f:
                setup_sql = f.read()
            cursor.execute(setup_sql)
        self._connection.commit()

    def store_embeddings(self, image_embeddings: list[ImageEmbedding]) -> None:
        """Store image embeddings in the database.

        Args:
            image_embeddings (list[ImageEmbedding]): A list of ImageEmbedding objects to store.
        """
        with self._connection.cursor() as cursor:
            embeddings_data = [(img.filename, img.embedding) for img in image_embeddings]
            insert_query = pgsql.SQL(f"""
            INSERT INTO {self._table_name} (filename, embedding)
            VALUES %s
            ON CONFLICT (filename) DO UPDATE SET
                filename = EXCLUDED.filename,
                embedding = EXCLUDED.embedding;
            """)  # noqa: S608
            execute_values(cursor, insert_query, embeddings_data)
        self._connection.commit()

    def get_neighbours(
        self, image_embedding: ImageEmbedding, threshold: Optional[float] = None, limit: int = 10
    ) -> list[ImageEmbeddingNeighbour]:
        """Get the neighbours of an image embedding.

        This method accepts a threshold to filter the neighbours by distance and a limit to restrict the number of
        returned neigbours. A combination of both parameters is possible.

        Args:
            image_embedding (ImageEmbedding): The image embedding to search for.
            threshold (Optional[float]): The maximum distance to consider a neighbour. If None, no threshold is applied.
            limit (int): The maximum number of neighbours to return. Defaults to 10.

        Returns:
            list[ImageEmbeddingNeighbour]: A list of ImageEmbeddingNeighbour objects.
        """
        with self._connection.cursor() as cursor:
            query = """
            SELECT filename, embedding, distance
            FROM get_neighbours(%s::VECTOR(2048), %s::FLOAT, %s::INTEGER);
            """
            cursor.execute(query, (image_embedding.embedding, threshold, limit))
            rows = cursor.fetchall()
            return [
                ImageEmbeddingNeighbour(filename=row[0], embedding=ast.literal_eval(row[1]), distance=row[2])
                for row in rows
            ]

    def get_neighbours_top_n(self, image_embedding: ImageEmbedding, limit: int = 10) -> list[ImageEmbeddingNeighbour]:
        """Get the top N neighbours of an image embedding."""
        return self.get_neighbours(image_embedding, threshold=None, limit=limit)

    def get_neighbours_threshold(
        self, image_embedding: ImageEmbedding, threshold: float
    ) -> list[ImageEmbeddingNeighbour]:
        """Get all neighbours of an image embedding that are closer than the given threshold."""
        return self.get_neighbours(image_embedding, threshold=threshold, limit=100)

    def close(self) -> None:
        """Close the database connection."""
        self._logger.info("Closing pgVector database connection.")
        if self._connection:
            self._connection.close()
