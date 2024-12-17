import logging
from typing import Optional

import chromadb

from ..config import config
from ..models import ImageEmbedding, ImageEmbeddingNeighbour
from .vector_db_repository import VectorDBRepository


class EmbeddedChromaDB(VectorDBRepository):
    """Repository class for the embedded ChromaDB.

    This database can be used to store and retrieve image embeddings if no remote pgVector DB is available.
    The class theoretically supports a remote ChromaDB but the primary use case is the embedded version.
    """

    _db: chromadb.ClientAPI
    _db_collection: chromadb.Collection
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._db = self._get_chroma_client()
        self._db_collection = self._db.get_or_create_collection(
            config.CHROMA_DB_DATABASE_NAME, metadata={"hnsw:space": "l2"}
        )

    def _get_chroma_client(self) -> chromadb.ClientAPI:
        # Embedded Client
        if config.CHROMA_DB_MODE == "embedded":
            self._logger.info(f"Using embedded ChromaDB with filepath: {config.CHROMA_DB_EMBEDDED_PATH}")
            return chromadb.PersistentClient(path=config.CHROMA_DB_EMBEDDED_PATH)
        # HTTP Client
        self._logger.info(f"Using HTTP ChromaDB on {config.CHROMA_DB_HTTP_HOST}:{config.CHROMA_DB_HTTP_PORT}")
        return chromadb.HttpClient(
            host=config.CHROMA_DB_HTTP_HOST,
            port=config.CHROMA_DB_HTTP_PORT,
            headers=config.CHROMA_DB_HTTP_HEADERS,
            ssl=config.CHROMA_DB_HTTP_SSL,
        )

    def store_embeddings(self, image_embeddings: list[ImageEmbedding]) -> None:
        """Store image embeddings in the ChromaDB."""
        if not image_embeddings:
            return
        ids = [emb.filename for emb in image_embeddings]
        embeddings = [emb.embedding for emb in image_embeddings]
        self._db_collection.upsert(ids=ids, documents=ids, embeddings=embeddings)

    def get_neighbours(
        self, image_embedding: ImageEmbedding, threshold: Optional[float] = None, limit: int = 50
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
        neighbours = self.get_neighbours_top_n(image_embedding, limit)
        if threshold is None:
            return neighbours
        return [neighbour for neighbour in neighbours if neighbour.distance <= threshold]

    def get_neighbours_top_n(self, image_embedding: ImageEmbedding, limit: int = 20) -> list[ImageEmbeddingNeighbour]:
        """Get the n closest neighbours of an image embedding."""
        query_res = self._db_collection.query(
            query_embeddings=[image_embedding.embedding], n_results=limit, include=["distances", "embeddings"]
        )
        return self._convert_chroma_results(query_res)

    def get_neighbours_threshold(
        self, image_embedding: ImageEmbedding, threshold: float
    ) -> list[ImageEmbeddingNeighbour]:
        """Get the neighbours of an image embedding with a distance threshold."""
        return self.get_neighbours(image_embedding, threshold, limit=100)

    def _convert_chroma_results(self, query_res: chromadb.QueryResult) -> list[ImageEmbeddingNeighbour]:
        # a list comprehension to convert the results to ImageEmbeddingNeighbour would be messy, thus it's a for loop
        neighbours = []
        for filename, emb, distance in zip(query_res["ids"][0], query_res["embeddings"][0], query_res["distances"][0]):
            neighbours.append(ImageEmbeddingNeighbour(filename=filename, embedding=emb, distance=distance))
        return neighbours

    def _clear_database(self) -> None:
        """Clear the database."""
        ids = self._db_collection.get()["ids"]
        if ids:
            self._db_collection.delete(ids)
