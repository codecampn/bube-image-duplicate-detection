from abc import ABC, abstractmethod

from ..models import ImageEmbedding, ImageEmbeddingNeighbour


class VectorDBRepository(ABC):
    """Abstract class for a Vector Database Repository.

    The repository ensures that different vector databases can be used in the application.
    Functionailty should include the storage of image embeddings and the retrieval of neighbours depending on a
    distance threshold or a limit.
    """

    @abstractmethod
    def store_embeddings(self, image_embeddings: list[ImageEmbedding]) -> None:
        """Abstract method which should store image embeddings in the database."""

    @abstractmethod
    def get_neighbours(
        self, image_embedding: ImageEmbedding, threshold: float, limit: int = 50
    ) -> list[ImageEmbeddingNeighbour]:
        """Abstract method which should return the neighbours of an image embedding."""

    @abstractmethod
    def get_neighbours_top_n(self, image_embedding: ImageEmbedding, limit: int = 50) -> list[ImageEmbeddingNeighbour]:
        """Abstract method which should return the n closest neighbours of an image embedding."""

    @abstractmethod
    def get_neighbours_threshold(
        self, image_embedding: ImageEmbedding, threshold: float
    ) -> list[ImageEmbeddingNeighbour]:
        """Abstract method which should return the neighbours of an image embedding with a distance threshold."""
