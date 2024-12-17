from pydantic import BaseModel


class ImageEmbedding(BaseModel):
    """Model for an image embedding."""

    embedding: list[float]
    filename: str


class ImageEmbeddingNeighbour(ImageEmbedding):
    """Extension for ImageEmbedding to include the distance to the original image."""

    distance: float
