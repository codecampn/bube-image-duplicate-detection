from typing import Optional

from ...models import ImageEmbedding
from ..image_embedding_model import ImageEmbeddingModel
from .local_img_reader import LocalImgReader


class LocalImageService:
    """Service class for embedding images from local storage."""

    _embedding_model: ImageEmbeddingModel

    def __init__(self):
        self._embedding_model = ImageEmbeddingModel()

    def embed_local_images(self, image_root: str, filenames: Optional[list[str]] = None) -> list[ImageEmbedding]:
        """Embeds images from local storage and returns a list of ImageEmbedding objects.

        Args:
            image_root (str): Path to the root directory containing images
            filenames (list[str], optional): filenames of images to embed in the root directory. If not provided,
                all images in the root directory will be embedded. Defaults to None.

        Returns:
            list[ImageEmbedding]: List of ImageEmbedding objects
        """
        if not filenames:
            file_reader = LocalImgReader(image_root=image_root, all_img_files=True)
        else:
            file_reader = LocalImgReader(image_root=image_root, filenames=filenames)

        embeddings = []
        for batch, batch_filenames in file_reader:
            # compute embedding for each image
            embeddings_batch = self._embedding_model.compute_embedding_batch(batch)

            # Embedding is currently a Numpy array, which should be converted to list[float]
            embeddings_list = [
                ImageEmbedding(embedding=embedding.tolist(), filename=filename)
                for embedding, filename in zip(embeddings_batch, batch_filenames)
            ]
            embeddings.extend(embeddings_list)
        return embeddings
