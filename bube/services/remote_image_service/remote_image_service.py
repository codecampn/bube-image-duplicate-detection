import datetime
import logging
from typing import BinaryIO, Optional

import numpy as np
from PIL import Image

from ...models import ImageEmbedding
from ..image_embedding_model import ImageEmbeddingModel


class RemoteImageService:
    """Service class for embedding images which were uploaded by the user through API."""

    _embedding_model: ImageEmbeddingModel
    _logger: logging.Logger

    def __init__(self):
        self._embedding_model = ImageEmbeddingModel()
        self._logger = logging.getLogger(__name__)

    def embed_images(self, images: list[BinaryIO], filenames: Optional[list[str]] = None) -> list[ImageEmbedding]:
        """Embeds images (uploaded to the API) and returns a list of ImageEmbedding objects.

        Args:
            images (list[BinaryIO]): List of images as BinaryIO objects
            filenames (list[str], optional): List of filenames for the images. If not provided, filenames will be
                generated with a timestamp.

        Returns:
            list[ImageEmbedding]: List of ImageEmbedding objects
        """
        self._logger.info(f"Embedding {len(images)} images.")
        # convert images to numpy array
        images = [np.array(Image.open(image).convert("RGB"), dtype=np.float32) for image in images]

        # if no filenames are provided, generate with timestamp
        if filenames is None or len(images) != len(filenames):
            self._logger.info("No filenames provided. Generating filenames with timestamp.")
            filenames = [f"{datetime.datetime.now(datetime.UTC)}_image_{i}" for i in range(len(images))]

        embedding_list = []
        for img, filename in zip(images, filenames):
            self._logger.info(f"Computing embedding for image: {filename}")
            emb = self._embedding_model.compute_embedding_single(img)
            embedding_list.extend([ImageEmbedding(embedding=emb.tolist(), filename=filename)])
        return embedding_list
