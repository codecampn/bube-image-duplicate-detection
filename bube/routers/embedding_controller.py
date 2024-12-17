from fastapi import APIRouter, Query, UploadFile

from ..models import ImageEmbedding
from ..services import LocalImageService, RemoteImageService


class EmbeddingController:
    """Controller class for embedding images.

    This controller class is responsible for handling requests related to image embeddings.
    Either images from a local directory or images uploaded through the API can be embedded.
    No duplicate checks are performed in this controller.
    """

    router: APIRouter
    _local_image_service: LocalImageService
    _remote_image_service: RemoteImageService

    def __init__(self):
        self.router = APIRouter(prefix="/embeddings", tags=["Embeddings"])
        self._local_image_service = LocalImageService()
        self._remote_image_service = RemoteImageService()

        self.router.add_api_route(
            "/local",
            self.embed_local_images,
            methods=["GET"],
            response_model=list[ImageEmbedding],
            summary="Calculate embeddings and compare them against db for images on the local disk",
            status_code=200,
        )

        self.router.add_api_route(
            "",
            self.calculate_embeddings,
            methods=["POST"],
            response_model=None,
            summary="Calculate embeddings and compare them for images",
            status_code=200,
        )

    async def embed_local_images(
        self, image_root: str, filenames: list[str] | None = Query(None)
    ) -> list[ImageEmbedding]:
        """Embed images from a local directory and compare them against the database."""
        return self._local_image_service.embed_local_images(image_root=image_root, filenames=filenames)

    async def calculate_embeddings(self, images: list[UploadFile]) -> list[ImageEmbedding]:
        """Calculate embeddings for images uploaded through the API."""
        # filter for valid image types
        images = [image for image in images if image.content_type.startswith("image/")]

        image_binariers = [image.file for image in images]
        image_filenames = [image.filename for image in images]
        return self._remote_image_service.embed_images(images=image_binariers, filenames=image_filenames)
