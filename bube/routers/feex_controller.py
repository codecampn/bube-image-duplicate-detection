from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from ..models import DuplicateReport
from ..services import FEEXService


class FEEXController:
    """Controller class for handling image embedding and duplicate detection."""

    router: APIRouter
    _feex_service: FEEXService

    def __init__(self):
        self.router = APIRouter(prefix="/feex", tags=["FEEX"])
        self._feex_service = FEEXService()

        self.router.add_api_route(
            "/insert",
            self.store_images,
            methods=["POST"],
            response_model=None,
            summary="Calculate embeddings and store them in the db",
            status_code=201,
        )

        self.router.add_api_route(
            "",
            self.calculate_duplicate_report,
            methods=["POST"],
            response_model=list[DuplicateReport],
            summary="Calculate embeddings for images on the local disk",
            status_code=200,
        )

    async def calculate_duplicate_report(
        self,
        images: list[UploadFile] = File(None),
        image_root: Optional[str] = Form(None),
        filenames: list[str] = Form(None),
        save_embeddings: Optional[bool] = Form(True),
    ) -> list[DuplicateReport]:
        """Calculate embeddings for images and check for duplicates.

        Args:
            images(list[UploadFile]): The images to embed and check for duplicates.
            image_root(Optional[str]): The root directory of the images if local images should be used
            filenames(list[str]): The filenames of the images if local images should be used. Optional.
            save_embeddings(Optional[bool]): Whether to save the embeddings in the database. Defaults to True.

        Returns:
            list[DuplicateReport]: A list of DuplicateReport objects. This includes the amount of duplicate and
                suspicious files with their filenames, distances and duplication_chance.

        """
        if images:
            images = [image for image in images if image.content_type.startswith("image/")]
            filenames = [image.filename for image in images]
            images = [image.file for image in images]
        return self._feex_service.check_duplicate(
            images=images, image_root=image_root, filenames=filenames, save_embeddings=save_embeddings
        )

    async def store_images(
        self,
        images: list[UploadFile] = File(None),
        image_root: Optional[str] = Form(None),
        filenames: list[str] = Form(None),
    ) -> None:
        """Embed images and store them in the database.

        This method just stores the embeddings in the database without checking for duplicates.
        This can be used for future duplicate searches.

        Args:
            images(list[UploadFile]): The images to embed and check for duplicates.
            image_root(Optional[str]): The root directory of the images if local images should be used
            filenames(list[str]): The filenames of the images if local images should be used. Optional.
        """
        if images:
            images = [image for image in images if image.content_type.startswith("image/")]
            filenames = [image.filename for image in images]
            images = [image.file for image in images]
        return self._feex_service.embed_and_store_images(images=images, image_root=image_root, filenames=filenames)
