import logging
from typing import BinaryIO, Optional

from ...config import config
from ...models import DuplicateReport, DuplicateReportPart, ImageEmbedding, SuspiciousFile
from ...repository import EmbeddedChromaDB, PgVector, VectorDBRepository
from ..local_image_service import LocalImageService
from ..remote_image_service import RemoteImageService


class FEEXService:
    """Service class for handling image embedding and duplicate detection.

    The service class can embed local images and images uploaded through the API.
    These images will be embedded and checked for duplicates in the database.
    As a default, these embeddings will also be saved for future duplicate checks.
    """

    _local_image_service: LocalImageService
    _remote_image_service: RemoteImageService

    __vector_db: VectorDBRepository

    _logger: logging.Logger

    duplicate_threshould: int

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._local_image_service = LocalImageService()
        self._remote_image_service = RemoteImageService()
        if config.DB_TYPE == "chroma":
            self._logger.info("Using ChromaDB")
            self.__vector_db = EmbeddedChromaDB()
        else:
            self._logger.info("Using PgVector")
            self.__vector_db = PgVector()

        self.duplicate_threshould = config.DUPLICATE_THRESHOLD_PERCENTAGE

    def check_duplicate(
        self,
        images: Optional[list[BinaryIO]] = None,
        image_root: Optional[str] = None,
        filenames: Optional[list[str]] = None,
        save_embeddings: bool = True,
    ) -> list[DuplicateReport]:
        """Checks for duplicates in the database.

        The images from the API or local storage (depending on provided params) will be embedded first.
        For each Image, a DuplicateReport will be created, containing duplicate and suspicious files.
        Optionally, the embeddings of the images will be saved in the database for future duplicate checks.

        Args:
            images (list[BinaryIO], optional): List of images as BinaryIO objects.
                If provided, these images will be used
            image_root (str,optional): Path to the root directory containing images.
                Will only be used, if no images are provided directly
            filenames (str, optional): List of filenames for the images. These can either be the filenames of the images
                from the API or the filenames of the images in the image_root directory.
            save_embeddings (bool,optinal): If True, the embeddings of the images will be saved in the database.
                Defaults to True.

        Returns:
            list[DuplicateReport]: A list of DuplicateReport objects. This includes the amount of duplicate and
                suspicious files with their filenames, distances and duplication_chance.

        """
        # create image embeddings
        if images:
            image_embeddings = self.embed_remote_images(images=images, filenames=filenames)
        else:
            image_embeddings = self.embed_local_images(image_root=image_root, filenames=filenames)

        # check for duplicates in db
        duplicate_reports = [self.create_duplicate_report(image_embedding) for image_embedding in image_embeddings]
        self._logger.info(f"Duplicate Report was created for {len(duplicate_reports)} images.")
        # Save the elements after inspection, so that the duplicate check doesn't operate on the images from same case
        if save_embeddings:
            self.store_image_embeddings(image_embeddings)

        return duplicate_reports

    def create_duplicate_report(self, image_embedding: ImageEmbedding) -> DuplicateReport:
        """Creates a DuplicateReport for a given image embedding.

        The DuplicateReport contains a list of duplicate files and a list of suspicious files.
        The duplicate files are files which are very similar to the original image (threshold at 80%).
        The suspicious files are files which are similar to the original image but not similar enough to be considered
        duplicates (similarity between 40% and 80%).

        Args:
            image_embedding (ImageEmbedding): Embedding of the image for which the duplicate report should be created

        Returns:
            DuplicateReport: Report containing duplicate and suspicious files with their filenames and similarity
        """
        neighbours = self.__vector_db.get_neighbours(image_embedding=image_embedding, threshold=0.6)
        neighbours = [SuspiciousFile.from_neighbour_embedding(neighbour) for neighbour in neighbours]

        duplicate_files = [
            neighbour for neighbour in neighbours if neighbour.duplicate_chance_in_percent >= self.duplicate_threshould
        ]
        suspicious_files = [
            neighbour for neighbour in neighbours if neighbour.duplicate_chance_in_percent < self.duplicate_threshould
        ]

        duplicate_files_report = DuplicateReportPart(num_of_files=len(duplicate_files), filenames=duplicate_files)
        suspicious_file_report = DuplicateReportPart(num_of_files=len(suspicious_files), filenames=suspicious_files)

        return DuplicateReport(
            original_filename=image_embedding.filename,
            duplicates=duplicate_files_report,
            suspicious=suspicious_file_report,
        )

    def embed_local_images(self, image_root: str, filenames: Optional[list[str]] = None) -> list[ImageEmbedding]:
        """Embeds images from local storage using the LocalImageService."""
        return self._local_image_service.embed_local_images(image_root=image_root, filenames=filenames)

    def embed_remote_images(
        self, images: list[BinaryIO], filenames: Optional[list[str]] = None
    ) -> list[ImageEmbedding]:
        """Embeds images uploaded through the API using the RemoteImageService."""
        return self._remote_image_service.embed_images(images=images, filenames=filenames)

    def store_image_embeddings(self, image_embeddings: list[ImageEmbedding]) -> None:
        """Stores the image embeddings in the database."""
        self.__vector_db.store_embeddings(image_embeddings)
        self._logger.info(f"Stored {len(image_embeddings)} image embeddings in the database.")

    def embed_and_store_images(
        self,
        images: Optional[list[BinaryIO]] = None,
        image_root: Optional[str] = None,
        filenames: Optional[list[str]] = None,
    ) -> None:
        """Embeds images and stores the embeddings in the database without performing a duplicate check.

        Depending on the provided arguments, the images are either from the API or from the local storage.

        Args:
            images (list[BinaryIO], optional): List of images as BinaryIO objects.
                If provided, these images will be used
            image_root (str,optional): Path to the root directory containing images.
                Will only be used, if no images are provided directly
            filenames (str, optional): List of filenames for the images. These can either be the filenames of the images
                from the API or the filenames of the images in the image_root directory.
        """
        if images:
            image_embeddings = self.embed_remote_images(images=images, filenames=filenames)
        else:
            image_embeddings = self.embed_local_images(image_root=image_root, filenames=filenames)

        self.store_image_embeddings(image_embeddings)
