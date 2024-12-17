import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import PIL
from PIL import Image
from pillow_heif import register_heif_opener

from ...config import config


class LocalImgReader:
    """Class to read images from local storage and return them as numpy arrays in batches.

    Images are grouped by resolution for efficient batch creation (and thus inference).
    Through the __len__ and __getitem__ methods, the class can be used as an iterable of batches.
    """

    _image_root: str
    _max_batch_size: int
    _filenames: list[str]
    _batches: list[list[str]]

    _logger: logging.Logger

    def __init__(
        self,
        filenames: Optional[list[str]] = None,
        image_root: str = "",
        max_batch_size: int = config.LOCAL_IMAGE_BATCH_SIZE,
        all_img_files: bool = False,
    ):
        register_heif_opener()  # support for HEIF images
        self._logger = logging.getLogger(__name__)
        filenames = filenames if filenames else []
        # if multiples filenames are passed as a single string (to the API) we split them up
        if len(filenames) == 1:
            filenames = filenames[0].split(",")

        self._logger.info(f"Reading images from folder: {image_root} with filenames: {filenames}")
        self._image_root = image_root
        self._max_batch_size = max_batch_size
        if not filenames and all_img_files:
            self._logger.info(f"No filenames provided. Reading all image files from folder: {image_root}")
            filenames = os.listdir(image_root)
            filenames = [name for name in filenames if name.lower().endswith((".jpg", ".jpeg", ".png", ".heif"))]
        self._filenames = [str(Path(image_root) / filename) for filename in filenames]
        self._batches = self._create_batches(self._filenames)
        self._logger.info(f"Found {len(self._filenames)} images in total. These are grouped into {len(self)} batches.")

    def __len__(self) -> int:
        """Get the number of available batches.

        Returns:
            int: number of available batches
        """
        return len(self._batches)

    def _create_batches(self, filenames: list[str]) -> list[list[str]]:
        img_dict = self._sort_img_by_res(filenames)
        return self._convert_dict_to_batches(img_dict)

    def _sort_img_by_res(self, filenames: list[str]) -> dict[tuple[int, int], list[str]]:
        """Group image files by resolution.

        Args:
            filenames (list[str]): filenames (including path) of images which should be grouped

        Returns:
            dict[tuple[int, int], list[str]]: dict, where key is the resolution of the images
                and value is a list of filenames with said res
        """
        img_files_by_res = {}
        for filename in filenames:
            # Read resolution without loading the whole image into memory
            try:
                with Image.open(filename) as img:
                    height, width = img.size
            except (PIL.UnidentifiedImageError, FileNotFoundError):
                self._logger.info(f"Could not read image resolution for file: {filename}. Skipping File.")
                continue

            # Group images files by image resolution
            key = (height, width)
            if key not in img_files_by_res:
                img_files_by_res[key] = []
            img_files_by_res[key].append(filename)

        return img_files_by_res

    def _convert_dict_to_batches(
        self, img_dict: dict[tuple[int, int], list[str]], batch_size: int = 0
    ) -> list[list[str]]:
        """Convert dictionary of images grouped by resolution to list of batches.

        Args:
            img_dict (dict[tuple[int, int], list[str]]): dict, where key is the resolution of the images
                and value is a list of filenames with said res
            batch_size (int, optional): maximum number of images in a batch

        Returns:
            list[list[str]]: list with batches of image filenames
        """
        batches = []
        batch_size = self._max_batch_size if batch_size == 0 else batch_size
        for filenames in img_dict.values():
            batches_per_resolution = [filenames[i : i + batch_size] for i in range(0, len(filenames), batch_size)]
            batches.extend(batches_per_resolution)
        return batches

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[str]]:
        """Get a batch of images.

        Args:
            index (int): index of the batch which should be returned

        Returns:
            tuple[np.ndarray, list[str]]: tuple with the batch as a numpy array and the corresponding list of filenames
        """
        if not 0 <= index <= len(self):
            error_msg = f"Index {index} out of range"
            raise IndexError(error_msg)
        # get filenames of images in the batch
        filenames = self._batches[index]

        # read images with PIL and convert them to a single numpy array
        batch_images = [np.array(Image.open(filename).convert("RGB"), dtype=np.float32) for filename in filenames]
        batch_images = np.stack(batch_images, axis=0)
        return batch_images, filenames
