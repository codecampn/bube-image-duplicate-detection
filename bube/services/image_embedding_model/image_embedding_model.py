import importlib.resources as impresources
import logging
from typing import Optional

import numpy as np
import onnxruntime as ort

from ...config import config
from .image_preprocessing import preprocess_imgs


class ImageEmbeddingModel:
    """Class to compute image embeddings using the provided image embedding model."""

    _instance = None
    _is_initialized = False

    _model: any
    _inference_dtype: np.dtype
    _execution_provider_list: list[str]

    __input_name: ort.NodeArg
    __output_name: ort.NodeArg

    _logger: logging.Logger

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one instance of the model is loaded."""
        if not cls._instance:
            cls._instance = super(ImageEmbeddingModel, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: Optional[str] = None, inference_dtype: np.dtype = np.float32):
        """Init the model.

        Args:
            model_path (str, optional): Path to the model (if it should be exchanged)
            inference_dtype (np.dtype, optional): Data type for inference. Defaults to np.float32.
            execution_provider_list (list[str], optional): List of execution providers if e.g.GPU should be used.
                Defaults to ['CPUExecutionProvider'] for CPU usage.
        """
        if self._is_initialized:
            return

        self._logger = logging.getLogger(__name__)
        self._inference_dtype = inference_dtype
        self._execution_provider_list = self._get_execution_providers()
        if not model_path:
            model_path = str(impresources.files("bube.services.image_embedding_model") / "resnet_mac_model.onnx")
        self._model = ort.InferenceSession(model_path, providers=self._execution_provider_list)
        self.__input_name = self._model.get_inputs()[0].name
        self.__output_name = self._model.get_outputs()[0].name
        self._is_initialized = True

        self._logger.info(f"Model loaded from: {model_path} successfully.")

    def compute_embedding_single(self, input_img: np.ndarray) -> np.ndarray:
        """Compute embeddings for a single image.

        Args:
            input_img (np.ndarray): Image to compute embeddings for. Shape should be: (Height, Width, Channel=3)

        Returns:
            np.ndarray: Embeddings for the input image in shape (Embedding_dim=2048)
        """
        # if input is single image, add batch dimension before inference
        input_img = np.expand_dims(input_img, axis=0)
        prediction = self.compute_embedding_batch(input_img)
        return prediction.squeeze()

    def compute_embedding_batch(self, input_img_batch: np.ndarray) -> np.ndarray:
        """Compute embeddings for a batch of images.

        Args:
            input_img_batch (np.ndarray): Batch of images to compute embeddings for. Shape should be:
            (Batch, Height, Width, Channel=3)

        Returns:
            np.ndarray: Embeddings for the input images in shape (Batch, Embedding_dim=2048)
        """
        input_img_batch = preprocess_imgs(input_img_batch)
        return self._model.run(
            [self.__output_name], {self.__input_name: input_img_batch.astype(self._inference_dtype)}
        )[0]

    def _get_execution_providers(self) -> list[str]:
        """Get the list of execution providers based on availability and user preference."""
        if not config.USE_GPU:
            self._logger.info("USE_GPU is set to False. Using only CPU for inference.")
            return ["CPUExecutionProvider"]

        available_providers = ort.get_available_providers()
        preferred_providers = []
        if "CUDAExecutionProvider" in available_providers:
            preferred_providers.append("CUDAExecutionProvider")
            self._logger.info("NVIDIA GPU with CUDA is availabe.")
        else:
            self._logger.info("No GPU available. Using CPU for inference.")

        preferred_providers.append("CPUExecutionProvider")

        return preferred_providers
