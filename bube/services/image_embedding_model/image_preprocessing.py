import numpy as np


def preprocess_imgs(input_imgs: np.ndarray) -> np.ndarray:
    """Preprocess images for the model.

    The model is based on a pretrained ResNet model, which was trained on ImageNet data.
    The images need to be preprocessed in the same way as the ImageNet data was preprocessed.
    This means that the images need to be normalized to have zero mean and unit variance.

    Args:
        input_imgs (np.ndarray): Batch of images to preprocess

    Returns:
        np.ndarray: Preprocessed images
    """
    # 'RGB'->'BGR'
    # Shape (Batch, Height, Width, Channel) remeins the same
    input_imgs = input_imgs[..., ::-1]

    # mean and std according to imagenet data (which was used to train the base model)
    mean = [103.939, 116.779, 123.68]
    std = None

    # Zero-center by mean pixel
    input_imgs[..., 0] -= mean[0]
    input_imgs[..., 1] -= mean[1]
    input_imgs[..., 2] -= mean[2]
    if std is not None:
        input_imgs[..., 0] /= std[0]
        input_imgs[..., 1] /= std[1]
        input_imgs[..., 2] /= std[2]

    return input_imgs
