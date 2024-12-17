import importlib.resources as impresources

import numpy as np

from bube.services.local_image_service import LocalImageService, LocalImgReader

asset_path = str(impresources.files("tests") / "test_assets")
filenames_assets = [f"feex_check00{i}.jpg" for i in range(1, 6)]
filenames_assets.append("feex_check001_duplicate.jpg")


def test_local_image_reader():
    # pass only image_root
    reader = LocalImgReader(image_root=asset_path, all_img_files=True)
    assert len(reader._filenames) == 75  # 5 images with 14 different variants->  5 + 5*14 = 75

    # pass image_root and just one filename
    reader = LocalImgReader(image_root=asset_path, filenames=[filenames_assets[0]])
    assert len(reader._filenames) == 1

    # pass image_root and multiple filenames
    reader = LocalImgReader(image_root=asset_path, filenames=filenames_assets)
    assert len(reader._filenames) == len(filenames_assets)

    # ImageReader Groups by resolution,
    # feex_check001.jpg, 002.jpg, 004.jpg and check001_duplicate.jpg  have the same resolution -> one batch
    # feex_check003.jpg and feex_check005.jpg have different resolutions -> two batches
    # -> total 3 batches
    assert len(reader) == 3

    # test iteration
    for img, filenames in reader:
        assert isinstance(img, np.ndarray)
        assert isinstance(filenames, list)
        # check dimension of the image
        assert len(img) == len(filenames)


def test_local_image_service():
    service = LocalImageService()
    embeddings = service.embed_local_images(image_root=asset_path, filenames=filenames_assets)

    # 1 embedding for each image
    assert len(embeddings) == len(filenames_assets)

    for embedding in embeddings:
        # embedding should be list[float] with length 2048
        assert isinstance(embedding.embedding, list)
        assert len(embedding.embedding) == 2048
        assert isinstance(embedding.filename, str)
