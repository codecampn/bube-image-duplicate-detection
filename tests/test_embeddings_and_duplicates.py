import importlib.resources as impresources
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bube.routers import EmbeddingController, FEEXController

# overwrite the config to use a test DB
os.environ["DB_TYPE"] = "chroma"
os.environ["CHROMA_DB_MODE"] = "embedded"
os.environ["CHROMA_DB_EMBEDDED_PATH"] = "./data/chroma_db/"
os.environ["CHROMA_DB_DATABASE_NAME"] = "test_database"

app = FastAPI()
feex_controller = FEEXController()
app.include_router(feex_controller.router)

embedding_controller = EmbeddingController()
app.include_router(embedding_controller.router)

test_client = TestClient(app)

image_root = str(impresources.files("tests") / "test_assets")
filenames_assets = ["feex_check001.jpg",
                    "feex_check002.jpg"]

# we clean the test db before running the tests
feex_controller._feex_service._FEEXService__vector_db._clear_database()


def test_embeddings_local():
    params = {"image_root": image_root, "filenames": filenames_assets}
    response = test_client.get("/embeddings/local", params=params)
    assert response.status_code == 200
    check_response_format(response.json())


def test_embeddings_remote():
    files = []
    for filename in filenames_assets:
        image_path = Path(image_root) / filename
        files.append(("images", (filename, Path.open(image_path, "rb"), "image/jpeg")))
    response = test_client.post("/embeddings", files=files)
    assert response.status_code == 200
    check_response_format(response.json())


def test_feex_insert():
    files = []
    for filename in filenames_assets:
        image_path = Path(image_root) / filename
        files.append(("images", (filename, Path.open(image_path, "rb"), "image/jpeg")))

    response = test_client.post("/feex/insert", files=files)
    assert response.status_code == 201


def test_duplication_report():
    image_path = Path(image_root) / "feex_check001_duplicate.jpg"
    files = [("images", ("feex_check001_duplicate.jpg", Path.open(image_path, "rb"), "image/jpeg"))]

    response = test_client.post("/feex", files=files)
    assert response.status_code == 200

    isinstance(response.json(), list)
    duplicate_report = response.json()[0]

    assert "original_filename" in duplicate_report
    assert "duplicates" in duplicate_report
    assert "suspicious" in duplicate_report

    # the file is a duplicate, but was created with PIL in the Experiment.ipynb notebook
    # this creation can lead to a very small difference in the embeddings -> this distance isn't 0 but rather very small
    duplicate_files = duplicate_report["duplicates"]
    assert duplicate_files["num_of_files"] == 1
    assert duplicate_files["filenames"][0]["filename"] == "feex_check001.jpg"
    assert duplicate_files["filenames"][0]["distance"] < 0.03
    assert duplicate_files["filenames"][0]["duplicate_chance_in_percent"] >= 97

    suspicious_files = duplicate_report["suspicious"]
    assert suspicious_files["num_of_files"] == 1
    assert suspicious_files["filenames"][0]["filename"] == "feex_check002.jpg"
    assert suspicious_files["filenames"][0]["distance"] > 0
    assert suspicious_files["filenames"][0]["duplicate_chance_in_percent"] < 100


def check_response_format(response_data):
    assert len(response_data) == len(filenames_assets)
    for data in response_data:
        assert "embedding" in data
        assert "filename" in data
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) == 2048
        assert isinstance(data["filename"], str)


def check_if_duplicate_have_same_embedding(response_data):
    # emb of feex_check001.jpg and feex_check001_duplicate.jpg should be the same
    # but emb of feex_check002.jpg should be different
    emb001 = response_data[0]
    emb001_duplicate = response_data[2]
    emb002 = response_data[1]
    assert check_if_embeddings_match(emb001["embedding"], emb001_duplicate["embedding"])
    assert not check_if_embeddings_match(emb001["embedding"], emb002["embedding"])


def check_if_embeddings_match(emb1: list[float], emb2: list[float]):
    if len(emb1) != len(emb2):
        return False
    return all(value1 == value2 for value1, value2 in zip(emb1, emb2))
