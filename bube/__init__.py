from fastapi import FastAPI

from .config import config
from .logger import setup_logging
from .routers import EmbeddingController, FEEXController

setup_logging()

app = FastAPI()

embedding_controller = EmbeddingController()
app.include_router(embedding_controller.router)

if config.BUBE_MODE == "app":
    feex_controller = FEEXController()
    app.include_router(feex_controller.router)
