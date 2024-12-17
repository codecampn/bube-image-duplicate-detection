import uvicorn

from .config import config

uvicorn.run("bube:app", host=config.BUBE_APP_HOST, port=config.BUBE_APP_PORT)
