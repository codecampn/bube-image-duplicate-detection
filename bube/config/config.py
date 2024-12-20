from typing import Literal, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration class for the whole application. Values can be overwrittren by environment variables."""

    BUBE_MODE: Literal["embedding", "app"] = "app"
    BUBE_APP_PORT: int = 8000
    BUBE_APP_HOST: str = "0.0.0.0"  # noqa: S104

    # DB Config (if BUBE_MODE == "app")
    DB_TYPE: Literal["chroma", "pgvector"] = "chroma"

    CHROMA_DB_MODE: Literal["embedded", "http"] = "embedded"
    CHROMA_DB_EMBEDDED_PATH: str = "./data/chroma_db/"
    CHROMA_DB_HTTP_HOST: str = "localhost"
    CHROMA_DB_HTTP_PORT: int = 8000
    CHROMA_DB_HTTP_HEADERS: Optional[dict[str, str]] = None
    CHROMA_DB_HTTP_SSL: bool = True
    CHROMA_DB_DATABASE_NAME: str = "img_embeddings"

    PGVECTOR_DB_HOST: str = "localhost"
    PGVECTOR_DB_PORT: int = 5432
    PGVECTOR_DB_USER: str = "postgres"
    PGVECTOR_DB_PWD: SecretStr = "mypassword" # noqa: S105
    PGVECTOR_DB_HTTP_SSL: bool = False
    PGVECTOR_DB_DATABASE_NAME: str = "postgres"
    PGVECTOR_DB_TABLE_NAME: str = "feex_embeddings"

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    USE_GPU: bool = True
    LOCAL_IMAGE_BATCH_SIZE: int = 4

    DUPLICATE_THRESHOLD_PERCENTAGE: int = 80


config = Config()
