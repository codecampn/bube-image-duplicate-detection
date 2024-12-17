FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libffi-dev libpq-dev&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry==1.8.4

COPY pyproject.toml poetry.lock README.md /app/
COPY bube /app/bube

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

RUN apt-get purge -y --auto-remove build-essential libffi-dev libpq-dev

EXPOSE 8000

ENTRYPOINT ["uvicorn", "bube:app", "--port", "8000", "--host", "0.0.0.0"]
