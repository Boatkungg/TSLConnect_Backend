FROM python:3.12.1-slim-bookworm as base

FROM base as builder

RUN apt-get update

RUN apt-get install -y --no-install-recommends gcc build-essential

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml /install/
COPY README.md /install/
COPY .python-version /install/

WORKDIR /install

RUN pip install .

FROM base

COPY --from=builder /opt/venv /opt/venv

RUN apt-get update 

RUN apt-get install -y --no-install-recommends python3-opencv

COPY src/ /app/src
COPY data/ /app/data
COPY "thai_tokenizer.json" /app
COPY "sign_tokenizer.json" /app
COPY "sign2thai_model.bin" /app
COPY "thai2sign_model.bin" /app

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 5000

CMD ["uvicorn", "src.tslconnect_backend.main:app", "--host", "0.0.0.0", "--port", "5000"]
