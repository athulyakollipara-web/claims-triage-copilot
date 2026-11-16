# Local-mode demo: builds the sample data, features and model at image build time, then serves the API.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY claims_copilot ./claims_copilot
RUN pip install --no-cache-dir ".[spark]" && python -m claims_copilot.cli pipeline

ENV CLAIMS_MODE=local CLAIMS_LLM=template
EXPOSE 8000
CMD ["uvicorn", "claims_copilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
