FROM python:3.11-slim

# libgl/libglib are required by opencv-python even in headless use
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# Model weights are downloaded on first run; mount a volume to persist them
VOLUME ["/root/.cache/zonewatch"]

ENV HEADLESS=true
ENTRYPOINT ["zonewatch"]
