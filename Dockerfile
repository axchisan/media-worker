# media-worker — FFmpeg + edge-tts para el canal de contenido tech (CreationContent)
# Imagen liviana: el servidor tiene RAM justa, procesar 1 video a la vez.
FROM python:3.12-slim

# FFmpeg + fuentes para subtítulos quemados (ASS). fontconfig para que libass resuelva fuentes.
# libasound2 + ca-certificates: requeridos por el Azure Speech SDK (TTS multilingüe nativo).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        fontconfig \
        curl \
        libasound2 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY assets ./assets

# Directorio de trabajo temporal para render (se limpia tras cada job).
RUN mkdir -p /tmp/jobs

EXPOSE 8000

# Un solo worker: minimizar RAM y garantizar "un render a la vez".
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
