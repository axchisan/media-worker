"""
media-worker — microservicio de medios para el canal de contenido tech (CreationContent).

Stateless y liviano (el servidor tiene RAM justa → un render a la vez).
Expone:
  GET  /health  → healthcheck para Coolify/Traefik
  POST /tts     → texto -> MP3 (edge-tts) + timing por palabra (subtítulos karaoke sin Whisper)
  POST /render  → imágenes + audio (+ subtítulos ASS opcional) -> MP4 9:16

Auth: si la variable de entorno MEDIA_WORKER_API_KEY está definida, se exige
el header `X-API-Key` con ese valor en /tts y /render.
"""

import asyncio
import base64
import os
import shutil
import uuid
from typing import List, Optional

import edge_tts
import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

API_KEY = os.environ.get("MEDIA_WORKER_API_KEY", "").strip()
JOBS_DIR = "/tmp/jobs"
# Voz por defecto: español LatAm (decisión de marca de Fase 0).
DEFAULT_VOICE = os.environ.get("DEFAULT_TTS_VOICE", "es-CO-GonzaloNeural")

app = FastAPI(title="media-worker", version="0.1.0")

os.makedirs(JOBS_DIR, exist_ok=True)


def _check_auth(provided: Optional[str]) -> None:
    """Valida el API key si está configurado en el entorno."""
    if API_KEY and provided != API_KEY:
        raise HTTPException(status_code=401, detail="API key inválido o ausente.")


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {"status": "ok", "service": "media-worker", "version": "0.1.0"}


# --------------------------------------------------------------------------- #
# TTS
# --------------------------------------------------------------------------- #
class TTSRequest(BaseModel):
    text: str = Field(..., description="Texto a narrar.")
    voice: str = Field(default=DEFAULT_VOICE, description="Voz de edge-tts.")
    rate: str = Field(default="+0%", description="Velocidad, p.ej. '+10%'.")
    pitch: str = Field(default="+0Hz", description="Tono, p.ej. '+2Hz'.")
    volume: str = Field(default="+0%", description="Volumen, p.ej. '+0%'.")
    boundary: str = Field(
        default="WordBoundary",
        description="Granularidad del timing: 'WordBoundary' (karaoke) o 'SentenceBoundary'.",
    )


@app.post("/tts")
async def tts(req: TTSRequest, x_api_key: Optional[str] = Header(default=None)):
    """Genera narración MP3 y devuelve audio (base64) + timing por palabra."""
    _check_auth(x_api_key)

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' está vacío.")

    communicate = edge_tts.Communicate(
        req.text,
        voice=req.voice,
        rate=req.rate,
        pitch=req.pitch,
        volume=req.volume,
        boundary=req.boundary,
    )

    audio_chunks: List[bytes] = []
    words: List[dict] = []
    try:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                # edge-tts entrega offset/duration en ticks de 100ns.
                words.append(
                    {
                        "text": chunk["text"],
                        "offset_ms": chunk["offset"] / 10000,
                        "duration_ms": chunk["duration"] / 10000,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Fallo en edge-tts: {exc}")

    audio = b"".join(audio_chunks)
    if not audio:
        raise HTTPException(status_code=502, detail="edge-tts no devolvió audio.")

    duration_ms = (
        (words[-1]["offset_ms"] + words[-1]["duration_ms"]) if words else None
    )

    return JSONResponse(
        {
            "mime": "audio/mpeg",
            "audio_b64": base64.b64encode(audio).decode("ascii"),
            "duration_ms": duration_ms,
            "voice": req.voice,
            "words": words,
        }
    )


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
class ImageItem(BaseModel):
    b64: Optional[str] = Field(default=None, description="Imagen en base64.")
    url: Optional[str] = Field(default=None, description="URL de la imagen.")
    duration_sec: float = Field(default=3.0, description="Segundos en pantalla.")


class RenderRequest(BaseModel):
    images: List[ImageItem] = Field(..., description="Imágenes en orden.")
    audio_b64: Optional[str] = Field(default=None, description="Narración MP3 base64.")
    subtitles_ass: Optional[str] = Field(
        default=None, description="Contenido de un archivo .ass (subtítulos quemados)."
    )
    width: int = Field(default=1080)
    height: int = Field(default=1920)
    fps: int = Field(default=30)


async def _fetch_image_bytes(item: ImageItem) -> bytes:
    if item.b64:
        return base64.b64decode(item.b64)
    if item.url:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(item.url)
            resp.raise_for_status()
            return resp.content
    raise HTTPException(status_code=400, detail="Cada imagen requiere 'b64' o 'url'.")


@app.post("/render")
async def render(req: RenderRequest, x_api_key: Optional[str] = Header(default=None)):
    """Compone un MP4 9:16 a partir de imágenes + audio + subtítulos opcionales."""
    _check_auth(x_api_key)

    if not req.images:
        raise HTTPException(status_code=400, detail="Se requiere al menos una imagen.")

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    def cleanup():
        shutil.rmtree(job_dir, ignore_errors=True)

    try:
        # 1) Guardar imágenes y construir la lista del concat demuxer.
        concat_lines: List[str] = []
        for idx, item in enumerate(req.images):
            data = await _fetch_image_bytes(item)
            fname = f"img_{idx:03d}.png"
            with open(os.path.join(job_dir, fname), "wb") as fh:
                fh.write(data)
            concat_lines.append(f"file '{fname}'")
            concat_lines.append(f"duration {max(0.1, item.duration_sec)}")
        # El concat demuxer exige repetir el último archivo sin 'duration'.
        last = f"img_{len(req.images) - 1:03d}.png"
        concat_lines.append(f"file '{last}'")
        with open(os.path.join(job_dir, "list.txt"), "w") as fh:
            fh.write("\n".join(concat_lines) + "\n")

        # 2) Audio (opcional).
        has_audio = bool(req.audio_b64)
        if has_audio:
            with open(os.path.join(job_dir, "audio.mp3"), "wb") as fh:
                fh.write(base64.b64decode(req.audio_b64))

        # 3) Subtítulos ASS (opcional).
        has_subs = bool(req.subtitles_ass)
        if has_subs:
            with open(os.path.join(job_dir, "subs.ass"), "w", encoding="utf-8") as fh:
                fh.write(req.subtitles_ass)

        # 4) Filtro de video: cubrir 9:16 (scale+crop), fps fijo, opcional subtítulos.
        vf = (
            f"scale={req.width}:{req.height}:force_original_aspect_ratio=increase,"
            f"crop={req.width}:{req.height},setsar=1,fps={req.fps},format=yuv420p"
        )
        if has_subs:
            vf += ",subtitles=subs.ass"

        # 5) Comando ffmpeg. CPU débil -> preset veryfast.
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", "list.txt",
        ]
        if has_audio:
            cmd += ["-i", "audio.mp3"]
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += ["-movflags", "+faststart", "output.mp4"]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=job_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        out_path = os.path.join(job_dir, "output.mp4")
        if proc.returncode != 0 or not os.path.exists(out_path):
            cleanup()
            tail = stderr.decode("utf-8", "ignore")[-2000:]
            raise HTTPException(status_code=500, detail=f"ffmpeg falló:\n{tail}")

        # FileResponse + limpieza diferida del job al terminar el envío.
        return FileResponse(
            out_path,
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
            background=BackgroundTask(cleanup),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        cleanup()
        raise HTTPException(status_code=500, detail=f"Error en render: {exc}")
