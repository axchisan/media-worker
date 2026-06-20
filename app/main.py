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
# Voz por defecto: español LatAm (decisión de marca: es-MX-JorgeNeural).
DEFAULT_VOICE = os.environ.get("DEFAULT_TTS_VOICE", "es-MX-JorgeNeural")
# Música de fondo del canal (CC-BY, ver assets/music/CREDITS.md).
MUSIC_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "music", "carefree.mp3")
)

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


class MascotSeg(BaseModel):
    """Un clip de mascota (con alfa) a superponer durante una ventana de tiempo."""
    url: Optional[str] = Field(default=None, description="URL del clip de mascota (mp4 'alfa empacado': color arriba, alfa abajo).")
    b64: Optional[str] = Field(default=None, description="Clip de mascota en base64 (mismo formato empacado).")
    start_sec: float = Field(default=0.0, description="Inicio del overlay (s).")
    end_sec: float = Field(default=99999.0, description="Fin del overlay (s).")
    corner: str = Field(default="br", description="Esquina: br, bl, tr, tl.")
    scale: float = Field(default=0.34, description="Alto del clip como fracción del alto del video.")
    margin: int = Field(default=40, description="Margen en px desde el borde.")


class RenderRequest(BaseModel):
    images: List[ImageItem] = Field(..., description="Imágenes en orden.")
    audio_b64: Optional[str] = Field(default=None, description="Narración MP3 base64.")
    subtitles_ass: Optional[str] = Field(
        default=None, description="Contenido de un archivo .ass (subtítulos quemados)."
    )
    width: int = Field(default=1080)
    height: int = Field(default=1920)
    fps: int = Field(default=30)
    transition: str = Field(
        default="fade", description="Transición entre escenas (xfade) o 'none'."
    )
    transition_duration: float = Field(default=0.5, description="Duración de la transición (s).")
    ken_burns: bool = Field(default=True, description="Movimiento Ken Burns (zoom/pan lento).")
    motion_intensity: float = Field(default=0.12, description="Cuánto zoom del Ken Burns (0.12 = +12%).")
    background_music: bool = Field(default=True, description="Mezclar música de fondo del canal (CC-BY).")
    music_volume: float = Field(default=0.18, description="Volumen de la música respecto a la narración.")
    mascots: List[MascotSeg] = Field(default_factory=list, description="Clips de mascota (alfa) a superponer.")


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
        # 1) Guardar imágenes.
        for idx, item in enumerate(req.images):
            data = await _fetch_image_bytes(item)
            with open(os.path.join(job_dir, f"img_{idx:03d}.png"), "wb") as fh:
                fh.write(data)
        n = len(req.images)

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

        # 3b) Clips de mascota (opcional, .mov con alfa).
        mascot_files: List[tuple] = []  # (filename, seg)
        for k, seg in enumerate(req.mascots):
            try:
                if seg.b64:
                    data = base64.b64decode(seg.b64)
                elif seg.url:
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.get(seg.url)
                        r.raise_for_status()
                        data = r.content
                else:
                    continue
                fname = f"mascot_{k}.mp4"
                with open(os.path.join(job_dir, fname), "wb") as fh:
                    fh.write(data)
                mascot_files.append((fname, seg))
            except Exception:
                continue

        W, H, FPS = req.width, req.height, req.fps
        TD = max(0.0, req.transition_duration)
        durations = [max(0.6, it.duration_sec) for it in req.images]
        use_xfade = bool(req.transition) and req.transition != "none" and n > 1

        # 4) Inputs: cada imagen en bucle por su duración (frames a FPS fijo).
        cmd = ["ffmpeg", "-y"]
        for idx in range(n):
            cmd += [
                "-loop", "1", "-framerate", str(FPS),
                "-t", f"{durations[idx]:.3f}", "-i", f"img_{idx:03d}.png",
            ]
        if has_audio:
            cmd += ["-i", "audio.mp3"]
        music_on = req.background_music and os.path.exists(MUSIC_PATH)
        if music_on:
            cmd += ["-stream_loop", "-1", "-i", MUSIC_PATH]
        for fname, _seg in mascot_files:
            cmd += ["-stream_loop", "-1", "-i", fname]

        # 5) Filtro por imagen: cover 9:16 + Ken Burns (zoom alternado in/out).
        filters: List[str] = []
        for idx in range(n):
            frames = max(1, int(round(durations[idx] * FPS)))
            if req.ken_burns:
                k = req.motion_intensity / frames  # por frame para llegar a +intensity
                if idx % 2 == 0:
                    zexpr = f"min(1+on*{k:.6f},{1 + req.motion_intensity:.3f})"
                else:
                    zexpr = f"max({1 + req.motion_intensity:.3f}-on*{k:.6f},1.0)"
                filt = (
                    f"[{idx}:v]scale={W*2}:{H*2}:force_original_aspect_ratio=increase,"
                    f"crop={W*2}:{H*2},"
                    f"zoompan=z='{zexpr}':d=1:s={W}x{H}:fps={FPS}:"
                    f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',"
                    f"setsar=1,format=yuv420p[v{idx}]"
                )
            else:
                filt = (
                    f"[{idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                    f"crop={W}:{H},fps={FPS},setsar=1,format=yuv420p[v{idx}]"
                )
            filters.append(filt)

        # 6) Combinar: xfade encadenado, concat, o passthrough.
        if n == 1:
            last_label = "[v0]"
        elif use_xfade:
            timeline = durations[0]
            prev = "[v0]"
            for i in range(1, n):
                offset = max(0.0, timeline - TD)
                out = f"[x{i}]"
                filters.append(
                    f"{prev}[v{i}]xfade=transition={req.transition}:"
                    f"duration={TD}:offset={offset:.3f}{out}"
                )
                timeline = timeline + durations[i] - TD
                prev = out
            last_label = prev
        else:
            joined = "".join(f"[v{i}]" for i in range(n))
            filters.append(f"{joined}concat=n={n}:v=1:a=0[cv]")
            last_label = "[cv]"

        # 6b) Overlay de mascota(s) con alfa, sobre las escenas (debajo de subtítulos).
        base_m = n + (1 if has_audio else 0) + (1 if music_on else 0)
        cur = last_label
        for k, (fname, seg) in enumerate(mascot_files):
            midx = base_m + k
            th = max(2, int(H * seg.scale))
            m = seg.margin
            if seg.corner == "bl":
                pos = f"{m}:H-h-{m}"
            elif seg.corner == "tr":
                pos = f"W-w-{m}:{m}"
            elif seg.corner == "tl":
                pos = f"{m}:{m}"
            else:  # br por defecto
                pos = f"W-w-{m}:H-h-{m}"
            # Clip "alfa empacado": color arriba, máscara alfa (gris) abajo.
            filters.append(f"[{midx}:v]split=2[c{k}][a{k}]")
            filters.append(f"[c{k}]crop=iw:ih/2:0:0[col{k}]")
            filters.append(f"[a{k}]crop=iw:ih/2:0:ih/2,format=gray[alp{k}]")
            filters.append(f"[col{k}][alp{k}]alphamerge,scale=-1:{th}[mk{k}]")
            out = f"[mov{k}]"
            filters.append(
                f"{cur}[mk{k}]overlay={pos}:enable='between(t,{seg.start_sec},{seg.end_sec})'{out}"
            )
            cur = out
        last_label = cur

        # 7) Subtítulos quemados sobre el resultado final.
        if has_subs:
            filters.append(f"{last_label}subtitles=subs.ass[vout]")
        else:
            filters.append(f"{last_label}null[vout]")

        # 8) Audio: narración + música de fondo (volumen bajo) vía amix.
        music_idx = (n + 1) if has_audio else n
        audio_map: Optional[str] = None
        if has_audio and music_on:
            filters.append(
                f"[{music_idx}:a]volume={req.music_volume}[mus];"
                f"[{n}:a][mus]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
            )
            audio_map = "[aout]"
        elif has_audio:
            audio_map = f"{n}:a"
        elif music_on:
            filters.append(f"[{music_idx}:a]volume={req.music_volume}[aout]")
            audio_map = "[aout]"

        cmd += ["-filter_complex", ";".join(filters), "-map", "[vout]"]
        if audio_map:
            cmd += ["-map", audio_map, "-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "output.mp4",
        ]

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
