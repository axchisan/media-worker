# media-worker

Microservicio de medios para el canal automatizado de contenido tech (**CreationContent**).
Stateless y liviano — el servidor tiene RAM justa, así que procesa **un render a la vez**.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Healthcheck (Coolify/Traefik). |
| `POST` | `/tts` | Texto → MP3 (edge-tts) + timing por palabra para subtítulos karaoke. |
| `POST` | `/render` | Imágenes + audio (+ subtítulos ASS) → MP4 9:16 1080×1920. |

### Auth
Si `MEDIA_WORKER_API_KEY` está definida, `/tts` y `/render` exigen el header `X-API-Key`.

### `POST /tts`
```json
{ "text": "Hola mundo", "voice": "es-CO-GonzaloNeural", "rate": "+0%" }
```
Respuesta: `{ "audio_b64", "mime", "duration_ms", "words": [{ "text", "offset_ms", "duration_ms" }] }`

### `POST /render`
```json
{
  "images": [{ "b64": "...", "duration_sec": 3 }],
  "audio_b64": "...",
  "subtitles_ass": "[Script Info]...",
  "width": 1080, "height": 1920, "fps": 30
}
```
Respuesta: binario `video/mp4`.

## Variables de entorno
| Var | Default | Uso |
|---|---|---|
| `MEDIA_WORKER_API_KEY` | (vacío) | Si se define, exige `X-API-Key`. |
| `DEFAULT_TTS_VOICE` | `es-CO-GonzaloNeural` | Voz por defecto (español LatAm). |

## Desarrollo local
```bash
docker build -t media-worker .
docker run --rm -p 8000:8000 media-worker
curl localhost:8000/health
```

## Despliegue
Coolify (Dockerfile build) → expuesto por Traefik en `media.axchisan.com`. Consumido por n8n vía HTTP.
