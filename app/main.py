"""Irodori TTS — Qwen3-TTS 0.6B CustomVoice web UI."""

from __future__ import annotations

import io
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("irodori-tts")

MODEL_ID = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEVICE = os.getenv("TTS_DEVICE", "auto")  # auto | cpu | cuda | mps
OUTPUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", "/app/output"))
STATIC_DIR = Path(__file__).parent / "static"

LANGUAGES = [
    {"id": "Auto", "label": "Tự động", "flag": "🌐"},
    {"id": "Chinese", "label": "Tiếng Trung", "flag": "🇨🇳"},
    {"id": "English", "label": "Tiếng Anh", "flag": "🇺🇸"},
    {"id": "Japanese", "label": "Tiếng Nhật", "flag": "🇯🇵"},
    {"id": "Korean", "label": "Tiếng Hàn", "flag": "🇰🇷"},
    {"id": "German", "label": "Tiếng Đức", "flag": "🇩🇪"},
    {"id": "French", "label": "Tiếng Pháp", "flag": "🇫🇷"},
    {"id": "Russian", "label": "Tiếng Nga", "flag": "🇷🇺"},
    {"id": "Portuguese", "label": "Tiếng Bồ Đào Nha", "flag": "🇵🇹"},
    {"id": "Spanish", "label": "Tiếng Tây Ban Nha", "flag": "🇪🇸"},
    {"id": "Italian", "label": "Tiếng Ý", "flag": "🇮🇹"},
]

# 9 premium timbres from Qwen3-TTS CustomVoice
SPEAKERS = [
    {
        "id": "Vivian",
        "name": "Vivian",
        "gender": "female",
        "native": "Chinese",
        "tagline": "Năng động, trẻ trung",
        "description": "Giọng nữ trẻ sáng và hơi sắc nét — phù hợp hội thoại hiện đại.",
        "color": "#E85D75",
        "emoji": "🌸",
    },
    {
        "id": "Serena",
        "name": "Serena",
        "gender": "female",
        "native": "Chinese",
        "tagline": "Ấm áp, dịu dàng",
        "description": "Giọng nữ trẻ ấm áp, nhẹ nhàng — lý tưởng kể chuyện, ASMR nhẹ.",
        "color": "#6B9AC4",
        "emoji": "🕊️",
    },
    {
        "id": "Uncle_Fu",
        "name": "Uncle Fu",
        "gender": "male",
        "native": "Chinese",
        "tagline": "Trầm ấm, dày dặn",
        "description": "Giọng nam trung niên trầm mellow — kể chuyện, thuyết minh.",
        "color": "#8B6F47",
        "emoji": "🍵",
    },
    {
        "id": "Dylan",
        "name": "Dylan",
        "gender": "male",
        "native": "Chinese",
        "tagline": "Bắc Kinh trẻ",
        "description": "Giọng nam trẻ Bắc Kinh rõ ràng, tự nhiên — phương ngữ Bắc Kinh.",
        "color": "#3D5A80",
        "emoji": "🏙️",
    },
    {
        "id": "Eric",
        "name": "Eric",
        "gender": "male",
        "native": "Chinese",
        "tagline": "Thành Đô sống động",
        "description": "Giọng nam Thành Đô hơi khàn sáng — phương ngữ Tứ Xuyên.",
        "color": "#E07A3D",
        "emoji": "🌶️",
    },
    {
        "id": "Ryan",
        "name": "Ryan",
        "gender": "male",
        "native": "English",
        "tagline": "Nhịp điệu mạnh",
        "description": "Giọng nam Anh năng động, nhịp rõ — podcast, quảng cáo.",
        "color": "#2A9D8F",
        "emoji": "🎙️",
    },
    {
        "id": "Aiden",
        "name": "Aiden",
        "gender": "male",
        "native": "English",
        "tagline": "Sunny American",
        "description": "Giọng nam Mỹ sáng trung âm — thân thiện, dễ nghe.",
        "color": "#457B9D",
        "emoji": "☀️",
    },
    {
        "id": "Ono_Anna",
        "name": "Ono Anna",
        "gender": "female",
        "native": "Japanese",
        "tagline": "Nhẹ nhàng, tinh nghịch",
        "description": "Giọng nữ Nhật vui tươi, linh hoạt — anime / VTuber vibe.",
        "color": "#E07A9A",
        "emoji": "🍡",
    },
    {
        "id": "Sohee",
        "name": "Sohee",
        "gender": "female",
        "native": "Korean",
        "tagline": "Ấm, giàu cảm xúc",
        "description": "Giọng nữ Hàn ấm và giàu cảm xúc — ballad, narration.",
        "color": "#F4A261",
        "emoji": "🌙",
    },
]

_model = None
_model_lock = threading.Lock()
_model_error: str | None = None
_ready = False


def resolve_device() -> str:
    choice = DEVICE.lower().strip()
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if choice == "cuda":
        return "cuda:0"
    return choice


def resolve_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model() -> Any:
    global _model, _ready, _model_error
    with _model_lock:
        if _model is not None:
            return _model
        from qwen_tts import Qwen3TTSModel

        device = resolve_device()
        dtype = resolve_dtype(device)
        log.info("Loading %s on %s (%s)...", MODEL_ID, device, dtype)

        kwargs: dict[str, Any] = {
            "device_map": device,
            "dtype": dtype,
        }
        if device.startswith("cuda"):
            try:
                import flash_attn  # noqa: F401

                kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                log.info("flash-attn not available; using default attention")

        try:
            _model = Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)
            _ready = True
            _model_error = None
            log.info("Model ready.")
            return _model
        except Exception as exc:
            _model_error = str(exc)
            _ready = False
            log.exception("Failed to load model")
            raise


app = FastAPI(title="Irodori TTS", version="1.0.0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def startup() -> None:
    # Lazy-load in a background thread so the UI is reachable while weights download.
    def _bg() -> None:
        try:
            load_model()
        except Exception:
            pass

    threading.Thread(target=_bg, daemon=True).start()


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    language: str = "Auto"
    speaker: str = "Vivian"
    instruct: str = ""


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ready": _ready,
        "model": MODEL_ID,
        "device": resolve_device(),
        "error": _model_error,
    }


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "languages": LANGUAGES,
        "speakers": SPEAKERS,
        "ready": _ready,
        "device": resolve_device(),
    }


@app.post("/api/synthesize")
def synthesize(req: SynthesizeRequest) -> Response:
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Vui lòng nhập văn bản.")

    lang_ids = {x["id"] for x in LANGUAGES}
    speaker_ids = {x["id"] for x in SPEAKERS}
    if req.language not in lang_ids:
        raise HTTPException(400, f"Ngôn ngữ không hỗ trợ: {req.language}")
    if req.speaker not in speaker_ids:
        raise HTTPException(400, f"Nhân vật không hỗ trợ: {req.speaker}")

    try:
        model = load_model()
    except Exception as exc:
        raise HTTPException(503, f"Model chưa sẵn sàng: {exc}") from exc

    with _model_lock:
        try:
            kwargs: dict[str, Any] = {
                "text": text,
                "language": req.language,
                "speaker": req.speaker,
            }
            instruct = (req.instruct or "").strip()
            if instruct:
                kwargs["instruct"] = instruct
            wavs, sr = model.generate_custom_voice(**kwargs)
        except Exception as exc:
            log.exception("Synthesis failed")
            raise HTTPException(500, f"Tổng hợp thất bại: {exc}") from exc

    audio = np.asarray(wavs[0], dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, int(sr), format="WAV")
    data = buf.getvalue()

    out_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"
    out_path.write_bytes(data)

    return Response(
        content=data,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="irodori.wav"',
            "X-Sample-Rate": str(int(sr)),
        },
    )


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
