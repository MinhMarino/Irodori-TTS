"""RunPod Serverless worker for Qwen3-TTS 1.7B (CustomVoice + VoiceDesign)."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import numpy as np
import runpod
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("irodori-worker")

CUSTOM_VOICE_MODEL = os.getenv(
    "CUSTOM_VOICE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)
VOICE_DESIGN_MODEL = os.getenv(
    "VOICE_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)
LOAD_MODELS = os.getenv("LOAD_MODELS", "both")  # both | custom_voice | voice_design

_models: dict[str, Any] = {}


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _load_one(model_id: str) -> Any:
    from qwen_tts import Qwen3TTSModel

    device = _device()
    dtype = _dtype()
    kwargs: dict[str, Any] = {"device_map": device, "dtype": dtype}
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
            log.info("Using flash_attention_2 for %s", model_id)
        except Exception:
            log.info("flash-attn unavailable; default attention for %s", model_id)

    log.info("Loading %s on %s (%s)...", model_id, device, dtype)
    return Qwen3TTSModel.from_pretrained(model_id, **kwargs)


def ensure_models() -> None:
    if _models:
        return
    modes = LOAD_MODELS.lower().strip()
    if modes in ("both", "custom_voice", "all"):
        _models["custom_voice"] = _load_one(CUSTOM_VOICE_MODEL)
    if modes in ("both", "voice_design", "all"):
        _models["voice_design"] = _load_one(VOICE_DESIGN_MODEL)
    log.info("Loaded modes: %s", list(_models))


def _wav_to_b64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32), int(sr), format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        ensure_models()
    except Exception as exc:
        log.exception("Model load failed")
        return {"error": f"Model load failed: {exc}"}

    inp = job.get("input") or {}
    mode = str(inp.get("mode") or "custom_voice").strip().lower()
    text = str(inp.get("text") or "").strip()
    language = str(inp.get("language") or "Auto").strip()
    instruct = str(inp.get("instruct") or "").strip()
    speaker = str(inp.get("speaker") or "Vivian").strip()

    if not text:
        return {"error": "text is required"}

    if mode in ("custom", "customvoice", "custom_voice"):
        mode = "custom_voice"
    elif mode in ("design", "voicedesign", "voice_design"):
        mode = "voice_design"
    else:
        return {
            "error": "mode must be 'custom_voice' or 'voice_design'",
            "supported_modes": list(_models),
        }

    if mode not in _models:
        return {
            "error": f"mode '{mode}' is not loaded on this worker",
            "loaded_modes": list(_models),
        }

    model = _models[mode]

    try:
        if mode == "custom_voice":
            kwargs: dict[str, Any] = {
                "text": text,
                "language": language,
                "speaker": speaker,
            }
            if instruct:
                kwargs["instruct"] = instruct
            wavs, sr = model.generate_custom_voice(**kwargs)
            meta = {"speaker": speaker, "instruct": instruct or None}
        else:
            if not instruct:
                return {"error": "instruct is required for voice_design mode"}
            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
            meta = {"instruct": instruct}

        audio_b64 = _wav_to_b64(wavs[0], sr)
        return {
            "mode": mode,
            "language": language,
            "sample_rate": int(sr),
            "audio_base64": audio_b64,
            "format": "wav",
            **meta,
        }
    except Exception as exc:
        log.exception("Synthesis failed")
        return {"error": f"Synthesis failed: {exc}"}


# Warm models at import time so cold-start happens before first job when possible.
try:
    ensure_models()
except Exception:
    log.exception("Warm load failed; will retry on first request")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
