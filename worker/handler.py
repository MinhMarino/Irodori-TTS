"""RunPod Serverless worker — Qwen3-TTS 1.7B (CustomVoice + VoiceDesign).

Cold-start optimizations:
- Prefer Runpod model cache at /runpod-volume/huggingface-cache/hub
- Fall back to baked /models paths, then Hugging Face hub id
- Lazy-load each mode on first use (keeps unused model off GPU)
- Optional PRELOAD_MODELS to warm selected modes at boot
"""

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
# none | custom_voice | voice_design | both
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "none").strip().lower()
HF_CACHE_ROOT = os.getenv(
    "HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"
)
BAKED_MODEL_ROOT = os.getenv("BAKED_MODEL_ROOT", "/models")
# Extend init window for large GPU load (seconds)
os.environ.setdefault("RUNPOD_INIT_TIMEOUT", "800")

_models: dict[str, Any] = {}


def resolve_snapshot_path(model_id: str) -> str | None:
    """Return local snapshot dir for a HF model id if present on disk."""
    if "/" not in model_id:
        return None
    org, name = model_id.split("/", 1)

    candidates = [
        os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}"),
        os.path.join(BAKED_MODEL_ROOT, org, name),
        os.path.join(BAKED_MODEL_ROOT, f"models--{org}--{name}"),
    ]

    for model_root in candidates:
        refs_main = os.path.join(model_root, "refs", "main")
        snapshots_dir = os.path.join(model_root, "snapshots")

        if os.path.isfile(refs_main) and os.path.isdir(snapshots_dir):
            with open(refs_main, "r", encoding="utf-8") as f:
                snapshot_hash = f.read().strip()
            candidate = os.path.join(snapshots_dir, snapshot_hash)
            if os.path.isdir(candidate):
                log.info("Resolved %s -> %s (refs/main)", model_id, candidate)
                return candidate

        if os.path.isdir(snapshots_dir):
            versions = sorted(
                d
                for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))
            )
            if versions:
                chosen = os.path.join(snapshots_dir, versions[0])
                log.info("Resolved %s -> %s (first snapshot)", model_id, chosen)
                return chosen

        # Baked flat checkout (files directly under org/name)
        if os.path.isdir(model_root) and any(
            os.path.isfile(os.path.join(model_root, f))
            for f in ("config.json", "model.safetensors", "model.safetensors.index.json")
        ):
            log.info("Resolved %s -> %s (flat dir)", model_id, model_root)
            return model_root

    return None


def resolve_model_source(model_id: str) -> str:
    local = resolve_snapshot_path(model_id)
    if local:
        return local
    log.warning("No local cache for %s — will download from Hub (slow cold start)", model_id)
    return model_id


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _load_one(model_id: str) -> Any:
    from qwen_tts import Qwen3TTSModel

    source = resolve_model_source(model_id)
    device = _device()
    dtype = _dtype()
    kwargs: dict[str, Any] = {
        "device_map": device,
        "dtype": dtype,
    }
    # When loading from a local snapshot, avoid Hub round-trips.
    if source != model_id:
        kwargs["local_files_only"] = True
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
            log.info("Using flash_attention_2 for %s", model_id)
        except Exception:
            log.info("flash-attn unavailable; default attention for %s", model_id)

    log.info("Loading %s from %s on %s (%s)...", model_id, source, device, dtype)
    return Qwen3TTSModel.from_pretrained(source, **kwargs)


def get_model(mode: str) -> Any:
    if mode in _models:
        return _models[mode]
    model_id = CUSTOM_VOICE_MODEL if mode == "custom_voice" else VOICE_DESIGN_MODEL
    _models[mode] = _load_one(model_id)
    return _models[mode]


def preload() -> None:
    wanted = PRELOAD_MODELS
    if wanted in ("", "none", "0", "false"):
        log.info("PRELOAD_MODELS=%s — lazy load on first request", wanted or "none")
        return
    modes: list[str] = []
    if wanted in ("both", "all"):
        modes = ["custom_voice", "voice_design"]
    elif wanted in ("custom", "customvoice", "custom_voice"):
        modes = ["custom_voice"]
    elif wanted in ("design", "voicedesign", "voice_design"):
        modes = ["voice_design"]
    else:
        log.warning("Unknown PRELOAD_MODELS=%s; skipping preload", wanted)
        return
    for mode in modes:
        try:
            get_model(mode)
        except Exception:
            log.exception("Preload failed for %s", mode)


def _wav_to_b64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32), int(sr), format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def handler(job: dict[str, Any]) -> dict[str, Any]:
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
        return {"error": "mode must be 'custom_voice' or 'voice_design'"}

    try:
        model = get_model(mode)
    except Exception as exc:
        log.exception("Model load failed")
        return {"error": f"Model load failed: {exc}"}

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

        return {
            "mode": mode,
            "language": language,
            "sample_rate": int(sr),
            "audio_base64": _wav_to_b64(wavs[0], sr),
            "format": "wav",
            "loaded_modes": list(_models),
            **meta,
        }
    except Exception as exc:
        log.exception("Synthesis failed")
        return {"error": f"Synthesis failed: {exc}"}


# Optional warm load at import (default: none → fastest container start)
preload()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
