# Irodori TTS RunPod Worker

Serverless worker for **Qwen3-TTS 12Hz 1.7B** — CustomVoice (primary) + VoiceDesign (lazy).

Latency guide: [`scripts/create-runpod-endpoint.md`](../scripts/create-runpod-endpoint.md).

**GPU note:** image is PyTorch 2.4 — pin endpoint to `ADA_24` (4090). Blackwell MIG (e.g. RTX PRO 6000 MIG 1g.24gb) fails with `CUDA error: no kernel image is available for execution on the device`.

## Architecture (fast path)

1. Endpoint keeps **1 warm worker** (`workersMin=1`).
2. RunPod **Cached model** = CustomVoice only (1 model / endpoint limit).
3. Worker **preloads + warms** CustomVoice at boot.
4. VoiceDesign loads on first use (slower).

## Modes

| `mode` | Model | Required fields |
|---|---|---|
| `custom_voice` | CustomVoice | `text`, `language`, `speaker` (+ optional `instruct`) |
| `voice_design` | VoiceDesign | `text`, `language`, `instruct` |

## Env

| Var | Default | Notes |
|---|---|---|
| `PRELOAD_MODELS` | `custom_voice` | `none` / `custom_voice` / `voice_design` / `both` |
| `WARMUP_ON_LOAD` | `1` | Tiny generate after load |
| `STRICT_LOCAL_CACHE` | `0` | `1` = fail if cache missing (no Hub download) |
| `HF_CACHE_ROOT` | `/runpod-volume/huggingface-cache/hub` | RunPod model cache mount |

## Request

```json
{
  "input": {
    "mode": "custom_voice",
    "text": "Hello from Irodori.",
    "language": "English",
    "speaker": "Ryan"
  }
}
```

## Response extras

```json
{
  "warm": true,
  "cache_hit": true,
  "timings_ms": { "model_load": 0, "infer": 7900, "handler_total": 8100 },
  "audio_base64": "UklGRi..."
}
```

## Speakers

Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
