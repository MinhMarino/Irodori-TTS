# Tạo RunPod Endpoint (sau khi nạp credit)

Image đã build: `ghcr.io/minhmarino/irodori-tts-worker:1.7b-dual`

## Checklist trước khi tạo

1. Nạp credit RunPod: https://console.runpod.io/user/billing  
   (cần tối thiểu vài USD — API báo cần ≥ $0.01)
2. Public package GHCR (để RunPod pull không cần registry auth):  
   https://github.com/users/MinhMarino/packages/container/package/irodori-tts-worker  
   → **Package settings** → Change visibility → **Public**

## Tạo endpoint (Cursor + RunPod MCP)

Trong chat Cursor (đã login RunPod MCP), bảo agent:

> Tạo endpoint irodori-tts-1.7b-dual từ image `ghcr.io/minhmarino/irodori-tts-worker:1.7b-dual`, GPU pool ADA_24 + AMPERE_24, workers 0–2

Hoặc dùng cấu hình:

- **image**: `ghcr.io/minhmarino/irodori-tts-worker:1.7b-dual`
- **type**: QUEUE
- **GPU pools**: `ADA_24`, `AMPERE_24` (RTX 4090 / 3090 class)
- **workersMin**: 0, **workersMax**: 2
- **containerDisk**: 50 GB (tải 2 model 1.7B lần đầu)
- **idleTimeout**: 60s
- **env**:
  - `LOAD_MODELS=both`
  - `CUSTOM_VOICE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `VOICE_DESIGN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

## Test

```bash
export ENDPOINT_ID=...
export RUNPOD_API_KEY=...

# CustomVoice
curl -sS -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "custom_voice",
      "text": "Hello from Irodori on RunPod.",
      "language": "English",
      "speaker": "Ryan"
    }
  }' | tee /tmp/cv.json

# VoiceDesign
curl -sS -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "voice_design",
      "text": "哥哥，你回来啦，人家等了你好久好久了！",
      "language": "Chinese",
      "instruct": "Cute playful young female voice, high pitch, affectionate tone."
    }
  }' | tee /tmp/vd.json
```

Giải mã WAV từ `output.audio_base64`:

```bash
python3 - <<'PY'
import json, base64, pathlib
data = json.load(open("/tmp/cv.json"))
b64 = data["output"]["audio_base64"]
pathlib.Path("out.wav").write_bytes(base64.b64decode(b64))
print("wrote out.wav")
PY
```
