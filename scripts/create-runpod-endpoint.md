# RunPod Endpoint — cold-start optimized

Image: `ghcr.io/minhmarino/irodori-tts-worker:1.7b-dual`

## Chiến lược chống cold start

| Layer | Setting | Effect |
|---|---|---|
| 1. Model cache | Attach **2** HF models trên endpoint | Worker start trên host đã có weights → bỏ bước tải GB |
| 2. FlashBoot | `PRIORITY_FLASHBOOT` | Giữ container state, boot lại nhanh |
| 3. Lazy load | `PRELOAD_MODELS=none` | Chỉ load model của `mode` đang gọi lên GPU |
| 4. Idle keep-alive | `idleTimeout=300` | Worker ấm 5 phút giữa các request |
| 5. Active worker | `workersMin=1` | **Loại bỏ** cold start (trả phí worker standby) |

Khuyến nghị production latency-sensitive: bật **cả 1–5**.  
Tiết kiệm chi phí: `workersMin=0` + giữ 1–4 (request đầu sau idle vẫn có cold start nhẹ: load GPU).

## Checklist trước deploy

1. Nạp credit: https://console.runpod.io/user/billing  
2. Public GHCR package: https://github.com/users/MinhMarino/packages/container/package/irodori-tts-worker  

## Model references (bắt buộc cho cache)

Trong console **New/Edit Endpoint → Model**, thêm cả hai:

```
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

Hoặc CLI:

```bash
runpodctl serverless create \
  --name irodori-tts-1.7b-dual \
  --image ghcr.io/minhmarino/irodori-tts-worker:1.7b-dual \
  --gpu-id ADA_24 \
  --gpu-id AMPERE_24 \
  --gpu-id AMPERE_48 \
  --workers-min 1 \
  --workers-max 3 \
  --idle-timeout 300 \
  --flash-boot true \
  --execution-timeout 300 \
  --model-reference https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice:main \
  --model-reference https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign:main \
  --env PRELOAD_MODELS=none \
  --env RUNPOD_INIT_TIMEOUT=800
```

## Cấu hình tối ưu (MCP / API v2)

- **image**: `ghcr.io/minhmarino/irodori-tts-worker:1.7b-dual`
- **type**: QUEUE
- **gpuPoolIds**: `ADA_24`, `AMPERE_24`, `AMPERE_48`
- **workersMin**: `1` (zero cold start) hoặc `0` (rẻ hơn)
- **workersMax**: `3`
- **idleTimeout**: `300`
- **flashboot**: `PRIORITY_FLASHBOOT`
- **containerDiskInGb**: `20` (không cần lớn nếu dùng model cache)
- **executionTimeoutMs**: `300000`
- **scalerValue**: `2`
- **env**:
  - `PRELOAD_MODELS=none`
  - `CUSTOM_VOICE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `VOICE_DESIGN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
  - `HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub`
  - `RUNPOD_INIT_TIMEOUT=800`

> MCP `create-endpoint` hiện chưa expose `modelReferences` — sau khi tạo endpoint, vào console thêm 2 model URLs (hoặc dùng `runpodctl` như trên).

## Tradeoff chi phí

- `workersMin=1` trên RTX 4090-class: worker standby luôn chạy (giá standby thấp hơn active inference, vẫn > $0 khi không traffic).
- `workersMin=0` + FlashBoot + model cache: request đầu sau scale-to-zero ~ vài–chục giây (load GPU), không còn tải model từ internet.

## Test

```bash
export ENDPOINT_ID=...
export RUNPOD_API_KEY=...

curl -sS -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "custom_voice",
      "text": "Hello from warm Irodori.",
      "language": "English",
      "speaker": "Ryan"
    }
  }'
```
