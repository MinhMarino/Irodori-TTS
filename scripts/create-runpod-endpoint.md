# RunPod Endpoint — cấu trúc tối ưu latency

**Live endpoint ID:** `3gq6tivo3ms4ls`  
Console: https://console.runpod.io/serverless/user/endpoint/3gq6tivo3ms4ls  
Image: `ghcr.io/minhmarino/irodori-tts-worker:1.7b-dual`

## Vì sao chậm (trước khi tối ưu)

| Nguyên nhân | Ảnh hưởng |
|---|---|
| `workersMin=0` + `idleTimeout=5` | Mỗi request sau vài giây = cold start đầy đủ |
| Chưa gắn **Cached model** | Worker tải weights từ Hub (~phút) |
| `PRELOAD_MODELS=none` | Request đầu còn phải load GPU |
| GPU pool hẹp / host throttled | Job kẹt `IN_QUEUE` dù đã có worker |
| 2 model trên 1 endpoint | RunPod **chỉ cache 1 model**/endpoint |

## Cấu trúc khuyến nghị (latency)

```text
Client
  → RunPod /run + poll
      → Worker ấm (workersMin=1)
          → CustomVoice đã preload + warmup
          → Cache HF: Qwen3-TTS-12Hz-1.7B-CustomVoice
          → VoiceDesign: lazy (chậm hơn nếu gọi lần đầu)
```

### Profile đang áp dụng

| Setting | Giá trị | Mục đích |
|---|---|---|
| `workersMin` | `1` | Bỏ cold start scale-to-zero |
| `workersMax` | `2` | Spike nhẹ |
| `idleTimeout` | `300` | Giữ worker sau request |
| `flashboot` | `PRIORITY_FLASHBOOT` | Revive nhanh nếu scale |
| `PRELOAD_MODELS` | `custom_voice` | Ready inference khi worker Ready |
| `WARMUP_ON_LOAD` | `1` | Chạy 1 generate ngắn sau load |
| Cached model (console) | **CustomVoice only** | Cache nhanh nhất (1 model/endpoint) |
| GPU pools | AMPERE_24, ADA_24, AMPERE_48… | Giảm throttle |

### Tradeoff chi phí

- `workersMin=1` = trả tiền standby GPU (nhanh, đắt hơn scale-to-zero).
- Muốn rẻ lại: `workersMin=0`, `idleTimeout=120`, vẫn giữ cached model + FlashBoot.

## BẮT BUỘC trên Console — Cached model

MCP/API hiện **không** set được `modelReferences`. Làm tay:

1. Mở https://console.runpod.io/serverless/user/endpoint/3gq6tivo3ms4ls  
2. **Manage → Edit Endpoint → Model**  
3. Thêm đúng **một** model (khuyến nghị primary):

```text
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

hoặc URL:

```text
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

4. Save.

> VoiceDesign không cache trên cùng endpoint. Lần đầu gọi `voice_design` có thể chậm (download/load model 2). Nếu cần VoiceDesign nhanh: tạo endpoint thứ 2 chỉ cache VoiceDesign.

## Checklist deploy image mới

1. Push `worker/**` → GitHub Actions build `ghcr.io/minhmarino/irodori-tts-worker:1.7b-dual`
2. Package GHCR public (nếu pull lỗi)
3. Endpoint env:

```env
PRELOAD_MODELS=custom_voice
WARMUP_ON_LOAD=1
STRICT_LOCAL_CACHE=0
CUSTOM_VOICE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
VOICE_DESIGN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub
RUNPOD_INIT_TIMEOUT=800
```

4. Gắn Cached model CustomVoice (bước trên)
5. Test 2 lần liên tiếp: lần 2 phải `warm: true`, `timings_ms.infer` vài giây, `delayTime` thấp

## Kỳ vọng latency

| Tình huống | Kỳ vọng |
|---|---|
| Worker ấm + model preload + cache | **~3–15s** end-to-end (chủ yếu infer) |
| Cold scale từ 0, có cache | ~20–60s (boot + load GPU) |
| Không cache, tải Hub | **1–3+ phút** (tránh) |
| Host throttled | Queue lâu — thêm GPU pool / đổi region |

Response worker có thêm:

```json
{
  "warm": true,
  "cache_hit": true,
  "timings_ms": {
    "model_load": 0,
    "infer": 7900,
    "handler_total": 8100
  }
}
```

## Test

```bash
export ENDPOINT_ID=3gq6tivo3ms4ls
export RUNPOD_API_KEY=...

curl -sS -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
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
