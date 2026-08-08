# Irodori TTS

UI thân thiện cho **Qwen3-TTS 12Hz 0.6B CustomVoice** — chọn ngôn ngữ + nhân vật giọng, chạy bằng Docker.

## Tính năng

- 10 ngôn ngữ: Trung, Anh, Nhật, Hàn, Đức, Pháp, Nga, Bồ Đào Nha, Tây Ban Nha, Ý (+ Auto)
- 9 nhân vật giọng: Vivian, Serena, Uncle Fu, Dylan, Eric, Ryan, Aiden, Ono Anna, Sohee
- Phong cách / cảm xúc tuỳ chọn (instruct)
- Phát & tải file WAV ngay trên trình duyệt

## Chạy nhanh (Docker)

Máy bạn là **Apple M2** → Docker chạy **CPU** (lần đầu tải model ~2GB, tổng hợp sẽ chậm hơn GPU).

```bash
cd /Volumes/128GB-SSD/MyProject/Irodori-TTS
docker compose up --build
```

Mở: [http://localhost:8000](http://localhost:8000)

Lần đầu container sẽ tải weights từ Hugging Face vào volume `hf-cache` — đợi status **Sẵn sàng**.

### Token Hugging Face (tuỳ chọn)

Nếu bị rate-limit khi tải model:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_xxx
docker compose up --build
```

### Máy NVIDIA (GPU)

Trong `docker-compose.yml`, bỏ comment khối `deploy.resources...gpu`, đổi image base sang CUDA nếu cần, và:

```bash
TTS_DEVICE=cuda docker compose up --build
```

## Dùng UI

1. Chọn **nhân vật** (card bên trái)
2. Chọn **ngôn ngữ** (hoặc Auto)
3. Nhập văn bản → **Tạo giọng nói**
4. Nghe / tải WAV

Gợi ý: dùng ngôn ngữ bản địa của nhân vật cho chất lượng tốt nhất (VD: Ono Anna → Japanese, Ryan → English).

## API

```bash
# Health
curl http://localhost:8000/api/health

# Tổng hợp
curl -X POST http://localhost:8000/api/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Irodori","language":"English","speaker":"Ryan"}' \
  --output out.wav
```

## Ghi chú hiệu năng

| Môi trường | Ghi chú |
|---|---|
| Apple Silicon + Docker | CPU only, ổn để thử, chậm với câu dài |
| NVIDIA GPU | Nên dùng `TTS_DEVICE=cuda` + flash-attn |
| RAM khuyến nghị | ≥ 8GB trống khi load 0.6B |

Model (local Docker UI): [`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)

## RunPod Serverless (1.7B CustomVoice + VoiceDesign)

Worker code: [`worker/`](worker/). Image builds via GitHub Actions to:

`ghcr.io/minhmarino/irodori-tts-worker:1.7b-dual`

See [worker/README.md](worker/README.md) for API payload examples.
