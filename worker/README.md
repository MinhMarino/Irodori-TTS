# Irodori TTS RunPod Worker

Serverless worker for **Qwen3-TTS 12Hz 1.7B** with two modes.

Cold-start notes: attach both models as Runpod **cached models**, use FlashBoot, keep `PRELOAD_MODELS=none` (lazy GPU load). See [`scripts/create-runpod-endpoint.md`](../scripts/create-runpod-endpoint.md).

## Modes

| `mode` | Model | Required fields |
|---|---|---|
| `custom_voice` | CustomVoice | `text`, `language`, `speaker` (+ optional `instruct`) |
| `voice_design` | VoiceDesign | `text`, `language`, `instruct` |

## Request example

### CustomVoice

```json
{
  "input": {
    "mode": "custom_voice",
    "text": "Hello from Irodori.",
    "language": "English",
    "speaker": "Ryan",
    "instruct": "Speak cheerfully."
  }
}
```

### VoiceDesign

```json
{
  "input": {
    "mode": "voice_design",
    "text": "哥哥，你回来啦！",
    "language": "Chinese",
    "instruct": "Cute playful young female voice, high pitch, affectionate tone."
  }
}
```

## Response

```json
{
  "mode": "custom_voice",
  "language": "English",
  "speaker": "Ryan",
  "sample_rate": 24000,
  "format": "wav",
  "audio_base64": "UklGRi..."
}
```

## Speakers (CustomVoice)

Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee

## Call API

```bash
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"mode":"custom_voice","text":"Hi","language":"English","speaker":"Ryan"}}'
```
