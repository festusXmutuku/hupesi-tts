# Hupesi TTS - API Guide & AI Agent Integration

> **App version**: `1.0.6` (OpenAPI uses the same app version)  
> **Default address**: `http://127.0.0.1:8001` (or `http://localhost:8001`)  
> **Purpose**: Complete, accurate API reference with examples so **AI agents** (Cursor, Claude, ChatGPT, LangChain, AutoGen, CrewAI, Open WebUI, Dify, etc.) or developers can integrate immediately.

---

## 1. System Overview

**Hupesi TTS** is an offline, on-device AI text-to-speech platform for high-quality Vietnamese and multilingual speech:
- **Text-to-Speech (TTS)** via WAV streaming. Latency depends on the model and machine; Supertonic and Kokoro return audio by segment.
- **Sample rate**: 48 kHz (VieNeu-TTS v3 Turbo), 24 kHz (VieNeu-TTS v3 Nano, Kokoro), or 44.1 kHz (Supertonic 3). For `/stream`, read the WAV header or `GET /health`; `/stream` does not return `X-Sample-Rate`. File-generation endpoints return `X-Sample-Rate`, and subtitles endpoints include `sample_rate` in JSON. Do not hardcode sample rates.
- **Foreign-language models**: **Supertonic 3** reads 31 languages (English, Japanese, Korean, French, German, etc.) with 10 voices `F1`-`F5`, `M1`-`M5`. **Kokoro** reads 8 language groups (English, Spanish, French, Italian, Portuguese, Hindi; plus Japanese and Chinese when language packs are added) with native voices and **does not read Vietnamese**. Choose the language via the `language` parameter (section 3).
- **Emotion tags**: `[laugh]`, `[throat-clear]`, `[sigh]` (model-dependent).
- **Saved cloned voices**: in the app, record with a microphone or import an MP3 of your own/permitted voice and confirm the rights commitment. Then call the API using `voice_id`. There is no public clone-from-file API; voices and commitments are stored locally only.
- **Multi-speaker conversation & podcast**: automatically stitches dialogue turns between different characters.
- **Pause tags `[pause 2s]` / `[nghỉ 2s]`**: insert precise silence, supported by all models.
- **Pronunciation dictionary**: user-defined readings for abbreviations/proper nouns in the app are applied automatically to text-reading APIs.
- **SRT subtitles**: generate audio with per-sentence timestamps for video editing.
- **CORS**: enabled for cross-origin browser calls. Some timing/sample-rate headers may not be exposed via CORS; cross-origin JavaScript should read sample rate from `/health` or the WAV header.
- **Remote access (LAN)**: by default only the local machine can call the API. When enabled in Settings, other machines can access the APIs in section 4 (this modified build also allows the full UI remotely when enabled).

---

## 2. Available Voices (Studio Presets)

Always get the exact list from `GET /voices` (including user-saved cloned voices). `voice_id` is the **voice name**.

### VieNeu-TTS v3 Turbo (default, 48 kHz) - 25 voices

| Voice ID / Name | Gender | Region | Style |
| :--- | :--- | :--- | :--- |
| `Hải Đăng` *(default)* | Male | North | Natural |
| `Adam bựa` | Male | North | Natural |
| `Trúc Ly` | Female | North | Natural |
| `Tshow Minh` | Male | North | Storytelling |
| `Mai Anh` | Female | North | News |
| `Thùy Dung` | Female | South | News |
| `Thiền Tâm Đức` | Male | North | Storytelling |
| `Ngọc Huyền` | Female | North | Natural |
| `Quang Sơn` | Male | Central | Natural |
| `Ngọc Trân` | Female | Central | Natural |
| `Minh Đức` | Male | North | News |
| `Phạm Tuyên` | Male | North | Natural |
| `Thái Sơn` | Male | South | Storytelling |
| `Xuân Vĩnh` | Male | North | Natural |
| `Thanh Bình` | Male | North | Storytelling |
| `Ngọc Linh` | Female | North | Storytelling |
| `Đoan Trang` | Female | North | Natural |
| `Thục Đoan` | Female | South | Storytelling |
| `Minh Triết` | Male | South | News |
| `Mỹ Duyên` | Female | South | Audiobook |
| `Quỳnh Anh` | Female | North | Audiobook |
| `Đức Trí` | Male | South | Audiobook |
| `Kim Thanh` | Female | South | Audiobook |
| `Adam` | Male | South | Natural |
| `Quốc Tuấn` | Male | North | Natural |

Legacy names are still accepted and mapped to new voices: `Minh Quân`, `Minh Quân Pro` → `Hải Đăng`; `Anh Khôi` → `Tshow Minh`; `Mạnh Dũng` → `Quốc Tuấn`.

### VieNeu-TTS v3 Nano (24 kHz) - 11 voices

`Minh Quân` *(default)*, `Anh Khôi`, `Mạnh Dũng`, `Hữu Quân`, `Đức Trí`, `Adam` (male) · `Trúc Ly`, `Mai Anh`, `Xuân Tiên`, `Ái Hân`, `Mỹ Duyên` (female).

### Supertonic 3 (44.1 kHz, foreign voices) - 10 voices

`F1` *(default)*, `F2`, `F3`, `F4`, `F5` (female) · `M1`, `M2`, `M3`, `M4`, `M5` (male). Any voice can read any language supported by the model. This model **does not support voice cloning**.

### Kokoro-82M (24 kHz, foreign voices)

Downloadable model for **non-Vietnamese** text: English (US/UK), Spanish, French, Italian, Portuguese, Hindi, and Japanese/Chinese when language packs are added. Each voice is trained for a specific language; see each voice's `language` field in `GET /voices`. Default voice is `Heart` (US English). `voice_id` accepts display names (e.g. `Heart`, `Siwis (FR)`) or Kokoro's native IDs. This model **does not support voice cloning** and does not support emotion tags. Vietnamese text is rejected with `400`.

If `voice_id` is omitted, the server uses the default voice of the currently running model (first in `GET /voices`). An invalid `voice_id` for the running model returns `400`.

---

## 3. Input Text Conventions

### Language (`language`) - multilingual models only

Text-reading endpoints (`/stream`, `/api/tts/generate`, `/api/tts/generate-with-subtitles`, `/api/tts/conversation`) accept optional `language` (string), e.g. `"en"`, `"ja"`, `"ko"`, `"fr"`, `"de"`, `"vi"`. Available codes for the running model are in `languages` from `GET /health` (empty for VieNeu).

- Omit: Supertonic auto-detects from text (Vietnamese diacritics → `vi`, Japanese script → `ja`, Korean → `ko`, otherwise `en`).
- Kokoro always reads in the voice's own language and ignores `language` (valid codes are still accepted).
- VieNeu ignores this parameter, so clients can always send it safely.
- Unsupported code for the running model → `400`.
- For Vietnamese, use VieNeu for best quality.

### Emotion tags

Supported on **VieNeu-TTS-v3-Turbo**. Supertonic 3 uses `<laugh>`, `<breath>`, `<sigh>` and also accepts `[laugh]`, `[sigh]`, `[lấy hơi]` while ignoring `[throat-clear]`. Kokoro does not support emotion tags and ignores them. Emotion tags are not shown in SRT subtitles.

### Pause tags `[pause 2s]` / `[nghỉ 2s]`

Supported by **all models** and text-reading endpoints. The server splits text at tags, synthesizes segments, and inserts exact silence; tags are never spoken.

- Examples: `[nghỉ 2s]`, `[nghỉ 1.5s]`, `[nghỉ 500ms]`, `[pause 2s]`, `[nrecord 2]` (no unit = seconds), `[nghỉ]` = 1 second.
- Max **10 seconds** per tag. `speed` affects speech, not pauses.
- Text containing only pause tags and no words → `400`.

### Pronunciation dictionary

Users define readings for abbreviations/proper nouns in the app (e.g. `KTX` → "ký túc xá"). The server applies them automatically to `/stream`, `/api/tts/generate`, `/api/tts/generate-with-subtitles`, `/api/tts/conversation`. Applied for models that read Vietnamese (including Supertonic); not applied for Kokoro.

### AI-generated content labeling

Exported audio includes an "AI-generated" note in metadata. Do not remove it. When publishing publicly, clearly state that the voice was generated by AI.

---

## 4. API Endpoints

### 4.0. Authentication (when calling from another machine)

- **Same machine** as the app (`127.0.0.1` / `localhost`): **no key required**.
- **Another machine on LAN**: in the app, enable remote access in **Settings**, restart if required, then send the key via `X-API-Key: <key>` or `Authorization: Bearer <key>`. For `<audio src="...">` (no custom headers), add `&api_key=<key>` to the `GET /stream` URL.
- Requests through proxies/tunnels (ngrOK, Cloudflare Tunnel, nginx, etc.) are treated as remote and require the key.
- Keys are sent over unencrypted HTTP: use only on trusted networks. For Internet exposure, use an HTTPS tunnel instead of opening a router port.

```bash
curl.exe -H "X-API-Key: cit_xxxxxxxx" http://192.168.1.5:8001/health
```

This document lists APIs intended for external software to generate speech. Other APIs (history, model management, settings, dictionary, cloning, file reading, etc.) are internal to Hupesi TTS and may change between versions.

### 4.1. Health Check - `GET /health`
Check connectivity, model status, device (CPU/GPU), and sample rate.

- **Response `200 OK` (JSON)** example:
```json
{
  "status": "OK",
  "engine": "VieNeu-TTS-v3-Turbo (int8)",
  "sample_rate": 48000,
  "precision": "int8",
  "device": "cpu",
  "backend": "onnx",
  "message": "Ready",
  "languages": [],
  "supports_cloning": true
}
```

### 4.2. List Voices - `GET /voices`
Returns all available voices (including user-saved cloned voices).

### 4.3. Real-time Streaming TTS - `GET /stream` or `POST /stream`
Best for chatbots, assistants, and interactive apps. Returns chunked **WAV** audio (44-byte header + PCM16 chunks).

**GET**: `/stream?text=Hello&voice_id=Minh+Đức`  
**POST** JSON:
```json
{
  "text": "Hello everyone, this is real-time streaming audio.",
  "voice_id": "Minh Đức"
}
```

### 4.4. Generate Complete Audio - `POST /api/tts/generate`
Generate a complete audio file with advanced options (speed, pauses, etc.). Returns audio (WAV/MP3 depending on request/options).

### 4.5. Generate Audio with Subtitles - `POST /api/tts/generate-with-subtitles`
Generate audio and SRT subtitles with per-sentence timestamps.

### 4.6. Multi-speaker Conversation - `POST /api/tts/conversation`
Generate a conversation/podcast from multiple speaker turns.

---

For the exact, up-to-date request/response schemas, use:
- `GET /docs` (interactive Swagger UI)
- `GET /openapi.json` (OpenAPI spec)
- `GET /hupesi-tts.md` (this document, if served by the app)
