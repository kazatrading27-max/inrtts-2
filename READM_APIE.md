# IndexTTS2 API 文档

基于 FastAPI 的 IndexTTS2 零样本语音合成 API，支持声音克隆、情感控制、音色缓存和推理参数调优。

## 快速开始

```bash
# 安装依赖
uv sync --all-extras

# 启动服务器（首次启动自动下载模型）
uv run python api_server.py --host 0.0.0.0 --port 8002 --fp16
```

启动后：
- WebUI：`http://localhost:8002/`
- Swagger 文档：`http://localhost:8002/docs`
- 健康检查：`http://localhost:8002/health`

## 接口列表

| 接口 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/audio/speech` | POST | 语音合成 |
| `/v1/audio/voices` | POST | 上传并注册音色 |
| `/v1/audio/voices` | GET | 获取所有已注册音色 |
| `/v1/audio/voices/{voice_id}` | DELETE | 删除指定音色 |
| `/ws` | WebSocket | 流式合成 |

## 接口详情

### 健康检查

```
GET /health
```

返回模型加载状态和设备信息。

### 语音合成

```
POST /v1/audio/speech
```

兼容 OpenAI Speech API 规范的语音合成接口。

**请求体（JSON）**：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `input` | string | **是** | - | 要合成的文本 |
| `voice` | string | 否 | `examples/voice_01.wav` | voice_id 或音频文件路径 |
| `response_format` | string | 否 | `wav` | 输出格式：`wav`、`mp3`、`pcm` |

**情感参数（可选）**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `emo_audio_prompt` | string | - | 情感参考音频路径 |
| `emo_alpha` | float | 1.0 | 情感强度（0.0-1.0），文本情感模式建议 0.6 |
| `emo_vector` | array | - | 8 维情感向量 `[高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]` |
| `use_emo_text` | bool | false | 根据文本自动生成情感向量 |
| `emo_text` | string | - | 独立情感文本描述（需 `use_emo_text=true`） |
| `use_random` | bool | false | 随机情感采样（会降低克隆保真度） |

**推理参数（可选）**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_beams` | int | 3 | beam search 宽度，1=贪心 |
| `do_sample` | bool | true | 是否采样 |
| `top_k` | int | 30 | top-k 采样范围 |
| `top_p` | float | 0.8 | nucleus 采样阈值 |
| `temperature` | float | 0.8 | 采样温度 |
| `max_mel_tokens` | int | 1500 | 最大生成长度 |
| `length_penalty` | float | 0.0 | 长度惩罚 |
| `repetition_penalty` | float | 10.0 | 重复惩罚 |

**返回**：音频二进制流（Content-Type 由 `response_format` 决定）

### 上传音色

```
POST /v1/audio/voices
```

上传音频文件并注册音色，返回 `voice_id` 用于后续合成。上传后自动预热缓存。

**请求（multipart/form-data）**：

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `audio` | file | 是 | 音色参考音频（3-10秒最佳），支持 mp3/wav/m4a/flac/ogg/aac |
| `speaker_name` | string | 否 | 音色名称 |

**返回**：

```json
{
  "voice_id": "spk_abc12345",
  "md5": "abc12345...",
  "status": "new",
  "message": "音色注册成功，可使用 /v1/audio/speech 的 voice 参数合成语音"
}
```

### 获取音色列表

```
GET /v1/audio/voices
```

返回所有已注册的音色。

### 删除音色

```
DELETE /v1/audio/voices/{voice_id}
```

删除指定音色及其音频文件。

### WebSocket 流式合成

```
WS /ws
```

发送 JSON 消息进行合成，参数同 `/v1/audio/speech`。

**发送**：
```json
{"type": "tts", "text": "你好", "voice": "spk_abc12345"}
```

**接收**：
```json
{"type": "completed", "audio_base64": "...", "sample_rate": 22050}
```

支持的消息类型：`tts`、`tts_stream`、`ping`、`get_voices`

## 使用示例

### 基础合成

```bash
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，这是一个测试"}' \
  -o test.wav
```

### 注册音色后合成

```bash
# 1. 上传音色
curl -X POST http://localhost:8002/v1/audio/voices \
  -F "audio=@examples/voice_01.wav" \
  -F "speaker_name=我的音色"

# 返回 {"voice_id": "spk_xxxxxxxx", ...}

# 2. 用 voice_id 合成
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好世界","voice":"spk_xxxxxxxx"}' \
  -o test.wav
```

### 情感控制

```bash
# 情感向量（悲伤）
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"对不起嘛！","voice":"spk_xxxxxxxx","emo_vector":[0,0,0.8,0,0,0,0,0]}' \
  -o test.wav

# 文本情感识别
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"快躲起来！是他要来了！","voice":"spk_xxxxxxxx","use_emo_text":true,"emo_alpha":0.6}' \
  -o test.wav

# 情感文本分离
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"快躲起来！","voice":"spk_xxxxxxxx","use_emo_text":true,"emo_text":"你吓死我了！","emo_alpha":0.6}' \
  -o test.wav
```

### 输出格式

```bash
# MP3 格式
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好","voice":"spk_xxxxxxxx","response_format":"mp3"}' \
  -o test.mp3

# PCM 格式
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好","voice":"spk_xxxxxxxx","response_format":"pcm"}' \
  -o test.pcm
```

### 推理参数调优

```bash
# 速度优先：贪心搜索
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好","voice":"spk_xxxxxxxx","num_beams":1,"do_sample":false,"top_k":10}' \
  -o test.wav
```

## 错误码

| 状态码 | 场景 |
|--------|------|
| 400 | 参数错误（缺少 input、格式不支持、音色不存在等） |
| 404 | 删除不存在的音色 |
| 500 | 模型推理失败 |
| 503 | 模型未初始化 |
