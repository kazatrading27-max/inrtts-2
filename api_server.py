"""
IndexTTS2 API Server
Supports voice cloning, emotion control, voice caching, and inference parameter tuning.
Root route mounts the WebUI, /v1/ routes provide the API.

DeepSpeed Support:
  DeepSpeed accelerates GPT autoregressive inference but imposes constraints:
    1. Beam search (num_beams > 1) is NOT supported → auto-falls back to greedy decoding
    2. KV cache is pre-allocated for a fixed max sequence length (default 1024 tokens)
       → max_mel_tokens is auto-capped to fit
    3. Certain HF generation flags (length_penalty, early_stopping) are unsupported
       → automatically stripped out
  All normalisation is logged so users can see exactly what was adjusted.
  Quality presets ('max'/'balanced'/'fast') simplify parameter selection.
"""
import os
import sys
import io
import uuid
import time
import json
import base64
import hashlib
import asyncio
import traceback
import subprocess
import inspect
from typing import Optional, Literal, Any, Dict, List, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import librosa

from fastapi import FastAPI, Body, Response, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager

import uvicorn
import argparse
from loguru import logger

from indextts.infer_v2 import IndexTTS2

# Import custom WebUI and its TTS instance to avoid reloading models into VRAM
import webui_enhanced

# ── AudioX (joint SFX/music/background generation) — ADDITIVE, does not touch TTS ──
import importlib.util as _importlib_util

_AUDIOX_MODEL_ALIASES = {
    "AudioX": "HKUSTAudio/AudioX",
    "AudioX-MAF": "HKUSTAudio/AudioX-MAF",
    "AudioX-MAF-MMDiT": "HKUSTAudio/AudioX-MAF-MMDiT",
    "base": "HKUSTAudio/AudioX",
    "maf": "HKUSTAudio/AudioX-MAF",
    "mmdit": "HKUSTAudio/AudioX-MAF-MMDiT",
}


def _audiox_available() -> bool:
    try:
        return _importlib_util.find_spec("audiox") is not None
    except Exception:
        return False


# ============== Global Variables ==============
tts: Optional[IndexTTS2] = None
args: argparse.Namespace = None

audiox_model: Any = None
audiox_config: Optional[dict] = None
audiox_loaded_name: Optional[str] = None

SPEAKER_CACHE_DIR = "assets/speaker_cache"
SPEAKER_META_FILE = os.path.join(SPEAKER_CACHE_DIR, "meta.json")

_gpu_semaphore: Optional[asyncio.Semaphore] = None
_queue_slots: Optional[asyncio.Semaphore] = None
_meta_cache: Optional[dict] = None
_meta_mtime: Optional[float] = None
SUPPORTED_RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
MEDIA_TYPES = {
    "mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac",
    "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm",
}
OPENAPI_TAGS = [
    {"name": "Base", "description": "Health check, concurrency queue configuration, and running status."},
    {"name": "WebSocket", "description": "WebSocket long-connection synthesis instructions."},
    {"name": "Voice Management", "description": "Upload, query, and delete voices."},
    {"name": "Speech Synthesis", "description": "OpenAI-compatible speech synthesis interface."},
    {"name": "Plain Speech Synthesis", "description": "Plain JSON speech synthesis interface."},
    {"name": "Audio Generation", "description": "AudioX joint SFX/music/background generation."},
]
AUDIO_RESPONSES = {
    200: {"description": "Synthesis successful.", "content": {m: {} for m in MEDIA_TYPES.values()}},
    400: {"description": "Invalid request parameters."},
    429: {"description": "Inference queue is full or wait timed out."},
    500: {"description": "Synthesis failed or internal server error."},
}

DEEPSPEED_MAX_TOTAL_TOKENS = int(os.environ.get("INDEXTTS_DEEPSPEED_MAX_TOKENS", "1024"))
DEEPSPEED_SPECIAL_TOKEN_BUFFER = int(os.environ.get("INDEXTTS_DEEPSPEED_TOKEN_BUFFER", "24"))

_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "max": {
        "diffusion_steps": 40, "do_sample": True, "num_beams": 1, "top_k": 30, "top_p": 0.8,
        "temperature": 0.8, "max_mel_tokens": 1500, "inference_cfg_rate": 0.7,
        "max_text_tokens_per_segment": 120, "interval_silence": 200,
    },
    "balanced": {
        "diffusion_steps": 25, "do_sample": True, "num_beams": 1, "top_k": 30, "top_p": 0.8,
        "temperature": 0.8, "max_mel_tokens": 1500, "inference_cfg_rate": 0.7,
        "max_text_tokens_per_segment": 120, "interval_silence": 200,
    },
    "fast": {
        "diffusion_steps": 12, "do_sample": False, "num_beams": 1, "top_k": 10, "top_p": 0.8,
        "temperature": 0.8, "max_mel_tokens": 900, "inference_cfg_rate": 0.7,
        "max_text_tokens_per_segment": 120, "interval_silence": 200,
    },
}

def _default_args() -> argparse.Namespace:
    return argparse.Namespace(
        host="0.0.0.0", port=8002, model_dir="./checkpoints", device="auto", fp16=False,
        max_concurrency=1, queue_size=16, queue_timeout=120.0, deepspeed=False,
        accel=False, cuda_kernel=False, audiox_model="HKUSTAudio/AudioX-MAF", audiox_preload=False,
    )

def _is_deepspeed_enabled() -> bool:
    """Check if DeepSpeed inference is actually active."""
    global tts
    if tts is not None:
        if getattr(tts, "use_deepspeed", False): return True
        if getattr(tts, "deepspeed", False): return True
        if hasattr(tts, "gpt") and getattr(tts.gpt, "use_deepspeed", False): return True
    return args is not None and getattr(args, "deepspeed", False)


# ============== Audio Trimming Logic ==============

def _detect_silence_regions(y: np.ndarray, sr: int, window_ms: int = 25, rms_threshold: float = 0.005) -> List[Tuple[int, int]]:
    window_samples = max(1, int(sr * window_ms / 1000))
    step = max(1, window_samples // 2)
    regions, in_silence, silence_start, pos = [], False, 0, 0
    while pos + window_samples <= len(y):
        seg = y[pos: pos + window_samples]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        if rms < rms_threshold and not in_silence:
            silence_start, in_silence = pos, True
        elif rms >= rms_threshold and in_silence:
            regions.append((silence_start, pos))
            in_silence = False
        pos += step
    if in_silence: regions.append((silence_start, len(y)))
    return regions

def _rms_trim_edges(y: np.ndarray, sr: int, rms_threshold: float = 0.005, window_ms: int = 25, margin_ms: int = 80) -> np.ndarray:
    if len(y) == 0: return y
    window_samples = max(1, int(sr * window_ms / 1000))
    step = max(1, window_samples // 2)
    end_pos = len(y)
    while end_pos > window_samples:
        seg = y[end_pos - window_samples: end_pos]
        if float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) > rms_threshold: break
        end_pos -= step
    start_pos = 0
    while start_pos + window_samples < end_pos:
        seg = y[start_pos: start_pos + window_samples]
        if float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) > rms_threshold: break
        start_pos += step
    margin_samples = int(sr * margin_ms / 1000)
    start_pos, end_pos = max(0, start_pos - margin_samples), min(len(y), end_pos + margin_samples)
    return y[start_pos:end_pos] if end_pos > start_pos else y

def _remove_trailing_noise(y: np.ndarray, sr: int, min_gap_ms: int = 200, max_trailing_ms: int = 500, energy_ratio: float = 0.3, rms_threshold: float = 0.005) -> np.ndarray:
    if len(y) < int(sr * 0.3): return y
    regions = _detect_silence_regions(y, sr, rms_threshold=rms_threshold)
    if not regions: return y
    min_gap_samples, max_trailing_samples, very_short_samples = int(sr * min_gap_ms / 1000), int(sr * max_trailing_ms / 1000), int(sr * 0.1)
    last_gap = next(((s, e) for s, e in reversed(regions) if (e - s) >= min_gap_samples), None)
    if not last_gap: return y
    gap_start, gap_end = last_gap
    if gap_end >= len(y): return y
    trailing = y[gap_end:]
    trailing_len = len(trailing)

    def _cut() -> np.ndarray:
        fade_s = min(int(sr * 0.02), gap_start)
        result = y[:gap_start].copy()
        if fade_s > 0:
            fade = np.linspace(1.0, 0.0, fade_s).astype(result.dtype)
            result[-fade_s:] *= fade
        return result

    if trailing_len < very_short_samples: return _cut()
    if trailing_len < max_trailing_samples:
        main_speech = y[:gap_start]
        if len(main_speech) > 0:
            main_rms = float(np.sqrt(np.mean(main_speech.astype(np.float64) ** 2)))
            trail_rms = float(np.sqrt(np.mean(trailing.astype(np.float64) ** 2)))
            if main_rms > 0 and (trail_rms / main_rms) < energy_ratio: return _cut()
    return y

def _compress_internal_silence(y: np.ndarray, sr: int, max_silence_ms: int = 600, compress_to_ms: int = 200, rms_threshold: float = 0.005) -> np.ndarray:
    regions = _detect_silence_regions(y, sr, rms_threshold=rms_threshold)
    if not regions: return y
    max_s, compress_s, min_boundary = int(sr * max_silence_ms / 1000), int(sr * compress_to_ms / 1000), int(sr * 0.05)
    long_silences = [(s, e) for s, e in regions if s >= min_boundary and e <= len(y) - min_boundary and (e - s) > max_s]
    if not long_silences: return y
    pieces, prev_end = [], 0
    for sil_start, sil_end in long_silences:
        pieces.append(y[prev_end:sil_start])
        edge = min(compress_s // 2, (sil_end - sil_start) // 2)
        pieces.append(y[sil_start: sil_start + edge])
        fill = compress_s - edge * 2
        if fill > 0: pieces.append(np.zeros(fill, dtype=y.dtype))
        pieces.append(y[sil_end - edge: sil_end])
        prev_end = sil_end
    pieces.append(y[prev_end:])
    return np.concatenate(pieces)

def trim_audio_array(y: np.ndarray, sr: int, top_db: int = 30, margin_ms: int = 80, max_internal_silence_ms: int = 600, compress_silence_to_ms: int = 200) -> np.ndarray:
    if len(y) == 0: return y
    y_float = y.astype(np.float32) / 32768.0
    original_duration = len(y_float) / sr
    yt, _ = librosa.effects.trim(y_float, top_db=top_db)
    if len(yt) == 0: yt = y_float
    yt = _rms_trim_edges(yt, sr, margin_ms=margin_ms)
    yt = _remove_trailing_noise(yt, sr)
    yt = _compress_internal_silence(yt, sr, max_silence_ms=max_internal_silence_ms, compress_to_ms=compress_silence_to_ms)
    trimmed_duration = len(yt) / sr
    if len(yt) > int(sr * 0.1):
        removed = original_duration - trimmed_duration
        if removed > 0.05: logger.info(f"Trimmed audio: {original_duration:.2f}s -> {trimmed_duration:.2f}s (removed {removed:.2f}s)")
        return np.clip(yt * 32767.0, -32768, 32767).astype(np.int16)
    return y


# ============== /docs Request Body Models ==============

ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
QualityPreset = Literal["max", "balanced", "fast"]

class SpeechParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    voice: str = Field(..., description="Voice ID from POST /v1/audio/voices voice_id; can also be a server-accessible audio path.")
    response_format: ResponseFormat | None = Field(None, description="Return audio format. OpenAI protocol defaults to mp3, plain /tts defaults to wav.")
    speaker_id: str | None = Field(None, description="Compatible alias for voice.")
    spk_audio_prompt: str | None = Field(None, description="Server-accessible voice reference audio path.")
    spk_audio_base64: str | None = Field(None, description="Base64 encoded audio data for voice cloning.")
    emo_audio_prompt: str | None = Field(None, description="Server-accessible emotion reference audio path.")
    emo_audio_base64: str | None = Field(None, description="Base64 encoded emotion reference audio.")
    emo_alpha: float = Field(1.0, ge=0.0, le=1.0, description="Emotion control strength, 0-1.")
    emo_vector: list[float] | None = Field(None, min_length=8, max_length=8, description="8-dimensional emotion vector.")
    use_emo_text: bool = Field(False, description="Enable text emotion recognition.")
    emo_text: str | None = Field(None, description="Text used for emotion recognition.")
    use_random: bool = Field(False, description="Whether to randomly sample emotions.")
    interval_silence: int = Field(200, ge=0, le=1000, description="Silence inserted between segments in milliseconds.")
    max_text_tokens_per_segment: int = Field(120, ge=20, le=240, description="Maximum text tokens per segment.")
    quick_streaming_tokens: int = Field(60, ge=0, le=240, description="Streaming first packet target token count.")
    quality_preset: QualityPreset | None = Field(None, description="Quality preset: 'max', 'balanced', 'fast'. Only fills unspecified params.")
    num_beams: int = Field(3, ge=1, le=10, description="Search width. Note: beam search >1 is not supported under DeepSpeed.")
    do_sample: bool = Field(True, description="Enable sampling for natural variation. Set to False for deterministic, consistent output (greedy decoding).")
    top_k: int = Field(30, ge=1, le=100, description="Top-K sampling parameter.")
    top_p: float = Field(0.8, ge=0.0, le=1.0, description="Top-P sampling parameter.")
    temperature: float = Field(0.8, gt=0.0, le=2.0, description="Sampling temperature.")
    max_mel_tokens: int = Field(1500, ge=100, le=3000, description="Maximum generated mel token count. Auto-capped under DeepSpeed.")
    length_penalty: float = Field(0.0, ge=0.0, le=2.0, description="Length penalty. (Ignored under DeepSpeed)")
    repetition_penalty: float = Field(10.0, gt=0.0, le=20.0, description="Repetition penalty.")
    diffusion_steps: int = Field(25, ge=1, le=50, description="Diffusion vocoder steps. 'max'=40, 'balanced'=25, 'fast'=12.")
    inference_cfg_rate: float = Field(0.7, ge=0.0, le=2.0, description="Classifier-free guidance strength for vocoder.")
    trim_silence: bool = Field(True, description="Automatically trim leading/trailing silence and trailing noise.")

class OpenAISpeechRequest(SpeechParams):
    input: str = Field(..., min_length=1, description="Text to synthesize.")
    model: str | None = Field(None, description="OpenAI protocol field. Not validated.")

class PlainTTSRequest(SpeechParams):
    text: str = Field(..., min_length=1, description="Text to synthesize.")


# ============== Voice Cache Management ==============

def _load_meta() -> dict:
    global _meta_cache, _meta_mtime
    if os.path.exists(SPEAKER_META_FILE):
        mtime = os.path.getmtime(SPEAKER_META_FILE)
        if _meta_cache is not None and _meta_mtime == mtime: return _meta_cache
        with open(SPEAKER_META_FILE, "r", encoding="utf-8") as f: _meta_cache = json.load(f)
        _meta_mtime = mtime
        return _meta_cache
    _meta_cache, _meta_mtime = {}, None
    return _meta_cache

def _save_meta(meta: dict):
    global _meta_cache, _meta_mtime
    _meta_cache = meta
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
    with open(SPEAKER_META_FILE, "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, indent=2)
    _meta_mtime = os.path.getmtime(SPEAKER_META_FILE)

def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

async def _register_audio_from_base64(b64_data: str, speaker_name: str = "dynamic_base64") -> str:
    try: audio_bytes = base64.b64decode(b64_data)
    except Exception: raise ValueError("Invalid base64 audio data")
    if len(audio_bytes) > 20 * 1024 * 1024: raise ValueError("Base64 audio file too large (exceeds 20MB)")
    md5 = hashlib.md5(audio_bytes).hexdigest()
    voice_id = f"spk_{md5[:8]}"
    meta = _load_meta()
    if voice_id in meta:
        logger.info(f"Dynamic voice cache hit: {voice_id}")
        return meta[voice_id]["audio_path"]
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
    tmp_path = os.path.join(SPEAKER_CACHE_DIR, f"tmp_{uuid.uuid4().hex[:8]}.wav")
    try:
        wav_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        sf.write(tmp_path, wav_data, sr)
    except Exception as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise ValueError(f"Failed to decode audio bytes. Must be valid WAV/FLAC/OGG: {e}")
    final_path = os.path.join(SPEAKER_CACHE_DIR, f"{voice_id}.wav")
    os.rename(tmp_path, final_path)
    meta[voice_id] = {"voice_name": speaker_name or voice_id, "audio_path": final_path, "md5": md5, "original_filename": "inline_base64", "created_at": time.time(), "embedding_cached": False}
    _save_meta(meta)
    logger.info(f"Registered dynamic voice from base64: {voice_id}")
    return final_path


# ============== Quality Preset & DeepSpeed Normalisation ==============

def _apply_quality_preset(params: dict, preset: str) -> None:
    defaults = _QUALITY_PRESETS.get(preset.lower().strip())
    if defaults is None:
        logger.warning(f"Unknown quality preset '{preset}', ignoring.")
        return
    applied = [f"{k}={v}" for k, v in defaults.items() if k not in params]
    if applied: logger.info(f"Quality preset '{preset}' applied: {', '.join(applied)}")
    for k, v in defaults.items():
        if k not in params: params[k] = v

def _normalize_for_deepspeed(params: dict) -> None:
    changes = []
    user_beams = params.get("num_beams", 3)
    user_do_sample = params.get("do_sample", True)
    beams_was_specified = "num_beams" in params

    if user_beams > 1:
        # User requested beam search (deterministic quality) → DeepSpeed doesn't support it.
        # Fall back to greedy decoding (num_beams=1, do_sample=False) to preserve determinism!
        params["num_beams"] = 1
        params["do_sample"] = False
        changes.append(f"num_beams: {user_beams}→1, do_sample: forced False (DeepSpeed fallback to greedy decoding for deterministic output)")
    elif not beams_was_specified:
        # User didn't specify num_beams; IndexTTS2 would default to 3 which crashes DeepSpeed.
        params["num_beams"] = 1
        changes.append(f"num_beams: set to 1 (overriding IndexTTS2 default of 3 for DeepSpeed compatibility)")

    max_text_tokens = params.get("max_text_tokens_per_segment", 120)
    max_allowed_mel = max(100, DEEPSPEED_MAX_TOTAL_TOKENS - max_text_tokens - DEEPSPEED_SPECIAL_TOKEN_BUFFER)
    current_max_mel = params.get("max_mel_tokens", 1500)
    if current_max_mel > max_allowed_mel:
        params["max_mel_tokens"] = max_allowed_mel
        changes.append(f"max_mel_tokens: {current_max_mel}→{max_allowed_mel} (DeepSpeed KV cache limit)")

    if changes:
        logger.info("DeepSpeed parameter normalisation:")
        for c in changes: logger.info(f"  • {c}")


# ============== Inference Parameter Extraction ==============

def _extract_params(data: dict) -> dict:
    def as_bool(v): return v if isinstance(v, bool) else str(v).strip().lower() in {"1", "true", "yes", "on"}
    params = {}

    for key in ("emo_audio_prompt", "emo_alpha", "emo_vector", "use_emo_text", "emo_text", "use_random"):
        if key in data: params[key] = data[key]
    if "emo_alpha" not in params: params["emo_alpha"] = 1.0
    else: params["emo_alpha"] = float(params["emo_alpha"])
    if "emo_vector" in params:
        v = params["emo_vector"]
        if not isinstance(v, list) or len(v) != 8: raise ValueError("emo_vector must be an array of length 8")
        params["emo_vector"] = [float(x) for x in v]

    for key in ("use_emo_text", "use_random", "do_sample"):
        if key in params: params[key] = as_bool(params[key])
        elif key in data: params[key] = as_bool(data[key])

    for key in ("interval_silence", "max_text_tokens_per_segment", "num_beams", "top_k", "max_mel_tokens", "diffusion_steps"):
        if key in data and data[key] is not None: params[key] = int(data[key])

    for key in ("top_p", "temperature", "length_penalty", "repetition_penalty", "inference_cfg_rate"):
        if key in data and data[key] is not None: params[key] = float(data[key])

    if params.get("repetition_penalty", 10.0) <= 0: raise ValueError("repetition_penalty must be > 0")
    if params.get("temperature", 0.8) <= 0: raise ValueError("temperature must be > 0")
    if "top_p" in params and not 0 <= params["top_p"] <= 1: raise ValueError("top_p must be between 0 and 1")

    if data.get("quality_preset"): _apply_quality_preset(params, data["quality_preset"])

    do_sample = params.get("do_sample", True)
    num_beams = params.get("num_beams", 3)
    
    # Beam search implies determinism. Overriding do_sample=True to False if num_beams > 1.
    if num_beams > 1 and do_sample:
        logger.info(f"Beam search requested (num_beams={num_beams}). Overriding do_sample=True to False for deterministic output.")
        params["do_sample"] = False

    if _is_deepspeed_enabled(): _normalize_for_deepspeed(params)

    return params


def _encode_audio(wav: np.ndarray, sr: int, fmt: str) -> Tuple[bytes, str]:
    if fmt == "pcm": return wav.tobytes(), MEDIA_TYPES[fmt]
    if fmt not in SUPPORTED_RESPONSE_FORMATS: raise ValueError(f"Unsupported format: {fmt}")
    buf = io.BytesIO()
    if fmt in ("wav", "flac"):
        sf.write(buf, wav, sr, format=fmt.upper())
        return buf.getvalue(), MEDIA_TYPES[fmt]
    wav_buf = io.BytesIO()
    sf.write(wav_buf, wav, sr, format="WAV")
    codec_args = {"mp3": ["-f", "mp3", "-codec:a", "libmp3lame"], "opus": ["-f", "opus", "-codec:a", "libopus"], "aac": ["-f", "adts", "-codec:a", "aac"]}[fmt]
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *codec_args, "pipe:1"], input=wav_buf.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0: raise RuntimeError(f"ffmpeg transcoding failed: {proc.stderr.decode('utf-8', errors='ignore')}")
    return proc.stdout, MEDIA_TYPES[fmt]

def _resolve_speaker(data: dict) -> Optional[str]:
    speaker_id = data.get("speaker_id") or data.get("voice")
    if speaker_id:
        meta = _load_meta()
        if speaker_id in meta: return meta[speaker_id]["audio_path"]
        return speaker_id
    return data.get("spk_audio_prompt")

def _unique_output_path(prefix: str = "") -> str:
    name = f"{prefix}{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:12]
    return f"outputs/{name}.wav"


# ============== Synchronous Inference ==============

def _do_infer(text: str, spk_audio_prompt: str, output_path: str, params: dict):
    # Set the requested diffusion steps on the patched cfm object
    if hasattr(tts, 'cfm'):
        tts.cfm._api_diffusion_steps = params.get("diffusion_steps", 25)

    result = tts.infer(spk_audio_prompt=spk_audio_prompt, text=text, output_path=output_path, **params)
    if isinstance(result, tuple) and len(result) == 2: return result
    if isinstance(result, str) and os.path.exists(result):
        wav, sr = sf.read(result, dtype="int16"); return sr, wav
    if result is None and output_path and os.path.exists(output_path):
        wav, sr = sf.read(output_path, dtype="int16"); return sr, wav
    return result

def _wav_chunk_to_numpy(wav: Any) -> np.ndarray:
    if hasattr(wav, "detach"): wav = wav.detach().cpu()
    if hasattr(wav, "numpy"): wav = wav.numpy()
    arr = np.asarray(wav)
    if arr.dtype != np.int16: arr = np.clip(arr, -32767, 32767).astype(np.int16)
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[0] <= 8: arr = arr.T
    return np.ascontiguousarray(arr)

def _concat_wav_chunks(chunks: List[np.ndarray]) -> np.ndarray:
    valid_chunks = [c for c in chunks if c.size > 0]
    return np.ascontiguousarray(np.concatenate(valid_chunks, axis=0)) if valid_chunks else np.zeros((0, 1), dtype=np.int16)

_STREAM_DONE = object()

def _next_stream_item(generator):
    try: return next(generator)
    except StopIteration: return _STREAM_DONE

async def _acquire_with_timeout(semaphore: asyncio.Semaphore, timeout: float) -> bool:
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        return True
    except asyncio.TimeoutError: return False

async def _guarded_infer(text: str, spk: str, output_path: str, params: dict) -> dict:
    request_start = time.perf_counter()
    if _queue_slots is None or _gpu_semaphore is None: raise RuntimeError("Inference queue not initialized")
    accepted = await _acquire_with_timeout(_queue_slots, args.queue_timeout)
    if not accepted: raise TimeoutError(f"Inference queue is full (wait exceeded {args.queue_timeout:.1f}s)")
    try:
        await _gpu_semaphore.acquire()
        queue_elapsed = time.perf_counter() - request_start
        infer_start = time.perf_counter()
        try: result = await asyncio.to_thread(_do_infer, text, spk, output_path, params)
        finally: _gpu_semaphore.release()
        return {"result": result, "queue_time": queue_elapsed, "infer_time": time.perf_counter() - infer_start, "total_time": time.perf_counter() - request_start}
    finally: _queue_slots.release()

async def _guarded_stream_to_ws(ws: WebSocket, text: str, spk: str, output_path: str, params: dict, output_format: str, quick_streaming_tokens: int, trim_silence: bool = True):
    request_start = time.perf_counter()
    if _queue_slots is None or _gpu_semaphore is None: raise RuntimeError("Inference queue not initialized")
    accepted = await _acquire_with_timeout(_queue_slots, args.queue_timeout)
    if not accepted: raise TimeoutError(f"Inference queue is full (wait exceeded {args.queue_timeout:.1f}s)")
    chunk_count, chunk_bytes, chunks, gpu_acquired, generator = 0, 0, [], False, None
    try:
        await _gpu_semaphore.acquire()
        gpu_acquired = True
        queue_elapsed = time.perf_counter() - request_start
        infer_start = time.perf_counter()

        await ws.send_json({"type": "stream_started", "format": output_format, "media_type": MEDIA_TYPES[output_format], "sample_rate": 22050, "queue_time": round(queue_elapsed, 3), "quick_streaming_tokens": quick_streaming_tokens, "deepspeed": _is_deepspeed_enabled()})

        if hasattr(tts, 'cfm'): tts.cfm._api_diffusion_steps = params.get("diffusion_steps", 25)
        generator = tts.infer(spk_audio_prompt=spk, text=text, output_path=None, stream_return=True, more_segment_before=quick_streaming_tokens, **params)
        
        while True:
            item = await asyncio.to_thread(_next_stream_item, generator)
            if item is _STREAM_DONE: break
            chunk = _wav_chunk_to_numpy(item)
            if chunk.size == 0: continue
            chunks.append(chunk)
            chunk_count += 1
            audio_bytes, media_type = _encode_audio(chunk, 22050, output_format)
            chunk_bytes += len(audio_bytes)
            await ws.send_json({"type": "stream_chunk", "chunk_index": chunk_count, "audio_base64": base64.b64encode(audio_bytes).decode(), "format": output_format, "media_type": media_type, "sample_rate": 22050, "byte_count": len(audio_bytes), "queue_time": round(queue_elapsed, 3), "infer_time": round(time.perf_counter() - infer_start, 3), "total_time": round(time.perf_counter() - request_start, 3)})

        full_wav = _concat_wav_chunks(chunks)
        if trim_silence and full_wav.size > 0: full_wav = trim_audio_array(full_wav, 22050)
        if output_path and full_wav.size > 0: sf.write(output_path, full_wav, 22050, subtype="PCM_16")
        audio_bytes, media_type = _encode_audio(full_wav, 22050, output_format)
        await ws.send_json({"type": "stream_completed", "audio_base64": base64.b64encode(audio_bytes).decode(), "format": output_format, "media_type": media_type, "sample_rate": 22050, "chunk_count": chunk_count, "chunk_bytes": chunk_bytes, "byte_count": len(audio_bytes), "queue_time": round(queue_elapsed, 3), "infer_time": round(time.perf_counter() - infer_start, 3), "total_time": round(time.perf_counter() - request_start, 3)})
    finally:
        if generator is not None:
            try: generator.close()
            except Exception: pass
        if gpu_acquired: _gpu_semaphore.release()
        _queue_slots.release()


# ============== WebSocket Management ==============

class ConnectionManager:
    def __init__(self): self.active_connections: List[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept(); self.active_connections.append(ws)
        logger.info(f"WebSocket connection count: {len(self.active_connections)}")
    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections: self.active_connections.remove(ws)
    async def send_json(self, message: dict, ws: WebSocket):
        try: await ws.send_json(message)
        except Exception as e: logger.error(f"Failed to send message: {e}")

manager = ConnectionManager()


# ============== Application Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts, _gpu_semaphore, _queue_slots, args
    if args is None:
        args = _default_args()
        logger.warning("args not initialized, using defaults.")
    _gpu_semaphore = asyncio.Semaphore(args.max_concurrency)
    _queue_slots = asyncio.Semaphore(args.max_concurrency + args.queue_size)

    logger.info("Initializing IndexTTS2...")
    try:
        if hasattr(webui_enhanced, 'tts') and webui_enhanced.tts is not None:
            tts = webui_enhanced.tts
            ds_active = _is_deepspeed_enabled()
            if args.deepspeed and not ds_active:
                logger.warning("DeepSpeed requested but reused instance lacks it. Re-initializing model with DeepSpeed...")
                tts = None
            else:
                logger.info(f"IndexTTS2 initialized successfully (reusing instance, DeepSpeed: {'enabled' if ds_active else 'disabled'})")
        
        if tts is None:
            device = None if args.device == "auto" else args.device
            tts = IndexTTS2(cfg_path=os.path.join(args.model_dir, "config.yaml"), model_dir=args.model_dir, use_fp16=args.fp16, device=device, use_cuda_kernel=args.cuda_kernel, use_deepspeed=args.deepspeed, use_accel=args.accel, use_torch_compile=False)
            webui_enhanced.tts = tts
            ds_active = _is_deepspeed_enabled()
            logger.info(f"IndexTTS2 initialized successfully (DeepSpeed: {'enabled' if ds_active else 'disabled'})")
    except Exception as e:
        logger.warning(f"Model initialization failed: {e}")
        tts = None

    # Monkey-patch GPT generate to drop leaking vocoder kwargs and unsupported DeepSpeed flags
    if tts is not None and hasattr(tts, 'gpt') and hasattr(tts.gpt, 'inference_model') and hasattr(tts.gpt.inference_model, 'generate'):
        _orig_generate = tts.gpt.inference_model.generate
        if not getattr(_orig_generate, "_api_kwarg_filtered", False):
            def _safe_generate(*args, **kwargs):
                kwargs.pop("diffusion_steps", None)
                kwargs.pop("inference_cfg_rate", None)
                if _is_deepspeed_enabled():
                    kwargs.pop("length_penalty", None)
                    kwargs.pop("early_stopping", None)
                return _orig_generate(*args, **kwargs)
            _safe_generate._api_kwarg_filtered = True
            tts.gpt.inference_model.generate = _safe_generate
            logger.info("Patched GPT generate() to filter vocoder kwargs and unsupported DeepSpeed flags")

    # Monkey-patch Vocoder (cfm.inference) to correctly apply requested diffusion steps
    if tts is not None and hasattr(tts, 'cfm') and hasattr(tts.cfm, 'inference'):
        _orig_cfm_inference = tts.cfm.inference
        if not getattr(_orig_cfm_inference, "_api_step_overridden", False):
            _step_param_name = None
            try:
                sig = inspect.signature(_orig_cfm_inference)
                for name in sig.parameters:
                    if name in ('steps', 'N', 'num_steps', 'diffusion_steps', 'timesteps'):
                        _step_param_name = name; break
            except Exception: pass
            
            if _step_param_name:
                def _patched_cfm_inference(codes, *args, **kwargs):
                    desired_steps = getattr(tts.cfm, "_api_diffusion_steps", None)
                    if desired_steps is not None:
                        kwargs[_step_param_name] = desired_steps
                        if hasattr(tts.cfm, "_api_diffusion_steps"): del tts.cfm._api_diffusion_steps
                    return _orig_cfm_inference(codes, *args, **kwargs)
                _patched_cfm_inference._api_step_overridden = True
                tts.cfm.inference = _patched_cfm_inference
                logger.info(f"Patched cfm.inference() to dynamically override vocoder steps via '{_step_param_name}'")
            else:
                logger.warning("Could not find step parameter name in cfm.inference to override diffusion_steps")

    _load_meta()
    logger.info(f"Loaded {len(_meta_cache or {})} voices")

    if getattr(args, "audiox_preload", False) and _audiox_available():
        try: await asyncio.to_thread(_audiox_ensure_loaded, _resolve_audiox_model(args.audiox_model))
        except Exception as e: logger.warning(f"AudioX preload failed: {e}")

    yield
    logger.info("Shutting down...")

app = FastAPI(
    lifespan=lifespan, title="IndexTTS2 API", version="2.4",
    description=("IndexTTS2 zero-shot speech synthesis API. DeepSpeed support with automatic parameter normalisation. "
                 "Deterministic output enforced for beam search requests. Vocoder step bug patched for max quality output."),
    openapi_tags=OPENAPI_TAGS,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
logger.add("logs/api.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

@app.get("/health", tags=["Base"])
async def health_check():
    if tts is None: return {"status": "partial", "model_loaded": False}
    deepspeed_active = _is_deepspeed_enabled()
    max_text_tokens = 120
    ds_max_mel = max(100, DEEPSPEED_MAX_TOTAL_TOKENS - max_text_tokens - DEEPSPEED_SPECIAL_TOKEN_BUFFER)
    return {
        "status": "healthy", "model_loaded": True, "device": str(tts.device),
        "deepspeed": {"enabled": deepspeed_active, "max_mel_tokens_cap": ds_max_mel if deepspeed_active else None, "beam_search_supported": not deepspeed_active},
        "quality_presets": {"available": list(_QUALITY_PRESETS.keys())},
        "concurrency": {"max_concurrency": args.max_concurrency, "queue_size": args.queue_size, "queue_timeout": args.queue_timeout}
    }

@app.get("/api/v1/health", tags=["Base"], include_in_schema=False)
async def health_check_v1(): return await health_check()

@app.post("/v1/audio/voices", tags=["Voice Management"])
async def upload_speaker(audio: UploadFile = File(...), speaker_name: str = Form("")):
    if tts is None: return JSONResponse(status_code=503, content={"error": "Model not initialized"})
    try:
        if not audio.filename: return JSONResponse(status_code=400, content={"error": "Please provide an audio file"})
        ext = os.path.splitext(audio.filename)[-1].lower()
        if ext not in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}: return JSONResponse(status_code=400, content={"error": f"Unsupported format: {ext}"})
        audio_bytes = await audio.read()
        if len(audio_bytes) < 1024: return JSONResponse(status_code=400, content={"error": "Audio file is too small"})
        if len(audio_bytes) > 20 * 1024 * 1024: return JSONResponse(status_code=400, content={"error": "Audio file is too large"})
        os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
        tmp_path = os.path.join(SPEAKER_CACHE_DIR, f"tmp_{uuid.uuid4().hex[:8]}{ext}")
        with open(tmp_path, "wb") as f: f.write(audio_bytes)
        md5 = _md5_file(tmp_path)
        voice_id = f"spk_{md5[:8]}"
        meta = _load_meta()
        if voice_id in meta:
            os.remove(tmp_path)
            return {"voice_id": voice_id, "status": "exists"}
        final_path = os.path.join(SPEAKER_CACHE_DIR, f"{voice_id}{ext}")
        os.rename(tmp_path, final_path)
        try:
            warmup_path = _unique_output_path("warmup_")
            os.makedirs("outputs", exist_ok=True)
            await _guarded_infer("Test", final_path, warmup_path, {})
            if os.path.exists(warmup_path): os.remove(warmup_path)
            if voice_id in _load_meta():
                _meta_cache[voice_id]["embedding_cached"] = True
                _save_meta(_meta_cache)
        except Exception as warmup_err:
            logger.warning(f"Voice warmup failed: {warmup_err}")
        meta = _load_meta()
        meta[voice_id] = {"voice_name": speaker_name or voice_id, "audio_path": final_path, "md5": md5, "original_filename": audio.filename, "created_at": time.time(), "embedding_cached": True}
        _save_meta(meta)
        return {"voice_id": voice_id, "md5": md5, "status": "new"}
    except Exception as e:
        logger.error(f"Failed to upload voice: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/v1/audio/voices", tags=["Voice Management"])
async def list_speakers():
    meta = _load_meta()
    return {"object": "list", "data": [{"voice_id": vid, "name": i.get("voice_name", ""), "original_filename": i.get("original_filename", ""), "created_at": i.get("created_at")} for vid, i in meta.items()]}

@app.delete("/v1/audio/voices/{voice_id}", tags=["Voice Management"])
async def delete_speaker(voice_id: str):
    meta = _load_meta()
    if voice_id not in meta: return JSONResponse(status_code=404, content={"error": f"Voice {voice_id} does not exist"})
    info = meta.pop(voice_id)
    _save_meta(meta)
    if info.get("audio_path", "") and os.path.exists(info["audio_path"]): os.remove(info["audio_path"])
    return {"status": "deleted", "voice_id": voice_id}

async def _speech_response_from_payload(data: dict | BaseModel, *, text_field: str = "input", default_format: str = "wav") -> Response | JSONResponse:
    if tts is None: return JSONResponse(status_code=503, content={"error": "Model not initialized"})
    if isinstance(data, BaseModel): data = data.model_dump(exclude_none=True)
    text = data.get(text_field) or data.get("input") or data.get("text")
    if not text or not isinstance(text, str) or not text.strip(): return JSONResponse(status_code=400, content={"error": f"{text_field} is required"})
    
    if data.get("emo_audio_base64"):
        try: data["emo_audio_prompt"] = await _register_audio_from_base64(data["emo_audio_base64"], "emo_base64")
        except ValueError as e: return JSONResponse(status_code=400, content={"error": f"emo_audio_base64: {e}"})
    if data.get("emo_audio_prompt") and not os.path.exists(data["emo_audio_prompt"]): data.pop("emo_audio_prompt", None)

    try: params = _extract_params(data)
    except ValueError as e: return JSONResponse(status_code=400, content={"error": str(e)})

    voice_label = data.get("voice") or data.get("speaker_id") or data.get("spk_audio_prompt") or ""
    spk_audio_prompt = None
    if data.get("spk_audio_base64"):
        try: spk_audio_prompt = await _register_audio_from_base64(data["spk_audio_base64"], data.get("voice"))
        except ValueError as e: return JSONResponse(status_code=400, content={"error": str(e)})
    if not spk_audio_prompt: spk_audio_prompt = _resolve_speaker(data) or "examples/voice_01.wav"
    if not os.path.exists(spk_audio_prompt): return JSONResponse(status_code=400, content={"error": f"Voice file does not exist: {voice_label}"})

    output_format = data.get("response_format") or default_format
    if output_format not in SUPPORTED_RESPONSE_FORMATS: return JSONResponse(status_code=400, content={"error": f"Unsupported format: {output_format}"})

    try:
        output_path = _unique_output_path()
        os.makedirs("outputs", exist_ok=True)
        infer_record = await _guarded_infer(text.strip(), spk_audio_prompt, output_path, params)
        result = infer_record["result"]
        if result is None: return JSONResponse(status_code=500, content={"error": "Synthesis failed"})
        sr, wav = result
        if data.get("trim_silence", True): wav = trim_audio_array(wav, sr)
        logger.info(f"TTS completed: {len(text)} chars, queue={infer_record['queue_time']:.2f}s, infer={infer_record['infer_time']:.2f}s, format: {output_format}")
        audio_bytes, media_type = _encode_audio(wav, sr, output_format)
        return Response(content=audio_bytes, media_type=media_type, headers={"X-IndexTTS-Voice": str(voice_label), "X-IndexTTS-Sample-Rate": str(sr), "X-IndexTTS-Queue-Time": f"{infer_record['queue_time']:.3f}", "X-IndexTTS-Infer-Time": f"{infer_record['infer_time']:.3f}", "X-IndexTTS-Total-Time": f"{infer_record['total_time']:.3f}", "X-IndexTTS-Output-Format": output_format, "X-IndexTTS-DeepSpeed": str(_is_deepspeed_enabled())})
    except TimeoutError as e: return JSONResponse(status_code=429, content={"error": str(e)})
    except Exception as e:
        logger.error(f"TTS failed: {e}"); traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Synthesis failed: {str(e)}"})

@app.post("/v1/audio/speech", tags=["Speech Synthesis"], responses=AUDIO_RESPONSES)
async def openai_speech(request: OpenAISpeechRequest = Body(...)): return await _speech_response_from_payload(request, text_field="input", default_format="mp3")

@app.post("/tts", tags=["Plain Speech Synthesis"], responses=AUDIO_RESPONSES)
async def plain_tts(request: PlainTTSRequest = Body(...)): return await _speech_response_from_payload(request, text_field="text", default_format="wav")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await manager.send_json({"type": "connected", "deepspeed": _is_deepspeed_enabled()}, ws)
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            try:
                if msg_type in ("tts", "tts_stream"):
                    if tts is None: await manager.send_json({"type": "error", "message": "Model not initialized"}, ws); continue
                    text = data.get("text", "")
                    if not text.strip(): await manager.send_json({"type": "error", "message": "text cannot be empty"}, ws); continue
                    if data.get("emo_audio_base64"):
                        try: data["emo_audio_prompt"] = await _register_audio_from_base64(data["emo_audio_base64"], "emo_base64")
                        except ValueError as e: await manager.send_json({"type": "error", "message": str(e)}, ws); continue
                    if data.get("emo_audio_prompt") and not os.path.exists(data["emo_audio_prompt"]): data.pop("emo_audio_prompt", None)
                    try: params = _extract_params(data)
                    except ValueError as e: await manager.send_json({"type": "error", "message": str(e)}, ws); continue
                    quick_streaming_tokens = max(0, min(int(data.get("quick_streaming_tokens") or 60), 240))
                    spk = None
                    if data.get("spk_audio_base64"):
                        try: spk = await _register_audio_from_base64(data["spk_audio_base64"], data.get("voice"))
                        except ValueError as e: await manager.send_json({"type": "error", "message": str(e)}, ws); continue
                    if not spk: spk = _resolve_speaker(data) or "examples/voice_01.wav"
                    output_format = data.get("response_format") or "wav"
                    if output_format not in SUPPORTED_RESPONSE_FORMATS: await manager.send_json({"type": "error", "message": "Unsupported format"}, ws); continue

                    if msg_type == "tts_stream":
                        await _guarded_stream_to_ws(ws, text.strip(), spk, _unique_output_path("ws_stream_"), params, output_format, quick_streaming_tokens, data.get("trim_silence", True)); continue
                    infer_record = await _guarded_infer(text, spk, _unique_output_path("ws_"), params)
                    if infer_record["result"]:
                        sr, wav = infer_record["result"]
                        if data.get("trim_silence", True): wav = trim_audio_array(wav, sr)
                        audio_bytes, media_type = _encode_audio(wav, sr, output_format)
                        await manager.send_json({"type": "completed", "audio_base64": base64.b64encode(audio_bytes).decode(), "format": output_format, "sample_rate": sr, "deepspeed": _is_deepspeed_enabled()}, ws)
                elif msg_type == "ping": await manager.send_json({"type": "pong"}, ws)
                elif msg_type == "get_voices": await manager.send_json({"type": "voices_list", "voices": _load_meta()}, ws)
            except Exception as e:
                logger.error(f"WS error: {e}"); await manager.send_json({"type": "error", "message": str(e)}, ws)
    except WebSocketDisconnect: manager.disconnect(ws)


# ============== AudioX Generation ==============

def _audiox_device() -> str:
    if args is not None and getattr(args, "device", "auto") not in (None, "auto"): return args.device
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    except Exception: pass
    return "cpu"

def _resolve_audiox_model(name: Optional[str]) -> str:
    return _AUDIOX_MODEL_ALIASES.get(name, name) if name else getattr(args, "audiox_model", "HKUSTAudio/AudioX-MAF")

def _audiox_ensure_loaded(model_name: str) -> None:
    global audiox_model, audiox_config, audiox_loaded_name
    if audiox_model is not None and audiox_loaded_name == model_name: return
    import torch
    from audiox import get_pretrained_model
    if audiox_model is not None:
        try: del audiox_model; audiox_model = None; torch.cuda.empty_cache()
        except Exception: pass
    device = _audiox_device()
    m, cfg = get_pretrained_model(model_name)
    m = m.to(device); m.eval()
    audiox_model, audiox_config, audiox_loaded_name = m, cfg, model_name

def _audiox_generate_wav(prompt: str, video_path: Optional[str], seconds_req: Optional[float], model_name: str, steps: int, cfg_scale: float) -> Tuple[np.ndarray, int, bool]:
    import torch
    from einops import rearrange
    from audiox.inference.generation import generate_diffusion_cond
    from audiox.data.utils import read_video
    device = _audiox_device()
    sr, sample_size = int(audiox_config["sample_rate"]), int(audiox_config["sample_size"])
    target_fps = int(audiox_config.get("video_fps", 5))
    seconds_total = max(1, int(round(sample_size / sr)))
    is_maf = "MAF" in model_name
    used_video, sync_features = False, None
    if video_path and os.path.exists(video_path):
        try:
            video_tensors = read_video(video_path, seek_time=0, duration=seconds_total, target_fps=target_fps)
            used_video = True
            if is_maf:
                try:
                    from audiox.data.utils import encode_video_with_synchformer
                    sync_features = encode_video_with_synchformer(video_path, model_name, 0, seconds_total, device)
                except Exception: sync_features = torch.zeros(1, 240, 768).to(device)
        except Exception:
            video_tensors = torch.zeros(int(target_fps * seconds_total), 3, 224, 224)
            if is_maf: sync_features = torch.zeros(1, 240, 768).to(device)
    else:
        video_tensors = torch.zeros(int(target_fps * seconds_total), 3, 224, 224)
        if is_maf: sync_features = torch.zeros(1, 240, 768).to(device)
    audio_tensor = torch.zeros((2, int(sr * seconds_total))).to(device)
    conditioning = [{"video_prompt": {"video_tensors": video_tensors.unsqueeze(0), "video_sync_frames": sync_features}, "text_prompt": prompt or "", "audio_prompt": audio_tensor.unsqueeze(0), "seconds_start": 0, "seconds_total": seconds_total}]
    output = generate_diffusion_cond(audiox_model, steps=steps, cfg_scale=cfg_scale, conditioning=conditioning, sample_size=sample_size, sigma_min=0.3, sigma_max=500, sampler_type="dpmpp-3m-sde", device=device)
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output)).clamp(min=1e-8)).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    arr = output.numpy().T
    if seconds_req and seconds_req > 0:
        want = int(seconds_req * sr)
        if 0 < want < arr.shape[0]: arr = arr[:want]
    return np.ascontiguousarray(arr), sr, used_video

async def _guarded_audiox_infer(prompt: str, video_path: Optional[str], seconds_req: Optional[float], model_name: str, steps: int, cfg_scale: float) -> dict:
    request_start = time.perf_counter()
    if _queue_slots is None or _gpu_semaphore is None: raise RuntimeError("Inference queue not initialized")
    accepted = await _acquire_with_timeout(_queue_slots, args.queue_timeout)
    if not accepted: raise TimeoutError(f"Inference queue is full (wait exceeded {args.queue_timeout:.1f}s)")
    try:
        await _gpu_semaphore.acquire()
        queue_elapsed = time.perf_counter() - request_start
        infer_start = time.perf_counter()
        def _work():
            _audiox_ensure_loaded(model_name)
            return _audiox_generate_wav(prompt, video_path, seconds_req, model_name, steps, cfg_scale)
        try: arr, sr, used_video = await asyncio.to_thread(_work)
        finally: _gpu_semaphore.release()
        return {"arr": arr, "sr": sr, "used_video": used_video, "queue_time": queue_elapsed, "infer_time": time.perf_counter() - infer_start, "total_time": time.perf_counter() - request_start}
    finally: _queue_slots.release()

class AudioXGenerateRequest(BaseModel):
    prompt: str | None = Field(None)
    text: str | None = Field(None)
    seconds_total: float | None = Field(None)
    duration: float | None = Field(None)
    video_base64: str | None = Field(None)
    model: str | None = Field(None)
    steps: int = Field(100, ge=1, le=500)
    cfg_scale: float = Field(7.0, ge=0.0, le=20.0)
    response_format: ResponseFormat | None = Field("wav")

@app.post("/v1/audio/generate", tags=["Audio Generation"], responses=AUDIO_RESPONSES)
async def audiox_generate(request: AudioXGenerateRequest = Body(...)):
    if not _audiox_available(): return JSONResponse(status_code=503, content={"error": "AudioX is not installed on the server."})
    data = request.model_dump(exclude_none=True)
    prompt = (data.get("prompt") or data.get("text") or "").strip()
    seconds_req = data.get("seconds_total") or data.get("duration")
    try: seconds_req = float(seconds_req) if seconds_req is not None else None
    except: seconds_req = None
    output_format = data.get("response_format") or "wav"
    model_name = _resolve_audiox_model(data.get("model"))
    steps, cfg_scale = int(data.get("steps", 100)), float(data.get("cfg_scale", 7.0))
    video_path = None
    if data.get("video_base64"):
        try: video_bytes = base64.b64decode(data["video_base64"])
        except: return JSONResponse(status_code=400, content={"error": "Invalid base64 video data"})
        if len(video_bytes) > 200 * 1024 * 1024: return JSONResponse(status_code=400, content={"error": "Video clip too large"})
        os.makedirs("outputs", exist_ok=True)
        video_path = os.path.join("outputs", f"audiox_vid_{uuid.uuid4().hex[:8]}.mp4")
        with open(video_path, "wb") as f: f.write(video_bytes)
    try:
        rec = await _guarded_audiox_infer(prompt, video_path, seconds_req, model_name, steps, cfg_scale)
        arr, sr = rec["arr"], rec["sr"]
        audio_bytes, media_type = _encode_audio(arr, sr, output_format)
        return Response(content=audio_bytes, media_type=media_type, headers={"X-AudioX-Model": model_name, "X-AudioX-Used-Video": str(rec["used_video"])})
    except TimeoutError as e: return JSONResponse(status_code=429, content={"error": str(e)})
    except Exception as e: return JSONResponse(status_code=500, content={"error": f"Generation failed: {str(e)}"})
    finally:
        if video_path and os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass

@app.get("/v1/audio/generate/health", tags=["Audio Generation"])
async def audiox_health(): return {"available": _audiox_available(), "loaded": audiox_model is not None, "loaded_model": audiox_loaded_name}

# ============== Mount WebUI ==============
import gradio as gr
webui_enhanced.demo.queue(20)
app = gr.mount_gradio_app(app, webui_enhanced.demo, path="/")

# ============== Main Program ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_concurrency", type=int, default=int(os.environ.get("INDEXTTS_MAX_CONCURRENCY", "1")))
    parser.add_argument("--queue_size", type=int, default=int(os.environ.get("INDEXTTS_QUEUE_SIZE", "16")))
    parser.add_argument("--queue_timeout", type=float, default=float(os.environ.get("INDEXTTS_QUEUE_TIMEOUT", "120")))
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed GPT inference acceleration.")
    parser.add_argument("--accel", action="store_true")
    parser.add_argument("--cuda_kernel", action="store_true")
    parser.add_argument("--audiox_model", type=str, default=os.environ.get("AUDIOX_MODEL", "HKUSTAudio/AudioX-MAF"))
    parser.add_argument("--audiox_preload", action="store_true")
    args = parser.parse_args()
    args.max_concurrency, args.queue_size, args.queue_timeout = max(1, args.max_concurrency), max(0, args.queue_size), max(0.1, args.queue_timeout)
    os.makedirs("outputs", exist_ok=True); os.makedirs("logs", exist_ok=True); os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
    logger.info(f"IndexTTS2 API Server - http://{args.host}:{args.port}")
    logger.info(f"Device: {args.device}, FP16: {args.fp16}, DeepSpeed: {args.deepspeed}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)