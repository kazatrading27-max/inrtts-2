"""
IndexTTS2 API Server
Supports voice cloning, emotion control, voice caching, and inference parameter tuning.
Root route mounts the WebUI, /v1/ routes provide the API.
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


# ============== Global Variables ==============
tts: Optional[IndexTTS2] = None
args: argparse.Namespace = None
SPEAKER_CACHE_DIR = "assets/speaker_cache"
SPEAKER_META_FILE = os.path.join(SPEAKER_CACHE_DIR, "meta.json")

_gpu_semaphore: Optional[asyncio.Semaphore] = None
_queue_slots: Optional[asyncio.Semaphore] = None
_meta_cache: Optional[dict] = None
_meta_mtime: Optional[float] = None
SUPPORTED_RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
OPENAPI_TAGS = [
    {"name": "Base", "description": "Health check, concurrency queue configuration, and running status."},
    {"name": "WebSocket", "description": "WebSocket long-connection synthesis instructions. FastAPI Swagger does not natively display ws:// routes, so /ws/docs is provided."},
    {"name": "Voice Management", "description": "Upload, query, and delete voices. Returns voice_id after upload for use in HTTP and WebSocket synthesis."},
    {"name": "Speech Synthesis", "description": "OpenAI-compatible speech synthesis interface. Uses the input field for text; model is optional and not validated."},
    {"name": "Plain Speech Synthesis", "description": "Plain JSON speech synthesis interface. Uses the text field for text."},
]
AUDIO_RESPONSES = {
    200: {
        "description": "Synthesis successful, returns binary audio in the specified response_format.",
        "content": {media_type: {} for media_type in MEDIA_TYPES.values()},
    },
    400: {"description": "Invalid request parameters."},
    429: {"description": "Inference queue is full or wait timed out."},
    500: {"description": "Synthesis failed or internal server error."},
}


# ============== Audio Trimming Logic (Ported from WebUI) ==============

def _detect_silence_regions(y: np.ndarray, sr: int, window_ms: int = 25, rms_threshold: float = 0.005) -> List[Tuple[int, int]]:
    window_samples = max(1, int(sr * window_ms / 1000))
    step = max(1, window_samples // 2)
    regions = []
    in_silence = False
    silence_start = 0
    pos = 0
    while pos + window_samples <= len(y):
        seg = y[pos: pos + window_samples]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        if rms < rms_threshold and not in_silence:
            silence_start = pos
            in_silence = True
        elif rms >= rms_threshold and in_silence:
            regions.append((silence_start, pos))
            in_silence = False
        pos += step
    if in_silence:
        regions.append((silence_start, len(y)))
    return regions


def _rms_trim_edges(y: np.ndarray, sr: int, rms_threshold: float = 0.005, window_ms: int = 25, margin_ms: int = 80) -> np.ndarray:
    if len(y) == 0:
        return y
    window_samples = max(1, int(sr * window_ms / 1000))
    step = max(1, window_samples // 2)
    end_pos = len(y)
    while end_pos > window_samples:
        seg = y[end_pos - window_samples: end_pos]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        if rms > rms_threshold:
            break
        end_pos -= step
    start_pos = 0
    while start_pos + window_samples < end_pos:
        seg = y[start_pos: start_pos + window_samples]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        if rms > rms_threshold:
            break
        start_pos += step
    margin_samples = int(sr * margin_ms / 1000)
    start_pos = max(0, start_pos - margin_samples)
    end_pos = min(len(y), end_pos + margin_samples)
    if end_pos <= start_pos:
        return y
    return y[start_pos:end_pos]


def _remove_trailing_noise(y: np.ndarray, sr: int, min_gap_ms: int = 200, max_trailing_ms: int = 500, energy_ratio: float = 0.3, rms_threshold: float = 0.005) -> np.ndarray:
    if len(y) < int(sr * 0.3):
        return y
    regions = _detect_silence_regions(y, sr, rms_threshold=rms_threshold)
    if not regions:
        return y
    min_gap_samples = int(sr * min_gap_ms / 1000)
    max_trailing_samples = int(sr * max_trailing_ms / 1000)
    very_short_samples = int(sr * 0.1)
    last_gap = None
    for start, end in reversed(regions):
        if (end - start) >= min_gap_samples:
            last_gap = (start, end)
            break
    if last_gap is None:
        return y
    gap_start, gap_end = last_gap
    if gap_end >= len(y):
        return y
    trailing = y[gap_end:]
    trailing_len = len(trailing)

    def _cut() -> np.ndarray:
        fade_s = min(int(sr * 0.02), gap_start)
        result = y[:gap_start].copy()
        if fade_s > 0:
            fade = np.linspace(1.0, 0.0, fade_s).astype(result.dtype)
            result[-fade_s:] *= fade
        return result

    if trailing_len < very_short_samples:
        return _cut()
    if trailing_len < max_trailing_samples:
        main_speech = y[:gap_start]
        if len(main_speech) > 0:
            main_rms = float(np.sqrt(np.mean(main_speech.astype(np.float64) ** 2)))
            trail_rms = float(np.sqrt(np.mean(trailing.astype(np.float64) ** 2)))
            if main_rms > 0 and (trail_rms / main_rms) < energy_ratio:
                return _cut()
    return y


def _compress_internal_silence(y: np.ndarray, sr: int, max_silence_ms: int = 600, compress_to_ms: int = 200, rms_threshold: float = 0.005) -> np.ndarray:
    regions = _detect_silence_regions(y, sr, rms_threshold=rms_threshold)
    if not regions:
        return y
    max_s = int(sr * max_silence_ms / 1000)
    compress_s = int(sr * compress_to_ms / 1000)
    min_boundary = int(sr * 0.05)
    long_silences = [
        (s, e) for s, e in regions
        if s >= min_boundary and e <= len(y) - min_boundary and (e - s) > max_s
    ]
    if not long_silences:
        return y
    pieces = []
    prev_end = 0
    for sil_start, sil_end in long_silences:
        pieces.append(y[prev_end:sil_start])
        edge = min(compress_s // 2, (sil_end - sil_start) // 2)
        pieces.append(y[sil_start: sil_start + edge])
        fill = compress_s - edge * 2
        if fill > 0:
            pieces.append(np.zeros(fill, dtype=y.dtype))
        pieces.append(y[sil_end - edge: sil_end])
        prev_end = sil_end
    pieces.append(y[prev_end:])
    return np.concatenate(pieces)


def trim_audio_array(y: np.ndarray, sr: int, top_db: int = 30, margin_ms: int = 80, max_internal_silence_ms: int = 600, compress_silence_to_ms: int = 200) -> np.ndarray:
    """4-pass post-processing to remove extended silence and trailing noise. In-memory version."""
    if len(y) == 0:
        return y

    # Convert to float32 for librosa processing
    y_float = y.astype(np.float32) / 32768.0
    original_duration = len(y_float) / sr
    
    # Pass 1: Librosa top_db trim
    yt, _ = librosa.effects.trim(y_float, top_db=top_db)
    if len(yt) == 0:
        yt = y_float
        
    # Pass 2: RMS edge trim
    yt = _rms_trim_edges(yt, sr, margin_ms=margin_ms)
    
    # Pass 3: Trailing noise/hallucination removal
    yt = _remove_trailing_noise(yt, sr)
    
    # Pass 4: Compress long internal silences
    yt = _compress_internal_silence(
        yt, sr, max_silence_ms=max_internal_silence_ms, compress_to_ms=compress_silence_to_ms
    )
    
    trimmed_duration = len(yt) / sr
    if len(yt) > int(sr * 0.1):
        removed = original_duration - trimmed_duration
        if removed > 0.05:
            logger.info(f"Trimmed audio: {original_duration:.2f}s -> {trimmed_duration:.2f}s (removed {removed:.2f}s)")
        
        # Convert back to int16
        yt = np.clip(yt * 32767.0, -32768, 32767).astype(np.int16)
        return yt
        
    return y


# ============== /docs Request Body Models ==============

ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class SpeechParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    voice: str = Field(
        ...,
        description="Voice ID from POST /v1/audio/voices voice_id; can also be a server-accessible audio path.",
        examples=["spk_0dc59f90"],
    )
    response_format: ResponseFormat | None = Field(
        None,
        description="Return audio format. OpenAI protocol defaults to mp3, plain /tts defaults to wav.",
        examples=["wav"],
    )
    speaker_id: str | None = Field(
        None,
        description="Compatible alias for voice. voice takes precedence; speaker_id can be used if voice is empty.",
        examples=["spk_0dc59f90"],
    )
    spk_audio_prompt: str | None = Field(
        None,
        description="Server-accessible voice reference audio path. Can be passed directly if not using the voice library.",
        examples=["assets/speaker_cache/spk_0dc59f90.wav"],
    )
    spk_audio_base64: str | None = Field(
        None,
        description="Base64 encoded audio data for voice cloning. Automatically cached via MD5 hash to avoid re-processing on subsequent requests.",
        examples=["UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="]
    )
    emo_audio_prompt: str | None = Field(
        None,
        description="Server-accessible emotion reference audio path.",
        examples=["examples/emotion.wav"],
    )
    emo_alpha: float = Field(1.0, ge=0.0, le=1.0, description="Emotion control strength, 0-1. Text emotion mode suggested not to exceed 0.6.")
    emo_vector: list[float] | None = Field(
        None,
        min_length=8,
        max_length=8,
        description="8-dimensional emotion vector: [happy, angry, sad, fear, disgust, gloomy, surprised, calm].",
        examples=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    use_emo_text: bool = Field(False, description="Enable text emotion recognition.")
    emo_text: str | None = Field(None, description="Text used for emotion recognition; can use synthesis text if empty.")
    use_random: bool = Field(False, description="Whether to randomly sample emotions.")
    interval_silence: int = Field(200, ge=0, le=1000, description="Silence inserted between segments in milliseconds.")
    max_text_tokens_per_segment: int = Field(120, ge=20, le=240, description="Maximum text tokens per segment.")
    quick_streaming_tokens: int = Field(
        60,
        ge=0,
        le=240,
        description="Streaming first packet target token count, used only for WebSocket tts_stream. Smaller means faster first packet, more segments.",
    )
    num_beams: int = Field(3, ge=1, le=10, description="Search width. Speed priority suggests 1, quality priority can use 3.")
    do_sample: bool = Field(True, description="Enable sampling. Can be disabled for speed priority.")
    top_k: int = Field(30, ge=1, le=100, description="Top-K sampling parameter.")
    top_p: float = Field(0.8, ge=0.0, le=1.0, description="Top-P sampling parameter.")
    temperature: float = Field(0.8, gt=0.0, le=2.0, description="Sampling temperature, must be greater than 0.")
    max_mel_tokens: int = Field(1500, ge=100, le=3000, description="Maximum generated mel token count.")
    length_penalty: float = Field(0.0, ge=0.0, le=2.0, description="Length penalty.")
    repetition_penalty: float = Field(10.0, gt=0.0, le=20.0, description="Repetition penalty, must be greater than 0.")
    diffusion_steps: int = Field(25, ge=1, le=50, description="Diffusion steps. Speed priority suggests 12, quality priority can use 25.")
    inference_cfg_rate: float = Field(0.7, ge=0.0, le=2.0, description="CFG strength.")
    trim_silence: bool = Field(True, description="Automatically trim leading/trailing silence and trailing noise, fixing model generation of overly long silent tails.")
    target_dur: float | None = Field(
        None,
        ge=0.2,
        le=30.0,
        description=(
            "Target speech duration in seconds (for lip-sync / SRT dubbing). "
            "When set, enables IndexTTS2 native duration control (num_codes = round(target_dur * 50)) "
            "if the loaded model supports it. Works best within ~0.75x-1.25x of the text's natural length; "
            "combine with fit_to_target for an exact match. Leave unset for natural duration."
        ),
        examples=[3.5],
    )
    fit_to_target: bool = Field(
        False,
        description=(
            "After synthesis, pitch-preservingly time-stretch the audio to exactly match target_dur. "
            "Requires target_dur. Auto-enabled when the loaded model lacks native duration support, so "
            "target_dur is always honored. Ratio is clamped by max_fit_ratio to protect quality."
        ),
    )
    native_duration: bool = Field(
        True,
        description="Use the model's native duration conditioning (target_dur) when available. Set False to rely only on fit_to_target post-stretch.",
    )
    max_fit_ratio: float = Field(
        1.5,
        ge=1.0,
        le=4.0,
        description="Max speed-up/slow-down factor allowed when fit_to_target stretches audio (e.g. 1.5 = between 0.67x and 1.5x). Prevents artifacts on impossible targets.",
    )


class OpenAISpeechRequest(SpeechParams):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "model": "gpt-4o-mini-tts",
                    "input": "Hello, this is an OpenAI protocol speech synthesis test.",
                    "voice": "spk_0dc59f90",
                    "response_format": "wav",
                    "num_beams": 1,
                    "do_sample": False,
                    "top_k": 10,
                    "diffusion_steps": 12,
                }
            ]
        },
    )

    input: str = Field(..., min_length=1, description="Text to synthesize.")
    model: str | None = Field(
        None,
        description="OpenAI protocol field. This service does not enforce or validate the model name; ignored if passed.",
        examples=["gpt-4o-mini-tts"],
    )


class PlainTTSRequest(SpeechParams):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "text": "Hello, this is a plain TTS interface test.",
                    "voice": "spk_0dc59f90",
                    "response_format": "wav",
                    "num_beams": 1,
                    "do_sample": False,
                    "top_k": 10,
                    "diffusion_steps": 12,
                }
            ]
        },
    )

    text: str = Field(..., min_length=1, description="Text to synthesize.")


# ============== Voice Cache Management ==============

def _load_meta() -> dict:
    global _meta_cache, _meta_mtime
    if os.path.exists(SPEAKER_META_FILE):
        mtime = os.path.getmtime(SPEAKER_META_FILE)
        if _meta_cache is not None and _meta_mtime == mtime:
            return _meta_cache
        with open(SPEAKER_META_FILE, "r", encoding="utf-8") as f:
            _meta_cache = json.load(f)
        _meta_mtime = mtime
        return _meta_cache
    _meta_cache = {}
    _meta_mtime = None
    return _meta_cache


def _save_meta(meta: dict):
    global _meta_cache, _meta_mtime
    _meta_cache = meta
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
    with open(SPEAKER_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _meta_mtime = os.path.getmtime(SPEAKER_META_FILE)


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def _register_audio_from_base64(b64_data: str, speaker_name: str = "dynamic_base64") -> str:
    """Decodes base64 audio, caches it as a file, and registers it to skip GPU re-processing on subsequent calls."""
    try:
        audio_bytes = base64.b64decode(b64_data)
    except Exception:
        raise ValueError("Invalid base64 audio data")

    if len(audio_bytes) > 20 * 1024 * 1024:
        raise ValueError("Base64 audio file too large (exceeds 20MB)")

    md5 = hashlib.md5(audio_bytes).hexdigest()
    voice_id = f"spk_{md5[:8]}"
    
    meta = _load_meta()
    if voice_id in meta:
        logger.info(f"Dynamic voice cache hit: {voice_id}")
        return meta[voice_id]["audio_path"]
    
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
    tmp_path = os.path.join(SPEAKER_CACHE_DIR, f"tmp_{uuid.uuid4().hex[:8]}.wav")
    try:
        buf = io.BytesIO(audio_bytes)
        wav_data, sr = sf.read(buf, dtype="int16")
        sf.write(tmp_path, wav_data, sr)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise ValueError(f"Failed to decode audio bytes. Must be valid WAV/FLAC/OGG: {e}")

    final_path = os.path.join(SPEAKER_CACHE_DIR, f"{voice_id}.wav")
    os.rename(tmp_path, final_path)

    meta[voice_id] = {
        "voice_name": speaker_name or voice_id,
        "audio_path": final_path,
        "md5": md5,
        "original_filename": "inline_base64",
        "created_at": time.time(),
        "embedding_cached": False,
    }
    _save_meta(meta)
    logger.info(f"Registered dynamic voice from base64: {voice_id}")
    
    return final_path


# ============== Inference Parameter Extraction ==============

def _extract_params(data: dict) -> dict:
    """Extract inference parameters from request body, returning kwargs needed by infer()."""
    def as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    params = {}
    for key in ("emo_audio_prompt", "emo_alpha", "emo_vector", "use_emo_text",
                "emo_text", "use_random"):
        if key in data:
            params[key] = data[key]
    if "emo_alpha" not in params:
        params["emo_alpha"] = 1.0
    else:
        params["emo_alpha"] = float(params["emo_alpha"])
    if "emo_vector" in params:
        v = params["emo_vector"]
        if not isinstance(v, list) or len(v) != 8:
            raise ValueError("emo_vector must be an array of length 8 [happy, angry, sad, fear, disgust, gloomy, surprised, calm]")
        params["emo_vector"] = [float(x) for x in v]
    for key in ("use_emo_text", "use_random", "do_sample"):
        if key in params:
            params[key] = as_bool(params[key])
        elif key in data:
            params[key] = as_bool(data[key])
    for key in ("interval_silence", "max_text_tokens_per_segment",
                "num_beams", "top_k", "max_mel_tokens"):
        if key in data and data[key] is not None:
            params[key] = int(data[key])
    for key in ("top_p", "temperature", "length_penalty", "repetition_penalty"):
        if key in data and data[key] is not None:
            params[key] = float(data[key])
    if params.get("repetition_penalty", 10.0) <= 0:
        raise ValueError("repetition_penalty must be greater than 0")
    if params.get("temperature", 0.8) <= 0:
        raise ValueError("temperature must be greater than 0")
    if "top_p" in params and not 0 <= params["top_p"] <= 1:
        raise ValueError("top_p must be between 0 and 1")

    # ---- FIX 1: Prevent beam search and sampling from being enabled simultaneously ----
    do_sample = params.get("do_sample", True)
    num_beams = params.get("num_beams", 3)
    if do_sample and num_beams > 1:
        logger.warning(
            f"Unsafe parameter combination detected: do_sample=True & num_beams={num_beams}. "
            f"This can cause issues under DeepSpeed inference. "
            f"Automatically setting num_beams=1 (preserving sampling mode). "
            f"For beam search, explicitly pass do_sample=False."
        )
        params["num_beams"] = 1

    # ---- FIX 2: DeepSpeed KV cache limit ----
    if args.deepspeed:
        max_text_tokens = params.get("max_text_tokens_per_segment", 120)
        buffer_tokens = 24  # Safety buffer for BOS/EOS and other special tokens
        max_allowed_mel_tokens = max(100, 1024 - max_text_tokens - buffer_tokens)
        
        current_max_mel = params.get("max_mel_tokens", 1500)
        if current_max_mel > max_allowed_mel_tokens:
            logger.warning(
                f"DeepSpeed is enabled, which pre-allocates KV cache for max 1024 total tokens. "
                f"Capping max_mel_tokens from {current_max_mel} to {max_allowed_mel_tokens} to prevent CUDA illegal memory access."
            )
            params["max_mel_tokens"] = max_allowed_mel_tokens

    return params


# ============== Duration Control (native + time-stretch fallback) ==============

# IndexTTS2 semantic codec frame rate: num_codes = round(seconds * 50). Verified against
# released weights (8s target -> 398/400 tokens). Native duration is accurate roughly within
# 0.75x-1.25x of the text's natural spoken length; outside that band use fit_to_target.
TOKENS_PER_SECOND = 50
DURATION_MIN_SEC = 0.3
DURATION_NATIVE_MAX_SEC = 30.0   # keep num_codes within the trained mel-position table
DURATION_FIT_MAX_SEC = 60.0

_infer_named_params_cache: Optional[set] = None
_infer_named_params_for: Optional[int] = None


def _infer_named_params() -> set:
    """Explicit (non var-keyword) parameter names accepted by the loaded tts.infer.

    Used to detect capabilities at runtime, because the API may run against either
    infer_v2.IndexTTS2 (no duration control) or the fork's infer_indextts2.IndexTTS2
    (accepts use_speed/target_dur). tts.infer usually has **generation_kwargs, so we
    must check the *explicit* names, not just "does it accept the kwarg".
    """
    global _infer_named_params_cache, _infer_named_params_for
    if tts is None:
        return set()
    if _infer_named_params_cache is not None and _infer_named_params_for == id(tts):
        return _infer_named_params_cache
    try:
        sig = inspect.signature(tts.infer)
        named = {
            name for name, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
    except (TypeError, ValueError):
        named = set()
    _infer_named_params_cache = named
    _infer_named_params_for = id(tts)
    return named


def _supports_native_duration() -> bool:
    return "target_dur" in _infer_named_params()


def _finalize_params(params: dict) -> dict:
    """Adapt param names to the loaded inference signature before calling tts.infer.

    infer_v2 names the segmenter kwarg `max_text_tokens_per_segment`, while the fork's
    infer_indextts2 uses `max_text_tokens_per_sentence`. Remap so neither leaks into
    **generation_kwargs (which would reach HF .generate() and raise).
    """
    named = _infer_named_params()
    if named and "max_text_tokens_per_segment" in params and "max_text_tokens_per_segment" not in named:
        val = params.pop("max_text_tokens_per_segment")
        if "max_text_tokens_per_sentence" in named:
            params["max_text_tokens_per_sentence"] = val
    return params


def _extract_duration_controls(data: dict) -> dict:
    """Parse duration-control fields. Returns {} when no target_dur is requested."""
    raw = data.get("target_dur", None)
    if raw is None or raw == "":
        return {}
    try:
        target = float(raw)
    except (TypeError, ValueError):
        raise ValueError("target_dur must be a number of seconds")
    if target <= 0:
        return {}
    if target > DURATION_FIT_MAX_SEC:
        target = DURATION_FIT_MAX_SEC

    def as_bool(value, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    try:
        max_ratio = float(data.get("max_fit_ratio", 1.5))
    except (TypeError, ValueError):
        max_ratio = 1.5
    max_ratio = min(max(max_ratio, 1.0), 4.0)

    return {
        "target_dur": target,
        "fit_to_target": as_bool(data.get("fit_to_target"), False),
        "native": as_bool(data.get("native_duration"), True),
        "max_fit_ratio": max_ratio,
    }


def _apply_native_duration(params: dict, dctl: dict) -> bool:
    """If requested and supported by the loaded model, inject native duration kwargs.

    Returns True when native conditioning was applied.
    """
    if not dctl or not dctl.get("native"):
        return False
    if not _supports_native_duration():
        return False
    native_target = min(max(dctl["target_dur"], DURATION_MIN_SEC), DURATION_NATIVE_MAX_SEC)
    params["use_speed"] = True
    params["target_dur"] = native_target
    return True


def _time_stretch_to_duration(wav: np.ndarray, sr: int, target_sec: float, max_ratio: float = 1.5) -> Tuple[np.ndarray, dict]:
    """Pitch-preserving stretch of an int16 mono waveform toward target_sec.

    librosa rate > 1 shortens, < 1 lengthens. Ratio is clamped to [1/max_ratio, max_ratio]
    so impossible targets degrade gracefully instead of producing artifacts.
    """
    n = len(wav)
    cur = (n / sr) if sr else 0.0
    info = {"target": target_sec, "before": cur, "after": cur, "ratio": 1.0, "clamped": False, "applied": False}
    if n == 0 or sr <= 0 or target_sec <= 0 or cur <= 0:
        return wav, info

    ratio = cur / target_sec
    lo, hi = 1.0 / max_ratio, max_ratio
    clamped_ratio = min(max(ratio, lo), hi)
    info["clamped"] = abs(clamped_ratio - ratio) > 1e-6
    if abs(clamped_ratio - 1.0) < 1e-3:
        return wav, info

    # librosa.time_stretch needs mono 1-D; stream path may hand us (N, 1)
    mono = wav.reshape(-1) if wav.ndim > 1 and wav.shape[-1] == 1 else (
        wav.mean(axis=-1) if wav.ndim > 1 else wav
    )
    y = mono.astype(np.float32) / 32768.0
    try:
        try:
            y2 = librosa.effects.time_stretch(y, rate=clamped_ratio)
        except TypeError:
            y2 = librosa.effects.time_stretch(y, clamped_ratio)  # older librosa positional API
    except Exception as e:  # pragma: no cover - safety net
        logger.warning(f"time_stretch failed ({e}); returning unstretched audio")
        return wav, info

    out = np.clip(y2 * 32767.0, -32768, 32767).astype(np.int16)
    info.update({"after": len(out) / sr, "ratio": clamped_ratio, "applied": True})
    return out, info


def _resolve_duration_plan(dctl: dict, native_used: bool) -> bool:
    """Whether to run the post-synthesis stretch. Explicit fit request, or auto when native is unavailable."""
    if not dctl:
        return False
    return bool(dctl.get("fit_to_target")) or (not native_used)


def _apply_fit_and_headers(wav: np.ndarray, sr: int, dctl: dict, native_used: bool) -> Tuple[np.ndarray, dict]:
    """Apply fit-to-target stretch if planned and build X-IndexTTS-Duration-* headers."""
    if not dctl:
        return wav, {}
    target = dctl["target_dur"]
    do_fit = _resolve_duration_plan(dctl, native_used)
    fit_info = None
    if do_fit:
        wav, fit_info = _time_stretch_to_duration(wav, sr, target, dctl["max_fit_ratio"])
    actual = (len(wav) / sr) if sr else 0.0
    mode = (
        "native+fit" if native_used and do_fit else
        "native" if native_used else
        "fit" if do_fit else "none"
    )
    headers = {
        "X-IndexTTS-Duration-Mode": mode,
        "X-IndexTTS-Target-Duration": f"{target:.3f}",
        "X-IndexTTS-Actual-Duration": f"{actual:.3f}",
    }
    if fit_info and fit_info["applied"]:
        headers["X-IndexTTS-Fit-Ratio"] = f"{fit_info['ratio']:.4f}"
        if fit_info["clamped"]:
            headers["X-IndexTTS-Fit-Clamped"] = "1"
    return wav, headers


def _encode_audio(wav: np.ndarray, sr: int, fmt: str) -> Tuple[bytes, str]:
    if fmt == "pcm":
        return wav.tobytes(), MEDIA_TYPES[fmt]
    if fmt not in SUPPORTED_RESPONSE_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")

    buf = io.BytesIO()
    if fmt in ("wav", "flac"):
        sf.write(buf, wav, sr, format=fmt.upper())
        return buf.getvalue(), MEDIA_TYPES[fmt]

    wav_buf = io.BytesIO()
    sf.write(wav_buf, wav, sr, format="WAV")
    codec_args = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame"],
        "opus": ["-f", "opus", "-codec:a", "libopus"],
        "aac": ["-f", "adts", "-codec:a", "aac"],
    }[fmt]
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *codec_args, "pipe:1"],
        input=wav_buf.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcoding failed: {proc.stderr.decode('utf-8', errors='ignore')}")
    return proc.stdout, MEDIA_TYPES[fmt]


def _resolve_speaker(data: dict) -> Optional[str]:
    """Resolve voice: supports voice_id (cache) or spk_audio_prompt (file path)"""
    speaker_id = data.get("speaker_id") or data.get("voice")
    if speaker_id:
        meta = _load_meta()
        if speaker_id in meta:
            return meta[speaker_id]["audio_path"]
        return speaker_id
    return data.get("spk_audio_prompt")


def _unique_output_path(prefix: str = "") -> str:
    """Generate a unique output file path (UUID prevents concurrent conflicts)"""
    name = f"{prefix}{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:12]
    return f"outputs/{name}.wav"


# ============== Synchronous Inference ==============

def _do_infer(text: str, spk_audio_prompt: str, output_path: str, params: dict):
    params = _finalize_params(params)
    result = tts.infer(
        spk_audio_prompt=spk_audio_prompt,
        text=text,
        output_path=output_path,
        **params,
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result
    if isinstance(result, str) and os.path.exists(result):
        wav, sr = sf.read(result, dtype="int16")
        return sr, wav
    if result is None and output_path and os.path.exists(output_path):
        wav, sr = sf.read(output_path, dtype="int16")
        return sr, wav
    return result


def _wav_chunk_to_numpy(wav: Any) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu()
    if hasattr(wav, "numpy"):
        wav = wav.numpy()
    arr = np.asarray(wav)
    if arr.dtype != np.int16:
        arr = np.clip(arr, -32767, 32767).astype(np.int16)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[0] <= 8:
        arr = arr.T
    return np.ascontiguousarray(arr)


def _concat_wav_chunks(chunks: List[np.ndarray]) -> np.ndarray:
    valid_chunks = [chunk for chunk in chunks if chunk.size > 0]
    if not valid_chunks:
        return np.zeros((0, 1), dtype=np.int16)
    return np.ascontiguousarray(np.concatenate(valid_chunks, axis=0))


_STREAM_DONE = object()


def _next_stream_item(generator):
    try:
        return next(generator)
    except StopIteration:
        return _STREAM_DONE


async def _acquire_with_timeout(semaphore: asyncio.Semaphore, timeout: float) -> bool:
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def _guarded_infer(text: str, spk: str, output_path: str, params: dict) -> dict:
    """Inference guarded by semaphores to control GPU concurrency"""
    request_start = time.perf_counter()
    if _queue_slots is None or _gpu_semaphore is None:
        raise RuntimeError("Inference queue not initialized")

    accepted = await _acquire_with_timeout(_queue_slots, args.queue_timeout)
    if not accepted:
        raise TimeoutError(f"Inference queue is full, please try again later (wait exceeded {args.queue_timeout:.1f}s)")

    try:
        await _gpu_semaphore.acquire()
        queue_elapsed = time.perf_counter() - request_start
        infer_start = time.perf_counter()
        try:
            result = await asyncio.to_thread(_do_infer, text, spk, output_path, params)
        finally:
            _gpu_semaphore.release()
        return {
            "result": result,
            "queue_time": queue_elapsed,
            "infer_time": time.perf_counter() - infer_start,
            "total_time": time.perf_counter() - request_start,
        }
    finally:
        _queue_slots.release()


async def _guarded_stream_to_ws(
    ws: WebSocket,
    text: str,
    spk: str,
    output_path: str,
    params: dict,
    output_format: str,
    quick_streaming_tokens: int,
    trim_silence: bool = True,
    dctl: Optional[dict] = None,
    native_used: bool = False,
):
    request_start = time.perf_counter()
    if _queue_slots is None or _gpu_semaphore is None:
        raise RuntimeError("Inference queue not initialized")

    accepted = await _acquire_with_timeout(_queue_slots, args.queue_timeout)
    if not accepted:
        raise TimeoutError(f"Inference queue is full, please try again later (wait exceeded {args.queue_timeout:.1f}s)")

    chunk_count = 0
    chunk_bytes = 0
    chunks: List[np.ndarray] = []
    gpu_acquired = False
    generator = None

    try:
        await _gpu_semaphore.acquire()
        gpu_acquired = True
        queue_elapsed = time.perf_counter() - request_start
        infer_start = time.perf_counter()

        await ws.send_json({
            "type": "stream_started",
            "format": output_format,
            "media_type": MEDIA_TYPES[output_format],
            "sample_rate": 22050,
            "queue_time": round(queue_elapsed, 3),
            "quick_streaming_tokens": quick_streaming_tokens,
        })

        params = _finalize_params(params)
        generator = tts.infer(
            spk_audio_prompt=spk,
            text=text,
            output_path=None,
            stream_return=True,
            more_segment_before=quick_streaming_tokens,
            **params,
        )
        while True:
            item = await asyncio.to_thread(_next_stream_item, generator)
            if item is _STREAM_DONE:
                break

            chunk = _wav_chunk_to_numpy(item)
            if chunk.size == 0:
                continue
            chunks.append(chunk)
            elapsed = time.perf_counter() - infer_start
            total_elapsed = time.perf_counter() - request_start
            chunk_count += 1
            audio_bytes, media_type = _encode_audio(chunk, 22050, output_format)
            chunk_bytes += len(audio_bytes)
            await ws.send_json({
                "type": "stream_chunk",
                "chunk_index": chunk_count,
                "audio_base64": base64.b64encode(audio_bytes).decode(),
                "format": output_format,
                "media_type": media_type,
                "sample_rate": 22050,
                "byte_count": len(audio_bytes),
                "queue_time": round(queue_elapsed, 3),
                "infer_time": round(elapsed, 3),
                "total_time": round(total_elapsed, 3),
            })

        full_wav = _concat_wav_chunks(chunks)

        # Apply trimming to the final concatenated audio
        if trim_silence and full_wav.size > 0:
            full_wav = trim_audio_array(full_wav, 22050)

        # Duration control: fit-to-target stretch on the final concatenated audio
        duration_headers = {}
        if full_wav.size > 0:
            full_wav, duration_headers = _apply_fit_and_headers(full_wav, 22050, dctl or {}, native_used)

        if output_path and full_wav.size > 0:
            sf.write(output_path, full_wav, 22050, subtype="PCM_16")
        elapsed = time.perf_counter() - infer_start
        total_elapsed = time.perf_counter() - request_start
        audio_bytes, media_type = _encode_audio(full_wav, 22050, output_format)
        completed_msg = {
            "type": "stream_completed",
            "audio_base64": base64.b64encode(audio_bytes).decode(),
            "format": output_format,
            "media_type": media_type,
            "sample_rate": 22050,
            "chunk_count": chunk_count,
            "chunk_bytes": chunk_bytes,
            "byte_count": len(audio_bytes),
            "queue_time": round(queue_elapsed, 3),
            "infer_time": round(elapsed, 3),
            "total_time": round(total_elapsed, 3),
        }
        if duration_headers:
            completed_msg["duration_mode"] = duration_headers.get("X-IndexTTS-Duration-Mode")
            completed_msg["target_duration"] = float(duration_headers["X-IndexTTS-Target-Duration"])
            completed_msg["actual_duration"] = float(duration_headers["X-IndexTTS-Actual-Duration"])
        await ws.send_json(completed_msg)
        return
    finally:
        if generator is not None:
            try:
                generator.close()
            except Exception:
                pass
        if gpu_acquired:
            _gpu_semaphore.release()
        _queue_slots.release()


# ============== WebSocket Management ==============

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        logger.info(f"WebSocket connection count: {len(self.active_connections)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)
            logger.info(f"WebSocket connection count: {len(self.active_connections)}")

    async def send_json(self, message: dict, ws: WebSocket):
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")


manager = ConnectionManager()


# ============== Application Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts, _gpu_semaphore, _queue_slots
    _gpu_semaphore = asyncio.Semaphore(args.max_concurrency)
    _queue_slots = asyncio.Semaphore(args.max_concurrency + args.queue_size)

    logger.info("Initializing IndexTTS2...")

    try:
        # Reuse the model instance already initialized in webui_enhanced.py to avoid reloading and wasting VRAM
        if hasattr(webui_enhanced, 'tts') and webui_enhanced.tts is not None:
            tts = webui_enhanced.tts
            logger.info("IndexTTS2 initialized successfully (reusing instance from webui_enhanced)")
        else:
            device = None if args.device == "auto" else args.device
            tts = IndexTTS2(
                cfg_path=os.path.join(args.model_dir, "config.yaml"),
                model_dir=args.model_dir,
                use_fp16=args.fp16,
                device=device,
                use_cuda_kernel=args.cuda_kernel,
                use_deepspeed=args.deepspeed,
                use_accel=args.accel,
                use_torch_compile=False,
            )
            logger.info("IndexTTS2 initialized successfully")
    except Exception as e:
        logger.warning(f"Model initialization failed, will run with limited functionality: {e}")
        tts = None

    # Load voice cache into memory
    _load_meta()
    logger.info(f"Loaded {len(_meta_cache or {})} voices")

    yield
    logger.info("Shutting down...")


# ============== FastAPI Application ==============

app = FastAPI(
    lifespan=lifespan,
    title="IndexTTS2 API",
    version="2.1",
    description=(
        "IndexTTS2 zero-shot speech synthesis API. "
        "Supports voice management, OpenAI protocol TTS, plain TTS, WebSocket long-connection synthesis, emotion control, and inference parameter tuning. "
        "Concurrency strategy: FastAPI can receive requests concurrently; GPU inference uniformly enters a queue. Default single GPU serial inference to avoid VRAM and model state conflicts."
    ),
    openapi_tags=OPENAPI_TAGS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.add("logs/api.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)


# ============== Health Check ==============

@app.get("/health", tags=["Base"])
async def health_check():
    """Health check"""
    if tts is None:
        return {"status": "partial", "model_loaded": False,
                "message": "Model not initialized"}
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": time.time(),
        "device": str(tts.device),
        "concurrency": {
            "strategy": "FastAPI receives requests concurrently; GPU inference is rate-limited via queue and semaphores.",
            "max_concurrency": args.max_concurrency,
            "queue_size": args.queue_size,
            "queue_timeout": args.queue_timeout,
            "queue_capacity": args.max_concurrency + args.queue_size,
            "headers": [
                "X-IndexTTS-Queue-Time",
                "X-IndexTTS-Infer-Time",
                "X-IndexTTS-Total-Time",
            ],
        },
    }

@app.get("/api/v1/health", tags=["Base"], include_in_schema=False)
async def health_check_v1():
    """Lightning AI / API v1 health check alias"""
    return await health_check()
    
@app.get("/ws/docs", tags=["WebSocket"])
async def websocket_docs():
    """WebSocket interface documentation. OpenAPI does not natively display ws:// routes, so an HTTP endpoint exposes the docs."""
    return {
        "endpoint": "/ws",
        "protocol": "websocket",
        "url_example": "ws://localhost:8002/ws",
        "concurrency": {
            "strategy": "WebSocket connections can persist; each synthesis message still enters the same GPU inference queue.",
            "max_concurrency": args.max_concurrency,
            "queue_size": args.queue_size,
            "queue_timeout": args.queue_timeout,
        },
        "message_types": {
            "tts": "Plain WebSocket synthesis, returns base64 audio all at once on completion",
            "tts_stream": "True segmented streaming synthesis: first returns stream_started, then continuously returns stream_chunk, finally returns stream_completed",
            "ping": "Heartbeat check, returns pong",
            "get_voices": "Returns current voice metadata",
        },
        "tts_request_example": {
            "type": "tts",
            "text": "Hello, this is a WebSocket synthesis test.",
            "voice": "spk_xxxxxxxx",
            "num_beams": 1,
            "do_sample": False,
            "top_k": 10,
            "top_p": 0.8,
            "temperature": 0.8,
            "response_format": "wav",
            "max_mel_tokens": 900,
            "diffusion_steps": 12,
            "repetition_penalty": 10.0,
            "quick_streaming_tokens": 60,
        },
        "tts_response_example": {
            "type": "completed",
            "audio_base64": "<wav base64>",
            "format": "wav",
            "media_type": "audio/wav",
            "sample_rate": 22050,
            "queue_time": 0.0,
            "infer_time": 1.23,
            "total_time": 1.23,
        },
        "stream_response_sequence": [
            {
                "type": "stream_started",
                "format": "wav",
                "sample_rate": 22050,
                "queue_time": 0.0,
            },
            {
                "type": "stream_chunk",
                "chunk_index": 1,
                "audio_base64": "<chunk wav base64>",
                "format": "wav",
                "sample_rate": 22050,
                "infer_time": 0.8,
            },
            {
                "type": "stream_completed",
                "audio_base64": "<full wav base64>",
                "chunk_count": 3,
                "format": "wav",
                "sample_rate": 22050,
                "total_time": 5.6,
            },
        ],
        "supported_speech_params": [
            "voice",
            "text",
            "speaker_id",
            "spk_audio_prompt",
            "spk_audio_base64",
            "response_format",
            "emo_audio_prompt",
            "emo_alpha",
            "emo_vector",
            "use_emo_text",
            "emo_text",
            "use_random",
            "interval_silence",
            "max_text_tokens_per_segment",
            "quick_streaming_tokens",
            "num_beams",
            "do_sample",
            "top_k",
            "top_p",
            "temperature",
            "max_mel_tokens",
            "length_penalty",
            "repetition_penalty",
            "diffusion_steps",
            "inference_cfg_rate",
            "trim_silence",
            "target_dur",
            "fit_to_target",
            "native_duration",
            "max_fit_ratio",
        ],
        "supported_response_formats": list(SUPPORTED_RESPONSE_FORMATS),
    }


# ============== Voice Management (/v1/audio/voices) ==============

@app.post("/v1/audio/voices", tags=["Voice Management"])
async def upload_speaker(
    audio: UploadFile = File(..., description="Voice reference audio (3-10 seconds is best)"),
    speaker_name: str = Form("", description="Voice name (optional)"),
):
    """Upload audio and register a voice, returning voice_id for subsequent synthesis."""
    if tts is None:
        return JSONResponse(status_code=503, content={"error": "Model not initialized"})

    try:
        if not audio.filename:
            return JSONResponse(status_code=400, content={"error": "Please provide an audio file"})

        ext = os.path.splitext(audio.filename)[-1].lower()
        if ext not in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}:
            return JSONResponse(status_code=400, content={"error": f"Unsupported format: {ext}, supported: mp3, wav, m4a, flac, ogg, aac"})

        audio_bytes = await audio.read()
        if len(audio_bytes) < 1024:
            return JSONResponse(status_code=400, content={"error": "Audio file is too small, please upload a 3-10 second reference audio"})
        if len(audio_bytes) > 20 * 1024 * 1024:
            return JSONResponse(status_code=400, content={"error": "Audio file is too large (exceeds 20MB), please compress before uploading"})

        os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
        tmp_path = os.path.join(SPEAKER_CACHE_DIR, f"tmp_{uuid.uuid4().hex[:8]}{ext}")
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        md5 = _md5_file(tmp_path)
        voice_id = f"spk_{md5[:8]}"

        meta = _load_meta()
        if voice_id in meta:
            os.remove(tmp_path)
            return {"voice_id": voice_id, "status": "exists", "message": "This audio is already registered"}

        final_path = os.path.join(SPEAKER_CACHE_DIR, f"{voice_id}{ext}")
        os.rename(tmp_path, final_path)

        # Warmup: cache speaker embedding via a short inference
        try:
            warmup_path = _unique_output_path("warmup_")
            os.makedirs("outputs", exist_ok=True)
            await _guarded_infer("Test", final_path, warmup_path, {})
            if os.path.exists(warmup_path):
                os.remove(warmup_path)
            # Mark embedding as cached in meta
            if voice_id in _load_meta():
                _meta_cache[voice_id]["embedding_cached"] = True
                _save_meta(_meta_cache)
        except Exception as warmup_err:
            logger.warning(f"Voice warmup failed (does not affect usage): {warmup_err}")

        meta = _load_meta()
        meta[voice_id] = {
            "voice_name": speaker_name or voice_id,
            "audio_path": final_path,
            "md5": md5,
            "original_filename": audio.filename,
            "created_at": time.time(),
            "embedding_cached": True,
        }
        _save_meta(meta)

        logger.info(f"Voice registered successfully: {voice_id} ({speaker_name})")
        return {"voice_id": voice_id, "md5": md5, "status": "new",
                "message": "Voice registered successfully, use the voice parameter in /v1/audio/speech to synthesize speech"}

    except Exception as e:
        logger.error(f"Failed to upload voice: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/v1/audio/voices", tags=["Voice Management"])
async def list_speakers():
    """Get a list of all registered voices"""
    meta = _load_meta()
    voices = []
    for vid, info in meta.items():
        voices.append({
            "voice_id": vid,
            "name": info.get("voice_name", ""),
            "original_filename": info.get("original_filename", ""),
            "created_at": info.get("created_at"),
        })
    return {"object": "list", "data": voices}


@app.delete("/v1/audio/voices/{voice_id}", tags=["Voice Management"])
async def delete_speaker(voice_id: str):
    """Delete a specified voice"""
    meta = _load_meta()
    if voice_id not in meta:
        return JSONResponse(status_code=404, content={"error": f"Voice {voice_id} does not exist"})

    info = meta.pop(voice_id)
    _save_meta(meta)

    audio_path = info.get("audio_path", "")
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    logger.info(f"Voice deleted: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}


# ============== Speech Synthesis (/v1/audio/speech) ==============

async def _speech_response_from_payload(
    data: dict | BaseModel,
    *,
    text_field: str = "input",
    default_format: str = "wav",
) -> Response | JSONResponse:
    """Build a speech response from either OpenAI-style or plain TTS payloads."""
    if tts is None:
        return JSONResponse(status_code=503, content={"error": "Model not initialized"})
    if isinstance(data, BaseModel):
        data = data.model_dump(exclude_none=True)

    text = data.get(text_field)
    if text is None and text_field != "input":
        text = data.get("input")
    if text is None and text_field != "text":
        text = data.get("text")
    if not text or not isinstance(text, str) or not text.strip():
        return JSONResponse(status_code=400, content={"error": f"{text_field} is a required parameter and cannot be empty"})

    try:
        params = _extract_params(data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        dctl = _extract_duration_controls(data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    native_used = _apply_native_duration(params, dctl)

    voice_label = data.get("voice") or data.get("speaker_id") or data.get("spk_audio_prompt") or ""
    
    spk_audio_prompt = None
    if data.get("spk_audio_base64"):
        try:
            spk_audio_prompt = await _register_audio_from_base64(data["spk_audio_base64"], data.get("voice"))
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
            
    if not spk_audio_prompt:
        spk_audio_prompt = _resolve_speaker(data) or "examples/voice_01.wav"
        
    if not os.path.exists(spk_audio_prompt):
        return JSONResponse(
            status_code=400,
            content={"error": f"Voice file does not exist: {voice_label}, please upload via /v1/audio/voices first or provide a valid spk_audio_prompt"},
        )

    output_format = data.get("response_format") or default_format
    if output_format not in SUPPORTED_RESPONSE_FORMATS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported format: {output_format}, options {', '.join(SUPPORTED_RESPONSE_FORMATS)}"},
        )

    try:
        output_path = _unique_output_path()
        os.makedirs("outputs", exist_ok=True)

        infer_record = await _guarded_infer(text.strip(), spk_audio_prompt, output_path, params)
        result = infer_record["result"]
        if result is None:
            return JSONResponse(status_code=500, content={"error": "Synthesis failed, model returned empty result"})

        sr, wav = result

        # Auto trim silence and trailing noise if requested (default: True)
        if data.get("trim_silence", True):
            wav = trim_audio_array(wav, sr)

        # Duration control: fit-to-target stretch (after trim) + duration headers
        wav, duration_headers = _apply_fit_and_headers(wav, sr, dctl, native_used)

        logger.info(
            f"TTS completed: {len(text)} chars, queue={infer_record['queue_time']:.2f}s, "
            f"infer={infer_record['infer_time']:.2f}s, format: {output_format}"
            + (f", duration={duration_headers.get('X-IndexTTS-Duration-Mode')}"
               f"->{duration_headers.get('X-IndexTTS-Actual-Duration')}s"
               f"/target {duration_headers.get('X-IndexTTS-Target-Duration')}s" if duration_headers else "")
        )

        audio_bytes, media_type = _encode_audio(wav, sr, output_format)
        resp_headers = {
            "X-IndexTTS-Voice": str(voice_label),
            "X-IndexTTS-Sample-Rate": str(sr),
            "X-IndexTTS-Queue-Time": f"{infer_record['queue_time']:.3f}",
            "X-IndexTTS-Infer-Time": f"{infer_record['infer_time']:.3f}",
            "X-IndexTTS-Total-Time": f"{infer_record['total_time']:.3f}",
            "X-IndexTTS-Output-Format": output_format,
        }
        resp_headers.update(duration_headers)
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers=resp_headers,
        )
    except TimeoutError as e:
        logger.warning(f"TTS queue timeout: {e}")
        return JSONResponse(status_code=429, content={"error": str(e)})
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Synthesis failed: {str(e)}"})


@app.post(
    "/v1/audio/speech",
    tags=["Speech Synthesis"],
    summary="OpenAI protocol speech synthesis",
    description=(
        "Local implementation compatible with OpenAI Speech API. "
        "Requires input and voice; model is optional and not validated; response_format supports mp3/opus/aac/flac/wav/pcm. "
        "Request enters the unified GPU inference queue, response headers return queue, inference, and total time."
    ),
    responses=AUDIO_RESPONSES,
)
async def openai_speech(
    request: OpenAISpeechRequest = Body(
        ...,
        title="OpenAI protocol speech synthesis request",
        description="Compatible with OpenAI Speech API. model field is optional and not strictly validated.",
    )
):
    """Speech synthesis (OpenAI Speech API specification)."""
    return await _speech_response_from_payload(request, text_field="input", default_format="mp3")


@app.post(
    "/tts",
    tags=["Plain Speech Synthesis"],
    summary="Plain JSON speech synthesis",
    description=(
        "Plain TTS interface. Requires text and voice; response_format can specify return format, defaults to wav. "
        "Supports the same emotion control and inference parameters as the OpenAI interface."
    ),
    responses=AUDIO_RESPONSES,
)
async def plain_tts(
    request: PlainTTSRequest = Body(
        ...,
        title="Plain TTS request",
        description="Plain JSON TTS interface. Uses the text field for synthesis text, response_format specifies return format.",
    )
):
    """Plain TTS interface. JSON input uses text, also compatible with voice/speaker_id/spk_audio_prompt."""
    return await _speech_response_from_payload(request, text_field="text", default_format="wav")


# ============== WebSocket ==============

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket streaming synthesis"""
    await manager.connect(ws)
    try:
        await manager.send_json({"type": "connected", "message": "IndexTTS2 WebSocket"}, ws)

        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            try:
                if msg_type in ("tts", "tts_stream"):
                    if tts is None:
                        await manager.send_json({"type": "error", "message": "Model not initialized"}, ws)
                        continue

                    text = data.get("text", "")
                    if not text or not text.strip():
                        await manager.send_json({"type": "error", "message": "text cannot be empty"}, ws)
                        continue

                    try:
                        params = _extract_params(data)
                    except ValueError as e:
                        await manager.send_json({"type": "error", "message": str(e)}, ws)
                        continue
                    try:
                        dctl = _extract_duration_controls(data)
                    except ValueError as e:
                        await manager.send_json({"type": "error", "message": str(e)}, ws)
                        continue
                    native_used = _apply_native_duration(params, dctl)
                    quick_streaming_tokens = int(data.get("quick_streaming_tokens") or 60)
                    quick_streaming_tokens = max(0, min(quick_streaming_tokens, 240))

                    spk = None
                    if data.get("spk_audio_base64"):
                        try:
                            spk = await _register_audio_from_base64(data["spk_audio_base64"], data.get("voice"))
                        except ValueError as e:
                            await manager.send_json({"type": "error", "message": str(e)}, ws)
                            continue
                            
                    if not spk:
                        spk = _resolve_speaker(data) or "examples/voice_01.wav"
                        
                    output_format = data.get("response_format") or "wav"
                    if output_format not in SUPPORTED_RESPONSE_FORMATS:
                        await manager.send_json({
                            "type": "error",
                            "message": f"Unsupported format: {output_format}, options {', '.join(SUPPORTED_RESPONSE_FORMATS)}",
                        }, ws)
                        continue

                    if msg_type == "tts_stream":
                        output_path = _unique_output_path("ws_stream_")
                        await _guarded_stream_to_ws(
                            ws,
                            text.strip(),
                            spk,
                            output_path,
                            params,
                            output_format,
                            quick_streaming_tokens,
                            data.get("trim_silence", True),
                            dctl,
                            native_used,
                        )
                        continue

                    output_path = _unique_output_path("ws_")
                    infer_record = await _guarded_infer(text, spk, output_path, params)
                    result = infer_record["result"]

                    if result:
                        sr, wav = result
                        # Auto trim silence and trailing noise if requested
                        if data.get("trim_silence", True):
                            wav = trim_audio_array(wav, sr)

                        # Duration control: fit-to-target stretch (after trim)
                        wav, duration_headers = _apply_fit_and_headers(wav, sr, dctl, native_used)

                        audio_bytes, media_type = _encode_audio(wav, sr, output_format)

                        completed_msg = {
                            "type": "stream_completed" if msg_type == "tts_stream" else "completed",
                            "audio_base64": base64.b64encode(audio_bytes).decode(),
                            "format": output_format,
                            "media_type": media_type,
                            "sample_rate": sr,
                            "queue_time": round(infer_record["queue_time"], 3),
                            "infer_time": round(infer_record["infer_time"], 3),
                            "total_time": round(infer_record["total_time"], 3),
                        }
                        if duration_headers:
                            completed_msg["duration_mode"] = duration_headers.get("X-IndexTTS-Duration-Mode")
                            completed_msg["target_duration"] = float(duration_headers["X-IndexTTS-Target-Duration"])
                            completed_msg["actual_duration"] = float(duration_headers["X-IndexTTS-Actual-Duration"])
                        await manager.send_json(completed_msg, ws)
                    else:
                        await manager.send_json({"type": "error", "message": "Synthesis failed"}, ws)

                elif msg_type == "ping":
                    await manager.send_json({"type": "pong"}, ws)

                elif msg_type == "get_voices":
                    await manager.send_json({"type": "voices_list", "voices": _load_meta()}, ws)

            except Exception as e:
                logger.error(f"WebSocket message processing failed: {e}")
                await manager.send_json({"type": "error", "message": str(e)}, ws)

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ============== Mount WebUI to Root Route ==============

import gradio as gr

def _get_tts():
    return tts

# Ensure Gradio enables queueing before mounting, to support multi-user concurrency and progress bars
webui_enhanced.demo.queue(20)
app = gr.mount_gradio_app(app, webui_enhanced.demo, path="/")


# ============== Main Program ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listening address")
    parser.add_argument("--port", type=int, default=8002, help="Listening port")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=int(os.environ.get("INDEXTTS_MAX_CONCURRENCY", "1")),
        help="Number of requests entering model inference simultaneously. Single GPU suggested to start from 1.",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=int(os.environ.get("INDEXTTS_QUEUE_SIZE", "16")),
        help="Inference queue wait slots, returns queue timeout error if exceeded.",
    )
    parser.add_argument(
        "--queue_timeout",
        type=float,
        default=float(os.environ.get("INDEXTTS_QUEUE_TIMEOUT", "120")),
        help="Maximum seconds a request waits to enter the inference queue.",
    )
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed GPT inference acceleration")
    parser.add_argument("--accel", action="store_true", help="Enable project's custom GPT accel engine")
    parser.add_argument("--cuda_kernel", action="store_true", help="Enable custom CUDA kernel for GPT inference")
    args = parser.parse_args()
    args.max_concurrency = max(1, args.max_concurrency)
    args.queue_size = max(0, args.queue_size)
    args.queue_timeout = max(0.1, args.queue_timeout)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)

    logger.info(f"IndexTTS2 API Server - http://{args.host}:{args.port}")
    logger.info(f"  WebUI: http://{args.host}:{args.port}/")
    logger.info(f"  API:   http://{args.host}:{args.port}/docs")
    logger.info(f"Device: {args.device}, FP16: {args.fp16}")

    uvicorn.run(app, host=args.host, port=args.port, workers=1)