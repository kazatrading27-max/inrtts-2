"""
IndexTTS2 LitServe API Server (speed-optimized, headless)
=========================================================

A LitServe (Lightning AI) port of the hot inference path of api_server.py.
Designed for maximum throughput on Lightning studios / single-GPU boxes:

  - LitServe worker model: each worker owns its own IndexTTS2 instance and
    processes requests serially on the GPU (same guarantee as the semaphore
    queue in api_server.py, but with LitServe's zero-overhead event loop).
  - Optional multi-worker (`--workers N`) for multi-GPU or high-VRAM cards —
    each worker loads a full model copy (~VRAM x N), giving true parallel
    inference instead of a queue.
  - FP16 enabled by default (biggest single speedup on GPU).
  - No Gradio WebUI, no WebSocket, no AudioX — pure TTS inference. Use
    api_server.py when you need those; both share the same voice cache
    directory, so voice_ids work interchangeably.

Serves EXACTLY the routes DubStudio's Logic_TTS.py uses:

  POST /tts                   -> synthesis (JSON: text, voice/spk_audio_base64, params)
  POST /v1/audio/voices       -> register a voice (multipart upload) -> voice_id
  GET  /v1/audio/voices       -> list voices
  GET  /api/v1/health         -> health probe (same shape as api_server.py)
  GET  /health                -> LitServe built-in liveness

Run:
  pip install litserve
  python api_server_lit.py --model_dir ./checkpoints --fp16
  # multi-GPU / big-VRAM parallel inference:
  python api_server_lit.py --workers 2
"""
import os
import sys
import io
import uuid
import time
import json
import re
import base64
import hashlib
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# fcntl is Unix-only; fallback gracefully on Windows (though Lightning is Linux)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import librosa

import litserve as ls
from fastapi import Response, UploadFile, File, Form
from fastapi.responses import JSONResponse
from loguru import logger

# ============== Shared constants (same values/paths as api_server.py) ==============

SPEAKER_CACHE_DIR = "assets/speaker_cache"
SPEAKER_META_FILE = os.path.join(SPEAKER_CACHE_DIR, "meta.json")
SUPPORTED_RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

ARGS: argparse.Namespace = None  # populated in __main__


# ============== Voice cache (same meta.json format as api_server.py) ==============

_meta_cache: Optional[dict] = None
_meta_mtime: Optional[float] = None


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
    """Save meta.json with file locking to prevent multi-worker race conditions."""
    global _meta_cache, _meta_mtime
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
    lock_path = os.path.join(SPEAKER_CACHE_DIR, "meta.lock")
    
    with open(lock_path, "w") as lock_file:
        if HAS_FCNTL:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            with open(SPEAKER_META_FILE, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            _meta_cache = meta
            _meta_mtime = os.path.getmtime(SPEAKER_META_FILE)
        finally:
            if HAS_FCNTL:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def _register_audio_bytes(audio_bytes: bytes, speaker_name: str = "dynamic") -> str:
    """Cache raw audio bytes as a voice file, return its server path (md5-keyed)."""
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise ValueError("Audio file too large (exceeds 20MB)")
    md5 = hashlib.md5(audio_bytes).hexdigest()
    voice_id = f"spk_{md5[:8]}"
    meta = _load_meta()
    if voice_id in meta:
        return meta[voice_id]["audio_path"]
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
    final_path = os.path.join(SPEAKER_CACHE_DIR, f"{voice_id}.wav")
    buf = io.BytesIO(audio_bytes)
    wav_data, sr = sf.read(buf, dtype="int16")
    sf.write(final_path, wav_data, sr)
    meta[voice_id] = {
        "voice_name": speaker_name or voice_id,
        "audio_path": final_path,
        "md5": md5,
        "original_filename": "inline_base64",
        "created_at": time.time(),
        "embedding_cached": False,
    }
    _save_meta(meta)
    logger.info(f"Registered voice: {voice_id}")
    return final_path


def _resolve_speaker(data: dict) -> Optional[str]:
    speaker_id = data.get("speaker_id") or data.get("voice")
    if speaker_id:
        meta = _load_meta()
        if speaker_id in meta:
            return meta[speaker_id]["audio_path"]
        return speaker_id
    return data.get("spk_audio_prompt")


# ============== Audio post-processing (ported 1:1 from api_server.py) ==============

def _detect_silence_regions(y, sr, window_ms=25, rms_threshold=0.005):
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
    if in_silence:
        regions.append((silence_start, len(y)))
    return regions


def _rms_trim_edges(y, sr, rms_threshold=0.005, window_ms=25, margin_ms=80):
    if len(y) == 0:
        return y
    window_samples = max(1, int(sr * window_ms / 1000))
    step = max(1, window_samples // 2)
    end_pos = len(y)
    while end_pos > window_samples:
        seg = y[end_pos - window_samples: end_pos]
        if float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) > rms_threshold:
            break
        end_pos -= step
    start_pos = 0
    while start_pos + window_samples < end_pos:
        seg = y[start_pos: start_pos + window_samples]
        if float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) > rms_threshold:
            break
        start_pos += step
    margin_samples = int(sr * margin_ms / 1000)
    start_pos = max(0, start_pos - margin_samples)
    end_pos = min(len(y), end_pos + margin_samples)
    if end_pos <= start_pos:
        return y
    return y[start_pos:end_pos]


def _remove_trailing_noise(y, sr, min_gap_ms=200, max_trailing_ms=500, energy_ratio=0.3, rms_threshold=0.005):
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

    def _cut():
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


def _compress_internal_silence(y, sr, max_silence_ms=600, compress_to_ms=200, rms_threshold=0.005):
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
    pieces, prev_end = [], 0
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


def trim_audio_array(y, sr, top_db=30, margin_ms=80, max_internal_silence_ms=600, compress_silence_to_ms=200):
    if len(y) == 0:
        return y
    y_float = y.astype(np.float32) / 32768.0
    original_duration = len(y_float) / sr
    yt, _ = librosa.effects.trim(y_float, top_db=top_db)
    if len(yt) == 0:
        yt = y_float
    yt = _rms_trim_edges(yt, sr, margin_ms=margin_ms)
    yt = _remove_trailing_noise(yt, sr)
    yt = _compress_internal_silence(yt, sr, max_silence_ms=max_internal_silence_ms, compress_to_ms=compress_silence_to_ms)
    trimmed_duration = len(yt) / sr
    if len(yt) > int(sr * 0.1):
        removed = original_duration - trimmed_duration
        if removed > 0.05:
            logger.info(f"Trimmed audio: {original_duration:.2f}s -> {trimmed_duration:.2f}s")
        return np.clip(yt * 32767.0, -32768, 32767).astype(np.int16)
    return y


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
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *codec_args, "pipe:1"],
            input=wav_buf.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=30.0  # Hard prevent ffmpeg hangs on weird edge cases
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg transcoding timed out (>30s)")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcoding failed: {proc.stderr.decode('utf-8', errors='ignore')}")
    return proc.stdout, MEDIA_TYPES[fmt]


# ============== Inference parameter extraction & Text Sanitization ==============

# Tuned defaults to prevent EOS-loops while keeping generation quality high.
# max_text_tokens_per_segment: 80 limits chunk size so a stuck loop bounds damage.
# max_mel_tokens: 800 limits worst-case generation time per chunk to ~30s.
REQUEST_DEFAULTS = {
    "emo_alpha": 1.0,
    "use_emo_text": False,
    "use_random": False,
    "interval_silence": 200,
    "max_text_tokens_per_segment": 80,
    "num_beams": 3,
    "do_sample": True,
    "top_k": 30,
    "top_p": 0.8,
    "temperature": 0.8,
    "max_mel_tokens": 1700,
    "length_penalty": 0.0,
    "repetition_penalty": 10.0,
}

def _sanitize_text_for_tts(text: str) -> str:
    """Strip control chars, normalize whitespace, ensure terminal punctuation."""
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Force terminal punctuation to help the LLM reliably emit EOS
    if text and text[-1] not in '.!?…,;:':
        text += '.'
    return text


def _extract_params(data: dict) -> dict:
    def as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    params = {}
    data = {**REQUEST_DEFAULTS, **{k: v for k, v in data.items() if v is not None}}
    for key in ("emo_audio_prompt", "emo_alpha", "emo_vector", "use_emo_text", "emo_text", "use_random"):
        if key in data and data[key] is not None:
            params[key] = data[key]
    params["emo_alpha"] = float(params.get("emo_alpha", 1.0))
    if "emo_vector" in params:
        v = params["emo_vector"]
        if not isinstance(v, list) or len(v) != 8:
            raise ValueError("emo_vector must be an array of length 8")
        params["emo_vector"] = [float(x) for x in v]
    for key in ("use_emo_text", "use_random", "do_sample"):
        if key in params:
            params[key] = as_bool(params[key])
        elif key in data and data[key] is not None:
            params[key] = as_bool(data[key])
    for key in ("interval_silence", "max_text_tokens_per_segment", "num_beams", "top_k", "max_mel_tokens"):
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

    if params.get("do_sample", True) and params.get("num_beams", 3) > 1:
        params["num_beams"] = 1

    emo_p = params.get("emo_audio_prompt")
    if emo_p and not os.path.exists(emo_p):
        logger.warning(f"emo_audio_prompt not on server, ignoring: {emo_p!r}")
        params.pop("emo_audio_prompt", None)
    return params


# ============== LitServe API ==============

class IndexTTS2LitAPI(ls.LitAPI):
    """LitServe wrapper around IndexTTS2."""

    def __init__(self, config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)
        self.config = dict(vars(config))

    def setup(self, device):
        import torch
        from indextts.infer_v2 import IndexTTS2

        cfg = self.config
        model_dir = cfg["model_dir"]

        # ENHANCEMENT: Honor LitServe's mapped device. If the CLI is "auto",
        # we must trust LitServe's `device` argument for multi-GPU to work.
        # Otherwise workers spawn randomly or pile on cuda:0.
        if cfg["device"] == "auto":
            dev = device if device and device != "cpu" else None
        else:
            dev = cfg["device"]

        cuda_ok = torch.cuda.is_available()
        on_cpu = (dev == "cpu") or (dev is None and not cuda_ok)

        use_accel = cfg["accel"]
        if use_accel:
            try:
                import flash_attn  # noqa: F401
                has_flash = True
            except ImportError:
                has_flash = False
            if on_cpu or not has_flash:
                logger.warning(
                    f"[worker] --accel requested but "
                    f"{'running on CPU' if on_cpu else 'flash_attn is not installed'} "
                    f"— falling back to the standard engine (no accel)."
                )
                use_accel = False

        fp16 = cfg["fp16"]
        if on_cpu and fp16:
            logger.warning("[worker] FP16 requested on CPU — disabling (CPU inference needs FP32).")
            fp16 = False

        logger.info(
            f"[worker] Loading IndexTTS2 from {model_dir} "
            f"(cuda_available={cuda_ok}, device_target={dev}, fp16={fp16}, accel={use_accel}, cuda_kernel={cfg['cuda_kernel']})..."
        )
        t0 = time.perf_counter()
        self.tts = IndexTTS2(
            cfg_path=os.path.join(model_dir, "config.yaml"),
            model_dir=model_dir,
            use_fp16=fp16,
            device=dev,
            use_cuda_kernel=cfg["cuda_kernel"],
            use_deepspeed=cfg["deepspeed"],
            use_accel=use_accel,
            use_torch_compile=False,
        )
        logger.info(f"[worker] IndexTTS2 ready in {time.perf_counter() - t0:.1f}s (device={self.tts.device})")

    def decode_request(self, request: dict) -> dict:
        text = request.get("text") or request.get("input")
        if not text or not str(text).strip():
            raise ValueError("text is a required parameter and cannot be empty")

        # ENHANCEMENT: Sanitize text before sending to TTS to drastically reduce EOS-loops
        text = _sanitize_text_for_tts(str(text))

        spk = None
        if request.get("spk_audio_base64"):
            try:
                audio_bytes = base64.b64decode(request["spk_audio_base64"])
            except Exception:
                raise ValueError("Invalid base64 audio data")
            spk = _register_audio_bytes(audio_bytes, request.get("voice") or "dynamic_base64")
        if not spk:
            spk = _resolve_speaker(request) or "examples/voice_01.wav"
        if not os.path.exists(spk):
            raise ValueError(f"Voice file does not exist: {spk}. Upload via /v1/audio/voices first.")

        if request.get("emo_audio_base64"):
            try:
                emo_bytes = base64.b64decode(request["emo_audio_base64"])
                request["emo_audio_prompt"] = _register_audio_bytes(emo_bytes, "emo_base64")
            except Exception as e:
                raise ValueError(f"emo_audio_base64: {e}")

        output_format = request.get("response_format") or "wav"
        if output_format not in SUPPORTED_RESPONSE_FORMATS:
            raise ValueError(f"Unsupported format: {output_format}")

        return {
            "text": text,
            "spk": spk,
            "params": _extract_params(request),
            "format": output_format,
            "trim_silence": bool(request.get("trim_silence", True)),
            "voice_label": request.get("voice") or request.get("speaker_id") or spk,
        }

    def predict(self, x: dict) -> dict:
        t0 = time.perf_counter()
        output_path = f"outputs/lit_{uuid.uuid4().hex[:12]}.wav"
        
        # ENHANCEMENT: Hard watchdog to prevent worker poisoning when EOS fails.
        # If 800 mel tokens are requested, ~120s is a highly generous upper bound.
        max_mel = x["params"].get("max_mel_tokens", 800)
        max_wait = 30.0 + (max_mel * 0.15)  # base 30s + 0.15s per token
        
        result_holder = {}
        
        def _run_infer():
            try:
                result_holder["r"] = self.tts.infer(
                    spk_audio_prompt=x["spk"],
                    text=x["text"],
                    output_path=output_path,
                    **x["params"],
                )
            except Exception as e:
                result_holder["e"] = e

        infer_thread = threading.Thread(target=_run_infer, daemon=True)
        infer_thread.start()
        infer_thread.join(timeout=max_wait)

        if infer_thread.is_alive():
            # The CUDA kernel is still running and we can't kill it natively from Python.
            # os._exit(1) kills the worker process. LitServe's manager will respawn it 
            # instantly. The dying process sends a SIGTERM to its CUDA context, freeing VRAM.
            logger.error(
                f"Watchdog timeout ({max_wait:.0f}s) hit! Worker stuck on text: {x['text'][:80]!r}. "
                f"Forcing worker respawn via os._exit(1)."
            )
            os._exit(1)
            
        if "e" in result_holder:
            raise result_holder["e"]

        result = result_holder.get("r")
        if isinstance(result, tuple) and len(result) == 2:
            sr, wav = result
        elif isinstance(result, str) and os.path.exists(result):
            wav, sr = sf.read(result, dtype="int16")
        elif result is None and os.path.exists(output_path):
            wav, sr = sf.read(output_path, dtype="int16")
        else:
            raise RuntimeError("Synthesis failed, model returned empty result")

        if x["trim_silence"]:
            wav = trim_audio_array(np.asarray(wav), sr)

        infer_time = time.perf_counter() - t0
        logger.info(f"TTS completed: {len(x['text'])} chars, infer={infer_time:.2f}s")
        return {"sr": sr, "wav": wav, "infer_time": infer_time,
                "format": x["format"], "voice_label": x["voice_label"]}

    def encode_response(self, output: dict) -> Response:
        audio_bytes, media_type = _encode_audio(output["wav"], output["sr"], output["format"])
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-IndexTTS-Voice": str(output["voice_label"]),
                "X-IndexTTS-Sample-Rate": str(output["sr"]),
                "X-IndexTTS-Infer-Time": f"{output['infer_time']:.3f}",
                "X-IndexTTS-Output-Format": output["format"],
                "X-IndexTTS-Backend": "litserve",
            },
        )


# ============== Lightweight compat routes (no GPU work) ==============

def _register_compat_routes(server: "ls.LitServer"):
    app = server.app

    @app.get("/api/v1/health")
    @app.get("/v1/health")
    async def health():
        return {
            "status": "healthy",
            "model_loaded": True,
            "backend": "litserve",
            "timestamp": time.time(),
            "concurrency": {
                "strategy": "LitServe workers: one model instance per worker, serial per worker.",
                "workers_per_device": ARGS.workers,
            },
        }

    @app.post("/v1/audio/voices")
    async def upload_speaker(
        audio: UploadFile = File(...),
        speaker_name: str = Form(""),
    ):
        try:
            if not audio.filename:
                return JSONResponse(status_code=400, content={"error": "Please provide an audio file"})
            ext = os.path.splitext(audio.filename)[-1].lower()
            if ext not in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}:
                return JSONResponse(status_code=400, content={"error": f"Unsupported format: {ext}"})
            audio_bytes = await audio.read()
            if len(audio_bytes) < 1024:
                return JSONResponse(status_code=400, content={"error": "Audio file is too small"})
            if len(audio_bytes) > 20 * 1024 * 1024:
                return JSONResponse(status_code=400, content={"error": "Audio file is too large (exceeds 20MB)"})

            md5 = hashlib.md5(audio_bytes).hexdigest()
            voice_id = f"spk_{md5[:8]}"
            meta = _load_meta()
            if voice_id in meta:
                return {"voice_id": voice_id, "status": "exists", "message": "This audio is already registered"}

            os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
            final_path = os.path.join(SPEAKER_CACHE_DIR, f"{voice_id}{ext}")
            with open(final_path, "wb") as f:
                f.write(audio_bytes)
            meta[voice_id] = {
                "voice_name": speaker_name or voice_id,
                "audio_path": final_path,
                "md5": md5,
                "original_filename": audio.filename,
                "created_at": time.time(),
                "embedding_cached": False,
            }
            _save_meta(meta)
            logger.info(f"Voice registered: {voice_id} ({speaker_name})")
            return {"voice_id": voice_id, "md5": md5, "status": "new",
                    "message": "Voice registered successfully, use the voice parameter in /tts"}
        except Exception as e:
            logger.error(f"Failed to upload voice: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/v1/audio/voices")
    async def list_speakers():
        meta = _load_meta()
        return {"object": "list", "data": [
            {"voice_id": vid, "name": info.get("voice_name", ""),
             "original_filename": info.get("original_filename", ""),
             "created_at": info.get("created_at")}
            for vid, info in meta.items()
        ]}

    @app.delete("/v1/audio/voices/{voice_id}")
    async def delete_speaker(voice_id: str):
        meta = _load_meta()
        if voice_id not in meta:
            return JSONResponse(status_code=404, content={"error": f"Voice {voice_id} does not exist"})
        info = meta.pop(voice_id)
        _save_meta(meta)
        audio_path = info.get("audio_path", "")
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        return {"status": "deleted", "voice_id": voice_id}


# ============== Hardware auto-detection ==============

VRAM_PER_WORKER_GB = float(os.environ.get("INDEXTTS_VRAM_PER_WORKER_GB", "10"))
VRAM_RESERVE_GB = float(os.environ.get("INDEXTTS_VRAM_RESERVE_GB", "1.5"))


def autodetect_workers() -> int:
    """Pick worker count from available VRAM."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("[auto] No CUDA — 1 worker (CPU).")
            return 1
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        cpu_cores = os.cpu_count() or 1
        budget_gb = total_gb - VRAM_RESERVE_GB
        workers = max(1, int(budget_gb // VRAM_PER_WORKER_GB))
        workers = min(workers, max(1, cpu_cores // 2))
        used_pct = min(100, workers * VRAM_PER_WORKER_GB / total_gb * 100)
        logger.info(
            f"[auto] GPU: {props.name}, VRAM {total_gb:.1f} GB (reserve {VRAM_RESERVE_GB:.1f}), "
            f"CPU cores {cpu_cores} → {workers} worker(s) x ~{VRAM_PER_WORKER_GB:.0f} GB "
            f"(~{used_pct:.0f}% VRAM)"
        )
        return workers
    except Exception as e:
        logger.warning(f"[auto] Hardware detection failed ({e}) — 1 worker.")
        return 1


# ============== Main ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 LitServe API Server (fast headless inference)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listening address")
    parser.add_argument("--port", type=int, default=8002, help="Listening port (same default as api_server.py)")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", default=os.environ.get("INDEXTTS_FP16", "1") == "1",
                        help="Use FP16 (default ON here — biggest speed win; pass INDEXTTS_FP16=0 to disable)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("INDEXTTS_WORKERS", "0")),
                        help="LitServe workers per device. 0 = auto-detect from VRAM. "
                             ">1 loads N model copies for parallel inference.")
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("INDEXTTS_TIMEOUT", "300")),
                        help="Per-request timeout in seconds before LitServe evicts it.")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed GPT inference acceleration")
    parser.add_argument("--accel", action="store_true", help="Enable project's custom GPT accel engine")
    parser.add_argument("--cuda_kernel", action="store_true", help="Enable custom CUDA kernel for GPT inference")
    ARGS = parser.parse_args()
    if ARGS.workers <= 0:
        ARGS.workers = autodetect_workers()
    ARGS.workers = max(1, ARGS.workers)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)
    logger.add("logs/api_lit.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

    api = IndexTTS2LitAPI(ARGS, api_path="/tts")  # DubStudio's Logic_TTS.py posts here
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices="auto",
        workers_per_device=ARGS.workers,
        timeout=ARGS.timeout,
        track_requests=True,
    )
    _register_compat_routes(server)

    logger.info(f"IndexTTS2 LitServe Server - http://{ARGS.host}:{ARGS.port}")
    logger.info(f"  TTS:    POST http://{ARGS.host}:{ARGS.port}/tts")
    logger.info(f"  Voices: POST/GET http://{ARGS.host}:{ARGS.port}/v1/audio/voices")
    logger.info(f"  Health: GET  http://{ARGS.host}:{ARGS.port}/api/v1/health")
    logger.info(f"FP16: {ARGS.fp16}, workers/device: {ARGS.workers}, device: {ARGS.device}")

    server.run(host=ARGS.host, port=ARGS.port, generate_client_file=False)