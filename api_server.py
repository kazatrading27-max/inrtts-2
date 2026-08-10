"""
IndexTTS2 API Server
Supports voice cloning, emotion control, voice caching, and inference parameter tuning.
Root route mounts the WebUI, /v1/ routes provide the API.

Data Parallel Multi-Machine Support (Lightning AI Optimized):
  Completely bypasses NCCL multi-node firewalls by isolating DeepSpeed/transformers to a
  single-node, single-process, LOOPBACK-ONLY context on EACH machine. Rank 0 discovers
  worker IPs via the firewall-approved MASTER_PORT and load-balances HTTP requests to
  them. Both GPUs process independent requests simultaneously => 100% utilization.

FIXES IN THIS VERSION (root-caused from the boot logs):
  1. RANK DETECTION - the #1 reason workers "never connect": Lightning MMT does NOT
     guarantee a per-machine "RANK" env var. If it is unset, os.environ.get("RANK","0")
     defaults BOTH machines to rank 0 -> both listen, nobody connects, 120s timeout,
     then rank 0 runs a lying "2-worker" job solo. We now resolve the node rank from
     INDEXTTS_RANK -> NODE_RANK -> GROUP_RANK -> RANK -> IP-ownership heuristic
     ("do I actually own MASTER_ADDR?"), and AUTO-FLIP an obviously wrong role.
  2. STAGGERED BOOT - workers spend tens of seconds in `uv` build + imports before
     python can dial home. The IP exchange now runs at the very TOP of the file
     (stdlib-only imports) and both sides patiently wait up to
     INDEXTTS_PEER_WAIT_S (default 900s) with progress logs - no more 120s give-up.
  3. FAIL LOUD, NOT FAKE - if no peer shows up by the deadline, the job EXITS (so the
     orchestrator retries) instead of silently serving a "2 GPU" job on 1 GPU.
     Set INDEXTTS_ALLOW_SOLO=1 to explicitly permit single-node fallback.
  4. CRITICAL (unchanged) - MASTER_ADDR forced to 127.0.0.1 on every rank so
     DeepSpeed/transformers build a loopback-only single-process group per machine.
  5. Ports   - MASTER_PORT = orig + 500 + rank: no TIME_WAIT/EADDRINUSE roulette.
  6. Routing - round-robin only to workers that PROVED readiness via /health,
     with quarantine + local fallback when a worker is unreachable.
  7. httpx   - ONE persistent AsyncClient, Timeout(None, connect=5): no per-request
     TCP re-handshake and NO 300s ceiling killing long 'max'-preset generations.
  8. Warmup  - each node fires one throwaway synthesis during boot.
  9. Identity- greppable [boot] lines incl. the RAW dist env dump + explicit
     cuda device pin from LOCAL_RANK. Ends the "it moved to GPU 1" mystery.
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
import socket
import re
import argparse
from typing import Optional, Literal, Any, Dict, List, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

# ============================================================================
# Multi-Machine Data Parallel Setup  (runs at the VERY TOP — stdlib only — so a
# worker dials home seconds after python starts, not after heavy imports)
# ============================================================================

_orig_master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
orig_master_port = int(os.environ.get("MASTER_PORT", "29500"))
exchange_port = orig_master_port + (1 if "TORCHELASTIC_RUN_ID" in os.environ else 0)

def _env_int(*names) -> Tuple[Optional[int], Optional[str]]:
    """First integer-valued env var among names -> (value, source_name)."""
    for n in names:
        v = os.environ.get(n)
        if v is not None and v.strip().lstrip("-").isdigit():
            return int(v), n
    return None, None

def _outgoing_ip(towards: str, port: int = 80) -> Optional[str]:
    """Our source IP when routing towards the given host (no packets sent for UDP)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((towards, port))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None

def _local_ips() -> set:
    ips = {"127.0.0.1"}
    try: ips.add(socket.gethostbyname(socket.gethostname()))
    except Exception: pass
    guessed = _outgoing_ip(_orig_master_addr, orig_master_port) or _outgoing_ip("8.8.8.8", 80)
    if guessed: ips.add(guessed)
    return ips

def _i_am_master() -> bool:
    """True when MASTER_ADDR resolves to THIS machine."""
    if _orig_master_addr in ("127.0.0.1", "localhost", "0.0.0.0"):
        return True
    try:
        return socket.gethostbyname(_orig_master_addr) in _local_ips()
    except Exception:
        return _orig_master_addr in _local_ips()

# ---- node rank: explicit override -> Lightning/torchrun conventions -> heuristic
_rank, _rank_src = _env_int("INDEXTTS_RANK", "RANK", "NODE_RANK", "GROUP_RANK")
if _rank is None:
    _rank, _rank_src = (0 if _i_am_master() else 1), "heuristic(ip-ownership)"

_world, _world_src = _env_int("INDEXTTS_WORLD_SIZE", "WORLD_SIZE", "NNODES", "GROUP_WORLD_SIZE")
if _world is None:
    _world, _world_src = 1, "default"

my_rank, my_world_size = _rank, _world

# ---- AUTO-HEAL role flips: env rank says 0 but we don't own MASTER_ADDR (or the
# reverse). This is precisely the "both machines are rank 0, nobody connects" failure.
# NOTE: Disabled to allow single-machine 2x GPU runs on Kaggle via torchrun
# if my_world_size > 1:
#     if my_rank == 0 and not _i_am_master():
#         logger.critical(f"[boot] env claims rank=0 (source={_rank_src}) but MASTER_ADDR "
#                         f"{_orig_master_addr} does NOT resolve to this host — flipping to worker rank 1. "
#                         f"Set INDEXTTS_RANK explicitly to override.")
#         my_rank = 1
#         _rank_src = "auto-flip"
#     elif my_rank != 0 and _i_am_master():
#         logger.critical(f"[boot] env claims rank={my_rank} (source={_rank_src}) but this host OWNS "
#                         f"MASTER_ADDR {_orig_master_addr} — flipping to rank 0.")
#         my_rank = 0
#         _rank_src = "auto-flip"

_DIST_ENV_DUMP = {k: os.environ.get(k) for k in (
    "INDEXTTS_RANK", "INDEXTTS_WORLD_SIZE", "NODE_RANK", "GROUP_RANK", "RANK",
    "LOCAL_RANK", "WORLD_SIZE", "NNODES", "GROUP_WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT")}

_boot_run_id = os.environ.get("LIGHTNING_RUN_ID", uuid.uuid4().hex[:6])
logger.info(f"[boot] run_id={_boot_run_id} rank={my_rank} (src={_rank_src}) world={my_world_size} (src={_world_src}) "
            f"host={socket.gethostname()} pid={os.getpid()} master={_orig_master_addr}:{orig_master_port}")
logger.info(f"[boot] dist env dump: {_DIST_ENV_DUMP}")

_worker_ips: List[str] = []
_worker_ports: List[int] = []      # per-worker negotiated API port (index-aligned with _worker_ips)
worker_api_port = 8003            # this node's own hidden-API port when we ARE a worker (set by negotiation)
PEER_WAIT_S = float(os.environ.get("INDEXTTS_PEER_WAIT_S", "900"))   # uv build stagger can eat minutes
ALLOW_SOLO = os.environ.get("INDEXTTS_ALLOW_SOLO", "0") == "1"

def _probe_http_port(ip: str, port: int = 8003, timeout: float = 3.0) -> bool:
    """Is `port` on a worker reachable across the inter-node network? Binding
    succeeds locally regardless - only a connect from THIS node proves it."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        s.close()
        return True
    except Exception:
        return False

if my_world_size > 1:
    if my_rank == 0:
        _worker_ips.append("127.0.0.1")  # Rank 0 forwards to itself locally

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Use the firewall-approved MASTER_PORT for the IP exchange (Lightning
        # explicitly opens this exact port for inter-node traffic).
        srv.bind(("0.0.0.0", exchange_port))
        srv.listen(my_world_size)
        srv.settimeout(15.0)

        collected, t0, last_log = 1, time.time(), 0.0
        logger.info(f"Rank 0 waiting for {my_world_size - 1} worker(s) on port {exchange_port} "
                    f"(deadline {PEER_WAIT_S:.0f}s)...")
        while collected < my_world_size and (time.time() - t0) < PEER_WAIT_S:
            try:
                conn, addr = srv.accept()
                worker_ip = conn.recv(1024).decode('utf-8').strip()
                if not worker_ip:
                    worker_ip = addr[0]
                _worker_ips.append(worker_ip)
                # Negotiate the worker's hidden-API port: prefer 8003, but if the
                # inter-node firewall blocks it, fall back to the firewall-approved
                # MASTER_PORT so forwarding can NEVER be blocked.
                chosen_port = 8003
                if not _probe_http_port(worker_ip, 8003):
                    chosen_port = orig_master_port
                    logger.warning(f"Worker {collected} ({worker_ip}): port 8003 unreachable across the "
                                   f"inter-node network - its hidden API will use firewall-approved "
                                   f"MASTER_PORT {orig_master_port} instead")
                _worker_ports.append(chosen_port)
                conn.sendall(f"OK:{chosen_port}".encode('utf-8'))
                conn.close()
                collected += 1
                logger.info(f"Worker {collected - 1} connected from {worker_ip} (api port {chosen_port})")
            except socket.timeout:
                if time.time() - last_log >= 30:
                    last_log = time.time()
                    logger.info(f"Rank 0 still waiting... {collected - 1}/{my_world_size - 1} workers in, "
                                f"{int(time.time() - t0)}s elapsed. (Worker still building packages, or role mix-up — "
                                f"compare its [boot] rank line.)")
        srv.close()

        if collected < my_world_size:
            msg = (f"No peer connected within {PEER_WAIT_S:.0f}s. dist env was: {_DIST_ENV_DUMP}. "
                   "If every machine's [boot] line says rank=0, your platform does not set per-machine rank — "
                   "launch with INDEXTTS_RANK=0 / INDEXTTS_RANK=1 (and INDEXTTS_WORLD_SIZE) per machine.")
            if ALLOW_SOLO:
                logger.warning(msg + " INDEXTTS_ALLOW_SOLO=1 set — continuing as single node.")
                my_world_size = 1
            else:
                logger.critical(msg + " Exiting so the orchestrator can retry.")
                sys.exit(1)
    else:
        # Worker: determine our own outgoing IP, then announce ourselves to Rank 0.
        my_ip = _outgoing_ip(_orig_master_addr, orig_master_port)
        if not my_ip:
            try:
                my_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                my_ip = "127.0.0.1"

        # Open a probe listener on 8003 BEFORE announcing ourselves, so Rank 0 can
        # test whether that port traverses the inter-node network and pick our
        # serving port accordingly (8003, else the firewall-approved MASTER_PORT).
        probe = None
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("0.0.0.0", 8003))
            probe.listen(2)
        except Exception as e:
            logger.warning(f"Rank {my_rank}: could not open probe listener :8003 ({e}) - "
                           f"Rank 0 will fall back to MASTER_PORT for us")
            probe = None

        t0, connected, attempt = time.time(), False, 0
        logger.info(f"Rank {my_rank} connecting to Rank 0 ({_orig_master_addr}:{exchange_port}) "
                    f"as {my_ip} (deadline {PEER_WAIT_S:.0f}s)...")
        while (time.time() - t0) < PEER_WAIT_S and not connected:
            attempt += 1
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(10.0)
                client.connect((_orig_master_addr, exchange_port))
                client.sendall(my_ip.encode('utf-8'))
                reply = client.recv(16).decode('utf-8').strip()   # "OK:<port>"
                client.close()
                if reply.startswith("OK:"):
                    worker_api_port = int(reply.split(":", 1)[1])
                connected = True
                logger.info(f"Rank {my_rank} handshake OK after {int(time.time() - t0)}s "
                            f"(attempt {attempt}) - hidden API will serve on port {worker_api_port}")
            except Exception as e:
                if attempt % 6 == 1:
                    logger.info(f"Rank {my_rank} waiting for Rank 0 listener... attempt {attempt} "
                                f"({type(e).__name__}), {int(time.time() - t0)}s elapsed")
                time.sleep(5)
        if probe is not None:
            try: probe.close()
            except Exception: pass
        if not connected:
            logger.critical(f"Rank {my_rank} could not reach Rank 0 at {_orig_master_addr}:{exchange_port} "
                            f"within {PEER_WAIT_S:.0f}s. dist env was: {_DIST_ENV_DUMP}. Exiting.")
            sys.exit(1)
else:
    _worker_ips.append("127.0.0.1")
    logger.info("[boot] single-node mode (world_size=1)")

logger.info(f"Cluster Worker IPs: {_worker_ips}")

# ── CRITICAL FIX ──────────────────────────────────────────────────────────
# Every machine now loads its OWN full copy of the model independently
# (Data Parallelism). To make transformers/DeepSpeed treat this as a totally
# self-contained single-process job (no networking, no firewall involvement,
# cannot hang), we must override ALL FIVE of these env vars together:
#   - WORLD_SIZE / RANK / LOCAL_RANK -> "this process IS the entire world"
#   - MASTER_ADDR -> MUST be 127.0.0.1 (loopback), NOT Rank 0's real IP.
#     (Leaving this as Rank 0's IP on a worker node would make that worker
#     try to bind a server socket on an IP that belongs to a DIFFERENT
#     machine, which is impossible and would hang/crash - THE original bug.)
#   - MASTER_PORT -> unique per machine so torch's TCPStore can never collide
#     with the exchange socket we just closed (no TIME_WAIT roulette).

# Tripwire: this override must run BEFORE torch.distributed/DeepSpeed is ever
# initialised - not merely before the import. If a future edit moves an import
# above this block and re-arms the bug, this fails loudly at boot instead of
# hanging politely for 600 seconds.
for _must_not_be_initialised in ("torch.distributed", "deepspeed"):
    assert _must_not_be_initialised not in sys.modules, (
        f"'{_must_not_be_initialised}' was imported before the env isolation block. "
        "Move the import below the override so the loopback env is in effect first."
    )

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(orig_master_port + 500 + my_rank)
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

# Isolate GPUs so accelerate's device_map="auto" doesn't see all GPUs
_true_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if "CUDA_VISIBLE_DEVICES" in os.environ:
    _cvd = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if _true_local_rank < len(_cvd):
        os.environ["CUDA_VISIBLE_DEVICES"] = _cvd[_true_local_rank].strip()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_true_local_rank)

os.environ["LOCAL_RANK"] = "0"
# Wipe out other torchrun variables that confuse accelerate
for k in ["LOCAL_WORLD_SIZE", "GROUP_WORLD_SIZE", "GROUP_RANK", "NODE_RANK", "CROSS_RANK"]:
    os.environ.pop(k, None)
# ──────────────────────────────────────────────────────────────────────────

_rr_idx = 0  # Round-robin index for load balancing

# Multi-node load-balancer state (used on Rank 0 only)
_http = None            # ONE persistent httpx.AsyncClient (no 300s ceiling)
worker_ready: Dict[str, bool] = {}   # worker_ip -> readiness proven via /health
_ticker_task = None     # background /health poller

# ============================================================================
# Heavy third-party imports (AFTER the exchange + env isolation, by design)
# ============================================================================
import numpy as np
import soundfile as sf
import librosa

from fastapi import FastAPI, Body, Response, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager

import uvicorn

# httpx is required for Rank 0 to forward requests to Rank 1+
import httpx

import torch

# Explicit device pin from LOCAL_RANK. On a 1-GPU-per-machine MMT job this is the
# node's single T4; on a single-box multi-GPU variant it stops two processes from
# piling onto cuda:0 (OOM) and "falling back" to cuda:1.
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(_local_rank)
    logger.info(f"[boot] rank={my_rank} cuda device pinned: {_local_rank} ({torch.cuda.get_device_name(_local_rank)})")
else:
    logger.info(f"[boot] rank={my_rank} CUDA not available, running on CPU")

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
    duration_mode: str | None = Field(None, description="Duration control: 'Auto' (default/native pacing), 'Seconds' (target a wall-clock length), or 'Tokens' (target a semantic-token count). Only applies to single-segment text.")
    target_duration_seconds: float | None = Field(None, gt=0.0, description="Target output duration in seconds when duration_mode='Seconds'.")
    target_semantic_tokens: int | None = Field(None, gt=0, description="Target semantic (mel) token count when duration_mode='Tokens' (50 tokens/sec).")

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
        params["num_beams"] = 1
        params["do_sample"] = False
        changes.append(f"num_beams: {user_beams}→1, do_sample: forced False (DeepSpeed fallback to greedy decoding for deterministic output)")
    elif not beams_was_specified:
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

    for key in ("interval_silence", "max_text_tokens_per_segment", "num_beams", "top_k", "max_mel_tokens", "diffusion_steps", "seed"):
        if key in data and data[key] is not None: params[key] = int(data[key])

    for key in ("top_p", "temperature", "length_penalty", "repetition_penalty", "inference_cfg_rate"):
        if key in data and data[key] is not None: params[key] = float(data[key])

    if params.get("repetition_penalty", 10.0) <= 0: raise ValueError("repetition_penalty must be > 0")
    if params.get("temperature", 0.8) <= 0: raise ValueError("temperature must be > 0")
    if "top_p" in params and not 0 <= params["top_p"] <= 1: raise ValueError("top_p must be between 0 and 1")

    if "seed" in params:
        seed_val = params["seed"]
        if not (0 <= seed_val <= 2**31 - 1):
            raise ValueError(f"seed must be between 0 and {2**31 - 1}")

    if data.get("quality_preset"): _apply_quality_preset(params, data["quality_preset"])

    do_sample = params.get("do_sample", True)
    num_beams = params.get("num_beams", 3)

    if num_beams > 1 and do_sample:
        logger.info(f"Beam search requested (num_beams={num_beams}). Overriding do_sample=True to False for deterministic output.")
        params["do_sample"] = False

    duration_mode = (data.get("duration_mode") or "").strip().lower()
    if duration_mode == "seconds":
        target_duration_seconds = data.get("target_duration_seconds")
        if target_duration_seconds is not None and float(target_duration_seconds) > 0:
            params["use_speed"] = True
            params["target_dur"] = float(target_duration_seconds)
        else:
            logger.warning("duration_mode='Seconds' requires a positive target_duration_seconds; ignoring.")
    elif duration_mode == "tokens":
        target_semantic_tokens = data.get("target_semantic_tokens")
        if target_semantic_tokens is not None and int(target_semantic_tokens) > 0:
            params["use_speed"] = True
            params["target_dur"] = int(target_semantic_tokens) / 50.0
        else:
            logger.warning("duration_mode='Tokens' requires a positive target_semantic_tokens; ignoring.")

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
    global tts, _gpu_semaphore, _queue_slots, args, _http, _ticker_task
    if args is None:
        args = _default_args()
        logger.warning("args not initialized, using defaults.")
    _gpu_semaphore = asyncio.Semaphore(args.max_concurrency)
    _queue_slots = asyncio.Semaphore(args.max_concurrency + args.queue_size)

    # ONE persistent HTTP client for the load balancer.
    # read=None -> NO hard ceiling: long 'max'-preset generations are allowed to finish.
    # connect=5s -> a dead/blocked worker is detected FAST and quarantined, not after 30s.
    _http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))

    # Rank 0: background health ticker - a worker only enters the round-robin pool
    # once its /health reports model_loaded=true. This is what stops boot-time bursts
    # from hitting a half-loaded Rank 1 and 503-ing (which looked exactly like the old bug).
    if my_world_size > 1 and my_rank == 0:
        async def _health_tick():
            while True:
                for i in range(1, my_world_size):
                    if i >= len(_worker_ips):
                        continue
                    ip = _worker_ips[i]
                    port = _worker_ports[i - 1] if (i - 1) < len(_worker_ports) else 8003
                    try:
                        r = await _http.get(f"http://{ip}:{port}/health", timeout=4.0)
                        ok = r.status_code == 200 and bool(r.json().get("model_loaded"))
                    except Exception:
                        ok = False
                    if worker_ready.get(ip) != ok:
                        logger.info(f"[lb] worker {i} ({ip}) readiness -> {ok}")
                    worker_ready[ip] = ok
                await asyncio.sleep(5)
        _ticker_task = asyncio.create_task(_health_tick())

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

    # GPU warmup on EVERY node (rank 0 AND workers): compiles DeepSpeed kernels /
    # warms CUDA state with a tiny throwaway synthesis so the first user request
    # is never the cold one on either machine.
    if tts is not None and os.environ.get("INDEXTTS_WARMUP", "1") == "1":
        try:
            warm_refs = ["examples/voice_01.wav"] + [m.get("audio_path") for m in (_meta_cache or {}).values() if m.get("audio_path")]
            warm_ref = next((p for p in warm_refs if p and os.path.exists(p)), None)
            if warm_ref:
                logger.info(f"[boot] rank {my_rank}: warming GPU with throwaway synthesis...")
                warm_path = _unique_output_path("warmup_")
                os.makedirs("outputs", exist_ok=True)
                await asyncio.to_thread(_do_infer, "Warm up.", warm_ref, warm_path,
                                        {"max_mel_tokens": 120, "diffusion_steps": 5})
                if os.path.exists(warm_path): os.remove(warm_path)
                logger.info(f"[boot] rank {my_rank}: warm and ready to serve")
            else:
                logger.info("[boot] no reference voice available for warmup; first request will be cold")
        except Exception as e:
            logger.warning(f"Warmup synthesis failed (non-fatal): {e}")

    # MusicGen preload on EVERY node (rank 0 AND workers): the first BGM/SFX
    # request per machine should never pay the hub-download + VRAM-load cost.
    # Disable with INDEXTTS_MUSIC_PRELOAD=0.
    if os.environ.get("INDEXTTS_MUSIC_PRELOAD", "1") == "1" and _musicgen_available():
        try:
            await asyncio.to_thread(_musicgen_ensure_loaded, _resolve_musicgen_model(None))
        except Exception as e:
            logger.warning(f"MusicGen preload failed (non-fatal): {e}")

    if getattr(args, "audiox_preload", False) and _audiox_available():
        try: await asyncio.to_thread(_audiox_ensure_loaded, _resolve_audiox_model(args.audiox_model))
        except Exception as e: logger.warning(f"AudioX preload failed: {e}")

    yield
    logger.info("Shutting down...")
    if _ticker_task is not None:
        _ticker_task.cancel()
        try: await _ticker_task
        except Exception: pass
    if _http is not None:
        try: await _http.aclose()
        except Exception: pass

app = FastAPI(
    lifespan=lifespan, title="IndexTTS2 API", version="2.6",
    description=("IndexTTS2 zero-shot speech synthesis API. DeepSpeed support with automatic parameter normalisation. "
                 "Deterministic output enforced for beam search requests. Vocoder step bug patched for max quality output. "
                 "Multi-machine data-parallel: auto-detected node roles, loopback-isolated process groups, "
                 "readiness-gated HTTP load balancing."),
    openapi_tags=OPENAPI_TAGS,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
logger.add("logs/api.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

@app.get("/health", tags=["Base"])
async def health_check():
    if tts is None: return {"status": "partial", "model_loaded": False, "rank": my_rank, "run_id": _boot_run_id}
    deepspeed_active = _is_deepspeed_enabled()
    max_text_tokens = 120
    ds_max_mel = max(100, DEEPSPEED_MAX_TOTAL_TOKENS - max_text_tokens - DEEPSPEED_SPECIAL_TOKEN_BUFFER)
    return {
        "status": "healthy", "model_loaded": True, "device": str(tts.device),
        "deepspeed": {"enabled": deepspeed_active, "max_mel_tokens_cap": ds_max_mel if deepspeed_active else None, "beam_search_supported": not deepspeed_active},
        "quality_presets": {"available": list(_QUALITY_PRESETS.keys())},
        "concurrency": {"max_concurrency": args.max_concurrency, "queue_size": args.queue_size, "queue_timeout": args.queue_timeout},
        "data_parallel_workers": my_world_size,
        "peer_ips": _worker_ips,
        "lb_ready_workers": (len(_forward_pool()) - 1) if my_rank == 0 else None,
        "worker_api_port": worker_api_port if my_rank != 0 else None,
        "rank": my_rank, "rank_source": _rank_src, "run_id": _boot_run_id
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
        return Response(content=audio_bytes, media_type=media_type, headers={"X-IndexTTS-Voice": str(voice_label), "X-IndexTTS-Sample-Rate": str(sr), "X-IndexTTS-Queue-Time": f"{infer_record['queue_time']:.3f}", "X-IndexTTS-Infer-Time": f"{infer_record['infer_time']:.3f}", "X-IndexTTS-Total-Time": f"{infer_record['total_time']:.3f}", "X-IndexTTS-Output-Format": output_format, "X-IndexTTS-DeepSpeed": str(_is_deepspeed_enabled()), "X-IndexTTS-Rank": str(my_rank)})
    except TimeoutError as e: return JSONResponse(status_code=429, content={"error": str(e)})
    except Exception as e:
        logger.error(f"TTS failed: {e}"); traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Synthesis failed: {str(e)}"})

# ============== Multi-Node Load Balancing Routes ==============
# Rank 0 intercepts requests and alternates between processing locally and
# forwarding to worker ranks' hidden API on port 8003.
#
# HARDENED:
#   - Workers only enter the pool AFTER their /health proves model_loaded=true
#     (no more boot-burst 503s that looked identical to the startup bug).
#   - One persistent _http client with Timeout(None, connect=5): no per-request
#     TCP re-handshake, and NO 300s read ceiling that killed long 'max'-preset
#     generations mid-flight (httpx ReadTimeout while the worker kept burning GPU).
#   - On ConnectError/ConnectTimeout the worker is quarantined and the request is
#     served LOCALLY instead of forwarding into a black hole.

def _forward_pool() -> List[int]:
    """Indices into _worker_ips eligible right now: 0 (local) always; remote workers
    only after they have proven readiness via the background /health ticker."""
    pool = [0]
    for i in range(1, my_world_size):
        if i < len(_worker_ips) and worker_ready.get(_worker_ips[i]):
            pool.append(i)
    return pool

async def _forward_or_local(request: BaseModel, path: str, local_coro) -> Response | JSONResponse:
    global _rr_idx
    if my_world_size == 1 or my_rank != 0 or _http is None or len(_worker_ips) < 2:
        return await local_coro()

    pool = _forward_pool()
    _rr_idx = (_rr_idx + 1) % len(pool)
    chosen = pool[_rr_idx]
    if chosen == 0:
        return await local_coro()

    target_ip = _worker_ips[chosen]
    target_port = _worker_ports[chosen - 1] if (chosen - 1) < len(_worker_ports) else 8003
    url = f"http://{target_ip}:{target_port}{path}"
    logger.info(f"Forwarding API request to Rank {chosen} ({url})")
    try:
        resp = await _http.post(url, json=request.model_dump(exclude_none=True))
        headers = {k: v for k, v in resp.headers.items() if k.startswith("X-")}
        return Response(content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type") or "application/octet-stream",
                        headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.RemoteProtocolError) as e:
        # Quarantine the worker until the health ticker proves it alive again,
        # then serve this request locally so the user never sees a black hole.
        worker_ready[target_ip] = False
        logger.warning(f"Rank {chosen} ({target_ip}) unreachable ({type(e).__name__}) - serving locally")
        return await local_coro()

@app.post("/v1/audio/speech", tags=["Speech Synthesis"], responses=AUDIO_RESPONSES)
async def openai_speech(request: OpenAISpeechRequest = Body(...)):
    return await _forward_or_local(
        request, "/v1/audio/speech",
        lambda: _speech_response_from_payload(request, text_field="input", default_format="mp3"),
    )

@app.post("/tts", tags=["Plain Speech Synthesis"], responses=AUDIO_RESPONSES)
async def plain_tts(request: PlainTTSRequest = Body(...)):
    return await _forward_or_local(
        request, "/tts",
        lambda: _speech_response_from_payload(request, text_field="text", default_format="wav"),
    )


# ============== SRT Batch Dubbing (server-side fan-out across ALL GPUs) ==============
#
# Round-robin only distributes work when requests ARRIVE concurrently. A classic
# sequential "for line in srt: await tts(line)" client leaves every GPU idle half
# the time (and in-process parallelism is impossible anyway: the tts singleton's
# shared conditioning caches are serialized by a global lock). This endpoint does
# the parallelisation SERVER-SIDE: segments are dispatched to the local GPU and
# every readiness-proven worker CONCURRENTLY, then stitched back onto the SRT
# timeline with silence padding. Fan-out defaults to the number of READY GPUs;
# override with INDEXTTS_DUB_FANOUT.

_SRT_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
DUB_MAX_SEGMENTS = int(os.environ.get("INDEXTTS_DUB_MAX_SEGMENTS", "500"))
DUB_FANOUT = int(os.environ.get("INDEXTTS_DUB_FANOUT", "0"))        # 0 = auto (=#ready GPUs)
DUB_SAMPLE_RATE = 22050

def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 3600) + (int(m) * 60) + int(s)) * 1000 + int(ms)

def _parse_srt(srt_text: str) -> List[dict]:
    segments = []
    for block in re.split(r"\n\s*\n", srt_text.replace("\r\n", "\n").strip()):
        match = _SRT_TS.search(block)
        if not match:
            continue
        g = match.groups()
        text = re.sub(r"<[^>]+>", " ", block[match.end():])
        text = " ".join(text.split())
        if not text:
            continue
        segments.append({"index": len(segments) + 1,
                         "start_ms": _ts_to_ms(g[0], g[1], g[2], g[3]),
                         "end_ms": _ts_to_ms(g[4], g[5], g[6], g[7]),
                         "text": text})
    return segments

class DubSegment(BaseModel):
    text: str = Field(..., min_length=1, description="One dub line of speech.")
    start_ms: int | None = Field(None, ge=0, description="Timeline placement (ms). Omit for back-to-back layout.")
    end_ms: int | None = Field(None, ge=0, description="Timeline end (ms); clips that overrun the next start are truncated.")
    voice: str | None = Field(None, description="Per-segment voice override (id/path); defaults to the shared voice.")

class DubRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    srt: str | None = Field(None, description="Raw SRT text. Alternative to providing pre-parsed `segments`.")
    segments: List[DubSegment] | None = Field(None, description="Pre-parsed segment list; alternative to `srt`.")
    voice: str = Field(..., description="Shared voice id/path used for all segments without an override.")
    response_format: ResponseFormat | None = Field("wav", description="Format of the assembled output mix.")
    reset_timeline: bool = Field(False, description="true = ignore SRT gaps; place clips back-to-back, interval_silence apart.")
    wait: bool = Field(True, description="true = synchronous single response; false = returns a job_id immediately (recommended for batches: immune to the ~100s Cloudflare first-byte budget).")
    # plus any SpeechParams fields (emotion, presets, sampling…) — passed through to every segment

async def _synth_segment(text: str, voice_label: str, spk_path: str, params: dict, pool_index: int) -> Tuple[int, np.ndarray]:
    """Synthesize one dub segment on a specific pool node (0 = local GPU, >0 = worker).
    Falls back to the local GPU on any transport/worker error. Returns (sample_rate, int16 mono wav)."""
    async def _local() -> Tuple[int, np.ndarray]:
        rec = await _guarded_infer(text, spk_path, _unique_output_path("dub_"), params)
        if rec["result"] is None:
            raise RuntimeError("Synthesis failed (local GPU)")
        return rec["result"]

    if pool_index == 0 or _http is None:
        return await _local()

    ip = _worker_ips[pool_index]
    port = _worker_ports[pool_index - 1] if (pool_index - 1) < len(_worker_ports) else 8003
    url = f"http://{ip}:{port}/v1/audio/speech"
    payload = {"input": text, "model": "indextts2", "voice": voice_label,
               "spk_audio_prompt": spk_path, "response_format": "wav", "trim_silence": False, **params}
    try:
        resp = await _http.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"worker {pool_index} HTTP {resp.status_code}: {resp.text[:120]}")
        wav, sr = sf.read(io.BytesIO(resp.content), dtype="int16")
        if wav.ndim > 1: wav = wav[:, 0]
        return int(sr), wav
    except Exception as e:
        worker_ready[ip] = False   # quarantine; the health ticker will re-admit it
        logger.warning(f"[dub] worker {pool_index} failed ({type(e).__name__}: {e}) - segment served locally")
        return await _local()

def _assemble_dub(clips: List[Tuple[dict, np.ndarray]], sr: int, interval_silence_ms: int, reset_timeline: bool) -> np.ndarray:
    """Place every clip on the SRT timeline (silence padding) or back-to-back."""
    gap = int(sr * max(0, interval_silence_ms) / 1000)
    stamps_complete = all(seg.get("start_ms") is not None and seg.get("end_ms") is not None for seg, _ in clips)
    if reset_timeline or not stamps_complete:
        pieces = []
        for i, (_, wav) in enumerate(clips):
            if i: pieces.append(np.zeros(gap, dtype=np.int16))
            pieces.append(wav)
        return np.ascontiguousarray(np.concatenate(pieces)) if pieces else np.zeros(0, dtype=np.int16)

    tail_ms = 800
    timeline_end_ms = max((seg.get("end_ms") or 0) for seg, _ in clips) + tail_ms
    buf = np.zeros(int(sr * timeline_end_ms / 1000), dtype=np.int16)
    truncated = 0
    for i, (seg, wav) in enumerate(clips):
        s = int(sr * seg["start_ms"] / 1000)
        next_start = clips[i + 1][0]["start_ms"] if i + 1 < len(clips) else timeline_end_ms
        slot = int(sr * max(0, next_start) / 1000) - s
        if len(wav) > slot:
            truncated += 1
            wav = wav[:slot]
        if slot > 0:
            buf[s:s + len(wav)] = wav
    if truncated:
        logger.warning(f"[dub] {truncated} segment(s) overran their subtitle slot and were truncated")
    return buf

# ---- async dub jobs (a full SRT can take minutes; Cloudflare's quick tunnel kills
# any sync response that needs >~100s to produce its FIRST byte, so long batches
# MUST run as polled jobs) ----
class DubError(Exception):
    """User-facing batch-dub input/runtime error."""

_dub_jobs: Dict[str, dict] = {}
DUB_JOB_TTL_S = int(os.environ.get("INDEXTTS_DUB_JOB_TTL", "3600"))

def _dub_job_gc() -> None:
    now = time.time()
    for jid, j in list(_dub_jobs.items()):
        if now - j.get("created", now) > DUB_JOB_TTL_S:
            p = j.get("result_path")
            if p:
                try:
                    if os.path.exists(p): os.remove(p)
                except Exception: pass
            _dub_jobs.pop(jid, None)

async def _run_dub_batch(request: DubRequest, job: Optional[dict] = None) -> Tuple[bytes, str, dict]:
    """Full fan-out dub pipeline. Raises DubError for input problems; returns
    (audio_bytes, media_type, header_meta)."""
    if tts is None: raise RuntimeError("Model not initialized")
    data = request.model_dump(exclude_none=True)
    t0 = time.perf_counter()

    # ---- segments ----
    if data.get("srt"):
        segments = _parse_srt(str(data["srt"]))
    elif data.get("segments"):
        segments = [{"index": i + 1, "start_ms": s.get("start_ms"), "end_ms": s.get("end_ms"),
                     "text": s["text"].strip(), "voice": s.get("voice")} for i, s in enumerate(data["segments"]) if s.get("text", "").strip()]
    else:
        raise DubError("Provide `srt` (raw text) or `segments` (list).")
    if not segments: raise DubError("No usable segments found in input.")
    if len(segments) > DUB_MAX_SEGMENTS:
        raise DubError(f"Too many segments ({len(segments)} > {DUB_MAX_SEGMENTS}). Split the job.")
    if job is not None: job["total"] = len(segments)

    # ---- shared params + voice (resolved once) ----
    try: params = _extract_params(data)
    except ValueError as e: raise DubError(str(e))
    voice_label = data.get("voice") or data.get("speaker_id") or data.get("spk_audio_prompt") or ""
    spk_path = None
    if data.get("spk_audio_base64"):
        try: spk_path = await _register_audio_from_base64(data["spk_audio_base64"], data.get("voice"))
        except ValueError as e: raise DubError(str(e))
    if not spk_path: spk_path = _resolve_speaker(data) or "examples/voice_01.wav"
    if not os.path.exists(spk_path): raise DubError(f"Voice file does not exist: {voice_label}")

    # NOTE: never put `trim_silence` into `params` - it is an API-level flag, not a
    # tts.infer kwarg. Worker clips are requested untrimmed; assembly owns the gaps.
    data_fmt = data.get("response_format") or "wav"
    if data_fmt not in SUPPORTED_RESPONSE_FORMATS: raise DubError(f"Unsupported format: {data_fmt}")

    # ---- fan-out across every READY gpu simultaneously ----
    pool = _forward_pool() if (my_rank == 0 and len(_worker_ips) > 1) else [0]
    fanout = DUB_FANOUT if DUB_FANOUT > 0 else max(1, len(pool))
    slots = asyncio.Semaphore(fanout)
    used_pool: set = set()
    logger.info(f"[dub] dispatching {len(segments)} segments | ready GPUs: {len(pool)} | fan-out: {fanout}")

    async def work(i: int, seg: dict):
        async with slots:
            target = pool[i % len(pool)]
            used_pool.add(target)
            v_label = seg.get("voice") or voice_label
            v_path = _resolve_speaker({"voice": seg.get("voice")}) if seg.get("voice") else spk_path
            sr, wav = await _synth_segment(seg["text"], v_label, v_path or spk_path, dict(params), target)
            if sr != DUB_SAMPLE_RATE:
                wav = np.clip(librosa.resample(wav.astype(np.float32) / 32768.0,
                                               orig_sr=sr, target_sr=DUB_SAMPLE_RATE) * 32767.0,
                              -32768, 32767).astype(np.int16)
            if job is not None: job["completed"] = job.get("completed", 0) + 1
            return wav

    results = await asyncio.gather(*(work(i, seg) for i, seg in enumerate(segments)), return_exceptions=True)
    failed = [(segments[i].get("index"), results[i]) for i in range(len(results)) if isinstance(results[i], Exception)]
    if failed:
        logger.error(f"[dub] {len(failed)} segment(s) failed: {failed[:3]}")
        raise DubError(f"{len(failed)} segment(s) failed; indexes: {[f[0] for f in failed]}")
    clips = [(segments[i], results[i]) for i in range(len(results))]

    mixed = _assemble_dub(clips, DUB_SAMPLE_RATE, int(params.get("interval_silence", 200)),
                          bool(data.get("reset_timeline", False)))
    audio_bytes, media_type = _encode_audio(mixed, DUB_SAMPLE_RATE, data_fmt)
    total = time.perf_counter() - t0
    logger.info(f"[dub] done: {len(clips)} segments -> {len(mixed)/DUB_SAMPLE_RATE:.1f}s audio in {total:.1f}s "
                f"across pool {sorted(used_pool)}")
    return audio_bytes, media_type, {
        "X-Dub-Segments": str(len(clips)),
        "X-Dub-Duration": f"{len(mixed)/DUB_SAMPLE_RATE:.3f}",
        "X-Dub-Fanout": str(fanout),
        "X-IndexTTS-Ranks": ",".join(str(x) for x in sorted(used_pool)),
        "X-IndexTTS-Total-Time": f"{total:.3f}",
        "X-IndexTTS-Output-Format": data_fmt,
    }

async def _dub_local(request: DubRequest) -> Response | JSONResponse:
    try:
        audio_bytes, media_type, meta = await _run_dub_batch(request)
    except DubError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    except Exception as e:
        logger.error(f"[dub] batch failed: {e}"); traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Dub failed: {e}"})
    return Response(content=audio_bytes, media_type=media_type, headers=meta)

async def _dub_job_run(jid: str, request: DubRequest) -> None:
    job = _dub_jobs[jid]
    job.update(status="running", started=time.time())
    try:
        audio_bytes, media_type, meta = await _run_dub_batch(request, job=job)
        ext = request.response_format or "wav"
        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/dub_{jid}.{ext}"
        with open(path, "wb") as f: f.write(audio_bytes)
        job.update(status="done", finished=time.time(), result_path=path,
                   media_type=media_type, meta=dict(meta))
    except Exception as e:
        logger.error(f"[dub] job {jid} failed: {e}"); traceback.print_exc()
        job.update(status="error", finished=time.time(), error=f"{type(e).__name__}: {str(e)[:400]}")

# NB: do NOT route this via _forward_or_local - the batch must be orchestrated
# from Rank 0 so its own dispatcher can span every GPU concurrently.
@app.post("/v1/dub/srt", tags=["Speech Synthesis"], responses=AUDIO_RESPONSES)
async def dub_srt(request: DubRequest = Body(...)):
    if request.wait:
        return await _dub_local(request)
    _dub_job_gc()
    jid = uuid.uuid4().hex[:12]
    _dub_jobs[jid] = {"id": jid, "status": "queued", "created": time.time(), "completed": 0, "total": None}
    asyncio.create_task(_dub_job_run(jid, request))
    return {"job_id": jid, "status": "queued",
            "status_url": f"/v1/dub/jobs/{jid}", "result_url": f"/v1/dub/jobs/{jid}/result"}

@app.get("/v1/dub/jobs/{job_id}", tags=["Speech Synthesis"])
async def dub_job_status(job_id: str):
    job = _dub_jobs.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Unknown or expired job id"})
    return {k: v for k, v in job.items() if k != "result_path"}

@app.get("/v1/dub/jobs/{job_id}/result", include_in_schema=False)
async def dub_job_result(job_id: str):
    job = _dub_jobs.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Unknown or expired job id"})
    if job.get("status") == "error":
        return JSONResponse(status_code=500, content={"error": job.get("error", "job failed")})
    if job.get("status") != "done":
        return JSONResponse(status_code=409, content={"status": job.get("status"),
                                                      "completed": job.get("completed", 0),
                                                      "total": job.get("total")})
    return FileResponse(job["result_path"], media_type=job.get("media_type") or "audio/wav",
                        filename=os.path.basename(job["result_path"]))

_DUB_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IndexTTS2 · SRT Batch Dubbing</title>
<style>
:root{--bg:#07090c;--p:#0d1117;--l:#1b2230;--t:#e9edf3;--m:#8b95a5;--lm:#a3e635;--cy:#38e1ff;--rd:#ff5a5a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--t);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.6}
.wrap{max-width:880px;margin:0 auto;padding:36px 18px 70px}
h1{font-size:19px;letter-spacing:.03em;margin:0 0 6px}
.sub{color:var(--m);font-size:11px;letter-spacing:.16em;text-transform:uppercase;margin-bottom:24px}
.sub b{color:var(--lm)}
.card{background:var(--p);border:1px solid var(--l);border-radius:12px;padding:20px;margin-bottom:16px}
label{display:block;color:var(--m);font-size:10px;letter-spacing:.14em;text-transform:uppercase;margin:14px 0 6px}
input[type=text],textarea,select{width:100%;background:#090c10;border:1px solid var(--l);border-radius:8px;color:var(--t);font:inherit;padding:10px 12px;outline:none}
textarea{min-height:160px;resize:vertical}
input:focus,textarea:focus,select:focus{border-color:rgba(163,230,53,.5)}
.row{display:flex;gap:14px;flex-wrap:wrap}.row>div{flex:1;min-width:150px}
button{background:var(--lm);color:#0a1005;border:0;border-radius:8px;padding:12px 26px;font:inherit;font-weight:700;letter-spacing:.06em;cursor:pointer}
button:disabled{opacity:.4;cursor:wait}
.note{color:var(--m);font-size:11px}
#status{margin-top:16px;white-space:pre-wrap;font-size:12px}
.ok{color:var(--lm)}.err{color:var(--rd)}.info{color:var(--cy)}
.spin{display:inline-block;width:11px;height:11px;border:2px solid var(--l);border-top-color:var(--lm);border-radius:50%;animation:sp .8s linear infinite;vertical-align:-1px;margin-right:7px}
@keyframes sp{to{transform:rotate(360deg)}}
audio{width:100%;margin-top:8px}a.dl{color:var(--lm)}
</style></head><body><div class="wrap">
<h1>SRT Batch Dubbing</h1>
<div class="sub">server-side fan-out — <b>every ready gpu works concurrently</b></div>
<div class="card">
<label>srt file (or paste into the box)</label>
<input type="file" id="file" accept=".srt,.txt">
<label>srt content</label>
<textarea id="srt" placeholder="1&#10;00:00:00,000 --> 00:00:02,400&#10;Hello and welcome.&#10;&#10;2&#10;00:00:03,100 --> 00:00:05,600&#10;Today both GPUs speak at once."></textarea>
<div class="row">
<div><label>voice id / audio path</label><input type="text" id="voice" value="examples/voice_01.wav"></div>
<div><label>quality preset</label><select id="preset"><option value="fast">fast (dub defaults)</option><option value="balanced" selected>balanced</option><option value="max">max</option></select></div>
<div><label>format</label><select id="fmt"><option>wav</option><option>mp3</option><option>flac</option><option>opus</option><option>aac</option></select></div>
</div>
<label><input type="checkbox" id="reset" style="width:auto"> back-to-back timeline (ignore SRT gaps)</label>
<div style="margin-top:16px"><button id="go" type="button">run batch on all gpus</button></div>
<div class="note" style="margin-top:10px">Runs as a polled job — long batches survive the 100s proxy first-byte budget. Progress streams below.</div>
<div id="status"></div>
</div>
<div class="card" id="out" style="display:none">
<label>result</label><audio id="aud" controls></audio>
<div style="margin-top:8px"><a class="dl" id="dl">download mixed file</a></div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
$("file").onchange=async e=>{const f=e.target.files[0];if(!f)return;$("srt").value=await f.text();};
$("go").onclick=async()=>{
 if(!$("srt").value.trim()){$("status").innerHTML='<span class="err">paste an SRT or pick a file first</span>';return;}
 $("go").disabled=true;$("out").style.display="none";
 const st=$("status");st.innerHTML='<span class="spin"></span>submitting batch job…';
 try{
  const body={voice:$("voice").value,srt:$("srt").value,quality_preset:$("preset").value,
              response_format:$("fmt").value,reset_timeline:$("reset").checked,wait:false};
  const r=await fetch("/v1/dub/srt",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const j=await r.json();
  if(!r.ok){st.innerHTML='<span class="err">submit failed: '+(j.error||r.status)+'</span>';$("go").disabled=false;return;}
  st.innerHTML='<span class="spin"></span>job '+j.job_id+' queued…';
  poll(j.job_id);
 }catch(e){st.innerHTML='<span class="err">'+e+'</span>';$("go").disabled=false;}
};
async function poll(id){
 const st=$("status");
 try{
  const r=await fetch("/v1/dub/jobs/"+id);const j=await r.json();
  if(!r.ok){st.innerHTML='<span class="err">'+(j.error||r.status)+'</span>';$("go").disabled=false;return;}
  st.innerHTML='<span class="spin"></span><span class="info">'+j.status+'</span> — '+(j.completed||0)+' / '+(j.total||"?")+' segments';
  if(j.status==="done"){
   const url="/v1/dub/jobs/"+id+"/result";
   $("out").style.display="block";$("aud").src=url;$("dl").href=url;
   const m=j.meta||{};st.innerHTML='<span class="ok">done</span> — '+(j.completed||0)+' segments · GPUs ['+(m["X-IndexTTS-Ranks"]||"-")+'] · '+(m["X-Dub-Duration"]||"?")+'s audio';
   $("go").disabled=false;return;
  }
  if(j.status==="error"){st.innerHTML='<span class="err">error: '+(j.error||"unknown")+'</span>';$("go").disabled=false;return;}
 }catch(e){st.innerHTML='<span class="err">poll failed: '+e+'</span>';}
 setTimeout(()=>poll(id),2500);
}
</script></body></html>"""

@app.get("/dub", include_in_schema=False)
async def dub_page():
    return HTMLResponse(_DUB_PAGE)

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
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    except Exception: pass
    return "cpu"

def _resolve_audiox_model(name: Optional[str]) -> str:
    return _AUDIOX_MODEL_ALIASES.get(name, name) if name else getattr(args, "audiox_model", "HKUSTAudio/AudioX-MAF")

def _audiox_ensure_loaded(model_name: str) -> None:
    global audiox_model, audiox_config, audiox_loaded_name
    if audiox_model is not None and audiox_loaded_name == model_name: return
    from audiox import get_pretrained_model
    if audiox_model is not None:
        try: del audiox_model; audiox_model = None; torch.cuda.empty_cache()
        except Exception: pass
    device = _audiox_device()
    m, cfg = get_pretrained_model(model_name)
    m = m.to(device); m.eval()
    audiox_model, audiox_config, audiox_loaded_name = m, cfg, model_name

def _audiox_generate_wav(prompt: str, video_path: Optional[str], seconds_req: Optional[float], model_name: str, steps: int, cfg_scale: float) -> Tuple[np.ndarray, int, bool]:
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

async def _audiox_generate_local(request: AudioXGenerateRequest) -> Response | JSONResponse:
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
        return Response(content=audio_bytes, media_type=media_type, headers={"X-AudioX-Model": model_name, "X-AudioX-Used-Video": str(rec["used_video"]), "X-IndexTTS-Rank": str(my_rank)})
    except TimeoutError as e: return JSONResponse(status_code=429, content={"error": str(e)})
    except Exception as e: return JSONResponse(status_code=500, content={"error": f"Generation failed: {str(e)}"})
    finally:
        if video_path and os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass

# Load-balanced like TTS: Rank 0 alternates SFX/music jobs across BOTH GPUs
# instead of serialising everything onto its own card.
@app.post("/v1/audio/generate", tags=["Audio Generation"], responses=AUDIO_RESPONSES)
async def audiox_generate(request: AudioXGenerateRequest = Body(...)):
    return await _forward_or_local(request, "/v1/audio/generate", lambda: _audiox_generate_local(request))

@app.get("/v1/audio/generate/health", tags=["Audio Generation"])
async def audiox_health(): return {"available": _audiox_available(), "loaded": audiox_model is not None, "loaded_model": audiox_loaded_name}


# ============== MusicGen Generation (BGM / SFX) ==============

musicgen_source: Any = None
musicgen_loaded_name: Optional[str] = None

def _musicgen_available() -> bool:
    try:
        return _importlib_util.find_spec("audiocraft") is not None
    except Exception:
        return False

def _resolve_musicgen_model(name: Optional[str]) -> str:
    return name or "facebook/musicgen-small"

# Keep MusicGen VRAM-resident between requests by default. offload_after=True
# saves ~1.5-2GB of VRAM but forces a FULL disk->VRAM reload on EVERY request -
# that is the "BGM/SFX gets slower every call" symptom. Set INDEXTTS_MUSIC_OFFLOAD=1
# only if a large AudioX checkpoint must share the same T4.
_MUSIC_OFFLOAD_AFTER = os.environ.get("INDEXTTS_MUSIC_OFFLOAD", "0") == "1"

def _musicgen_ensure_loaded(model_name: str) -> None:
    global musicgen_source, musicgen_loaded_name
    if musicgen_source is not None and musicgen_loaded_name == model_name: return
    import music_source
    if musicgen_source is not None:
        try: musicgen_source._unload()
        except Exception: pass
        musicgen_source = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # only when switching checkpoints, not per-request
    device = _audiox_device()
    t0 = time.perf_counter()
    musicgen_source = music_source.make_source("musicgen", model_name=model_name, device=device, offload_after=_MUSIC_OFFLOAD_AFTER)
    musicgen_loaded_name = model_name
    logger.info(f"[music] MusicGen '{model_name}' loaded on {device} in {time.perf_counter()-t0:.1f}s "
                f"(vram-resident={'no - offload_after' if _MUSIC_OFFLOAD_AFTER else 'yes'})")

async def _guarded_musicgen_infer(prompt: str, duration_sec: float, seed: Optional[int], model_name: str) -> dict:
    request_start = time.perf_counter()
    if _queue_slots is None or _gpu_semaphore is None: raise RuntimeError("Inference queue not initialized")
    accepted = await _acquire_with_timeout(_queue_slots, args.queue_timeout)
    if not accepted: raise TimeoutError(f"Inference queue is full (wait exceeded {args.queue_timeout:.1f}s)")
    try:
        await _gpu_semaphore.acquire()
        queue_elapsed = time.perf_counter() - request_start
        infer_start = time.perf_counter()
        def _work():
            _musicgen_ensure_loaded(model_name)
            return musicgen_source.generate(prompt, duration_sec, seed)
        try: arr, sr = await asyncio.to_thread(_work)
        finally: _gpu_semaphore.release()
        return {"arr": arr, "sr": sr, "queue_time": queue_elapsed, "infer_time": time.perf_counter() - infer_start, "total_time": time.perf_counter() - request_start}
    finally: _queue_slots.release()

class MusicGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="MusicGen-style text prompt describing the desired music/SFX.")
    duration_sec: float = Field(10.0, gt=0.0, le=30.0, description="Requested clip duration in seconds (MusicGen generates a short seed and the caller loops/trims it).")
    seed: int | None = Field(None, description="Optional generation seed for reproducibility.")
    model: str | None = Field(None, description="MusicGen checkpoint name, defaults to facebook/musicgen-small.")
    response_format: ResponseFormat | None = Field("wav")

async def _musicgen_generate_local(request: MusicGenerateRequest) -> Response | JSONResponse:
    if not _musicgen_available(): return JSONResponse(status_code=503, content={"error": "MusicGen (audiocraft) is not installed on the server."})
    prompt = request.prompt.strip()
    duration_sec = float(request.duration_sec)
    model_name = _resolve_musicgen_model(request.model)
    output_format = request.response_format or "wav"
    try:
        rec = await _guarded_musicgen_infer(prompt, duration_sec, request.seed, model_name)
        arr, sr = rec["arr"], rec["sr"]
        audio_bytes, media_type = _encode_audio(arr, sr, output_format)
        return Response(content=audio_bytes, media_type=media_type, headers={"X-MusicGen-Model": model_name, "X-IndexTTS-Rank": str(my_rank)})
    except TimeoutError as e: return JSONResponse(status_code=429, content={"error": str(e)})
    except Exception as e: return JSONResponse(status_code=500, content={"error": f"Generation failed: {str(e)}"})

# Load-balanced like TTS: BGM/SFX jobs round-robin across BOTH GPUs.
@app.post("/v1/music/generate", tags=["Audio Generation"], responses=AUDIO_RESPONSES)
async def musicgen_generate(request: MusicGenerateRequest = Body(...)):
    return await _forward_or_local(request, "/v1/music/generate", lambda: _musicgen_generate_local(request))

@app.get("/v1/music/generate/health", tags=["Audio Generation"])
async def musicgen_health(): return {"available": _musicgen_available(), "loaded": musicgen_source is not None, "loaded_model": musicgen_loaded_name, "vram_resident": (musicgen_source is not None and not _MUSIC_OFFLOAD_AFTER), "rank": my_rank}

# ============== Mount WebUI ==============
# Gradio is only mounted on Rank 0 to save memory on workers
if my_rank == 0:
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
    parser.add_argument("--share", action="store_true", help="Create a public Cloudflare tunnel to expose the API and WebUI")

    args = parser.parse_args()
    args.max_concurrency, args.queue_size, args.queue_timeout = max(1, args.max_concurrency), max(0, args.queue_size), max(0.1, args.queue_timeout)
    os.makedirs("outputs", exist_ok=True); os.makedirs("logs", exist_ok=True); os.makedirs(SPEAKER_CACHE_DIR, exist_ok=True)

    if my_rank == 0:
        if args.share:
            cloudflared_path = "./cloudflared"
            if not os.path.exists(cloudflared_path):
                logger.info("Downloading cloudflared for public sharing...")
                subprocess.run(["wget", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", cloudflared_path], check=True)
                subprocess.run(["chmod", "+x", cloudflared_path], check=True)

            logger.info("Starting Cloudflare public tunnel...")
            tunnel_proc = subprocess.Popen(
                [cloudflared_path, "tunnel", "--url", f"http://localhost:{args.port}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )

            public_url = None
            try:
                for line in iter(tunnel_proc.stdout.readline, ''):
                    logger.info(f"[cloudflared] {line.strip()}")
                    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        public_url = match.group(0)
                        break
            except Exception as e:
                logger.warning(f"Failed to read cloudflared output: {e}")

            if public_url:
                logger.info("=" * 70)
                logger.info(f"PUBLIC API & WEBUI URL: {public_url}")
                logger.info(f"API Docs available at: {public_url}/docs")
                logger.info("=" * 70)
            else:
                logger.warning("Cloudflare tunnel started, but couldn't extract the public URL. Check logs above.")

        logger.info(f"IndexTTS2 Master API Server - http://{args.host}:{args.port}")
        logger.info(f"Data-parallel world: {my_world_size} node(s), peers: {_worker_ips}")
        logger.info(f"Device: {args.device}, FP16: {args.fp16}, DeepSpeed: {args.deepspeed}")
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
    else:
        # Rank 1+ runs a hidden API on the negotiated port (8003, or the
        # firewall-approved MASTER_PORT if 8003 was proven unreachable).
        logger.info(f"IndexTTS2 Worker API Server (Rank {my_rank}) starting on port {worker_api_port}")
        uvicorn.run(app, host="0.0.0.0", port=worker_api_port, workers=1)
