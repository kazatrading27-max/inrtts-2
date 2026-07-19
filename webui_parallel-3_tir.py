from __future__ import annotations

import argparse
import atexit
import itertools
import inspect
import json
import logging
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import threading
import time
import gc
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# ARGUMENT PARSING — Must happen before any heavy imports
# ============================================================================
parser = argparse.ArgumentParser(description="IndexTTS2 Enhanced WebUI")
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--port", type=int, default=7862)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--model_dir", type=str, default="checkpoints")
parser.add_argument("--is_fp16", action="store_true", default=False)
parser.add_argument("--share", action="store_true", default=False)
cmd_args = parser.parse_args()

# ============================================================================
# ENVIRONMENT SETUP — Before any heavy imports that read env vars
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

hf_cache_dir = os.path.join(cmd_args.model_dir, "hf_cache")
torch_cache_dir = os.path.join(cmd_args.model_dir, "torch_cache")
os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")
os.environ.setdefault("HF_HOME", hf_cache_dir)
os.environ.setdefault("HF_HUB_CACHE", hf_cache_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
os.environ.setdefault("TORCH_HOME", torch_cache_dir)
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(torch_cache_dir, exist_ok=True)
os.makedirs(os.path.join(current_dir, "outputs", "tasks"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "prompts"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "temp", "batch"), exist_ok=True)

# ============================================================================
# LIGHTWEIGHT IMPORTS ONLY — Gradio loads fast, heavy deps load lazily
# ============================================================================
import gradio as gr

# These are lightweight and needed for UI structure
from omegaconf import OmegaConf
from pathlib import Path

logger = logging.getLogger("webui_enhanced")

# ============================================================================
# CONFIG VALIDATION — Fast, just file I/O
# ============================================================================
if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist.")
    sys.exit(1)

required_files = ["config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]
for file_name in required_files:
    file_path = os.path.join(cmd_args.model_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Required file {file_name} does not exist.")
        sys.exit(1)

try:
    BASE_CFG = OmegaConf.load(os.path.join(cmd_args.model_dir, "config.yaml"))
except Exception as exc:
    print(f"Failed to load config.yaml: {exc}")
    sys.exit(1)

# ============================================================================
# LAZY IMPORT MANAGER
# Heavy libs (torch, librosa, numpy, IndexTTS2) load only when first needed
# ============================================================================

class _LazyImports:
    """
    Defers all heavy imports until first use.
    Gradio UI appears instantly; model loads only when user clicks Load.
    """
    _lock = threading.Lock()
    _loaded: Dict[str, Any] = {}

    @classmethod
    def _load(cls, name: str):
        if name in cls._loaded:
            return cls._loaded[name]
        with cls._lock:
            if name in cls._loaded:
                return cls._loaded[name]
            print(f">> Lazy loading: {name}", flush=True)
            t0 = time.perf_counter()

            if name == "numpy":
                import numpy as _m
            elif name == "torch":
                import torch as _m
            elif name == "librosa":
                import librosa as _m
            elif name == "soundfile":
                import soundfile as _m
            elif name == "indextts":
                sys.path.append(current_dir)
                sys.path.append(os.path.join(current_dir, "indextts"))
                from indextts.infer_v2_modded import IndexTTS2 as _m
            elif name == "i18n":
                sys.path.append(current_dir)
                from tools.i18n.i18n import I18nAuto
                _m = I18nAuto(language="Auto")
            else:
                raise ImportError(f"Unknown lazy import: {name}")

            cls._loaded[name] = _m
            elapsed = time.perf_counter() - t0
            print(f">> Loaded {name} in {elapsed:.2f}s", flush=True)
            return _m

    # Convenience accessors
    @classmethod
    def np(cls):
        return cls._load("numpy")

    @classmethod
    def torch(cls):
        return cls._load("torch")

    @classmethod
    def librosa(cls):
        return cls._load("librosa")

    @classmethod
    def sf(cls):
        return cls._load("soundfile")

    @classmethod
    def IndexTTS2(cls):
        return cls._load("indextts")

    @classmethod
    def i18n(cls):
        return cls._load("i18n")


_lazy = _LazyImports()

# ============================================================================
# CONSTANTS
# ============================================================================
TOKENS_PER_SECOND_DEFAULT = 50.0

PHONEME_RATES = {
    "en": 0.085,
    "ja": 0.10,
    "zh": 0.27,
    "ko": 0.085,
    "fr": 0.085,
    "de": 0.085,
}

DURATION_SCALE_MIN = 0.5
DURATION_SCALE_MAX = 2.0
DURATION_SCALE_PRESETS = {
    "×0.75 (Fast)": 0.75,
    "×0.875 (Slightly Fast)": 0.875,
    "×1.0 (Natural)": 1.0,
    "×1.125 (Slightly Slow)": 1.125,
    "×1.25 (Slow)": 1.25,
}

DURATION_PARAM_CANDIDATES = [
    "speech_token_num",
    "target_speech_token_num",
    "token_num",
    "target_token_num",
    "target_code_len",
    "code_len",
    "target_len",
    "target_sem_len",
    "target_mel_tokens",
]

EMO_CHOICES = [
    "Match prompt audio",
    "Use emotion reference audio",
    "Use emotion vector",
    "Use emotion text description",
]

parallel_worker_config: Dict[str, Any] = {
    "model_dir": cmd_args.model_dir,
    "is_fp16": cmd_args.is_fp16,
    "verbose": cmd_args.verbose,
    "hf_cache": hf_cache_dir,
    "torch_cache": torch_cache_dir,
    "gpt_path": None,
    "bpe_path": None,
    "tokens_per_second": TOKENS_PER_SECOND_DEFAULT,
}

# ============================================================================
# EXAMPLES — Pure JSON, no heavy deps
# ============================================================================
example_cases: List[List[Any]] = []
examples_path = Path(current_dir) / "examples" / "cases.jsonl"
if examples_path.exists():
    with examples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                emo_audio = example.get("emo_audio")
                emo_audio_path = (
                    os.path.join("examples", emo_audio) if emo_audio else None
                )
                example_cases.append([
                    os.path.join("examples", example.get("prompt_audio", "")),
                    example.get("emo_mode", 0),
                    example.get("text", ""),
                    emo_audio_path,
                    example.get("emo_weight", 1.0),
                    example.get("emo_text", ""),
                    example.get("emo_vec_1", 0),
                    example.get("emo_vec_2", 0),
                    example.get("emo_vec_3", 0),
                    example.get("emo_vec_4", 0),
                    example.get("emo_vec_5", 0),
                    example.get("emo_vec_6", 0),
                    example.get("emo_vec_7", 0),
                    example.get("emo_vec_8", 0),
                ])
            except Exception:
                pass

# ============================================================================
# LANGUAGE PREFIX HELPER
# ============================================================================

LANGUAGE_PREFIXES = {
    'ti': 'tir ',   # Tigrinya → activates ▁tir▁ tokens
    'tigrinya': 'tir ',
}

def apply_language_prefix(text: str, language: str) -> str:
    """
    Prepend language-specific prefix to activate
    language-specific tokenizer tokens.
    
    Required for Tigrinya because shared Geez characters
    (ሀ, ሐ, ዐ, ፀ etc) have different phonemes in AM vs TI.
    The 'tir ' prefix activates TI-specific tokens:
      tir ሀ → ▁tir▁ሀ (token 26137) instead of ▁ሀ (token 12614)
    
    For all other languages: returns text unchanged.
    """
    lang = (language or '').strip().lower()
    prefix = LANGUAGE_PREFIXES.get(lang)
    if prefix and not text.startswith(prefix):
        return prefix + text
    return text
    
# ============================================================================
# DURATION ESTIMATOR
# ============================================================================

class DurationEstimator:
    """
    Lazy-initialized duration estimator.
    Does NOT import torch/librosa at construction time.
    """

    def __init__(self, tokens_per_second: float = TOKENS_PER_SECOND_DEFAULT):
        self.tokens_per_second = tokens_per_second
        self._espeak_checked = False
        self._espeak_available = False
        self._pyopenjtalk_checked = False
        self._pyopenjtalk_available = False

    def _check_espeak(self) -> bool:
        if self._espeak_checked:
            return self._espeak_available
        self._espeak_checked = True
        try:
            import subprocess
            result = subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True, timeout=5,
            )
            self._espeak_available = result.returncode == 0
        except Exception:
            self._espeak_available = False
        return self._espeak_available

    def _check_pyopenjtalk(self) -> bool:
        if self._pyopenjtalk_checked:
            return self._pyopenjtalk_available
        self._pyopenjtalk_checked = True
        try:
            import pyopenjtalk  # noqa
            self._pyopenjtalk_available = True
        except ImportError:
            self._pyopenjtalk_available = False
        return self._pyopenjtalk_available

    def count_phonemes(self, text: str, language: str = "en") -> int:
        lang = language.lower().strip()

        if lang in ("zh", "zh-cn", "zh-tw", "chinese", "mandarin"):
            cjk = re.findall(
                r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text
            )
            return max(len(cjk), len(text.split()))

        if lang in ("ja", "japanese") and self._check_pyopenjtalk():
            try:
                import pyopenjtalk
                njd = pyopenjtalk.run_frontend(text)
                count = sum(
                    len(f.get("read", "")) for f in njd if f.get("read")
                )
                return max(count, 1)
            except Exception:
                pass

        if self._check_espeak():
            try:
                import subprocess
                lang_map = {
                    "en": "en", "ja": "ja", "ko": "ko",
                    "fr": "fr", "de": "de", "es": "es",
                }
                result = subprocess.run(
                    ["espeak-ng", "-v", lang_map.get(lang, "en"),
                     "-q", "--ipa", text],
                    capture_output=True, text=True, timeout=10,
                )
                ipa_clean = re.sub(r'[ˈˌ\s\|]', '', result.stdout.strip())
                return max(len(ipa_clean), 1)
            except Exception:
                pass

        char_count = len(re.sub(r'\s+', '', text))
        return max(int(char_count * 0.6), len(text.split()), 1)

    def estimate_from_reference(
        self,
        ref_duration_seconds: float,
        ref_text: str,
        target_text: str,
        language: str = "en",
        scale_factor: float = 1.0,
    ) -> Tuple[int, float, str]:
        n_ref = self.count_phonemes(ref_text, language)
        n_tgt = self.count_phonemes(target_text, language)

        if n_ref > 0:
            d_target = ref_duration_seconds * (n_tgt / n_ref) * scale_factor
            method = (
                f"Ref-proportional: {ref_duration_seconds:.2f}s × "
                f"({n_tgt}/{n_ref}) × {scale_factor:.3f} = {d_target:.2f}s"
            )
        else:
            lang_rate = PHONEME_RATES.get(language.lower(), PHONEME_RATES["en"])
            d_target = n_tgt * lang_rate * scale_factor
            method = (
                f"Phoneme-rate: {n_tgt} × {lang_rate} × "
                f"{scale_factor:.3f} = {d_target:.2f}s"
            )

        tokens = max(1, int(math.floor(d_target * self.tokens_per_second)))
        return tokens, d_target, method

    def estimate_from_seconds(
        self,
        target_seconds: float,
        scale_factor: float = 1.0,
    ) -> Tuple[int, float, str]:
        scaled = target_seconds * scale_factor
        tokens = max(1, int(math.floor(scaled * self.tokens_per_second)))
        method = (
            f"Explicit: {target_seconds:.2f}s × {scale_factor:.3f} "
            f"= {scaled:.2f}s → {tokens} tokens"
        )
        return tokens, scaled, method

    def get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Lazy-loads librosa only when called."""
        try:
            librosa = _lazy.librosa()
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except Exception:
            return None

    def validate_token_count(
        self,
        token_count: int,
        max_tokens: int = 2048,
    ) -> Tuple[int, bool, str]:
        if token_count <= 0:
            return 1, True, "Token count ≤ 0, clamped to 1"
        if token_count > max_tokens:
            return max_tokens, True, f"Clamped {token_count} → {max_tokens}"
        return token_count, False, ""


_duration_estimator = DurationEstimator(TOKENS_PER_SECOND_DEFAULT)


# ============================================================================
# DATACLASS
# ============================================================================

@dataclass
class GenerationJob:
    row_id: int
    prompt_path: str
    text: str
    output_path: str
    emo_mode: int
    emo_weight: float
    emo_vector: Optional[List[float]]
    emo_text: str
    emo_random: bool
    emo_ref_path: Optional[str]
    max_tokens: int
    generation_kwargs: Dict[str, Any]
    verbose: bool
    trim_silence: bool = True
    use_batch_latent_cache: bool = False
    batch_latent_group: Optional[str] = None
    duration_mode: str = "Auto"
    target_tokens: int = 0
    target_seconds: Optional[float] = None
    duration_scale: float = 1.0
    ref_text: str = ""
    language: str = "en"
    srt_duration_seconds: Optional[float] = None
    enable_post_stretch: bool = False
    tokens_per_second: float = TOKENS_PER_SECOND_DEFAULT


# ============================================================================
# MODEL STATE — All None until user clicks Load
# ============================================================================

_PRIMARY_TTS = None
_MODEL_SELECTION: Dict[str, Optional[str]] = {"gpt": None, "bpe": None}
_MODEL_LOAD_LOCK = threading.Lock()


def _candidate_paths(bases: List[Path], suffixes: List[str]) -> List[str]:
    results: List[str] = []
    seen: set = set()
    for base in bases:
        if not base or not base.exists():
            continue
        for suffix in suffixes:
            for path in base.glob(f"*{suffix}"):
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    results.append(resolved)
    results.sort()
    return results


def _is_gpt_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".pth"):
        return False
    excluded = ("s2mel", "campplus", "bigvgan", "wav2vec", "emo", "spk", "cfm")
    return not any(t in name for t in excluded)


def _discover_gpt_checkpoints() -> List[str]:
    bases = [
        Path(cmd_args.model_dir),
        Path(current_dir) / "models",
        Path(current_dir) / "trained_ckpts_multi_am2src_om_ti_en_zh_h200_phase3",
        Path(current_dir) / "trained_ckpts_om_ti_microscopic",
        Path(current_dir) / "trained_ckpts_ti_om_refine_v6",
        Path(current_dir) / "trained_ckpts_duration-6",
        Path(current_dir) / "trained_ckpts_ti_om_refine_v7",
    ]
    candidates = _candidate_paths(bases, [".pth"])
    return [p for p in candidates if _is_gpt_checkpoint(Path(p))]


def _discover_bpe_models() -> List[str]:
    bases = [
        Path(cmd_args.model_dir),
        Path(current_dir) / "tokenizers",
        Path(cmd_args.model_dir) / "tokenizers",
    ]
    return _candidate_paths(bases, [".model"])


def _format_label(path: str) -> str:
    path_obj = Path(path)
    try:
        rel = os.path.relpath(path, cmd_args.model_dir)
        if not rel.startswith(".."):
            prefix = Path(cmd_args.model_dir).name or "checkpoints"
            return f"{prefix}/{rel}".replace("\\", "/")
    except ValueError:
        pass
    try:
        rel = os.path.relpath(path, current_dir)
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    except ValueError:
        pass
    return path_obj.name


def _format_dropdown_choices(
    paths: List[str],
    current_selection: Optional[str],
) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    labels: List[str] = []
    mapping: Dict[str, str] = {}
    selected_label: Optional[str] = None
    for path in paths:
        label = _format_label(path)
        base_label = label
        suffix = 1
        while label in mapping:
            label = f"{base_label} ({suffix})"
            suffix += 1
        mapping[label] = path
        labels.append(label)
        if (current_selection
                and os.path.abspath(path) == os.path.abspath(current_selection)):
            selected_label = label
    if labels and selected_label is None:
        selected_label = labels[0]
    return labels, mapping, selected_label


def _resolve_duration_param_name(tts) -> Optional[str]:
    try:
        sig = inspect.signature(tts.infer)
        params = sig.parameters
        for name in DURATION_PARAM_CANDIDATES:
            if name in params:
                return name
    except Exception:
        pass
    return None


def _model_status_text() -> str:
    if _PRIMARY_TTS is None:
        return (
            "⚠️ **No model loaded.** Select a GPT checkpoint and BPE tokenizer, "
            "then click **Load Models**."
        )
    gpt_path = _MODEL_SELECTION.get("gpt")
    bpe_path = _MODEL_SELECTION.get("bpe")
    gpt_name = Path(gpt_path).name if gpt_path else "?"
    bpe_name = Path(bpe_path).name if bpe_path else "?"
    param_name = _resolve_duration_param_name(_PRIMARY_TTS)
    dur_status = (
        f"✅ Paper duration param: `{param_name}`"
        if param_name
        else "⚠️ Duration: fallback (max_mel_tokens ceiling)"
    )
    return (
        f"✅ GPT: **{gpt_name}** | BPE: **{bpe_name}**\n\n{dur_status}"
    )


def dispose_primary_tts():
    global _PRIMARY_TTS
    if _PRIMARY_TTS is not None:
        try:
            if hasattr(_PRIMARY_TTS, "gr_progress"):
                _PRIMARY_TTS.gr_progress = None
        finally:
            _PRIMARY_TTS = None
            gc.collect()
            try:
                torch = _lazy.torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _do_load_model(gpt_path: str, bpe_path: str):
    """
    Actual model loading — called in background thread.
    Imports torch + IndexTTS2 here (lazy).
    """
    IndexTTS2 = _lazy.IndexTTS2()
    return IndexTTS2(
        model_dir=cmd_args.model_dir,
        cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
        is_fp16=cmd_args.is_fp16,
        use_cuda_kernel=False,
        gpt_checkpoint_path=gpt_path,
        bpe_model_path=bpe_path,
    )


def ensure_primary_tts():
    if _PRIMARY_TTS is None:
        raise RuntimeError(
            "No model loaded. Select a checkpoint and click **Load Models**."
        )
    return _PRIMARY_TTS


# ============================================================================
# AUDIO POST-PROCESSING — All use lazy numpy/librosa
# ============================================================================

def _detect_silence_regions(
    y, sr: int,
    window_ms: int = 25,
    rms_threshold: float = 0.005,
) -> List[Tuple[int, int]]:
    np = _lazy.np()
    window_samples = max(1, int(sr * window_ms / 1000))
    step = max(1, window_samples // 2)
    regions: List[Tuple[int, int]] = []
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


def _rms_trim_edges(y, sr: int, rms_threshold: float = 0.005,
                    window_ms: int = 25, margin_ms: int = 80):
    np = _lazy.np()
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


def _remove_trailing_noise(
    y, sr: int,
    min_gap_ms: int = 200,
    max_trailing_ms: int = 500,
    energy_ratio: float = 0.3,
    rms_threshold: float = 0.005,
):
    np = _lazy.np()
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


def _compress_internal_silence(
    y, sr: int,
    max_silence_ms: int = 600,
    compress_to_ms: int = 200,
    rms_threshold: float = 0.005,
):
    np = _lazy.np()
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


def trim_audio_wav(
    file_path: str,
    top_db: int = 30,
    margin_ms: int = 80,
    max_internal_silence_ms: int = 600,
    compress_silence_to_ms: int = 200,
):
    """4-pass post-processing. Lazy-loads librosa/soundfile."""
    try:
        librosa = _lazy.librosa()
        sf = _lazy.sf()
        y, sr = librosa.load(file_path, sr=None)
        if len(y) == 0:
            return
        original_duration = len(y) / sr
        basename = os.path.basename(file_path)
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        if len(yt) == 0:
            yt = y
        yt = _rms_trim_edges(yt, sr, margin_ms=margin_ms)
        yt = _remove_trailing_noise(yt, sr)
        yt = _compress_internal_silence(
            yt, sr,
            max_silence_ms=max_internal_silence_ms,
            compress_to_ms=compress_silence_to_ms,
        )
        trimmed_duration = len(yt) / sr
        if len(yt) > int(sr * 0.1):
            sf.write(file_path, yt, sr)
            removed = original_duration - trimmed_duration
            if removed > 0.05:
                print(
                    f">> Trimmed {basename}: {original_duration:.2f}s "
                    f"→ {trimmed_duration:.2f}s (removed {removed:.2f}s)",
                    flush=True,
                )
    except Exception as e:
        logger.error(f"Failed to process audio {file_path}: {e}")


def compute_duration_accuracy(
    generated_tokens: int,
    target_tokens: int,
    tolerance: float = 0.10,
) -> Dict[str, Any]:
    if target_tokens <= 0:
        return {"error": "No target specified"}
    token_error = abs(generated_tokens - target_tokens)
    token_error_rate = token_error / target_tokens
    within = token_error_rate <= tolerance
    return {
        "token_error_rate_pct": token_error_rate * 100,
        "token_error": token_error,
        "within_10pct_tolerance": within,
        "duration_accuracy": 1.0 if within else 0.0,
        "generated_tokens": generated_tokens,
        "target_tokens": target_tokens,
        "generated_seconds": generated_tokens / TOKENS_PER_SECOND_DEFAULT,
        "target_seconds": target_tokens / TOKENS_PER_SECOND_DEFAULT,
        "duration_delta_seconds": (
            (generated_tokens - target_tokens) / TOKENS_PER_SECOND_DEFAULT
        ),
    }


def post_process_duration_alignment(
    file_path: str,
    target_seconds: float,
    tolerance: float = 0.10,
    max_stretch_ratio: float = 1.3,
) -> Dict[str, Any]:
    try:
        librosa = _lazy.librosa()
        sf = _lazy.sf()
        y, sr = librosa.load(file_path, sr=None)
        generated_seconds = len(y) / sr
        gen_tokens = int(generated_seconds * TOKENS_PER_SECOND_DEFAULT)
        tgt_tokens = int(target_seconds * TOKENS_PER_SECOND_DEFAULT)
        metrics_before = compute_duration_accuracy(gen_tokens, tgt_tokens, tolerance)
        if metrics_before.get("within_10pct_tolerance", True):
            return {"adjusted": False, "before": metrics_before, "after": metrics_before}
        stretch_ratio = target_seconds / generated_seconds
        if (stretch_ratio > max_stretch_ratio
                or stretch_ratio < (1.0 / max_stretch_ratio)):
            return {
                "adjusted": False, "before": metrics_before, "after": metrics_before,
                "reason": f"Stretch ratio {stretch_ratio:.2f} exceeds safe limit",
            }
        y_stretched = librosa.effects.time_stretch(y, rate=1.0 / stretch_ratio)
        sf.write(file_path, y_stretched, sr)
        adj_seconds = len(y_stretched) / sr
        adj_tokens = int(adj_seconds * TOKENS_PER_SECOND_DEFAULT)
        metrics_after = compute_duration_accuracy(adj_tokens, tgt_tokens, tolerance)
        print(
            f">> Post-stretch: {generated_seconds:.2f}s → {adj_seconds:.2f}s "
            f"(target {target_seconds:.2f}s, ×{stretch_ratio:.3f})",
            flush=True,
        )
        return {
            "adjusted": True, "before": metrics_before,
            "after": metrics_after, "stretch_ratio": stretch_ratio,
        }
    except Exception as e:
        logger.error(f"Post-stretch failed: {e}")
        return {"adjusted": False, "error": str(e)}


# ============================================================================
# SRT PARSING
# ============================================================================

# ============================================================================
# SRT PARSING
# ============================================================================

def _get_filepath(file_obj):
    """Safely extract file path from Gradio File component (handles str or object)."""
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, 'name'):
        return file_obj.name
    return str(file_obj)

def parse_srt_file(file_path: str) -> List[Dict[str, Any]]:
    if not file_path or not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
    except Exception:
        return []
    
    # FIX: Normalize line endings to prevent regex failures on Windows CRLF (\r\n)
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    pattern = re.compile(
        r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n*$)'
    )
    segments = []
    for match in pattern.findall(content):
        # Clean HTML tags and normalize whitespace
        text = re.sub(r'<[^>]+>', '', match[3].strip()).replace('\n', ' ').strip()
        if not text:
            continue  # Skip empty subtitle blocks
        segments.append({
            "index": int(match[0]),
            "start": match[1], "end": match[2], "text": text,
        })
    segments.sort(key=lambda x: x['index'])
    return segments


def _srt_time_to_seconds(time_str: str) -> float:
    try:
        parts = time_str.replace(',', '.').split(':')
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0


def enrich_srt_with_durations(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for seg in segments:
        start_s = _srt_time_to_seconds(seg["start"])
        end_s = _srt_time_to_seconds(seg["end"])
        duration_s = max(0.1, end_s - start_s)
        target_tokens = max(1, int(math.floor(duration_s * TOKENS_PER_SECOND_DEFAULT)))
        enriched.append({
            **seg,
            "start_seconds": start_s,
            "end_seconds": end_s,
            "duration_seconds": duration_s,
            "target_tokens": target_tokens,
        })
    return enriched


# ============================================================================
# DURATION KWARGS BUILDING
# ============================================================================

# ============================================================================
# FIXED: _build_duration_kwargs
# Now properly overrides max_mel_tokens in gen_kwargs too
# ============================================================================
def resolve_job_duration(
    job: GenerationJob,
    worker_tts,
    estimator: DurationEstimator,
    gen_kwargs_ref: Optional[Dict[str, Any]] = None,  # ← NEW
) -> Tuple[Dict[str, Any], str]:
    gpt_cfg = getattr(BASE_CFG, "gpt", {})
    max_limit = max(100, int(getattr(gpt_cfg, "max_mel_tokens", 2048)))
    tps = job.tokens_per_second or estimator.tokens_per_second
    mode = job.duration_mode or "Auto"

    print(f">> resolve_job_duration: mode={mode}, tps={tps}", flush=True)

    # ── SRT Mode ─────────────────────────────────────────────────────────────
    if mode == "SRT" and job.srt_duration_seconds and job.srt_duration_seconds > 0:
        tokens = max(1, int(math.floor(job.srt_duration_seconds * tps)))
        tokens = min(tokens, max_limit)
        kwargs = _build_duration_kwargs(
            worker_tts, tokens, max_limit, True, gen_kwargs_ref
        )
        method = (
            f"SRT: {job.srt_duration_seconds:.3f}s × {tps} "
            f"= {tokens} tokens → speech_token_num={tokens}"
        )
        print(f">> {method}", flush=True)
        return kwargs, method

    # ── Tokens Mode ──────────────────────────────────────────────────────────
    if mode == "Tokens" and job.target_tokens > 0:
        tokens = min(job.target_tokens, max_limit)
        kwargs = _build_duration_kwargs(
            worker_tts, tokens, max_limit, True, gen_kwargs_ref
        )
        method = f"Tokens: speech_token_num={tokens}"
        print(f">> {method}", flush=True)
        return kwargs, method

    # ── Seconds Mode ─────────────────────────────────────────────────────────
    if mode == "Seconds":
        # Defensive: check all possible sources of target_seconds
        target_s = None

        if job.target_seconds and job.target_seconds > 0:
            target_s = float(job.target_seconds)
        
        if target_s is None or target_s <= 0:
            print(
                f">> Duration WARNING: mode=Seconds but target_seconds="
                f"{job.target_seconds!r} is invalid. Falling back to Auto.",
                flush=True,
            )
            return {}, "Auto (Seconds mode had no valid target)"

        tokens, secs, desc = estimator.estimate_from_seconds(
            target_s, job.duration_scale
        )
        tokens = min(tokens, max_limit)
        kwargs = _build_duration_kwargs(
            worker_tts, tokens, max_limit, True, gen_kwargs_ref
        )
        method = f"Seconds: {desc} → speech_token_num={tokens}"
        print(f">> {method}", flush=True)
        return kwargs, method

    # ── Reference / Scale Mode ───────────────────────────────────────────────
    if mode in ("Reference", "Scale") and job.prompt_path:
        ref_dur = estimator.get_audio_duration(job.prompt_path)
        if ref_dur and ref_dur > 0:
            if job.ref_text:
                tokens, secs, desc = estimator.estimate_from_reference(
                    ref_dur, job.ref_text, job.text,
                    job.language, job.duration_scale,
                )
            else:
                tokens, secs, desc = estimator.estimate_from_seconds(
                    ref_dur, job.duration_scale
                )
            tokens = min(tokens, max_limit)
            kwargs = _build_duration_kwargs(
                worker_tts, tokens, max_limit, True, gen_kwargs_ref
            )
            method = f"Reference: {desc} → speech_token_num={tokens}"
            print(f">> {method}", flush=True)
            return kwargs, method
        else:
            print(
                f">> Duration WARNING: mode={mode} but could not read "
                f"audio duration from {job.prompt_path}. Falling back to Auto.",
                flush=True,
            )

    # ── Auto ─────────────────────────────────────────────────────────────────
    print(">> Duration: Auto (p=0 free generation)", flush=True)
    return {}, "Auto: Free generation (p=0)"
# ============================================================================
# SEED HELPERS
# ============================================================================

def _normalize_seed(seed_value: Any) -> Optional[int]:
    if seed_value is None:
        return None
    if isinstance(seed_value, str):
        v = seed_value.strip()
        if not v:
            return None
        try:
            return abs(int(v))
        except ValueError:
            try:
                return abs(int(float(v)))
            except ValueError:
                return None
    if isinstance(seed_value, bool):
        return int(seed_value)
    if isinstance(seed_value, float):
        if math.isnan(seed_value):
            return None
        return abs(int(seed_value))
    try:
        return abs(int(seed_value))
    except (TypeError, ValueError):
        return None


def _apply_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random as _random
    np = _lazy.np()
    torch = _lazy.torch()
    _random.seed(int(seed % (2 ** 32)))
    np.random.seed(int(seed % (2 ** 32)))
    torch.manual_seed(int(seed % (2 ** 63 - 1)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed % (2 ** 63 - 1)))


def _prepare_generation_kwargs(raw_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(raw_kwargs or {})
    seed = _normalize_seed(kwargs.pop("seed", None))
    _apply_seed(seed)
    return kwargs


# ============================================================================
# LATENT CACHE HELPER
# ============================================================================

def _extract_and_cache_latents(worker_tts, spk_audio_prompt, emo_mode, emo_ref_path):
    spk_latents = None
    emo_latents = None
    if hasattr(worker_tts, 'get_speaker_conditioning'):
        try:
            spk_latents = worker_tts.get_speaker_conditioning(spk_audio_prompt)
        except Exception as e:
            print(f">> get_speaker_conditioning failed: {e}", flush=True)
    if hasattr(worker_tts, 'get_emotion_conditioning'):
        try:
            ref = emo_ref_path if (emo_mode == 1 and emo_ref_path) else spk_audio_prompt
            emo_latents = worker_tts.get_emotion_conditioning(ref)
        except Exception as e:
            print(f">> get_emotion_conditioning failed: {e}", flush=True)
    return spk_latents, emo_latents


# ============================================================================
# WORKER LOOP — Runs in spawned subprocess, imports everything fresh
# ============================================================================

def _worker_loop(job_queue: mp.Queue, result_queue: mp.Queue, config: Dict[str, Any]):
    """
    Subprocess worker. All heavy imports happen here, isolated from UI process.
    """
    # Setup environment
    for env_key, env_val in [
        ("HF_HOME", config.get("hf_cache")),
        ("HF_HUB_CACHE", config.get("hf_cache")),
        ("TRANSFORMERS_CACHE", config.get("hf_cache")),
        ("TORCH_HOME", config.get("torch_cache")),
        ("INDEXTTS_USE_DEEPSPEED", "0"),
    ]:
        if env_val:
            os.environ.setdefault(env_key, str(env_val))
            os.makedirs(str(env_val), exist_ok=True)

    gpt_path = config.get("gpt_path")
    bpe_path = config.get("bpe_path")
    if not gpt_path or not bpe_path:
        result_queue.put({
            "type": "init_error",
            "error": "No model loaded. Use the Load button.",
        })
        return

    # Import heavy deps in worker process
    import sys as _sys
    _sys.path.append(config.get("current_dir", "."))
    _sys.path.append(os.path.join(config.get("current_dir", "."), "indextts"))

    try:
        from indextts.infer_v2_modded import IndexTTS2
        import librosa as _librosa
        import soundfile as _sf
        import numpy as _np
    except ImportError as e:
        result_queue.put({"type": "init_error", "error": f"Import failed: {e}"})
        return

    try:
        worker_tts = IndexTTS2(
            model_dir=config["model_dir"],
            cfg_path=os.path.join(config["model_dir"], "config.yaml"),
            is_fp16=config.get("is_fp16", False),
            use_cuda_kernel=False,
            gpt_checkpoint_path=gpt_path,
            bpe_model_path=bpe_path,
        )
    except Exception as exc:
        result_queue.put({"type": "init_error", "error": str(exc)})
        return

    worker_estimator = DurationEstimator(
        tokens_per_second=config.get("tokens_per_second", TOKENS_PER_SECOND_DEFAULT)
    )
    latent_cache: Dict[str, tuple] = {}
    worker_pid = os.getpid()

    while True:
        raw_job = job_queue.get()
        if isinstance(raw_job, dict) and raw_job.get("type") == "stop":
            break

        try:
            job = GenerationJob(**{
                k: v for k, v in raw_job.items()
                if k in GenerationJob.__dataclass_fields__
            })

            emo_audio_prompt = job.emo_ref_path if job.emo_mode == 1 else None
            emo_alpha = job.emo_weight if job.emo_mode == 1 else 1.0
            emo_vector = job.emo_vector if job.emo_mode == 2 else None
            use_emo_text = job.emo_mode == 3

# ============================================================================
# FIXED: Worker loop duration resolution
# duration_kwargs must win over gen_kwargs
# ============================================================================

# Inside _worker_loop, replace the infer section with:

            generation_kwargs = _prepare_generation_kwargs(job.generation_kwargs.copy())

            # Strip internal _ prefixed keys
            for k in list(generation_kwargs.keys()):
                if k.startswith("_"):
                    generation_kwargs.pop(k, None)

            # ── DURATION RESOLUTION ───────────────────────────────────────
            # Pass generation_kwargs as ref so fallback can override max_mel_tokens
            duration_kwargs, duration_method = resolve_job_duration(
                job, worker_tts, worker_estimator,
                gen_kwargs_ref=generation_kwargs,   # ← FIXED
            )

            print(
                f">> [Worker {worker_pid}] Row {job.row_id}: "
                f"mode={job.duration_mode}, "
                f"duration_kwargs={duration_kwargs}, "
                f"method={duration_method}",
                flush=True,
            )

            # Merge: duration OVERWRITES gen_kwargs on conflict
            final_infer_kwargs = {
                **generation_kwargs,   # sampling settings
                **duration_kwargs,     # ← duration wins
            }

            infer_args = {
                "spk_audio_prompt": job.prompt_path,
                "text": apply_language_prefix(job.text, job.language),
                "output_path": job.output_path,
                "emo_audio_prompt": emo_audio_prompt,
                "emo_alpha": emo_alpha,
                "emo_vector": emo_vector,
                "use_emo_text": use_emo_text,
                "emo_text": job.emo_text,
                "use_random": job.emo_random,
                "verbose": job.verbose,
                "max_text_tokens_per_sentence": job.max_tokens,
                **final_infer_kwargs,   # ← merged, duration wins
            }
            # ── LATENT EXTRACTION ────────────────────────────────────────
            cache_key = job.prompt_path
            if cache_key in latent_cache:
                spk_latents, emo_latents = latent_cache[cache_key]
            else:
                spk_latents, emo_latents = _extract_and_cache_latents(
                    worker_tts, job.prompt_path, job.emo_mode, job.emo_ref_path
                )
                latent_cache[cache_key] = (spk_latents, emo_latents)

            supports_cache = True  # latent availability already guarded by _extract_and_cache_latents
            if supports_cache and spk_latents is not None:
                infer_args["spk_latents"] = spk_latents
            if supports_cache and emo_latents is not None:
                infer_args["emo_latents"] = emo_latents

            t_start = time.perf_counter()
            worker_tts.infer(**infer_args)
            t_elapsed = time.perf_counter() - t_start

            # Post-process
            if job.trim_silence:
                # Inline trim in worker to avoid re-import
                try:
                    y, sr = _librosa.load(job.output_path, sr=None)
                    yt, _ = _librosa.effects.trim(y, top_db=30)
                    if len(yt) > int(sr * 0.1):
                        _sf.write(job.output_path, yt, sr)
                except Exception:
                    pass

            # Duration accuracy
            duration_metrics = {}
            if duration_kwargs and os.path.exists(job.output_path):
                try:
                    y_check, sr_check = _librosa.load(job.output_path, sr=None)
                    gen_secs = len(y_check) / sr_check
                    gen_tokens = int(gen_secs * worker_estimator.tokens_per_second)

                    tgt = job.target_tokens
                    if job.srt_duration_seconds and job.srt_duration_seconds > 0:
                        tgt = int(
                            job.srt_duration_seconds * worker_estimator.tokens_per_second
                        )
                    elif job.target_seconds and job.target_seconds > 0:
                        tgt = int(
                            job.target_seconds * worker_estimator.tokens_per_second
                        )

                    if tgt > 0:
                        duration_metrics = compute_duration_accuracy(gen_tokens, tgt)
                        err = duration_metrics.get("token_error_rate_pct", 0)
                        print(
                            f">> Row {job.row_id}: {gen_secs:.3f}s generated, "
                            f"error={err:.4f}% (benchmark <0.02%)",
                            flush=True,
                        )

                        if (
                            job.enable_post_stretch
                            and not duration_metrics.get("within_10pct_tolerance", True)
                            and job.srt_duration_seconds
                        ):
                            try:
                                stretch = job.srt_duration_seconds / gen_secs
                                if 0.77 < stretch < 1.3:
                                    y_s = _librosa.effects.time_stretch(
                                        y_check, rate=1.0 / stretch
                                    )
                                    _sf.write(job.output_path, y_s, sr_check)
                                    duration_metrics["post_stretched"] = True
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Duration metrics failed: {e}")

            result_queue.put({
                "type": "result",
                "row_id": job.row_id,
                "status": "Completed",
                "output_path": job.output_path,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": None,
                "batch_id": raw_job.get("batch_id"),
                "duration_method": duration_method,
                "duration_metrics": duration_metrics,
                "generation_time": t_elapsed,
            })

        except Exception as exc:
            logger.exception("Worker error")
            result_queue.put({
                "type": "result",
                "row_id": raw_job.get("row_id", -1),
                "status": f"Error: {exc}",
                "output_path": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(exc),
                "batch_id": raw_job.get("batch_id"),
                "duration_method": "Error",
                "duration_metrics": {},
                "generation_time": 0,
            })

    latent_cache.clear()
    try:
        if hasattr(worker_tts, "unload"):
            worker_tts.unload()
    except Exception:
        pass


# ============================================================================
# WORKER POOL
# ============================================================================

class WorkerPool:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ctx = mp.get_context("spawn")
        self.job_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.processes: List[mp.Process] = []
        self.worker_count = 0
        self.lock = threading.Lock()
        self.batch_counter = itertools.count()

    def _all_alive(self) -> bool:
        return all(p.is_alive() for p in self.processes)

    def ensure(self, count: int):
        count = max(1, int(count))
        with self.lock:
            if (self.worker_count == count
                    and self.processes and self._all_alive()):
                return
            self.stop_locked()
            self.start_locked(count)

    def start_locked(self, count: int):
        self.job_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.processes = []
        self.worker_count = count
        cfg = {**self.config, "current_dir": current_dir}
        for _ in range(count):
            p = self.ctx.Process(
                target=_worker_loop,
                args=(self.job_queue, self.result_queue, cfg),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

    def stop_locked(self):
        if not self.processes:
            return
        if self.job_queue:
            for _ in self.processes:
                self.job_queue.put({"type": "stop"})
        for p in self.processes:
            p.join(timeout=5)
        self.processes = []
        if self.job_queue:
            self.job_queue.close()
            self.job_queue = None
        if self.result_queue:
            self.result_queue.close()
            self.result_queue = None
        self.worker_count = 0

    def stop(self):
        with self.lock:
            self.stop_locked()

    def run_jobs(
        self,
        jobs: List[GenerationJob],
        progress: Optional[gr.Progress],
    ) -> Dict[int, Dict[str, Any]]:
        if not jobs:
            return {}
        with self.lock:
            if not self.processes or not self.job_queue or not self.result_queue:
                raise RuntimeError("Worker pool not initialized")
            batch_id = next(self.batch_counter)
            for job in jobs:
                payload = job.__dict__.copy()
                payload["batch_id"] = batch_id
                self.job_queue.put(payload)

        row_results: Dict[int, Dict[str, Any]] = {}
        processed = 0
        total = len(jobs)
        while processed < total:
            msg = self.result_queue.get()
            if msg.get("type") == "init_error":
                raise RuntimeError(f"Worker failed: {msg['error']}")
            if msg.get("batch_id") != batch_id:
                continue
            row_results[msg["row_id"]] = msg
            processed += 1
            _update_progress(
                progress,
                min(processed / total, 0.999),
                f"Processed {processed}/{total}",
            )
        _update_progress(progress, 1.0, "Complete")
        return row_results


worker_pool = WorkerPool(parallel_worker_config)
atexit.register(lambda: worker_pool.stop())


def _update_progress(progress, value, desc=""):
    if progress is None:
        return
    try:
        progress(value, desc=desc)
    except Exception:
        pass


# ============================================================================
# UI HELPERS
# ============================================================================

def refresh_model_lists():
    gpt_choices = _discover_gpt_checkpoints()
    bpe_choices = _discover_bpe_models()
    gpt_labels, gpt_map, init_gpt = _format_dropdown_choices(
        gpt_choices, _MODEL_SELECTION["gpt"]
    )
    bpe_labels, bpe_map, init_bpe = _format_dropdown_choices(
        bpe_choices, _MODEL_SELECTION["bpe"]
    )
    return (
        gr.update(choices=gpt_labels, value=init_gpt),
        gr.update(choices=bpe_labels, value=init_bpe),
        gpt_map, bpe_map,
        _model_status_text(),
    )


def handle_model_load(gpt_label, bpe_label, gpt_map, bpe_map):
    """
    Called when user clicks Load Models.
    This is where torch + IndexTTS2 are imported for the first time.
    """
    global _PRIMARY_TTS
    gpt_path = gpt_map.get(gpt_label)
    bpe_path = bpe_map.get(bpe_label)
    if not gpt_path or not bpe_path:
        return "❌ Invalid selection."

    with _MODEL_LOAD_LOCK:
        try:
            dispose_primary_tts()
            _MODEL_SELECTION["gpt"] = os.path.abspath(gpt_path)
            _MODEL_SELECTION["bpe"] = os.path.abspath(bpe_path)
            _PRIMARY_TTS = _do_load_model(
                _MODEL_SELECTION["gpt"], _MODEL_SELECTION["bpe"]
            )
            parallel_worker_config["gpt_path"] = _MODEL_SELECTION["gpt"]
            parallel_worker_config["bpe_path"] = _MODEL_SELECTION["bpe"]
            worker_pool.stop()

            # Update token rate from config
            try:
                codec_fps = getattr(BASE_CFG, "codec_fps", None)
                if codec_fps:
                    _duration_estimator.tokens_per_second = float(codec_fps)
            except Exception:
                pass

            return _model_status_text()
        except Exception as e:
            logger.exception("Model load failed")
            _MODEL_SELECTION["gpt"] = None
            _MODEL_SELECTION["bpe"] = None
            _PRIMARY_TTS = None
            return f"❌ Load failed: {e}"


# ============================================================================
# FIXED build_generation_kwargs - place this OUTSIDE create_demo()
# ============================================================================

def build_generation_kwargs(
    do_sample_value, top_p_value, top_k_value, temperature_value,
    length_penalty_value, num_beams_value, repetition_penalty_value,
    max_mel_tokens_value, seed_value, trim_silence_value,
    duration_mode_value, target_tokens_value, target_seconds_value,
    duration_scale_value, ref_text_value, language_value,
    tokens_per_second_value, enable_post_stretch_value,
) -> Dict[str, Any]:
    try:
        top_k_int = int(top_k_value)
    except (TypeError, ValueError):
        top_k_int = 0
    try:
        num_beams_int = int(num_beams_value)
    except (TypeError, ValueError):
        num_beams_int = 1

    _target_seconds = None
    if target_seconds_value is not None:
        try:
            v = float(target_seconds_value)
            if v > 0:
                _target_seconds = v
        except (TypeError, ValueError):
            pass

    print(
        f">> build_generation_kwargs: "
        f"duration_mode={duration_mode_value!r}, "
        f"target_seconds={target_seconds_value!r} → {_target_seconds!r}, "
        f"target_tokens={target_tokens_value!r}",
        flush=True,
    )

    kwargs: Dict[str, Any] = {
        "do_sample": bool(do_sample_value),
        "top_p": float(top_p_value),
        "top_k": top_k_int if top_k_int > 0 else None,
        "temperature": float(temperature_value),
        "length_penalty": float(length_penalty_value),
        "num_beams": num_beams_int,
        "repetition_penalty": float(repetition_penalty_value),
        "max_mel_tokens": int(max_mel_tokens_value),
        "_duration_mode": str(duration_mode_value or "Auto"),
        "_target_tokens": int(target_tokens_value or 0),
        "_target_seconds": _target_seconds,
        "_duration_scale": float(duration_scale_value or 1.0),
        "_ref_text": str(ref_text_value or ""),
        "_language": str(language_value or "en"),
        "_tokens_per_second": float(
            tokens_per_second_value or TOKENS_PER_SECOND_DEFAULT
        ),
        "_trim_silence": bool(trim_silence_value),
        "_enable_post_stretch": bool(enable_post_stretch_value),
    }

    seed_int = _normalize_seed(seed_value)
    if seed_int is not None:
        kwargs["seed"] = seed_int

    return kwargs


# ============================================================================
# FIXED _extract_duration_fields - place this OUTSIDE create_demo()
# ============================================================================

def _extract_duration_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    duration_mode = str(kwargs.pop("_duration_mode", "Auto") or "Auto").strip()

    try:
        target_tokens = int(kwargs.pop("_target_tokens", 0) or 0)
    except (TypeError, ValueError):
        target_tokens = 0

    raw_target_seconds = kwargs.pop("_target_seconds", None)
    try:
        target_seconds = (
            float(raw_target_seconds)
            if raw_target_seconds is not None
            else None
        )
        if target_seconds is not None and target_seconds <= 0:
            target_seconds = None
    except (TypeError, ValueError):
        target_seconds = None

    try:
        duration_scale = float(kwargs.pop("_duration_scale", 1.0) or 1.0)
        if duration_scale <= 0:
            duration_scale = 1.0
    except (TypeError, ValueError):
        duration_scale = 1.0

    ref_text = str(kwargs.pop("_ref_text", "") or "").strip()
    language = str(kwargs.pop("_language", "en") or "en").strip()

    try:
        tps = float(
            kwargs.pop("_tokens_per_second", TOKENS_PER_SECOND_DEFAULT)
            or TOKENS_PER_SECOND_DEFAULT
        )
        if tps <= 0:
            tps = TOKENS_PER_SECOND_DEFAULT
    except (TypeError, ValueError):
        tps = TOKENS_PER_SECOND_DEFAULT

    trim_silence = bool(kwargs.pop("_trim_silence", True))
    enable_post_stretch = bool(kwargs.pop("_enable_post_stretch", False))

    if duration_mode == "Seconds" and target_seconds is None:
        print(
            f">> WARNING: duration_mode=Seconds but target_seconds is None. "
            f"Falling back to Auto.",
            flush=True,
        )
        duration_mode = "Auto"

    if duration_mode == "Tokens" and target_tokens <= 0:
        print(
            f">> WARNING: duration_mode=Tokens but target_tokens={target_tokens}. "
            f"Falling back to Auto.",
            flush=True,
        )
        duration_mode = "Auto"

    return {
        "duration_mode": duration_mode,
        "target_tokens": target_tokens,
        "target_seconds": target_seconds,
        "duration_scale": duration_scale,
        "ref_text": ref_text,
        "language": language,
        "tokens_per_second": tps,
        "trim_silence": trim_silence,
        "enable_post_stretch": enable_post_stretch,
    }


# ============================================================================
# FIXED _build_duration_kwargs - place this OUTSIDE create_demo()
# ============================================================================

def _build_duration_kwargs(
    tts,
    target_tokens: int,
    max_mel_tokens_limit: int,
    enable: bool,
    gen_kwargs_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not enable or target_tokens <= 0:
        return {}

    target_tokens = max(1, min(target_tokens, max_mel_tokens_limit))
    param_name = _resolve_duration_param_name(tts)

    if param_name:
        print(
            f">> Duration: {param_name}={target_tokens} "
            f"[W_num[T]=W_sem[T] paper mechanism]",
            flush=True,
        )
        return {param_name: target_tokens}
    else:
        print(
            f">> Duration: fallback max_mel_tokens={target_tokens}",
            flush=True,
        )
        if gen_kwargs_ref is not None:
            gen_kwargs_ref["max_mel_tokens"] = target_tokens
            print(
                f">> Duration: gen_kwargs max_mel_tokens overridden → {target_tokens}",
                flush=True,
            )
        return {"max_mel_tokens": target_tokens}



def create_demo() -> gr.Blocks:
    gpt_choices = _discover_gpt_checkpoints()
    bpe_choices = _discover_bpe_models()
    gpt_labels, gpt_map, init_gpt = _format_dropdown_choices(
        gpt_choices, _MODEL_SELECTION["gpt"]
    )
    bpe_labels, bpe_map, init_bpe = _format_dropdown_choices(
        bpe_choices, _MODEL_SELECTION["bpe"]
    )

    gpt_cfg = getattr(BASE_CFG, "gpt", {})
    max_mel_tokens_limit = max(100, int(getattr(gpt_cfg, "max_mel_tokens", 2048)))
    default_mel_value = min(1700, max_mel_tokens_limit)
    max_text_tokens_limit = max(40, int(getattr(gpt_cfg, "max_text_tokens", 256)))
    default_text_tokens = min(120, max_text_tokens_limit)
    cfg_version = getattr(BASE_CFG, "version", "1.0")

    with gr.Blocks(title="IndexTTS2 Enhanced WebUI") as demo:

        model_status = gr.Markdown(value=_model_status_text())
        gpt_map_state = gr.State(gpt_map)
        bpe_map_state = gr.State(bpe_map)

        with gr.Row():
            gpt_dropdown = gr.Dropdown(
                choices=gpt_labels, value=init_gpt,
                label="GPT Checkpoint (.pth)", interactive=True,
            )
            bpe_dropdown = gr.Dropdown(
                choices=bpe_labels, value=init_bpe,
                label="BPE Tokenizer (.model)", interactive=True,
            )
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            load_btn = gr.Button("⚡ Load Models", variant="primary")

        gr.HTML(
            "<h2 style='text-align:center;'>IndexTTS2 Enhanced — "
            "Duration Control + Emotion</h2>"
        )

        refresh_btn.click(
            refresh_model_lists, [],
            [gpt_dropdown, bpe_dropdown, gpt_map_state, bpe_map_state, model_status],
        )
        load_btn.click(
            handle_model_load,
            [gpt_dropdown, bpe_dropdown, gpt_map_state, bpe_map_state],
            [model_status],
        )

        batch_rows_state = gr.State([])
        next_batch_id_state = gr.State(1)

        # ── Emotion ──────────────────────────────────────────────────────────
        with gr.Accordion("🎭 Emotion Control", open=True):
            emo_control_method = gr.Radio(
                choices=EMO_CHOICES, type="index", value=0,
                label="Emotion Control Mode",
            )

        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(
                    label="Emotion Reference Audio", type="filepath"
                )
                emo_weight = gr.Slider(
                    label="Emotion Weight", minimum=0.0,
                    maximum=1.6, value=0.8, step=0.01,
                )

        with gr.Row():
            emo_random = gr.Checkbox(
                label="Random Emotion Sampling", value=False, visible=False
            )

        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label="Joy/Happiness", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec6 = gr.Slider(label="Low Mood", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec8 = gr.Slider(label="Calm/Neutral", minimum=0.0, maximum=1.4, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            emo_text = gr.Textbox(
                label="Emotion Description",
                placeholder="e.g. 'angry and frustrated'",
                value="",
            )

        # ── Advanced Settings ─────────────────────────────────────────────────
        with gr.Accordion("⚙️ Advanced Generation Settings", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**GPT Sampling**")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True)
                        temperature = gr.Slider(
                            label="temperature", minimum=0.1,
                            maximum=2.0, value=0.8, step=0.1,
                        )
                    with gr.Row():
                        top_p = gr.Slider(
                            label="top_p", minimum=0.0,
                            maximum=1.0, value=0.8, step=0.01,
                        )
                        top_k = gr.Slider(
                            label="top_k", minimum=0,
                            maximum=100, value=30, step=1,
                        )
                        num_beams = gr.Slider(
                            label="num_beams", value=3,
                            minimum=1, maximum=10, step=1,
                        )
                    with gr.Row():
                        repetition_penalty = gr.Number(
                            label="repetition_penalty", value=10.0,
                            minimum=0.1, maximum=20.0, step=0.1,
                        )
                        length_penalty = gr.Number(
                            label="length_penalty", value=0.0,
                            minimum=-2.0, maximum=2.0, step=0.1,
                        )
                    max_mel_tokens = gr.Slider(
                        label="max_mel_tokens (safety ceiling)",
                        value=default_mel_value, minimum=50,
                        maximum=max_mel_tokens_limit, step=10,
                        info="Duration control overrides this when active.",
                    )
                    seed_value = gr.Number(
                        label="Seed", value=None,
                        precision=0, minimum=0, step=1,
                    )
                    trim_silence = gr.Checkbox(
                        label="Trim Silence (post-processing)", value=True,
                    )
                with gr.Column(scale=2):
                    gr.Markdown("**Sentence Settings**")
                    max_text_tokens_per_sentence = gr.Slider(
                        label="Max tokens per sentence",
                        value=default_text_tokens, minimum=20,
                        maximum=max_text_tokens_limit, step=2,
                    )
                    with gr.Accordion("Preview sentences", open=True):
                        sentences_preview = gr.Dataframe(
                            headers=["Index", "Sentence", "Token Count"],
                            wrap=True,
                        )

        # ── Duration Control ──────────────────────────────────────────────────
        with gr.Accordion(
            "⏱️ Duration Control (IndexTTS2: speech_token_num → W_num[T]=W_sem[T])",
            open=True,
        ):
            gr.Markdown(
                "**Paper mechanism**: `p = W_num[T]` where `W_num = W_sem`. "
                "Achieves **~0.02% token error** (Table 4)."
            )
            with gr.Row():
                duration_mode = gr.Radio(
                    choices=["Auto", "Tokens", "Seconds", "Reference", "Scale"],
                    value="Auto",
                    label="Duration Mode",
                )
            duration_mode_info = gr.Markdown(
                "ℹ️ **Auto**: Free generation (p=0 vector)."
            )

            with gr.Group(visible=False) as dur_tokens_grp:
                with gr.Row():
                    target_tokens_slider = gr.Slider(
                        label="speech_token_num (T)",
                        minimum=1, maximum=max_mel_tokens_limit,
                        value=100, step=1,
                        info=f"At {TOKENS_PER_SECOND_DEFAULT}tps: "
                             f"50=1s, 250=5s, 500=10s",
                    )
                    tokens_seconds_display = gr.Markdown("≈ 2.00 seconds")

            with gr.Group(visible=False) as dur_seconds_grp:
                gr.Markdown(
                    "**Seconds Mode**: `speech_token_num = floor(seconds × tps)`"
                )
                with gr.Row():
                    target_seconds_input = gr.Number(
                        label="Target Duration (seconds)",
                        value=5.0, precision=2, minimum=0.1,
                        info="Must be > 0",
                    )
                    seconds_tokens_display = gr.Markdown("≈ 250 tokens")
                tokens_per_second_slider = gr.Slider(
                    label="Tokens/second (codec frame rate)",
                    value=TOKENS_PER_SECOND_DEFAULT,
                    minimum=10.0, maximum=100.0, step=1.0,
                )

            with gr.Group(visible=False) as dur_reference_grp:
                gr.Markdown(
                    "**Reference Mode** (T5Gemma-TTS): "
                    "`D_target = D_ref × (N_tgt/N_ref)` → tokens"
                )
                with gr.Row():
                    ref_text_input = gr.Textbox(
                        label="Reference Audio Transcript",
                        placeholder="Transcript of prompt audio",
                    )
                    language_dropdown = gr.Dropdown(
                    choices=["en", "zh", "ja", "ko", "fr", "de", "es", "am", "ti", "om"],
                    value="en", label="Language",
                    )
                reference_estimate_display = gr.Markdown(
                    "ℹ️ Upload prompt audio and enter transcript."
                )

            with gr.Group(visible=False) as dur_scale_grp:
                gr.Markdown(
                    "**Scale Mode**: ×factor on reference duration "
                    "(paper Table 4: ×0.75→0.067%, ×1.0→0.019% error)"
                )
                with gr.Row():
                    duration_scale_preset = gr.Radio(
                        choices=list(DURATION_SCALE_PRESETS.keys()),
                        value="×1.0 (Natural)",
                        label="Scale Preset",
                    )
                    duration_scale_custom = gr.Slider(
                        label="Custom Scale",
                        minimum=DURATION_SCALE_MIN,
                        maximum=DURATION_SCALE_MAX,
                        value=1.0, step=0.025,
                    )
                scale_estimate_display = gr.Markdown(
                    "ℹ️ Upload prompt audio to see estimate."
                )

            with gr.Row():
                enable_post_stretch = gr.Checkbox(
                    label="🔧 Post-stretch if >10% duration error",
                    value=False,
                )

        # Collect all advanced params (ORDER MUST MATCH build_generation_kwargs)
        advanced_params = [
            do_sample, top_p, top_k, temperature, length_penalty, num_beams,
            repetition_penalty, max_mel_tokens, seed_value, trim_silence,
            duration_mode, target_tokens_slider, target_seconds_input,
            duration_scale_custom, ref_text_input, language_dropdown,
            tokens_per_second_slider, enable_post_stretch,
        ]

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tab("🎙️ Single Generation"):
            with gr.Row():
                prompt_audio = gr.Audio(
                    label="Prompt Audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                with gr.Column():
                    input_text_single = gr.TextArea(
                        label="Text to Synthesize",
                        info=f"Model version {cfg_version}",
                    )
                    gen_button = gr.Button("🎯 Generate", variant="primary")
            output_audio = gr.Audio(label="Generated Audio", visible=True)
            single_duration_report = gr.Markdown("")

            if example_cases:
                gr.Examples(
                    examples=example_cases,
                    inputs=[
                        prompt_audio, emo_control_method, input_text_single,
                        emo_upload, emo_weight, emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                    ],
                )

        with gr.Tab("📦 Batch Generation"):
            with gr.Tab("Manual List"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            dataset_path_input = gr.Textbox(
                                label="Dataset train.txt path",
                                value="vivy_va_dataset/train.txt", scale=3,
                            )
                            load_dataset_button = gr.Button(
                                "Load Dataset", scale=1
                            )
                        batch_file_input = gr.Files(
                            label="Add prompt audio files",
                            file_types=["audio"],
                            file_count="multiple",
                            type="filepath",
                        )
                        worker_count = gr.Slider(
                            label="Parallel workers",
                            minimum=1, maximum=8, value=1, step=1,
                        )
                        batch_table = gr.Dataframe(
                            headers=[
                                "ID", "Prompt", "Text", "Output",
                                "Status", "Dur. Method", "Error%",
                            ],
                            datatype=[
                                "number", "str", "str", "str",
                                "str", "str", "str",
                            ],
                            row_count=(0, "dynamic"), col_count=7,
                            interactive=False, value=[],
                        )
                    with gr.Column():
                        selected_entry = gr.Dropdown(
                            label="Select entry", choices=[],
                            value=None, interactive=True,
                        )
                        batch_prompt_player = gr.Audio(
                            label="Prompt Audio", type="filepath",
                            interactive=False,
                        )
                        batch_output_player = gr.Audio(
                            label="Generated Audio", type="filepath",
                            interactive=False,
                        )
                        batch_text_input = gr.TextArea(label="Text")
                        batch_ref_text_input = gr.Textbox(
                            label="Ref Transcript (for Reference/Scale modes)",
                        )
                        apply_text_button = gr.Button(
                            "💾 Save Text", variant="secondary"
                        )
                        batch_status = gr.Markdown("No entry selected.")
                        with gr.Row():
                            generate_all_button = gr.Button(
                                "🚀 Generate All", variant="primary"
                            )
                            regenerate_button = gr.Button(
                                "🔄 Regenerate Selected"
                            )
                        with gr.Row():
                            delete_entry_button = gr.Button("🗑️ Delete")
                            clear_entries_button = gr.Button("🧹 Clear All")

            with gr.Tab("📺 SRT Dubbing"):
                gr.Markdown(
                    "**Precise sync**: Subtitle timestamps → "
                    "`speech_token_num` → `p = W_num[T]` (~0.02% error)."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        srt_file_input = gr.File(
                            label="Upload SRT File", file_types=[".srt"]
                        )
                        srt_ref_audio = gr.Audio(
                            label="Voice Reference", type="filepath"
                        )
                        srt_ref_transcript = gr.Textbox(
                            label="Reference Transcript (optional)",
                        )
                        srt_preview_df = gr.Dataframe(
                            headers=[
                                "Index", "Start", "End",
                                "Duration(s)", "Target Tokens", "Text",
                            ],
                            datatype=[
                                "number", "str", "str",
                                "number", "number", "str",
                            ],
                            row_count=10, col_count=6,
                        )
                        srt_status = gr.Markdown("Upload an SRT file to begin.")
                        with gr.Row():
                            generate_srt_button = gr.Button(
                                "🎬 Generate SRT Dubbing",
                                variant="primary", size="lg",
                            )
                            download_srt_zip = gr.File(
                                label="Download ZIP", visible=False
                            )
                    with gr.Column(scale=1):
                        srt_worker_count = gr.Slider(
                            label="Parallel workers",
                            minimum=1, maximum=8, value=1, step=1,
                        )
                        srt_consistent_voice = gr.Checkbox(
                            label="🔒 Consistent Voice", value=True,
                        )
                        srt_use_subtitle_duration = gr.Checkbox(
                            label="⏱️ Use Subtitle Timestamps",
                            value=True,
                            info="speech_token_num = floor(duration × tps)",
                        )
                        srt_post_stretch = gr.Checkbox(
                            label="🔧 Post-stretch safety net", value=False,
                        )
                        srt_language = gr.Dropdown(
                            choices=["en", "zh", "ja", "ko", "fr", "de", "am", "ti", "om"],
                            value="en", label="Language",
                        )
                        srt_progress_text = gr.Textbox(
                            label="Progress", value="", interactive=False
                        )
                        srt_duration_report = gr.Dataframe(
                            headers=[
                                "Seg", "Target(s)", "Generated(s)",
                                "Error%", "±10%",
                            ],
                            datatype=[
                                "number", "number", "number",
                                "number", "str",
                            ],
                            label="Duration Report", visible=False,
                        )

        # =================================================================
        # Event Handlers — all defined INSIDE create_demo()
        # =================================================================

        def update_duration_mode_ui(mode):
            descs = {
                "Auto": "ℹ️ **Auto**: `p=0` — model determines length freely.",
                "Tokens": (
                    "🎯 **Tokens**: `speech_token_num=T` → "
                    "`p=W_num[T]=W_sem[T]` (~0.02% error)."
                ),
                "Seconds": (
                    "⏱️ **Seconds**: `speech_token_num=floor(s×tps)` → "
                    "`p=W_num[T]`."
                ),
                "Reference": (
                    "📐 **Reference** (T5Gemma): "
                    "`D=D_ref×(N_tgt/N_ref)` → tokens."
                ),
                "Scale": (
                    "📊 **Scale**: ×factor on ref duration "
                    "(paper Table 4 experiments)."
                ),
            }
            return (
                gr.update(value=descs.get(mode, "")),
                gr.update(visible=mode == "Tokens"),
                gr.update(visible=mode == "Seconds"),
                gr.update(visible=mode == "Reference"),
                gr.update(visible=mode == "Scale"),
            )

        duration_mode.change(
            update_duration_mode_ui,
            [duration_mode],
            [
                duration_mode_info,
                dur_tokens_grp,
                dur_seconds_grp,
                dur_reference_grp,
                dur_scale_grp,
            ],
        )

        def _tokens_display(tokens):
            s = tokens / TOKENS_PER_SECOND_DEFAULT
            return gr.update(value=f"≈ **{s:.2f}s**")

        def _seconds_display(secs, tps):
            if secs and tps:
                t = int(math.floor(float(secs) * float(tps)))
                return gr.update(value=f"≈ **{t} tokens** (speech_token_num={t})")
            return gr.update(value="")

        target_tokens_slider.change(
            _tokens_display,
            [target_tokens_slider],
            [tokens_seconds_display],
        )
        target_seconds_input.change(
            _seconds_display,
            [target_seconds_input, tokens_per_second_slider],
            [seconds_tokens_display],
        )
        tokens_per_second_slider.change(
            _seconds_display,
            [target_seconds_input, tokens_per_second_slider],
            [seconds_tokens_display],
        )

        def _scale_preset_change(preset):
            return gr.update(value=DURATION_SCALE_PRESETS.get(preset, 1.0))

        duration_scale_preset.change(
            _scale_preset_change,
            [duration_scale_preset],
            [duration_scale_custom],
        )

        def _reference_estimate(audio, ref_text, target_text, lang, tps):
            if not audio:
                return gr.update(value="ℹ️ Upload prompt audio.")
            dur = _duration_estimator.get_audio_duration(audio)
            if not dur:
                return gr.update(value="❌ Cannot read audio.")
            if ref_text and target_text:
                tokens, secs, method = _duration_estimator.estimate_from_reference(
                    dur, ref_text, target_text, lang or "en"
                )
                return gr.update(
                    value=(
                        f"📐 **{secs:.2f}s** → "
                        f"`speech_token_num={tokens}`\n{method}"
                    )
                )
            return gr.update(
                value=f"ℹ️ Ref audio: **{dur:.2f}s**. Add transcript for estimate."
            )

        def _scale_estimate(audio, scale, tps):
            if not audio:
                return gr.update(value="ℹ️ Upload prompt audio.")
            dur = _duration_estimator.get_audio_duration(audio)
            if not dur:
                return gr.update(value="❌ Cannot read audio.")
            scaled = dur * float(scale or 1.0)
            tokens = int(
                math.floor(scaled * float(tps or TOKENS_PER_SECOND_DEFAULT))
            )
            return gr.update(
                value=(
                    f"📊 {dur:.2f}s × {scale:.3f} = **{scaled:.2f}s** → "
                    f"`speech_token_num={tokens}`"
                )
            )

        def on_method_select(v):
            if v == 1:
                return (
                    gr.update(visible=True), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                )
            if v == 2:
                return (
                    gr.update(visible=False), gr.update(visible=True),
                    gr.update(visible=True), gr.update(visible=False),
                )
            if v == 3:
                return (
                    gr.update(visible=False), gr.update(visible=True),
                    gr.update(visible=False), gr.update(visible=True),
                )
            return (
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False),
            )

        emo_control_method.select(
            on_method_select,
            [emo_control_method],
            [
                emotion_reference_group, emo_random,
                emotion_vector_group, emo_text_group,
            ],
        )

        def on_text_change(text, max_tok):
            if not text:
                return gr.update(value=[], visible=True)
            try:
                tts = ensure_primary_tts()
                tokenized = tts.tokenizer.tokenize(text)
                sentences = tts.tokenizer.split_segments(
                    tokenized,
                    max_text_tokens_per_segment=int(max_tok),
                )
                data = [
                    [i, "".join(s), len(s)]
                    for i, s in enumerate(sentences)
                ]
                return gr.update(value=data, visible=True)
            except RuntimeError:
                return gr.update(value=[], visible=True)

        input_text_single.change(
            on_text_change,
            [input_text_single, max_text_tokens_per_sentence],
            [sentences_preview],
        )
        max_text_tokens_per_sentence.change(
            on_text_change,
            [input_text_single, max_text_tokens_per_sentence],
            [sentences_preview],
        )

        # ── Single Generation ─────────────────────────────────────────────────

        def gen_single(
            emo_mode_val, prompt, text,
            emo_ref_path, emo_weight_val,
            v1, v2, v3, v4, v5, v6, v7, v8,
            emo_text_val, emo_random_val, max_tokens_val,
            *args,
            progress=gr.Progress(),
        ):
            if not prompt:
                gr.Warning("Upload prompt audio first.")
                return gr.update(), ""
            if not (text or "").strip():
                gr.Warning("Enter text to synthesize.")
                return gr.update(), ""

            try:
                tts = ensure_primary_tts()
            except RuntimeError as exc:
                gr.Warning(str(exc))
                return gr.update(), str(exc)

            output_path = os.path.join(
                current_dir, "outputs", f"spk_{int(time.time())}.wav"
            )
            tts.gr_progress = progress

            adv = list(args)
            if len(adv) < len(advanced_params):
                adv.extend([None] * (len(advanced_params) - len(adv)))

            raw_kwargs = build_generation_kwargs(*adv[:len(advanced_params)])
            dur_fields = _extract_duration_fields(raw_kwargs)
            gen_kwargs = _prepare_generation_kwargs(raw_kwargs)

            print(
                f">> gen_single: mode={dur_fields['duration_mode']!r}, "
                f"target_seconds={dur_fields['target_seconds']!r}, "
                f"target_tokens={dur_fields['target_tokens']!r}",
                flush=True,
            )

            try:
                emo_mode = int(emo_mode_val)
            except Exception:
                emo_mode = 0

            emo_vector = None
            if emo_mode == 2:
                if sum([v1, v2, v3, v4, v5, v6, v7, v8]) > 1.5:
                    gr.Warning("Emotion vector sum > 1.5")
                    return gr.update(), "Error: vector sum > 1.5"
                emo_vector = [v1, v2, v3, v4, v5, v6, v7, v8]

            job = GenerationJob(
                row_id=0,
                prompt_path=prompt,
                text=apply_language_prefix(text, dur_fields["language"]),
                output_path=output_path,
                emo_mode=emo_mode,
                emo_weight=float(emo_weight_val) if emo_mode == 1 else 1.0,
                emo_vector=emo_vector,
                emo_text=emo_text_val,
                emo_random=bool(emo_random_val),
                emo_ref_path=emo_ref_path if emo_mode == 1 else None,
                max_tokens=int(max_tokens_val),
                generation_kwargs={},
                verbose=cmd_args.verbose,
                trim_silence=dur_fields["trim_silence"],
                duration_mode=dur_fields["duration_mode"],
                target_tokens=dur_fields["target_tokens"],
                target_seconds=dur_fields["target_seconds"],
                duration_scale=dur_fields["duration_scale"],
                ref_text=dur_fields["ref_text"],
                language=dur_fields["language"],
                tokens_per_second=dur_fields["tokens_per_second"],
                enable_post_stretch=dur_fields["enable_post_stretch"],
            )

            # KEY FIX: pass gen_kwargs so duration can override max_mel_tokens
            duration_kwargs, duration_method = resolve_job_duration(
                job, tts, _duration_estimator,
                gen_kwargs_ref=gen_kwargs,
            )

            if dur_fields["duration_mode"] != "Auto" and not duration_kwargs:
                print(
                    f">> WARNING: mode={dur_fields['duration_mode']} "
                    f"but duration_kwargs is empty!",
                    flush=True,
                )

            _update_progress(progress, 0.1, "Generating...")

            # duration_kwargs comes AFTER gen_kwargs so it wins on conflicts
            final_infer_kwargs = {
                **gen_kwargs,
                **duration_kwargs,
            }

            print(
                f">> infer: max_mel_tokens={final_infer_kwargs.get('max_mel_tokens')}, "
                f"speech_token_num={final_infer_kwargs.get('speech_token_num')}, "
                f"method={duration_method}",
                flush=True,
            )

            tts.infer(
                spk_audio_prompt=prompt,
                text=apply_language_prefix(text, dur_fields["language"]),
                output_path=output_path,
                emo_audio_prompt=emo_ref_path if emo_mode == 1 else None,
                emo_alpha=float(emo_weight_val) if emo_mode == 1 else 1.0,
                emo_vector=emo_vector if emo_mode == 2 else None,
                use_emo_text=(emo_mode == 3),
                emo_text=emo_text_val,
                use_random=bool(emo_random_val),
                verbose=cmd_args.verbose,
                max_text_tokens_per_sentence=int(max_tokens_val),
                **final_infer_kwargs,
            )

            _update_progress(progress, 0.85, "Post-processing...")
            if dur_fields["trim_silence"]:
                trim_audio_wav(output_path)

            # Build report
            report = [f"**Duration Method**: {duration_method}"]
            if duration_kwargs and os.path.exists(output_path):
                try:
                    librosa = _lazy.librosa()
                    y, sr = librosa.load(output_path, sr=None)
                    gen_secs = len(y) / sr
                    gen_toks = int(
                        gen_secs * _duration_estimator.tokens_per_second
                    )
                    tgt = 0
                    if job.target_tokens > 0:
                        tgt = job.target_tokens
                    elif job.target_seconds and job.target_seconds > 0:
                        tgt = int(
                            job.target_seconds
                            * _duration_estimator.tokens_per_second
                        )
                    if tgt > 0:
                        m = compute_duration_accuracy(gen_toks, tgt)
                        report += [
                            f"\n**Duration Accuracy**:",
                            f"- Generated: **{gen_secs:.3f}s** ({gen_toks} tokens)",
                            f"- Target:    **{m['target_seconds']:.3f}s** ({tgt} tokens)",
                            f"- Token error: **{m['token_error_rate_pct']:.4f}%** "
                            f"(paper benchmark: <0.02%)",
                            f"- Within ±10%: "
                            f"{'✅ Yes' if m['within_10pct_tolerance'] else '❌ No'}",
                        ]
                        if (
                            dur_fields["enable_post_stretch"]
                            and not m["within_10pct_tolerance"]
                            and job.target_seconds
                        ):
                            r = post_process_duration_alignment(
                                output_path, job.target_seconds
                            )
                            if r.get("adjusted"):
                                report.append(
                                    f"- 🔧 Post-stretched ×{r['stretch_ratio']:.3f}"
                                )
                except Exception as e:
                    report.append(f"- Metrics error: {e}")

            _update_progress(progress, 1.0, "Done")
            return (
                gr.update(value=output_path, visible=True),
                "\n".join(report),
            )

        # Wire up single generation button
        gen_button.click(
            gen_single,
            inputs=[
                emo_control_method, prompt_audio, input_text_single,
                emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, max_text_tokens_per_sentence,
                *advanced_params,
            ],
            outputs=[output_audio, single_duration_report],
            show_progress=True,
        )

        # Reference/Scale estimate updates
        prompt_audio.change(
            _reference_estimate,
            [
                prompt_audio, ref_text_input, input_text_single,
                language_dropdown, tokens_per_second_slider,
            ],
            [reference_estimate_display],
        )
        ref_text_input.change(
            _reference_estimate,
            [
                prompt_audio, ref_text_input, input_text_single,
                language_dropdown, tokens_per_second_slider,
            ],
            [reference_estimate_display],
        )
        prompt_audio.change(
            _scale_estimate,
            [prompt_audio, duration_scale_custom, tokens_per_second_slider],
            [scale_estimate_display],
        )
        duration_scale_custom.change(
            _scale_estimate,
            [prompt_audio, duration_scale_custom, tokens_per_second_slider],
            [scale_estimate_display],
        )

        # ── SRT Dubbing ───────────────────────────────────────────────────────

        def handle_srt_upload(file_obj):
            # FIX: Use safe path extraction
            file_path = _get_filepath(file_obj)
            if not file_path:
                return (
                    gr.update(value=[]),
                    "No file uploaded.",
                    gr.update(visible=False),
                )
            
            segs = parse_srt_file(file_path)
            if not segs:
                return (
                    gr.update(value=[]),
                    "❌ Parse failed or SRT is empty. Check file format.",
                    gr.update(visible=False),
                )
            
            enriched = enrich_srt_with_durations(segs)
            df_data = [
                [
                    s['index'], s['start'], s['end'],
                    f"{s['duration_seconds']:.3f}",
                    s['target_tokens'], s['text'],
                ]
                for s in enriched
            ]
            total = sum(s['duration_seconds'] for s in enriched)
            return (
                gr.update(value=df_data),
                f"✅ {len(segs)} segments parsed | Total duration: {total:.1f}s",
                gr.update(visible=False),
            )

        srt_file_input.upload(
            handle_srt_upload,
            [srt_file_input],
            [srt_preview_df, srt_status, download_srt_zip],
        )

        def run_srt_generation(
            srt_file_obj, ref_audio, ref_transcript, worker_count_val,
            emo_mode_val, emo_ref_val, emo_weight_val,
            v1, v2, v3, v4, v5, v6, v7, v8,
            emo_text_val, emo_rand_val, max_tokens_val,
            consistent_val, use_subtitle_dur_val,
            post_stretch_val, srt_lang_val,
            *adv_vals,
            progress=gr.Progress(),
        ):
            # FIX: Use safe path extraction
            file_path = _get_filepath(srt_file_obj)
            if not file_path:
                gr.Warning("Upload SRT file first.")
                return (gr.update(visible=False), "No SRT file.", gr.update(visible=False))
            if not ref_audio:
                gr.Warning("Upload voice reference first.")
                return (gr.update(visible=False), "No reference audio.", gr.update(visible=False))

            segs = parse_srt_file(file_path)
            if not segs:
                return (gr.update(visible=False), "❌ SRT parse failed.", gr.update(visible=False))
            
            enriched = enrich_srt_with_durations(segs)

            # FIX: Use a unique timestamped directory to avoid collision/deletion issues with Batch Tab
            out_dir = os.path.join(current_dir, "temp", f"srt_{int(time.time() * 1000)}")
            os.makedirs(out_dir, exist_ok=True)

            adv = list(adv_vals)
            if len(adv) < len(advanced_params):
                adv.extend([None] * (len(advanced_params) - len(adv)))
            base_kwargs = build_generation_kwargs(*adv[:len(advanced_params)])
            dur_fields = _extract_duration_fields(base_kwargs)

            # FIX: Safely cast emo_mode to int to prevent string/int mismatch bugs
            try:
                emo_mode = int(emo_mode_val)
            except Exception:
                emo_mode = 0

            vec_vals = [v1, v2, v3, v4, v5, v6, v7, v8]
            if emo_mode == 2 and sum(vec_vals) > 1.5:
                gr.Warning("Emotion vector sum > 1.5")
                return (gr.update(visible=False), "Error: Vector sum > 1.5", gr.update(visible=False))

            final_emo_vec = vec_vals if emo_mode == 2 else None
            use_cache = bool(consistent_val)
            batch_group = (
                f"srt_{int(time.time() * 1000)}_{id(ref_audio)}"
                if use_cache else None
            )

            jobs = []
            for seg in enriched:
                out_path = os.path.join(out_dir, f"segment-{seg['index']:04d}.wav")
                seg_mode = "SRT" if use_subtitle_dur_val else dur_fields["duration_mode"]
                
                jobs.append(GenerationJob(
                    row_id=seg['index'],
                    prompt_path=ref_audio,
                    text=seg['text'],
                    output_path=out_path,
                    emo_mode=emo_mode,  # FIX: Use the safely parsed integer
                    emo_weight=float(emo_weight_val) if emo_mode == 1 else 1.0,
                    emo_vector=final_emo_vec,
                    emo_text=emo_text_val,
                    emo_random=bool(emo_rand_val),
                    emo_ref_path=emo_ref_val if emo_mode == 1 else None,
                    max_tokens=int(max_tokens_val),
                    generation_kwargs=dict(base_kwargs),
                    verbose=cmd_args.verbose,
                    trim_silence=dur_fields["trim_silence"],
                    use_batch_latent_cache=use_cache,
                    batch_latent_group=batch_group,
                    duration_mode=seg_mode,
                    target_tokens=seg["target_tokens"],
                    target_seconds=seg["duration_seconds"],
                    duration_scale=dur_fields["duration_scale"],
                    ref_text=ref_transcript or "",
                    language=srt_lang_val or "en",
                    srt_duration_seconds=seg["duration_seconds"],
                    enable_post_stretch=post_stretch_val,
                    tokens_per_second=dur_fields["tokens_per_second"],
                ))

            try:
                _update_progress(progress, 0.0, "Starting SRT generation...")
                worker_pool.ensure(worker_count_val)
                results = worker_pool.run_jobs(jobs, progress)
                _update_progress(progress, 0.95, "Packaging ZIP...")

                report_rows = []
                for seg in enriched:
                    r = results.get(seg['index'], {})
                    m = r.get("duration_metrics", {})
                    if m and "token_error_rate_pct" in m:
                        report_rows.append([
                            seg['index'],
                            round(seg['duration_seconds'], 3),
                            round(m.get("generated_seconds", 0), 3),
                            round(m.get("token_error_rate_pct", 0), 4),
                            "✅" if m.get("within_10pct_tolerance") else "❌",
                        ])

                zip_path = os.path.join(out_dir, "dubbing_segments.zip")
                with zipfile.ZipFile(zip_path, 'w') as z:
                    for f in sorted(os.listdir(out_dir)):
                        if f.endswith('.wav'):
                            z.write(os.path.join(out_dir, f), arcname=f)

                ok = sum(1 for r in results.values() if r.get("status") == "Completed")
                avg_err = sum(
                    r.get("duration_metrics", {}).get("token_error_rate_pct", 0)
                    for r in results.values()
                ) / max(len(results), 1)

                _update_progress(progress, 1.0, "Done")
                return (
                    gr.update(value=zip_path, visible=True),
                    f"✅ {ok}/{len(segs)} done | Avg duration error: {avg_err:.4f}%",
                    gr.update(value=report_rows, visible=bool(report_rows)),
                )
            except Exception as e:
                logger.exception("SRT failed")
                return (
                    gr.update(visible=False),
                    f"❌ Error: {e}",
                    gr.update(visible=False),
                )

        generate_srt_button.click(
            run_srt_generation,
            inputs=[
                srt_file_input, srt_ref_audio, srt_ref_transcript,
                srt_worker_count,
                emo_control_method, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, max_text_tokens_per_sentence,
                srt_consistent_voice, srt_use_subtitle_duration,
                srt_post_stretch, srt_language,
                *advanced_params,
            ],
            outputs=[download_srt_zip, srt_status, srt_duration_report],
            show_progress=True,
        )

        # ── Batch Helpers ─────────────────────────────────────────────────────

        def _build_table(rows):
            return [
                [
                    r.get("id"),
                    os.path.basename(r.get("prompt_path", "") or ""),
                    (r.get("text", "") or "")[:57] + (
                        "..." if len(r.get("text", "") or "") > 60 else ""
                    ),
                    os.path.basename(r.get("output_path", "") or ""),
                    r.get("status", "Pending"),
                    r.get("duration_method", ""),
                    (
                        f"{r['token_error_pct']:.4f}%"
                        if isinstance(r.get("token_error_pct"), float)
                        else ""
                    ),
                ]
                for r in rows
            ]

        def _find_row(rows, row_id):
            return next(
                (r for r in (rows or []) if r.get("id") == row_id),
                None,
            )

        def _resolve_selection(rows, selected):
            choices = [str(r.get("id")) for r in (rows or [])]
            if not choices:
                return gr.update(choices=[], value=None), None
            sel_str = str(selected) if selected is not None else None
            if sel_str and sel_str in choices:
                return (
                    gr.update(choices=choices, value=sel_str),
                    int(sel_str),
                )
            return (
                gr.update(choices=choices, value=choices[-1]),
                int(choices[-1]),
            )

        def _prep_selection(rows, selected):
            dd, rid = _resolve_selection(rows, selected)
            row = _find_row(rows, rid)
            return (
                dd, rid,
                gr.update(
                    value=row.get("prompt_path") if row else None
                ),
                gr.update(
                    value=row.get("output_path") if row else None
                ),
                gr.update(value=row.get("text", "") if row else ""),
                row,
            )

        def _fmt_status(row, msg=None):
            if not row:
                base = "No entry selected."
            else:
                parts = [
                    f"Row {row.get('id')}: {row.get('status', 'Pending')}"
                ]
                if row.get("text"):
                    parts.append(f"Text: {row['text'][:100]}")
                if row.get("duration_method"):
                    parts.append(f"Duration: {row['duration_method']}")
                if isinstance(row.get("token_error_pct"), float):
                    parts.append(
                        f"Error: {row['token_error_pct']:.4f}% "
                        f"(<0.02% = paper benchmark)"
                    )
                if row.get("output_path"):
                    parts.append(f"Output: {row['output_path']}")
                base = "\n".join(parts)
            return gr.update(
                value=f"{base}\n{msg}" if msg else base
            )

        def add_batch_prompts(files, rows, next_id, selected):
            rows = list(rows or [])
            next_id = next_id or 1
            prompts_dir = os.path.join(current_dir, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)
            added = 0
            last_id = None
            for fp in (files or []):
                if not fp:
                    continue
                name = os.path.basename(fp)
                tgt = os.path.join(
                    prompts_dir,
                    f"batch_{next_id}_{int(time.time()*1000)}_{name}",
                )
                try:
                    shutil.copy(fp, tgt)
                except Exception as exc:
                    gr.Warning(f"Failed: {exc}")
                    continue
                rows.append({
                    "id": next_id, "prompt_path": tgt,
                    "output_path": None, "status": "Pending",
                    "last_generated": "", "text": "",
                    "duration_method": "", "token_error_pct": "",
                })
                added += 1
                last_id = next_id
                next_id += 1
            dd, rid, pu, ou, tu, row = _prep_selection(
                rows, last_id or selected
            )
            return (
                rows, next_id, gr.update(value=None),
                gr.update(value=_build_table(rows)),
                dd, pu, ou, tu,
                _fmt_status(
                    row,
                    f"Added {added}." if added else "None added.",
                ),
            )

        def load_dataset_entries(
            dataset_path, rows, next_id, selected,
            progress: gr.Progress = None,
        ):
            rows = list(rows or [])
            next_id = next_id or 1
            dp = (dataset_path or "").strip()

            def _unchanged(msg=None):
                dd, rid, pu, ou, tu, row = _prep_selection(rows, selected)
                return (
                    rows, next_id, gr.update(value=dp),
                    gr.update(value=_build_table(rows)),
                    dd, pu, ou, tu, _fmt_status(row, msg),
                )

            if not dp:
                return _unchanged("No path provided.")
            abs_dp = (
                dp if os.path.isabs(dp)
                else os.path.abspath(os.path.join(current_dir, dp))
            )
            if not os.path.exists(abs_dp):
                gr.Warning(f"Not found: {abs_dp}")
                return _unchanged()

            ddir = os.path.dirname(abs_dp)
            search_dirs = [
                ddir,
                os.path.join(ddir, "wavs"),
                os.path.join(ddir, "audio"),
            ]
            try:
                lines = Path(abs_dp).read_text(
                    encoding="utf-8"
                ).splitlines()
            except Exception as exc:
                gr.Warning(f"Read failed: {exc}")
                return _unchanged()

            prompts_dir = os.path.join(current_dir, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)
            existing = {
                os.path.basename(r.get("prompt_path", ""))
                for r in rows
            }
            added = missing = invalid = 0

            for idx, line in enumerate(lines):
                _update_progress(
                    progress,
                    min((idx + 1) / max(len(lines), 1), 0.95),
                    f"Line {idx+1}/{len(lines)}",
                )
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split("|", 1)
                if (len(parts) != 2
                        or not parts[0].strip()
                        or not parts[1].strip()):
                    invalid += 1
                    continue
                audio_name = parts[0].strip()
                text_val = parts[1].strip()
                src = next(
                    (
                        os.path.join(d, audio_name)
                        for d in search_dirs
                        if os.path.exists(os.path.join(d, audio_name))
                    ),
                    None,
                )
                if not src:
                    missing += 1
                    continue
                tgt_name = (
                    f"ds_{next_id}_{int(time.time()*1000)}_"
                    f"{os.path.basename(audio_name)}"
                )
                if tgt_name in existing:
                    tgt_name = (
                        f"ds_{next_id}_b_"
                        f"{os.path.basename(audio_name)}"
                    )
                tgt = os.path.join(prompts_dir, tgt_name)
                try:
                    shutil.copy(src, tgt)
                except Exception:
                    missing += 1
                    continue
                rows.append({
                    "id": next_id, "prompt_path": tgt,
                    "output_path": None, "status": "Pending",
                    "last_generated": "", "text": text_val,
                    "duration_method": "", "token_error_pct": "",
                })
                existing.add(tgt_name)
                added += 1
                next_id += 1

            _update_progress(progress, 1.0, "Done")
            dd, rid, pu, ou, tu, row = _prep_selection(
                rows, rows[-1]["id"] if added else selected
            )
            msg = ", ".join(filter(None, [
                f"Loaded {added}" if added else None,
                f"{missing} missing" if missing else None,
                f"{invalid} invalid" if invalid else None,
            ])) or "No entries."
            return (
                rows, next_id, gr.update(value=dp),
                gr.update(value=_build_table(rows)),
                dd, pu, ou, tu, _fmt_status(row, msg),
            )

        def _make_jobs_for_rows(
            rows, row_ids_to_run,
            emo_mode, emo_ref, emo_weight_val,
            vec_vals, emo_text_val, emo_rand_val,
            max_tokens_val, dur_fields, base_kwargs, emo_vector,
        ):
            jobs = []
            row_map = {}
            batch_ts = int(time.time() * 1000)
            group_map: Dict[str, str] = {}

            for row in rows:
                if row.get("id") not in row_ids_to_run:
                    continue
                nr = dict(row)
                pp = nr.get("prompt_path")
                if not pp or not os.path.exists(pp):
                    nr["status"] = "Error: Prompt missing"
                    row_map[nr["id"]] = nr
                    continue
                text = (nr.get("text") or "").strip()
                if not text:
                    nr["status"] = "Error: Text missing"
                    row_map[nr["id"]] = nr
                    continue
                out = os.path.join(
                    current_dir, "outputs", "tasks",
                    f"batch_{nr['id']}_{int(time.time()*1000)}.wav",
                )
                nr["status"] = "Running"
                nr["output_path"] = out
                row_map[nr["id"]] = nr

                abs_pp = os.path.abspath(pp)
                abs_emo = (
                    os.path.abspath(emo_ref)
                    if emo_mode == 1 and emo_ref
                    else abs_pp
                )
                gk = f"{abs_pp}||{abs_emo}"
                if gk not in group_map:
                    group_map[gk] = f"bg_{batch_ts}_{len(group_map)}"

                jobs.append(GenerationJob(
                    row_id=nr["id"],
                    prompt_path=pp,
                    text=apply_language_prefix(text, dur_fields["language"]),
                    output_path=out,
                    emo_mode=emo_mode,
                    emo_weight=(
                        float(emo_weight_val) if emo_mode == 1 else 1.0
                    ),
                    emo_vector=emo_vector if emo_mode == 2 else None,
                    emo_text=emo_text_val,
                    emo_random=bool(emo_rand_val),
                    emo_ref_path=emo_ref if emo_mode == 1 else None,
                    max_tokens=int(max_tokens_val),
                    generation_kwargs=dict(base_kwargs),
                    verbose=cmd_args.verbose,
                    trim_silence=dur_fields["trim_silence"],
                    use_batch_latent_cache=True,
                    batch_latent_group=group_map[gk],
                    duration_mode=dur_fields["duration_mode"],
                    target_tokens=dur_fields["target_tokens"],
                    target_seconds=dur_fields["target_seconds"],
                    duration_scale=dur_fields["duration_scale"],
                    ref_text=dur_fields["ref_text"],
                    language=dur_fields["language"],
                    tokens_per_second=dur_fields["tokens_per_second"],
                    enable_post_stretch=dur_fields["enable_post_stretch"],
                ))
            return jobs, row_map

        def _apply_results(rows, row_map, results):
            for rid, r in results.items():
                e = row_map.get(rid)
                if not e:
                    continue
                e["status"] = r["status"]
                e["last_generated"] = r.get("timestamp", "")
                e["duration_method"] = r.get("duration_method", "")
                m = r.get("duration_metrics", {})
                if m and "token_error_rate_pct" in m:
                    e["token_error_pct"] = m["token_error_rate_pct"]
                if r.get("output_path"):
                    e["output_path"] = r["output_path"]
            return list(row_map.values())

        def generate_all_batch(
            rows, selected, wc,
            emo_mode_val, emo_ref, emo_wt,
            v1, v2, v3, v4, v5, v6, v7, v8,
            emo_text_val, emo_rand_val, max_tok_val,
            *adv_vals,
            progress: gr.Progress = None,
        ):
            rows = list(rows or [])
            if not rows:
                gr.Warning("No entries.")
                dd, _, pu, ou, tu, row = _prep_selection(rows, selected)
                return (
                    rows, gr.update(value=_build_table(rows)),
                    dd, pu, ou, tu, _fmt_status(row),
                )
            if not parallel_worker_config.get("gpt_path"):
                gr.Warning("Load model first.")
                dd, _, pu, ou, tu, row = _prep_selection(rows, selected)
                return (
                    rows, gr.update(value=_build_table(rows)),
                    dd, pu, ou, tu,
                    _fmt_status(row, "No model."),
                )

            adv = list(adv_vals)
            if len(adv) < len(advanced_params):
                adv.extend([None] * (len(advanced_params) - len(adv)))
            base_kw = build_generation_kwargs(*adv[:len(advanced_params)])
            dur_f = _extract_duration_fields(base_kw)

            try:
                emo_mode = int(emo_mode_val)
            except Exception:
                emo_mode = 0

            emo_vector = None
            if emo_mode == 2:
                vv = [v1, v2, v3, v4, v5, v6, v7, v8]
                if sum(vv) > 1.5:
                    gr.Warning("Emotion vector sum > 1.5")
                    dd, _, pu, ou, tu, row = _prep_selection(rows, selected)
                    return (
                        rows, gr.update(value=_build_table(rows)),
                        dd, pu, ou, tu, _fmt_status(row),
                    )
                emo_vector = vv

            ids = {r.get("id") for r in rows}
            jobs, row_map = _make_jobs_for_rows(
                rows, ids, emo_mode, emo_ref, emo_wt,
                [v1, v2, v3, v4, v5, v6, v7, v8],
                emo_text_val, emo_rand_val, max_tok_val,
                dur_f, base_kw, emo_vector,
            )

            if not jobs:
                dd, _, pu, ou, tu, row = _prep_selection(
                    list(row_map.values()), selected
                )
                return (
                    list(row_map.values()),
                    gr.update(value=_build_table(list(row_map.values()))),
                    dd, pu, ou, tu,
                    _fmt_status(row, "No jobs."),
                )

            _update_progress(progress, 0.0, "Starting...")
            worker_pool.ensure(wc)
            results = worker_pool.run_jobs(jobs, progress)
            final = _apply_results(rows, row_map, results)
            dd, rid, pu, ou, tu, row = _prep_selection(final, selected)
            return (
                final, gr.update(value=_build_table(final)),
                dd, pu, ou, tu, _fmt_status(row, "Done."),
            )

        def regenerate_batch_entry(
            rows, selected, wc,
            emo_mode_val, emo_ref, emo_wt,
            v1, v2, v3, v4, v5, v6, v7, v8,
            emo_text_val, emo_rand_val, max_tok_val,
            *adv_vals,
            progress: gr.Progress = None,
        ):
            rows = list(rows or [])
            dd, rid, pu, ou, tu, sel_row = _prep_selection(rows, selected)
            if not sel_row:
                gr.Warning("Select an entry.")
                return (
                    rows, gr.update(value=_build_table(rows)),
                    dd, pu, ou, tu, _fmt_status(None),
                )

            adv = list(adv_vals)
            if len(adv) < len(advanced_params):
                adv.extend([None] * (len(advanced_params) - len(adv)))
            base_kw = build_generation_kwargs(*adv[:len(advanced_params)])
            dur_f = _extract_duration_fields(base_kw)

            try:
                emo_mode = int(emo_mode_val)
            except Exception:
                emo_mode = 0
            emo_vector = (
                [v1, v2, v3, v4, v5, v6, v7, v8]
                if emo_mode == 2 else None
            )

            jobs, row_map = _make_jobs_for_rows(
                rows, {sel_row["id"]}, emo_mode, emo_ref, emo_wt,
                [v1, v2, v3, v4, v5, v6, v7, v8],
                emo_text_val, emo_rand_val, max_tok_val,
                dur_f, base_kw, emo_vector,
            )
            if not jobs:
                return (
                    rows, gr.update(value=_build_table(rows)),
                    dd, pu, ou, tu,
                    _fmt_status(sel_row, "Nothing to run."),
                )

            _update_progress(progress, 0.0, "Regenerating...")
            worker_pool.ensure(wc)
            results = worker_pool.run_jobs(jobs, progress)
            updated = []
            for row in rows:
                if row.get("id") not in row_map:
                    updated.append(dict(row))
                    continue
                nr = dict(row)
                r = results.get(row["id"])
                if r:
                    nr["status"] = r["status"]
                    nr["output_path"] = r.get(
                        "output_path", nr.get("output_path")
                    )
                    nr["last_generated"] = r.get("timestamp", "")
                    nr["duration_method"] = r.get("duration_method", "")
                    m = r.get("duration_metrics", {})
                    if m and "token_error_rate_pct" in m:
                        nr["token_error_pct"] = m["token_error_rate_pct"]
                updated.append(nr)
            dd, rid, pu, ou, tu, row = _prep_selection(
                updated, sel_row["id"]
            )
            return (
                updated, gr.update(value=_build_table(updated)),
                dd, pu, ou, tu, _fmt_status(row, "Done."),
            )

        def delete_batch_entry(rows, selected):
            rows = list(rows or [])
            dd, rid, pu, ou, tu, sel_row = _prep_selection(rows, selected)
            if not sel_row:
                gr.Warning("Select entry.")
                return (
                    rows, gr.update(value=_build_table(rows)),
                    dd, pu, ou, tu, _fmt_status(None),
                )
            remaining = [
                dict(r) for r in rows
                if r.get("id") != sel_row.get("id")
            ]
            dd, rid, pu, ou, tu, row = _prep_selection(remaining, None)
            return (
                remaining, gr.update(value=_build_table(remaining)),
                dd, pu, ou, tu, _fmt_status(row, "Deleted."),
            )

        def clear_batch_rows(rows, next_id):
            return (
                [], 1, gr.update(value=[]),
                gr.update(choices=[], value=None),
                gr.update(value=None), gr.update(value=None),
                gr.update(value=""), _fmt_status(None, "Cleared."),
            )

        def on_select_entry(sel, rows):
            dd, rid, pu, ou, tu, row = _prep_selection(rows, sel)
            return dd, pu, ou, tu, _fmt_status(row)

        def update_batch_text(new_text, rows, selected):
            rows = list(rows or [])
            try:
                sel_id = (
                    int(selected) if selected is not None else None
                )
            except (TypeError, ValueError):
                sel_id = None
            if sel_id is None:
                gr.Warning("Select entry first.")
                dd, rid, pu, ou, tu, row = _prep_selection(rows, selected)
                return (
                    rows, gr.update(value=_build_table(rows)),
                    dd, pu, ou, tu, _fmt_status(row),
                )
            updated = []
            target_row = None
            for row in rows:
                nr = dict(row)
                if nr.get("id") == sel_id:
                    nr["text"] = new_text
                    if nr.get("output_path"):
                        nr["status"] = "Pending"
                    target_row = nr
                updated.append(nr)
            dd, rid, pu, ou, tu, row = _prep_selection(updated, sel_id)
            return (
                updated, gr.update(value=_build_table(updated)),
                dd, pu, ou, tu,
                _fmt_status(target_row, "Text updated."),
            )

        # Wire up all batch events
        batch_outputs = [
            batch_rows_state, batch_table, selected_entry,
            batch_prompt_player, batch_output_player,
            batch_text_input, batch_status,
        ]

        batch_file_input.upload(
            add_batch_prompts,
            [
                batch_file_input, batch_rows_state,
                next_batch_id_state, selected_entry,
            ],
            [
                batch_rows_state, next_batch_id_state,
                batch_file_input, batch_table,
                selected_entry, batch_prompt_player,
                batch_output_player, batch_text_input, batch_status,
            ],
        )
        load_dataset_button.click(
            load_dataset_entries,
            [
                dataset_path_input, batch_rows_state,
                next_batch_id_state, selected_entry,
            ],
            [
                batch_rows_state, next_batch_id_state,
                dataset_path_input, batch_table,
                selected_entry, batch_prompt_player,
                batch_output_player, batch_text_input, batch_status,
            ],
        )
        selected_entry.change(
            on_select_entry,
            [selected_entry, batch_rows_state],
            [
                selected_entry, batch_prompt_player,
                batch_output_player, batch_text_input, batch_status,
            ],
        )
        apply_text_button.click(
            update_batch_text,
            [batch_text_input, batch_rows_state, selected_entry],
            [
                batch_rows_state, batch_table, selected_entry,
                batch_prompt_player, batch_output_player,
                batch_text_input, batch_status,
            ],
        )
        generate_all_button.click(
            generate_all_batch,
            [
                batch_rows_state, selected_entry, worker_count,
                emo_control_method, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, max_text_tokens_per_sentence,
                *advanced_params,
            ],
            batch_outputs,
        )
        regenerate_button.click(
            regenerate_batch_entry,
            [
                batch_rows_state, selected_entry, worker_count,
                emo_control_method, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, max_text_tokens_per_sentence,
                *advanced_params,
            ],
            batch_outputs,
        )
        delete_entry_button.click(
            delete_batch_entry,
            [batch_rows_state, selected_entry],
            batch_outputs,
        )
        clear_entries_button.click(
            clear_batch_rows,
            [batch_rows_state, next_batch_id_state],
            [
                batch_rows_state, next_batch_id_state,
                batch_table, selected_entry,
                batch_prompt_player, batch_output_player,
                batch_text_input, batch_status,
            ],
        )

    return demo


def main():
    demo = create_demo()
    demo.queue(20)
    demo.launch(
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        share=cmd_args.share,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()