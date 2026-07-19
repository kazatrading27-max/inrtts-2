import html
import inspect
import json
import os
import re
import sys
import threading
import time
import warnings as warnings_module
import zipfile

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import torch

# === Deterministic CUDA/cuDNN behavior =====================================
# Purely for seed reproducibility; does not affect decoding strategy or
# sampling logic. Confirmed NOT implicated in any regression tested so far.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args, _ = parser.parse_known_args()

# ── Required files check ──────────────────────────────────────────────────────
required_files = [
    "bpe.model",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
    "gpt.pth"
]
missing = [f for f in required_files if not os.path.exists(os.path.join(cmd_args.model_dir, f))]
if missing:
    print(
        f"Model directory {cmd_args.model_dir} is incomplete (missing: {', '.join(missing)}). "
        "Downloading IndexTTS-2 model..."
    )
    from indextts.utils.model_download import snapshot_download
    try:
        snapshot_download("IndexTeam/IndexTTS-2", local_dir=cmd_args.model_dir)
    except Exception as e:
        print(f"Failed to download model to {cmd_args.model_dir}: {e}")
        sys.exit(1)
    missing = [f for f in required_files if not os.path.exists(os.path.join(cmd_args.model_dir, f))]
    if missing:
        print(f"Failed to download model to {cmd_args.model_dir} (still missing: {', '.join(missing)}). Please download it manually.")
        sys.exit(1)
    print("Model downloaded successfully.")

from indextts.utils.model_download import ensure_config_available
try:
    ensure_config_available(cmd_args.model_dir)
except Exception as e:
    print(f"Failed to download config.yaml: {e}")
    sys.exit(1)

import gradio as gr

from indextts.infer_v2 import IndexTTS2

from indextts.utils.examples_downloader import ensure_examples_available
from indextts.utils.presets import list_presets, save_preset, load_preset, delete_preset
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'

ensure_examples_available()

# ── IndexTTS2 initialization ──────────────────────────────────────────────────
tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
)

print(f"[WebUI] Standard PyTorch mode.")

# ---------------------------------------------------------------------------
# NOTE ON THE ACTUAL ROOT CAUSE (confirmed via indextts/infer_v2.py source,
# lines 549-551):
#
#   estimated_max = (text_tokens.shape[1] * 15) + 200
#   dynamic_max_mel_tokens = max(100, min(max_mel_tokens, estimated_max))
#
# This dynamic per-segment mel-token budget SILENTLY OVERRIDES the UI's
# max_mel_tokens setting downward for any segment shorter than ~87 text
# tokens (nearly every SRT line). The "15 mel-tokens per text-token"
# heuristic was evidently tuned for the base model's training languages
# and is too tight for Amharic's morphologically dense subword tokens,
# causing GPT generation to be forcibly cut off before reaching its
# natural end-of-speech token on some segments (confirmed directly via a
# RuntimeWarning fired by the vendor code at generation time) -- this is
# the actual mechanism behind "some segments perfect, some inconsistent",
# not concurrency, caching, decoding strategy, or reference-audio cropping
# (all separately tested and ruled out).
#
# THE PRIMARY FIX is a source patch to indextts/infer_v2.py itself:
#   estimated_max = (text_tokens.shape[1] * 30) + 400
# (roughly doubling headroom). Apply that patch to your local checkout.
#
# The code below (infer_with_truncation_detection) is a DEFENSE-IN-DEPTH
# safety net on top of that source patch: it detects that specific
# RuntimeWarning if it still fires on any outlier segment, and forces an
# automatic retry with a new seed through the existing QA retry loop,
# rather than silently shipping a truncated segment.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tts is a SINGLE GLOBAL SINGLETON shared by every request that hits this
# Gradio app, carrying mutable cross-call conditioning cache state. The app
# uses demo.queue(default_concurrency_limit=20), allowing up to 20 concurrent
# requests against this SAME object. global_infer_lock serializes access.
# Verified via logs to correctly serialize every call (clean
# Acquire -> ... -> Release, no interleaving) -- kept as a correct safety
# net for the documented concurrency model, even though concurrency has
# been separately ruled out as the cause of the voice-drift symptom.
# ---------------------------------------------------------------------------
global_infer_lock = threading.Lock()

try:
    _INFER_SIGNATURE = inspect.signature(tts.infer)
    _INFER_PARAMS = _INFER_SIGNATURE.parameters
    _INFER_NAMED_PARAMS = set(
        name for name, p in _INFER_PARAMS.items()
        if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    )
    _INFER_HAS_VAR_KEYWORD = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in _INFER_PARAMS.values()
    )
    print("=" * 70)
    print("[Introspection] Verified tts.infer() signature (ground truth):")
    print(f"  {_INFER_SIGNATURE}")
    print(f"[Introspection] Explicitly named parameters: {sorted(_INFER_NAMED_PARAMS)}")
    print(f"[Introspection] Accepts arbitrary extra kwargs (**generation_kwargs): {_INFER_HAS_VAR_KEYWORD}")
    print("=" * 70)
except Exception as e:
    print(f"[Introspection] WARNING: Could not introspect tts.infer signature: {e}")
    _INFER_NAMED_PARAMS = None
    _INFER_HAS_VAR_KEYWORD = True

_TTS_CACHE_ATTRS = [a for a in vars(tts).keys() if a.lower().startswith("cache")]
if _TTS_CACHE_ATTRS:
    print(f"[Introspection] Detected SHARED mutable conditioning cache attributes on the tts singleton: {_TTS_CACHE_ATTRS}")
    print("[Introspection] These are now protected by global_infer_lock during every tts.infer() call.")
else:
    print("[Introspection] No cache-like attributes detected on tts instance.")


def _cache_fingerprint():
    """Diagnostic only (verbose mode): hash of cache_* tensors, confirmed
    stable/unmutated across segments in prior testing."""
    import hashlib
    parts = {}
    for attr in _TTS_CACHE_ATTRS:
        v = getattr(tts, attr, None)
        try:
            if torch.is_tensor(v):
                parts[attr] = hashlib.md5(v.detach().cpu().numpy().tobytes()).hexdigest()[:10]
            elif v is None:
                parts[attr] = "None"
            else:
                parts[attr] = hashlib.md5(str(v).encode("utf-8", errors="ignore")).hexdigest()[:10]
        except Exception as e:
            parts[attr] = f"<err:{e}>"
    return parts


def safe_tts_infer(**kwargs):
    """
    Calls tts.infer() using its verified real signature, serializing ALL
    access to the shared tts singleton via global_infer_lock.
    """
    if _INFER_NAMED_PARAMS is not None and not _INFER_HAS_VAR_KEYWORD:
        filtered = {}
        dropped = []
        for k, v in kwargs.items():
            if k in _INFER_NAMED_PARAMS:
                filtered[k] = v
            else:
                dropped.append(k)
        if dropped:
            print(f"[safe_tts_infer] WARNING: Dropped unsupported kwargs (no **kwargs catch-all found): {dropped}")
        call_kwargs = filtered
    else:
        call_kwargs = kwargs

    if cmd_args.verbose:
        print(f"[safe_tts_infer] Acquiring global_infer_lock for call with text: {str(call_kwargs.get('text',''))[:60]}...")

    with global_infer_lock:
        if cmd_args.verbose:
            print(f"[safe_tts_infer] Lock acquired. Calling tts.infer() with kwargs: { {k: v for k, v in call_kwargs.items() if k != 'spk_audio_prompt'} }")
        result = tts.infer(**call_kwargs)
        if cmd_args.verbose:
            print(f"[safe_tts_infer] tts.infer() completed. Releasing global_infer_lock.")
            print(f"[safe_tts_infer] Post-call cache fingerprint: {_cache_fingerprint()}")
    return result


# === Truncation detection (defense-in-depth on top of the source patch) ===
# Confirmed exact vendor warning text (indextts/infer_v2.py ~line 591):
#   "generation stopped due to exceeding dynamic `max_mel_tokens` (...).
#    Input text tokens: .... Consider reducing
#    `max_text_tokens_per_segment`(...)."
# This fires via warnings.warn(...) whenever GPT decoding is forcibly cut
# off before reaching its natural end-of-speech token, due to the internal
# dynamic_max_mel_tokens budget (see note above). This is DIRECT evidence
# of an incomplete/corrupted generation -- not inferred, not guessed.
# =============================================================================
_TRUNCATION_WARNING_MARKERS = ("exceeding dynamic", "max_mel_tokens")

def infer_with_truncation_detection(**kwargs):
    """
    Wraps safe_tts_infer() while capturing Python warnings, specifically
    watching for the vendor's dynamic-max-mel-tokens truncation warning.
    Returns (result, was_truncated: bool, warning_text: str or None).
    """
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        result = safe_tts_infer(**kwargs)
        truncation_warning = None
        for w in caught:
            msg = str(w.message)
            if all(marker in msg for marker in _TRUNCATION_WARNING_MARKERS):
                truncation_warning = msg
                break
    return result, (truncation_warning is not None), truncation_warning


def build_generation_kwargs(do_sample, top_p, top_k, temperature,
                             length_penalty, num_beams, repetition_penalty,
                             max_mel_tokens):
    """No decoding-strategy override. do_sample+num_beams>1 (beam_sample)
    was tested and confirmed BENEFICIAL for this finetune -- it acts as a
    built-in best-of-N candidate filter by cumulative log-probability.
    Forcing num_beams=1 was tested and made output measurably worse."""
    return {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": int(num_beams),
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }


# === Reference audio pre-trim (kept from prior fix; harmless/beneficial
# even though the truncation-cropping hypothesis was separately ruled out
# by your grep results -- this still guarantees byte-identical reference
# conditioning across all SRT segments for any file >15s). ==================
_REF_AUDIO_MAX_DURATION_SEC = 15.0

def _prepare_fixed_reference_audio(path, out_dir, max_duration_sec=_REF_AUDIO_MAX_DURATION_SEC):
    if not path or not os.path.exists(path):
        return path
    try:
        y, sr = librosa.load(path, sr=None)
    except Exception as e:
        print(f">> WARNING: could not pre-check reference audio '{path}' for length ({e}); leaving as-is.")
        return path
    max_samples = int(max_duration_sec * sr)
    if len(y) <= max_samples:
        return path

    y_fixed = y[:max_samples]
    os.makedirs(out_dir, exist_ok=True)
    fixed_path = os.path.join(out_dir, f"_fixed_ref_{os.path.basename(path)}")
    if not fixed_path.lower().endswith(".wav"):
        fixed_path += ".wav"
    sf.write(fixed_path, y_fixed, sr)
    print(
        f">> Reference audio '{os.path.basename(path)}' is {len(y)/sr:.2f}s, exceeding the model's "
        f"{max_duration_sec:.1f}s cap. Pre-trimmed ONCE to a FIXED {max_duration_sec:.1f}s window at "
        f"'{fixed_path}' so every SRT segment conditions on identical reference bytes."
    )
    return fixed_path
# ============================================================================


LANGUAGES = {
    "Chinese": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("Same as voice reference audio"),
                i18n("Use emotion reference audio"),
                i18n("Use emotion vector control"),
                i18n("Use emotion description text control")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None

        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES_ALL[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def format_glossary_markdown():
    if not tts.normalizer.term_glossary:
        return i18n("No terms yet")
    lines = [f"| {i18n('Term')} | {i18n('Chinese Reading')} | {i18n('English Reading')} |"]
    lines.append("|---|---|---|")
    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SRT Parsing & Audio Trimming Helpers
# ---------------------------------------------------------------------------

def parse_srt_file(file_path):
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
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    pattern = re.compile(
        r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n*$)'
    )
    segments = []
    for match in pattern.findall(content):
        text = re.sub(r'<[^>]+>', '', match[3].strip()).replace('\n', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        if not text:
            continue
        segments.append({
            "index": int(match[0]),
            "start": match[1],
            "end": match[2],
            "text": text
        })
    segments.sort(key=lambda x: x['index'])
    return segments


def _detect_silence_regions(y, sr, window_ms=25, rms_threshold=0.005):
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

def _rms_trim_edges(y, sr, rms_threshold=0.005, window_ms=25, margin_ms=80):
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

def trim_audio_wav(file_path, top_db=30, margin_ms=80, max_internal_silence_ms=600, compress_silence_to_ms=200):
    try:
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
            yt, sr, max_silence_ms=max_internal_silence_ms, compress_to_ms=compress_silence_to_ms
        )
        trimmed_duration = len(yt) / sr
        if len(yt) > int(sr * 0.1):
            sf.write(file_path, yt, sr)
            removed = original_duration - trimmed_duration
            if removed > 0.05:
                print(f">> Trimmed {basename}: {original_duration:.2f}s -> {trimmed_duration:.2f}s (removed {removed:.2f}s)")
    except Exception as e:
        print(f">> Failed to process audio {file_path}: {e}")


# ---------------------------------------------------------------------------
# Audio Quality Gate
# ---------------------------------------------------------------------------

def evaluate_audio_quality(file_path, text, min_chars_per_sec=2.0, max_chars_per_sec=25.0):
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        return False, f"failed to load audio for QA: {e}"

    if len(y) == 0:
        return False, "empty audio file"

    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))

    text_len = max(1, len(text.strip()))
    expected_min_duration = text_len / max_chars_per_sec
    expected_max_duration = text_len / min_chars_per_sec

    if rms < 0.0015:
        return False, f"near-total silence detected (rms={rms:.5f})"

    clip_ratio = float(np.mean(np.abs(y) > 0.995))
    if clip_ratio > 0.02:
        return False, f"excessive clipping detected (clip_ratio={clip_ratio:.3f})"

    if duration < expected_min_duration * 0.5:
        return False, f"audio drastically shorter than expected for text length (got {duration:.2f}s, expected >= {expected_min_duration*0.5:.2f}s)"

    if duration > expected_max_duration * 1.8:
        return False, f"audio drastically longer than expected for text length (got {duration:.2f}s, expected <= {expected_max_duration*1.8:.2f}s)"

    return True, "ok"


# ── gen_srt and gen_single ───────────────────

def gen_srt(prompt, srt_file,
            emo_control_method, emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_text_tokens_per_segment=120,
            *args, progress=gr.Progress()):
    
    if not prompt:
        gr.Warning(i18n("Please upload a reference audio in the \"Audio Generation\" tab first"))
        return gr.update(visible=False), i18n("Error: Missing reference audio")
    if not srt_file:
        gr.Warning(i18n("Please upload an SRT file"))
        return gr.update(visible=False), i18n("Error: Missing SRT file")
    
    srt_path = srt_file.name if hasattr(srt_file, 'name') else srt_file
    segments = parse_srt_file(srt_path)
    
    if not segments:
        gr.Warning(i18n("SRT parsing failed or empty"))
        return gr.update(visible=False), i18n("Error: SRT parsing failed")
    
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens, trim_silence_val, \
        max_retries_val, base_seed_val = args

    generation_kwargs = build_generation_kwargs(
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty, max_mel_tokens
    )

    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value

    out_dir = os.path.join("outputs", "tasks", f"srt_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)

    prompt = _prepare_fixed_reference_audio(prompt, out_dir)

    if emo_control_method == 0:
        emo_ref_path = prompt
    elif emo_ref_path:
        emo_ref_path = _prepare_fixed_reference_audio(emo_ref_path, out_dir)

    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        vec = None

    if emo_text == "":
        emo_text = None

    tts.gr_progress = progress
    total = len(segments)
    success_count = 0
    max_retries = int(max_retries_val)
    base_seed = int(base_seed_val)
    
    progress(0, desc=f"Start generating SRT dubbing (total {total} segments)...")
    
    qa_log_lines = []
    
    for i, seg in enumerate(segments):
        progress((i + 1) / total, desc=f"Generating segment {i+1}/{total}...")
        out_path = os.path.join(out_dir, f"segment-{seg['index']:04d}.wav")

        segment_ok = False
        last_reason = ""
        for attempt in range(max_retries + 1):
            if attempt == 0:
                seed_for_attempt = base_seed
            else:
                seed_for_attempt = base_seed + (i * 1000) + attempt

            try:
                torch.manual_seed(seed_for_attempt)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_for_attempt)

                if cmd_args.verbose:
                    print(f">> Segment {seg['index']} attempt {attempt+1}: seed={seed_for_attempt}")

                _, was_truncated, trunc_msg = infer_with_truncation_detection(
                    spk_audio_prompt=prompt,
                    text=seg['text'],
                    output_path=out_path,
                    emo_audio_prompt=emo_ref_path,
                    emo_alpha=emo_weight,
                    emo_vector=vec,
                    use_emo_text=(emo_control_method==3),
                    emo_text=emo_text,
                    use_random=emo_random,
                    verbose=cmd_args.verbose,
                    max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                    **generation_kwargs
                )

                if was_truncated:
                    last_reason = f"GPT generation forcibly truncated by internal dynamic max_mel_tokens cap: {trunc_msg}"
                    print(f">> Segment {seg['index']} attempt {attempt+1}/{max_retries+1}: {last_reason} Retrying with new seed...")
                    continue

                is_ok, reason = evaluate_audio_quality(out_path, seg['text'])
                last_reason = reason
                if is_ok:
                    segment_ok = True
                    break
                else:
                    print(f">> Segment {seg['index']} attempt {attempt+1}/{max_retries+1} failed QA check: {reason}. Retrying with new seed...")
            except Exception as e:
                last_reason = str(e)
                print(f"Error generating segment {seg['index']} (attempt {attempt+1}): {e}")

        if segment_ok:
            if trim_silence_val:
                trim_audio_wav(out_path)
            success_count += 1
        else:
            qa_log_lines.append(f"Segment {seg['index']}: FAILED after {max_retries+1} attempts ({last_reason})")
            print(f">> Segment {seg['index']} permanently failed QA after {max_retries+1} attempts: {last_reason}")

    if success_count == 0:
        return gr.update(visible=False), i18n("All segments failed to generate, please check logs")
        
    zip_path = os.path.join(out_dir, "dubbing_segments.zip")
    with zipfile.ZipFile(zip_path, 'w') as z:
        for f in sorted(os.listdir(out_dir)):
            if f.endswith('.wav'):
                z.write(os.path.join(out_dir, f), arcname=f)
        if qa_log_lines:
            log_path = os.path.join(out_dir, "qa_failures.log")
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(qa_log_lines))
            z.write(log_path, arcname="qa_failures.log")

    status_msg = f"Done! Successfully generated {success_count}/{total} segments."
    if qa_log_lines:
        status_msg += f" ({len(qa_log_lines)} segment(s) failed QA even after retries — see qa_failures.log in the ZIP)"

    return gr.update(value=zip_path, visible=True), status_msg


def gen_single(emo_control_method, prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text, emo_random,
               max_text_tokens_per_segment=120,
               *args, progress=gr.Progress()):
    output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens, trim_silence_val, \
        max_retries_val, base_seed_val = args

    generation_kwargs = build_generation_kwargs(
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty, max_mel_tokens
    )

    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = prompt
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        vec = None
    if emo_text == "":
        emo_text = None

    base_seed = int(base_seed_val)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    _, was_truncated, trunc_msg = infer_with_truncation_detection(
        spk_audio_prompt=prompt,
        text=text,
        output_path=output_path,
        emo_audio_prompt=emo_ref_path,
        emo_alpha=emo_weight,
        emo_vector=vec,
        use_emo_text=(emo_control_method==3),
        emo_text=emo_text,
        use_random=emo_random,
        verbose=cmd_args.verbose,
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        **generation_kwargs
    )
    if was_truncated:
        gr.Warning(i18n("Generation was truncated by the model's internal length estimate; consider raising max_mel_tokens or applying the infer_v2.py source patch for this language.") + f" ({trunc_msg})")
    if trim_silence_val:
        trim_audio_wav(output_path)
    return gr.update(value=output_path, visible=True)


# ---------------------------------------------------------------------------
# Preset management (unchanged)
# ---------------------------------------------------------------------------

def _build_preset_data(
    emo_control_method, emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text, emo_random,
    do_sample, top_p, top_k, temperature,
    length_penalty, num_beams, repetition_penalty,
    max_mel_tokens, max_text_tokens_per_segment,
):
    return {
        "emo_control_method": int(emo_control_method) if emo_control_method is not None else 0,
        "emo_weight": float(emo_weight) if emo_weight is not None else 0.65,
        "emo_vector": [
            float(v) if v is not None else 0.0
            for v in [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        ],
        "emo_text": emo_text or "",
        "emo_random": bool(emo_random),
        "advanced_params": {
            "do_sample": bool(do_sample),
            "top_p": float(top_p) if top_p is not None else 0.8,
            "top_k": int(top_k) if top_k is not None else 30,
            "temperature": float(temperature) if temperature is not None else 0.8,
            "length_penalty": float(length_penalty) if length_penalty is not None else 0.0,
            "num_beams": int(num_beams) if num_beams is not None else 3,
            "repetition_penalty": float(repetition_penalty) if repetition_penalty is not None else 2.0,
            "max_mel_tokens": int(max_mel_tokens) if max_mel_tokens is not None else 1500,
            "max_text_tokens_per_segment": int(max_text_tokens_per_segment)
            if max_text_tokens_per_segment is not None else 120,
        },
    }


def on_preset_save(
    name, prompt_audio, emo_control_method, emo_ref_path, emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text, emo_random,
    do_sample, top_p, top_k, temperature,
    length_penalty, num_beams, repetition_penalty,
    max_mel_tokens, max_text_tokens_per_segment,
):
    name = name.strip() if name else ""
    if not name:
        gr.Warning(i18n("Preset name cannot be empty"))
        return gr.update()

    data = _build_preset_data(
        emo_control_method, emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text, emo_random,
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty,
        max_mel_tokens, max_text_tokens_per_segment,
    )

    existed = name in list_presets()
    try:
        save_preset(name, data, prompt_audio=prompt_audio, emo_audio=emo_ref_path)
        msg = i18n("Preset name already exists, overwritten") if existed else i18n("Preset saved")
        gr.Info(msg, duration=2)
    except Exception as e:
        gr.Error(f"{i18n('Failed to load preset')}: {e}")
        return gr.update(), gr.update()

    choices = [""] + list_presets()
    has_presets = len(choices) > 1
    return (
        gr.update(choices=choices, value=name, interactive=has_presets),
        gr.update(choices=choices, value=name, interactive=has_presets),
    )


def on_preset_load(name):
    if not name:
        return {}
    data = load_preset(name)
    if data is None:
        gr.Warning(i18n("Preset does not exist"))
        return {}
    try:
        emo_method = int(data.get("emo_control_method", 0))
        is_experimental = emo_method == 3
        prompt_path = data.get("prompt_audio", "") or None
        emo_path = data.get("emo_audio", "") or None
        missing = []
        if prompt_path and not os.path.exists(prompt_path):
            missing.append("prompt")
            prompt_path = None
        if emo_path and not os.path.exists(emo_path):
            missing.append("emotion")
            emo_path = None
        if missing:
            gr.Warning(i18n("Reference audio file missing, skipped"))
        emo_weight_value = data.get("emo_weight", 0.65)
        emo_vec = data.get("emo_vector", [0.0] * 8)
        emo_text_value = data.get("emo_text", "")
        emo_random_value = data.get("emo_random", False)
        advanced = data.get("advanced_params", {})
        if len(emo_vec) < 8:
            emo_vec = emo_vec + [0.0] * (8 - len(emo_vec))
        elif len(emo_vec) > 8:
            emo_vec = emo_vec[:8]
        emo_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        return {
            experimental_checkbox: gr.update(value=is_experimental),
            emo_control_method: gr.update(choices=emo_choices, value=emo_choices[emo_method]),
            prompt_audio: gr.update(value=prompt_path),
            emo_upload: gr.update(value=emo_path),
            emo_weight: gr.update(value=emo_weight_value),
            vec1: gr.update(value=emo_vec[0]),
            vec2: gr.update(value=emo_vec[1]),
            vec3: gr.update(value=emo_vec[2]),
            vec4: gr.update(value=emo_vec[3]),
            vec5: gr.update(value=emo_vec[4]),
            vec6: gr.update(value=emo_vec[5]),
            vec7: gr.update(value=emo_vec[6]),
            vec8: gr.update(value=emo_vec[7]),
            emo_text: gr.update(value=emo_text_value),
            emo_random: gr.update(value=emo_random_value),
            do_sample: gr.update(value=advanced.get("do_sample", True)),
            top_p: gr.update(value=advanced.get("top_p", 0.8)),
            top_k: gr.update(value=advanced.get("top_k", 30)),
            temperature: gr.update(value=advanced.get("temperature", 0.8)),
            length_penalty: gr.update(value=advanced.get("length_penalty", 0.0)),
            num_beams: gr.update(value=advanced.get("num_beams", 3)),
            repetition_penalty: gr.update(value=advanced.get("repetition_penalty", 2.0)),
            max_mel_tokens: gr.update(value=advanced.get("max_mel_tokens", 1500)),
            max_text_tokens_per_segment: gr.update(
                value=advanced.get("max_text_tokens_per_segment", 120)
            ),
        }
    except Exception as e:
        gr.Error(f"{i18n('Failed to load preset')}: {e}")
        return {}


def on_preset_delete(name):
    if not name:
        gr.Warning(i18n("Please select a preset"))
        return gr.update()
    if delete_preset(name):
        gr.Info(i18n("Preset deleted"), duration=2)
    else:
        gr.Warning(i18n("Preset does not exist"))
    choices = [""] + list_presets()
    has_presets = len(choices) > 1
    return (
        gr.update(value="", choices=choices, interactive=has_presets),
        gr.update(value="", choices=choices, interactive=has_presets),
    )


def update_save_preset_button(prompt_audio):
    return gr.update(interactive=bool(prompt_audio))

def update_delete_preset_button(preset_name):
    return gr.update(interactive=bool(preset_name))

def format_preset_details(name):
    if not name:
        return i18n("Please select a preset to manage")
    data = load_preset(name)
    if data is None:
        return i18n("Preset does not exist")
    try:
        emo_method = int(data.get("emo_control_method", 0))
        emo_label = EMO_CHOICES_ALL[emo_method] if 0 <= emo_method < len(EMO_CHOICES_ALL) else i18n("Unknown")
        emo_weight = data.get("emo_weight", 0.0)
        emo_vec = data.get("emo_vector", [0.0] * 8)
        emo_text = data.get("emo_text", "") or i18n("None")
        emo_random = data.get("emo_random", False)
        advanced = data.get("advanced_params", {})
        prompt_path = data.get("prompt_audio", "") or i18n("None")
        emo_path = data.get("emo_audio", "") or i18n("None")
        lines = [
            f"### {i18n('Preset Details')}: {name}",
            "",
            f"| {i18n('Attribute')} | {i18n('Value')} |",
            "|---|---|",
            f"| {i18n('Name')} | {name} |",
            f"| {i18n('Emotion Control Method')} | {emo_label} |",
            f"| {i18n('Emotion Weight')} | {emo_weight} |",
            f"| {i18n('Emotion Random Sampling')} | {'On' if emo_random else 'Off'} |",
            f"| {i18n('Voice Audio')} | `{prompt_path}` |",
            f"| {i18n('Emotion Audio')} | `{emo_path}` |",
            "",
            f"**{i18n('Emotion Vector')}**: `[{', '.join(str(round(v, 2)) for v in emo_vec)}]`",
            "",
            f"**{i18n('Emotion Description Text')}**: {emo_text}",
            "",
            f"**{i18n('Advanced Generation Parameters')}**:",
            "",
        ]
        for key, value in advanced.items():
            lines.append(f"- `{key}`: {value}")
        return "\n".join(lines)
    except Exception as e:
        return f"{i18n('Failed to load preset')}: {e}"

def refresh_preset_choices():
    choices = [""] + list_presets()
    has_presets = len(choices) > 1
    return (
        gr.update(choices=choices, value="", interactive=has_presets),
        gr.update(choices=choices, value="", interactive=has_presets),
    )

def _format_preset_preview(
    prompt_audio, emo_control_method, emo_ref_path, emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text, emo_random,
    do_sample, top_p, top_k, temperature,
    length_penalty, num_beams, repetition_penalty,
    max_mel_tokens, max_text_tokens_per_segment,
):
    emo_label = EMO_CHOICES_ALL[emo_control_method] if 0 <= emo_control_method < len(EMO_CHOICES_ALL) else i18n("Unknown")
    vec = [float(v) for v in [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]]
    prompt_label = os.path.basename(prompt_audio) if prompt_audio else i18n("None")
    emo_label_audio = os.path.basename(emo_ref_path) if emo_ref_path else i18n("None")
    lines = [
        f"**{i18n('Emotion Control Method')}**: {emo_label}",
        f"**{i18n('Emotion Weight')}**: {emo_weight}",
        f"**{i18n('Emotion Random Sampling')}**: {'On' if emo_random else 'Off'}",
        f"**{i18n('Voice Audio')}**: `{prompt_label}`",
        f"**{i18n('Emotion Audio')}**: `{emo_label_audio}`",
        "",
        f"**{i18n('Emotion Vector')}**: `[{', '.join(str(round(v, 2)) for v in vec)}]`",
        f"**{i18n('Emotion Description Text')}**: {emo_text or i18n('None')}",
        "",
        f"**{i18n('Advanced Generation Parameters')}**:",
        f"- do_sample: {do_sample}",
        f"- top_p: {top_p}",
        f"- top_k: {top_k}",
        f"- temperature: {temperature}",
        f"- length_penalty: {length_penalty}",
        f"- num_beams: {num_beams}",
        f"- repetition_penalty: {repetition_penalty}",
        f"- max_mel_tokens: {max_mel_tokens}",
        f"- max_text_tokens_per_segment: {max_text_tokens_per_segment}",
    ]
    return "\n".join(lines)

def open_save_preset_modal(
    prompt_audio, emo_control_method, emo_ref_path, emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text, emo_random,
    do_sample, top_p, top_k, temperature,
    length_penalty, num_beams, repetition_penalty,
    max_mel_tokens, max_text_tokens_per_segment,
):
    preview = _format_preset_preview(
        prompt_audio, emo_control_method, emo_ref_path, emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text, emo_random,
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty,
        max_mel_tokens, max_text_tokens_per_segment,
    )
    return (
        gr.update(visible=True),
        gr.update(value=preview),
        gr.update(value=""),
    )

def confirm_save_preset_from_modal(
    name, prompt_audio, emo_control_method, emo_ref_path, emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text, emo_random,
    do_sample, top_p, top_k, temperature,
    length_penalty, num_beams, repetition_penalty,
    max_mel_tokens, max_text_tokens_per_segment,
):
    result = on_preset_save(
        name, prompt_audio, emo_control_method, emo_ref_path, emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text, emo_random,
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty,
        max_mel_tokens, max_text_tokens_per_segment,
    )
    return (
        gr.update(visible=False),
        result[0],
        result[1],
    )

def close_save_preset_modal():
    return gr.update(visible=False)

def update_prompt_audio():
    return gr.update(interactive=True)

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('Note: This feature is experimental and results may be unstable. We are continuously optimizing it.'))

_mode_badge = (
    '<span style="background:#64748b;color:#fff;padding:2px 10px;'
    'border-radius:999px;font-size:0.85em;font-weight:bold;margin-left:8px">'
    'Standard Mode</span>'
)

with gr.Blocks(
    title="IndexTTS Demo",
    css="""
        #prompt_audio_compact .audio-container,
        #prompt_audio_compact .upload-container {
            min-height: 110px !important;
        }
        #prompt_audio_compact .empty {
            min-height: 80px !important;
        }
        .preset-modal-overlay {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            background: rgba(0, 0, 0, 0.5) !important;
            z-index: 1000 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .preset-modal-overlay > .column-wrap,
        .preset-modal-overlay > div {
            width: auto !important;
            height: auto !important;
        }
        .preset-modal-content {
            background: var(--body-background-fill) !important;
            padding: 24px !important;
            border-radius: 12px !important;
            width: 90vw !important;
            max-width: 560px !important;
            max-height: 80vh !important;
            overflow-y: auto !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2) !important;
        }
        .preset-modal-content > .column-wrap,
        .preset-modal-content > div {
            gap: 16px !important;
        }
    """,
) as demo:
    gr.HTML(f'''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech {_mode_badge}</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("Audio Generation")):
        os.makedirs("prompts", exist_ok=True)

        gr.Markdown(f"### {i18n('Voice Reference Audio')}")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                prompt_audio = gr.Audio(
                    label="",
                    key="prompt_audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                    elem_classes=["compact-audio"],
                    elem_id="prompt_audio_compact",
                )
                save_preset_btn = gr.Button(i18n("Save as Preset"), interactive=False)
            with gr.Column(scale=1):
                _has_presets = bool(list_presets())
                load_preset_dropdown = gr.Dropdown(
                    choices=[""] + list_presets(),
                    value="",
                    label=i18n("Load from Preset"),
                    info=i18n("Load voice and parameters from preset"),
                    allow_custom_value=False,
                    interactive=_has_presets,
                )

        gr.Markdown(f"### {i18n('Text')}")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                input_text_single = gr.TextArea(
                    label="",
                    key="input_text_single",
                    placeholder=i18n("Please enter the target text"),
                    info=f"{i18n('Current model version: ')}{tts.model_version or '1.0'}",
                    lines=5,
                )
            with gr.Column(scale=1):
                gen_button = gr.Button(i18n("Generate Speech"), key="gen_button", interactive=True)
                output_audio = gr.Audio(label=i18n("Generation Result"), visible=True, key="output_audio")

        with gr.Row():
            experimental_checkbox = gr.Checkbox(label=i18n("Show experimental features"), value=False)
            glossary_checkbox = gr.Checkbox(label=i18n("Enable term glossary pronunciation"), value=tts.normalizer.enable_glossary)
        with gr.Accordion(i18n("Function Settings")):
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0], label=i18n("Emotion Control Method"))
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("Emotion Control Method"),
                    visible=False)
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("Upload Emotion Reference Audio"), type="filepath")
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("Emotion Random Sampling"), value=False)
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("Happy"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("Angry"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("Sad"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("Fear"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("Disgust"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("Depressed"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("Surprise"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("Calm"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
        with gr.Group(visible=False) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(
                    label=i18n("Emotion Description Text"),
                    placeholder=i18n("Enter emotion description (or leave blank to use target text automatically)"),
                    value="",
                    info=i18n("For example: aggrieved, danger is quietly approaching"))
        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("Emotion Weight"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        with gr.Accordion(i18n("Custom Term Glossary Pronunciation"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
            gr.Markdown(i18n("Customize pronunciation of specific technical terms"))
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(label=i18n("Term"), placeholder="IndexTTS2")
                    glossary_reading_zh = gr.Textbox(label=i18n("Chinese Reading"), placeholder="Index T-T-S two")
                    glossary_reading_en = gr.Textbox(label=i18n("English Reading"), placeholder="Index T-T-S two")
                    btn_add_term = gr.Button(i18n("Add Term"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(value=format_glossary_markdown())

        with gr.Accordion(i18n("Advanced Generation Parameters"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 Sampling Settings')}** _{i18n('Parameters affect audio diversity and generation speed, see')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("Whether to sample"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=2.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("Max tokens to generate, too small will truncate audio"), key="max_mel_tokens")
                    trim_silence = gr.Checkbox(label=i18n("Auto trim head/tail silence and tail noise"), value=True, info=i18n("Fixes the issue of model generating overly long silent tails"))
                    with gr.Row():
                        seed_number = gr.Number(label=i18n("Base Random Seed"), value=42, precision=0, info=i18n("Fixed seed enables reproducible debugging of a specific segment"))
                        max_retries_number = gr.Slider(label=i18n("Max Retries per Segment on QA Failure"), value=2, minimum=0, maximum=5, step=1, info=i18n("SRT batch only: automatically retries a segment with a new seed if audio fails quality check OR is detected as truncated by the model's internal length estimate"))
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("Segmentation Settings")}** _{i18n("Parameters affect audio quality and generation speed")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("Max Tokens per Segment"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("Recommended 80~200. Larger value = longer segments; smaller = fragmented. Too large or too small may degrade quality"),
                        )
                    with gr.Accordion(i18n("Preview Segments"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("Index"), i18n("Segment Content"), i18n("Token Count")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                trim_silence, max_retries_number, seed_number,
            ]

        example_table = gr.Dataset(
            label="Examples",
            samples_per_page=20,
            samples=get_example_cases(include_experimental=False),
            type="values",
            components=[prompt_audio, emo_control_method_all, input_text_single, emo_upload,
                        emo_weight, emo_text, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        )

    with gr.Tab(i18n("SRT Batch Dubbing")):
        gr.Markdown(f"### {i18n('SRT Subtitle Batch Dubbing')}")
        gr.Markdown(i18n('Uses "Voice Reference Audio" and emotion/sampling settings from the "Audio Generation" tab above. After importing an SRT file, it will generate line-by-line and package as a ZIP download. Each segment is automatically quality-checked (including detection of internal generation truncation) and retried with a new seed on failure. All model calls are serialized to prevent shared-state race conditions during long batches. Reference audio longer than 15s is pre-trimmed ONCE to a fixed window so every segment conditions on identical reference bytes.'))
        with gr.Row():
            srt_file_input = gr.File(label=i18n("Upload SRT Subtitle File"), file_types=[".srt"])
            srt_gen_button = gr.Button(i18n("Start SRT Dubbing Generation"), variant="primary")
        srt_status = gr.Textbox(label=i18n("Processing Status"), interactive=False, value="")
        srt_output_zip = gr.File(label=i18n("Download Dubbing File (ZIP)"), visible=False)

        srt_gen_button.click(
            gen_srt,
            inputs=[
                prompt_audio, srt_file_input,
                emo_control_method, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment,
                *advanced_params
            ],
            outputs=[srt_output_zip, srt_status]
        )

    with gr.Tab(i18n("Preset Management")):
        gr.Markdown(f"## {i18n('Preset Management')}")
        with gr.Row():
            _has_presets = bool(list_presets())
            manage_preset_dropdown = gr.Dropdown(
                choices=[""] + list_presets(),
                value="",
                label=i18n("Preset List"),
                allow_custom_value=False,
                scale=2,
                interactive=_has_presets,
            )
            apply_preset_btn = gr.Button(i18n("Apply"), scale=1)
            delete_preset_btn = gr.Button(i18n("Delete"), scale=1, interactive=False)
            refresh_preset_btn = gr.Button(i18n("Refresh"), scale=1)
        preset_details_markdown = gr.Markdown(value=i18n("Please select a preset to manage"))
        with gr.Accordion(i18n("Create from current state"), open=False):
            with gr.Row():
                create_preset_name = gr.Textbox(
                    label=i18n("Preset Name"),
                    placeholder=i18n("Please enter preset name"),
                    value="",
                    scale=2,
                )
                create_preset_btn = gr.Button(i18n("Create"), scale=1)

    with gr.Column(visible=False, elem_classes=["preset-modal-overlay"]) as save_preset_modal:
        with gr.Column(elem_classes=["preset-modal-content"]):
            gr.Markdown(f"### {i18n('Save Preset')}")
            modal_preset_preview = gr.Markdown(label=i18n("Preset Preview"), value=i18n("Preset Preview"))
            modal_preset_name = gr.Textbox(label=i18n("Preset Name"), placeholder=i18n("Please enter preset name"), value="")
            with gr.Row():
                modal_cancel_btn = gr.Button(i18n("Cancel"), scale=1)
                modal_confirm_btn = gr.Button(i18n("Confirm"), scale=1, variant="primary")

    def on_example_click(example):
        return (
            gr.update(value=example[0]),
            gr.update(value=example[1]),
            gr.update(value=example[2]),
            gr.update(value=example[3]),
            gr.update(value=example[4]),
            gr.update(value=example[5]),
            gr.update(value=example[6]),
            gr.update(value=example[7]),
            gr.update(value=example[8]),
            gr.update(value=example[9]),
            gr.update(value=example[10]),
            gr.update(value=example[11]),
            gr.update(value=example[12]),
            gr.update(value=example[13]),
        )

    example_table.click(on_example_click,
                        inputs=[example_table],
                        outputs=[prompt_audio, emo_control_method, input_text_single, emo_upload,
                                 emo_weight, emo_text, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)
            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = [[i, ''.join(s), len(s)] for i, s in enumerate(segments)]
            return {segments_preview: gr.update(value=data, visible=True)}
        else:
            df = pd.DataFrame([], columns=[i18n("Index"), i18n("Segment Content"), i18n("Token Count")])
            return {segments_preview: gr.update(value=df)}

    def on_add_glossary_term(term, reading_zh, reading_en):
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()
        if not term:
            gr.Warning(i18n("Please enter a term"))
            return gr.update()
        if not reading_zh and not reading_en:
            gr.Warning(i18n("Please enter at least one reading"))
            return gr.update()
        if reading_zh and reading_en:
            reading = {"zh": reading_zh, "en": reading_en}
        elif reading_zh:
            reading = {"zh": reading_zh}
        else:
            reading = {"en": reading_en}
        tts.normalizer.term_glossary[term] = reading
        try:
            tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
            gr.Info(i18n("Glossary updated"), duration=1)
        except Exception as e:
            gr.Error(i18n("Error saving glossary"))
            print(f"Error details: {e}")
            return gr.update()
        return gr.update(value=format_glossary_markdown())

    def on_method_change(emo_control_method):
        if emo_control_method == 1:
            return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))
        elif emo_control_method == 2:
            return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True))
        elif emo_control_method == 3:
            return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True))
        else:
            return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group, emotion_randomize_group, emotion_vector_group, emo_text_group, emo_weight_group])

    def on_experimental_change(is_experimental, current_mode_index):
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0
        return (
            gr.update(choices=new_choices, value=new_choices[new_index]),
            gr.update(samples=get_example_cases(include_experimental=is_experimental)),
        )

    experimental_checkbox.change(on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method, example_table])

    def on_glossary_checkbox_change(is_enabled):
        tts.normalizer.enable_glossary = is_enabled
        return gr.update(visible=is_enabled)

    glossary_checkbox.change(on_glossary_checkbox_change,
        inputs=[glossary_checkbox],
        outputs=[glossary_accordion])

    input_text_single.change(on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview])

    max_text_tokens_per_segment.change(on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview])

    prompt_audio.upload(update_prompt_audio, inputs=[], outputs=[gen_button])
    prompt_audio.change(update_save_preset_button, inputs=[prompt_audio], outputs=[save_preset_btn])

    def on_demo_load():
        try:
            tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
        except Exception as e:
            gr.Error(i18n("Error loading glossary"))
            print(f"Failed to reload glossary on page load: {e}")
        return (gr.update(value=format_glossary_markdown()), *refresh_preset_choices())

    btn_add_term.click(on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table])

    demo.load(on_demo_load, inputs=[], outputs=[glossary_table, load_preset_dropdown, manage_preset_dropdown])

    _preset_load_outputs = [
        experimental_checkbox, emo_control_method, prompt_audio, emo_upload, emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text, emo_random,
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty,
        max_mel_tokens, max_text_tokens_per_segment,
    ]
    _preset_save_inputs = [
        prompt_audio, emo_control_method, emo_upload, emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text, emo_random,
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty,
        max_mel_tokens, max_text_tokens_per_segment,
    ]

    load_preset_dropdown.change(on_preset_load, inputs=[load_preset_dropdown], outputs=_preset_load_outputs)
    save_preset_btn.click(open_save_preset_modal, inputs=_preset_save_inputs, outputs=[save_preset_modal, modal_preset_preview, modal_preset_name])
    modal_confirm_btn.click(confirm_save_preset_from_modal, inputs=[modal_preset_name] + _preset_save_inputs, outputs=[save_preset_modal, load_preset_dropdown, manage_preset_dropdown])
    modal_cancel_btn.click(close_save_preset_modal, inputs=[], outputs=[save_preset_modal])
    manage_preset_dropdown.change(format_preset_details, inputs=[manage_preset_dropdown], outputs=[preset_details_markdown])
    manage_preset_dropdown.change(update_delete_preset_button, inputs=[manage_preset_dropdown], outputs=[delete_preset_btn])
    apply_preset_btn.click(on_preset_load, inputs=[manage_preset_dropdown], outputs=_preset_load_outputs)
    delete_preset_btn.click(on_preset_delete, inputs=[manage_preset_dropdown], outputs=[load_preset_dropdown, manage_preset_dropdown])
    refresh_preset_btn.click(refresh_preset_choices, inputs=[], outputs=[load_preset_dropdown, manage_preset_dropdown])
    create_preset_btn.click(on_preset_save, inputs=[create_preset_name] + _preset_save_inputs, outputs=[load_preset_dropdown, manage_preset_dropdown])

    gen_button.click(gen_single,
        inputs=[emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment,
                *advanced_params],
        outputs=[output_audio])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)