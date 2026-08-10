import html
import json
import os
import re
import sys
import threading
import time
import zipfile
import asyncio

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import librosa
import soundfile as sf
import pandas as pd

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
parser.add_argument("--vllm", action="store_true", default=False,
                    help="Use vLLM to accelerate GPT inference (requires converted checkpoints/gpt/)")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.40,
                    help="vLLM GPU memory utilization fraction (0.0-1.0). Default 0.40 = 40%% of VRAM.")
parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10,
                    help="vLLM GPU memory for Qwen emotion model. Default 0.10 = 10%%.")
cmd_args, _ = parser.parse_known_args()

# ── Required files check ──────────────────────────────────────────────────────
required_files_base = [
    "bpe.model",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
]

if cmd_args.vllm:
    # vLLM mode: check for converted HF format weights in checkpoints/gpt/
    required_files = required_files_base.copy()
    missing = [f for f in required_files if not os.path.exists(os.path.join(cmd_args.model_dir, f))]
    
    # Check config.json exists
    gpt_config = os.path.join(cmd_args.model_dir, "gpt", "config.json")
    if not os.path.exists(gpt_config):
        missing.append("gpt/config.json")
    
    # Accept EITHER model.safetensors OR pytorch_model.bin (your conversion produced .bin)
    gpt_weight_safetensors = os.path.join(cmd_args.model_dir, "gpt", "model.safetensors")
    gpt_weight_bin = os.path.join(cmd_args.model_dir, "gpt", "pytorch_model.bin")
    
    if not os.path.exists(gpt_weight_safetensors) and not os.path.exists(gpt_weight_bin):
        missing.append("gpt/model.safetensors OR gpt/pytorch_model.bin")
    
    if missing:
        has_pth = os.path.exists(os.path.join(cmd_args.model_dir, "gpt.pth"))
        if has_pth and any("gpt/" in str(f) for f in missing):
            print(
                f"[vLLM] ERROR: Found gpt.pth but missing converted HF format files: {missing}\n"
                f"[vLLM] Please run the conversion first:\n"
                f"[vLLM]   bash convert_hf_format.sh {cmd_args.model_dir}\n"
                f"[vLLM] Or run without --vllm to use original gpt.pth."
            )
        else:
            print(
                f"[vLLM] Model directory {cmd_args.model_dir} is incomplete (missing: {', '.join(missing)}). "
                "Downloading IndexTTS-2 model..."
            )
            from indextts.utils.model_download import snapshot_download
            try:
                snapshot_download("IndexTeam/IndexTTS-2", local_dir=cmd_args.model_dir)
            except Exception as e:
                print(f"Failed to download model to {cmd_args.model_dir}: {e}")
                sys.exit(1)
        
        # Re-check after potential download
        missing = [f for f in required_files if not os.path.exists(os.path.join(cmd_args.model_dir, f))]
        if not os.path.exists(gpt_config):
            missing.append("gpt/config.json")
        if not os.path.exists(gpt_weight_safetensors) and not os.path.exists(gpt_weight_bin):
            missing.append("gpt/model.safetensors OR gpt/pytorch_model.bin")
        
        if missing:
            print(f"Model directory still incomplete (missing: {', '.join(missing)}). Please check manually.")
            sys.exit(1)
    
    gpt_weight_file = "model.safetensors" if os.path.exists(gpt_weight_safetensors) else "pytorch_model.bin"
    print(f"[vLLM] ✅ Converted GPT weights found at: {os.path.join(cmd_args.model_dir, 'gpt', gpt_weight_file)}")
    print(f"[vLLM] GPU memory utilization: {cmd_args.gpu_memory_utilization}")
else:
    # Standard mode: check for original gpt.pth
    required_files = required_files_base + ["gpt.pth"]
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

if cmd_args.vllm:
    from indextts.infer_vllm_v2 import IndexTTS2
else:
    from indextts.infer_v2 import IndexTTS2
    
from indextts.utils.examples_downloader import ensure_examples_available
from indextts.utils.presets import list_presets, save_preset, load_preset, delete_preset
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'

ensure_examples_available()

# ── IndexTTS2 initialization ──────────────────────────────────────────────────
if cmd_args.vllm:
    tts = IndexTTS2(
        model_dir=cmd_args.model_dir,
        is_fp16=cmd_args.fp16,
        gpu_memory_utilization=cmd_args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=cmd_args.qwenemo_gpu_memory_utilization,
    )
else:
    tts = IndexTTS2(
        model_dir=cmd_args.model_dir,
        cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
        use_fp16=cmd_args.fp16,
        use_deepspeed=cmd_args.deepspeed,
        use_cuda_kernel=cmd_args.cuda_kernel,
    )

if cmd_args.vllm:
    print(f"")
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  🚀 vLLM ACCELERATION ACTIVE                     ║")
    print(f"║  GPT model: checkpoints/gpt/{gpt_weight_file:20s} ║")
    print(f"║  VRAM pool: {cmd_args.gpu_memory_utilization:.0%} of available GPU memory         ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"")
else:
    print(f"[WebUI] Standard PyTorch mode (no vLLM). Use --vllm to enable acceleration.")

LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
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
        return i18n("暂无术语")
    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
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


# ── Conditional gen_srt (async for vLLM, sync for non-vLLM) ───────────────────

if cmd_args.vllm:
    async def gen_srt(prompt, srt_file,
                emo_control_method, emo_ref_path, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
        
        if not prompt:
            gr.Warning(i18n("请先在\"音频生成\"选项卡中上传参考音频"))
            return gr.update(visible=False), i18n("错误：缺少参考音频")
        if not srt_file:
            gr.Warning(i18n("请上传SRT文件"))
            return gr.update(visible=False), i18n("错误：缺少SRT文件")
        
        srt_path = srt_file.name if hasattr(srt_file, 'name') else srt_file
        segments = parse_srt_file(srt_path)
        
        if not segments:
            gr.Warning(i18n("SRT解析失败或为空"))
            return gr.update(visible=False), i18n("错误：SRT解析失败")
        
        do_sample, top_p, top_k, temperature, \
            length_penalty, num_beams, repetition_penalty, max_mel_tokens, trim_silence_val = args
            
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
        if emo_control_method == 2:
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            vec = None

        if emo_text == "":
            emo_text = None

        out_dir = os.path.join("outputs", "tasks", f"srt_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)
        
        tts.gr_progress = progress
        total = len(segments)
        success_count = 0
        
        progress(0, desc=f"开始生成 SRT 配音 (共 {total} 段)...")
        
        for i, seg in enumerate(segments):
            progress((i + 1) / total, desc=f"生成第 {i+1}/{total} 段...")
            out_path = os.path.join(out_dir, f"segment-{seg['index']:04d}.wav")
            try:
                await tts.infer(
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
                    max_text_tokens_per_sentence=int(max_text_tokens_per_segment),
                    **kwargs
                )
                if trim_silence_val:
                    # ONLY CHANGE THIS LINE IN THE ASYNC FUNCTION
                    await asyncio.to_thread(trim_audio_wav, out_path)
                success_count += 1
            except Exception as e:
                print(f"Error generating segment {seg['index']}: {e}")
                
        if success_count == 0:
            return gr.update(visible=False), i18n("所有片段生成失败，请检查日志")
            
        zip_path = os.path.join(out_dir, "dubbing_segments.zip")
        with zipfile.ZipFile(zip_path, 'w') as z:
            for f in sorted(os.listdir(out_dir)):
                if f.endswith('.wav'):
                    z.write(os.path.join(out_dir, f), arcname=f)
                    
        return gr.update(value=zip_path, visible=True), f"完成！成功生成 {success_count}/{total} 段。"
else:
    def gen_srt(prompt, srt_file,
                emo_control_method, emo_ref_path, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
        
        if not prompt:
            gr.Warning(i18n("请先在\"音频生成\"选项卡中上传参考音频"))
            return gr.update(visible=False), i18n("错误：缺少参考音频")
        if not srt_file:
            gr.Warning(i18n("请上传SRT文件"))
            return gr.update(visible=False), i18n("错误：缺少SRT文件")
        
        srt_path = srt_file.name if hasattr(srt_file, 'name') else srt_file
        segments = parse_srt_file(srt_path)
        
        if not segments:
            gr.Warning(i18n("SRT解析失败或为空"))
            return gr.update(visible=False), i18n("错误：SRT解析失败")
        
        do_sample, top_p, top_k, temperature, \
            length_penalty, num_beams, repetition_penalty, max_mel_tokens, trim_silence_val = args
            
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
        if emo_control_method == 2:
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            vec = None

        if emo_text == "":
            emo_text = None

        out_dir = os.path.join("outputs", "tasks", f"srt_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)
        
        tts.gr_progress = progress
        total = len(segments)
        success_count = 0
        
        progress(0, desc=f"开始生成 SRT 配音 (共 {total} 段)...")
        
        for i, seg in enumerate(segments):
            progress((i + 1) / total, desc=f"生成第 {i+1}/{total} 段...")
            out_path = os.path.join(out_dir, f"segment-{seg['index']:04d}.wav")
            try:
                tts.infer(
                    spk_audio_prompt=prompt,
                    text=seg['text'],
                    output_path=out_path,
                    # ...
                )
                if trim_silence_val:
                    # KEEP THIS ONE STANDARD (NO AWAIT)
                    trim_audio_wav(out_path)
                success_count += 1
            except Exception as e:
                print(f"Error generating segment {seg['index']}: {e}")
                
        if success_count == 0:
            return gr.update(visible=False), i18n("所有片段生成失败，请检查日志")
            
        zip_path = os.path.join(out_dir, "dubbing_segments.zip")
        with zipfile.ZipFile(zip_path, 'w') as z:
            for f in sorted(os.listdir(out_dir)):
                if f.endswith('.wav'):
                    z.write(os.path.join(out_dir, f), arcname=f)
                    
        return gr.update(value=zip_path, visible=True), f"完成！成功生成 {success_count}/{total} 段。"


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
            "repetition_penalty": float(repetition_penalty) if repetition_penalty is not None else 10.0,
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
        gr.Warning(i18n("预设名称不能为空"))
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
        msg = i18n("预设名称已存在，已覆盖") if existed else i18n("预设已保存")
        gr.Info(msg, duration=2)
    except Exception as e:
        gr.Error(f"{i18n('加载预设失败')}: {e}")
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
        gr.Warning(i18n("预设不存在"))
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
            gr.Warning(i18n("参考音频文件缺失，已跳过"))
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
            repetition_penalty: gr.update(value=advanced.get("repetition_penalty", 10.0)),
            max_mel_tokens: gr.update(value=advanced.get("max_mel_tokens", 1500)),
            max_text_tokens_per_segment: gr.update(
                value=advanced.get("max_text_tokens_per_segment", 120)
            ),
        }
    except Exception as e:
        gr.Error(f"{i18n('加载预设失败')}: {e}")
        return {}


def on_preset_delete(name):
    if not name:
        gr.Warning(i18n("请选择预设"))
        return gr.update()
    if delete_preset(name):
        gr.Info(i18n("预设已删除"), duration=2)
    else:
        gr.Warning(i18n("预设不存在"))
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
        return i18n("请选择要管理的预设")
    data = load_preset(name)
    if data is None:
        return i18n("预设不存在")
    try:
        emo_method = int(data.get("emo_control_method", 0))
        emo_label = EMO_CHOICES_ALL[emo_method] if 0 <= emo_method < len(EMO_CHOICES_ALL) else i18n("未知")
        emo_weight = data.get("emo_weight", 0.0)
        emo_vec = data.get("emo_vector", [0.0] * 8)
        emo_text = data.get("emo_text", "") or i18n("无")
        emo_random = data.get("emo_random", False)
        advanced = data.get("advanced_params", {})
        prompt_path = data.get("prompt_audio", "") or i18n("无")
        emo_path = data.get("emo_audio", "") or i18n("无")
        lines = [
            f"### {i18n('预设详情')}: {name}",
            "",
            f"| {i18n('属性')} | {i18n('值')} |",
            "|---|---|",
            f"| {i18n('名称')} | {name} |",
            f"| {i18n('情感控制方式')} | {emo_label} |",
            f"| {i18n('情感权重')} | {emo_weight} |",
            f"| {i18n('情感随机采样')} | {'On' if emo_random else 'Off'} |",
            f"| {i18n('音色音频')} | `{prompt_path}` |",
            f"| {i18n('情感音频')} | `{emo_path}` |",
            "",
            f"**{i18n('情感向量')}**: `[{', '.join(str(round(v, 2)) for v in emo_vec)}]`",
            "",
            f"**{i18n('情感描述文本')}**: {emo_text}",
            "",
            f"**{i18n('高级生成参数设置')}**:",
            "",
        ]
        for key, value in advanced.items():
            lines.append(f"- `{key}`: {value}")
        return "\n".join(lines)
    except Exception as e:
        return f"{i18n('加载预设失败')}: {e}"

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
    emo_label = EMO_CHOICES_ALL[emo_control_method] if 0 <= emo_control_method < len(EMO_CHOICES_ALL) else i18n("未知")
    vec = [float(v) for v in [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]]
    prompt_label = os.path.basename(prompt_audio) if prompt_audio else i18n("无")
    emo_label_audio = os.path.basename(emo_ref_path) if emo_ref_path else i18n("无")
    lines = [
        f"**{i18n('情感控制方式')}**: {emo_label}",
        f"**{i18n('情感权重')}**: {emo_weight}",
        f"**{i18n('情感随机采样')}**: {'On' if emo_random else 'Off'}",
        f"**{i18n('音色音频')}**: `{prompt_label}`",
        f"**{i18n('情感音频')}**: `{emo_label_audio}`",
        "",
        f"**{i18n('情感向量')}**: `[{', '.join(str(round(v, 2)) for v in vec)}]`",
        f"**{i18n('情感描述文本')}**: {emo_text or i18n('无')}",
        "",
        f"**{i18n('高级生成参数设置')}**:",
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


# ── Conditional gen_single (async for vLLM, sync for non-vLLM) ────────────────

if cmd_args.vllm:
    async def gen_single(emo_control_method, prompt, text,
                   emo_ref_path, emo_weight,
                   vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                   emo_text, emo_random,
                   max_text_tokens_per_segment=120,
                   *args, progress=gr.Progress()):
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
        tts.gr_progress = progress
        do_sample, top_p, top_k, temperature, \
            length_penalty, num_beams, repetition_penalty, max_mel_tokens, trim_silence_val = args
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
        if emo_control_method == 2:
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            vec = None
        if emo_text == "":
            emo_text = None
        
        output = await tts.infer(
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
            max_text_tokens_per_sentence=int(max_text_tokens_per_segment),
            **kwargs
        )
        if trim_silence_val:
            trim_audio_wav(output_path)
        return gr.update(value=output_path, visible=True)
else:
    def gen_single(emo_control_method, prompt, text,
                   emo_ref_path, emo_weight,
                   vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                   emo_text, emo_random,
                   max_text_tokens_per_segment=120,
                   *args, progress=gr.Progress()):
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
        tts.gr_progress = progress
        do_sample, top_p, top_k, temperature, \
            length_penalty, num_beams, repetition_penalty, max_mel_tokens, trim_silence_val = args
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
        if emo_control_method == 2:
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            vec = None
        if emo_text == "":
            emo_text = None
        
        output = tts.infer(
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
            **kwargs
        )
        if trim_silence_val:
            trim_audio_wav(output_path)
        return gr.update(value=output, visible=True)

def update_prompt_audio():
    return gr.update(interactive=True)

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

_vllm_badge = (
    '<span style="background:#22c55e;color:#fff;padding:2px 10px;'
    'border-radius:999px;font-size:0.85em;font-weight:bold;margin-left:8px">'
    '🚀 vLLM ON</span>'
) if cmd_args.vllm else (
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
    mutex = threading.Lock()
    gr.HTML(f'''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech {_vllm_badge}</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("音频生成")):
        os.makedirs("prompts", exist_ok=True)

        gr.Markdown(f"### {i18n('音色参考音频')}")
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
                save_preset_btn = gr.Button(i18n("保存为预设"), interactive=False)
            with gr.Column(scale=1):
                _has_presets = bool(list_presets())
                load_preset_dropdown = gr.Dropdown(
                    choices=[""] + list_presets(),
                    value="",
                    label=i18n("从预设加载"),
                    info=i18n("从预设加载音色和参数"),
                    allow_custom_value=False,
                    interactive=_has_presets,
                )

        gr.Markdown(f"### {i18n('文本')}")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                input_text_single = gr.TextArea(
                    label="",
                    key="input_text_single",
                    placeholder=i18n("请输入目标文本"),
                    info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}",
                    lines=5,
                )
            with gr.Column(scale=1):
                gen_button = gr.Button(i18n("生成语音"), key="gen_button", interactive=True)
                output_audio = gr.Audio(label=i18n("生成结果"), visible=True, key="output_audio")

        with gr.Row():
            experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
            glossary_checkbox = gr.Checkbox(label=i18n("开启术语词汇读音"), value=tts.normalizer.enable_glossary)
        with gr.Accordion(i18n("功能设置")):
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0], label=i18n("情感控制方式"))
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                    visible=False)
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
        with gr.Group(visible=False) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(
                    label=i18n("情感描述文本"),
                    placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                    value="",
                    info=i18n("例如：委屈巴巴、危险在悄悄逼近"))
        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        with gr.Accordion(i18n("自定义术语词汇读音"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
            gr.Markdown(i18n("自定义个别专业术语的读音"))
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(label=i18n("术语"), placeholder="IndexTTS2")
                    glossary_reading_zh = gr.Textbox(label=i18n("中文读法"), placeholder="Index T-T-S 二")
                    glossary_reading_en = gr.Textbox(label=i18n("英文读法"), placeholder="Index T-T-S two")
                    btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(value=format_glossary_markdown())

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    trim_silence = gr.Checkbox(label=i18n("自动裁剪首尾静音及尾部噪音"), value=True, info=i18n("修复模型生成超长无声尾音的问题"))
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                trim_silence,
            ]

        example_table = gr.Dataset(
            label="Examples",
            samples_per_page=20,
            samples=get_example_cases(include_experimental=False),
            type="values",
            components=[prompt_audio, emo_control_method_all, input_text_single, emo_upload,
                        emo_weight, emo_text, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        )

    with gr.Tab(i18n("SRT批量配音")):
        gr.Markdown(f"### {i18n('SRT字幕批量配音')}")
        gr.Markdown(i18n('使用上方"音频生成"选项卡中的"音色参考音频"和情感/采样设置。导入SRT文件后，将逐句生成并打包为ZIP下载。'))
        with gr.Row():
            srt_file_input = gr.File(label=i18n("上传SRT字幕文件"), file_types=[".srt"])
            srt_gen_button = gr.Button(i18n("开始生成SRT配音"), variant="primary")
        srt_status = gr.Textbox(label=i18n("处理状态"), interactive=False, value="")
        srt_output_zip = gr.File(label=i18n("下载配音文件(ZIP)"), visible=False)

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

    with gr.Tab(i18n("预设管理")):
        gr.Markdown(f"## {i18n('预设管理')}")
        with gr.Row():
            _has_presets = bool(list_presets())
            manage_preset_dropdown = gr.Dropdown(
                choices=[""] + list_presets(),
                value="",
                label=i18n("预设列表"),
                allow_custom_value=False,
                scale=2,
                interactive=_has_presets,
            )
            apply_preset_btn = gr.Button(i18n("应用"), scale=1)
            delete_preset_btn = gr.Button(i18n("删除"), scale=1, interactive=False)
            refresh_preset_btn = gr.Button(i18n("刷新"), scale=1)
        preset_details_markdown = gr.Markdown(value=i18n("请选择要管理的预设"))
        with gr.Accordion(i18n("从当前状态创建"), open=False):
            with gr.Row():
                create_preset_name = gr.Textbox(
                    label=i18n("预设名称"),
                    placeholder=i18n("请输入预设名称"),
                    value="",
                    scale=2,
                )
                create_preset_btn = gr.Button(i18n("创建"), scale=1)

    with gr.Column(visible=False, elem_classes=["preset-modal-overlay"]) as save_preset_modal:
        with gr.Column(elem_classes=["preset-modal-content"]):
            gr.Markdown(f"### {i18n('保存预设')}")
            modal_preset_preview = gr.Markdown(label=i18n("预设预览"), value=i18n("预设预览"))
            modal_preset_name = gr.Textbox(label=i18n("预设名称"), placeholder=i18n("请输入预设名称"), value="")
            with gr.Row():
                modal_cancel_btn = gr.Button(i18n("取消"), scale=1)
                modal_confirm_btn = gr.Button(i18n("确认"), scale=1, variant="primary")

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
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {segments_preview: gr.update(value=df)}

    def on_add_glossary_term(term, reading_zh, reading_en):
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()
        if not term:
            gr.Warning(i18n("请输入术语"))
            return gr.update()
        if not reading_zh and not reading_en:
            gr.Warning(i18n("请至少输入一种读法"))
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
            gr.Info(i18n("词汇表已更新"), duration=1)
        except Exception as e:
            gr.Error(i18n("保存词汇表时出错"))
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
            gr.Error(i18n("加载词汇表时出错"))
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