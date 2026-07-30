"""
music_source.py — pluggable background-music / SFX generator for the TTS UI.

Design:
  * Runs on the Lightning GPU, lazy-loaded ONLY when background music is enabled.
  * Model is intentionally mid-size (MusicGen-small ~300M / ~2-3GB VRAM) so it
    co-resides comfortably with IndexTTS2. Offloads after generation if asked,
    so the two models need not both sit at peak VRAM.
  * Source is pluggable: MusicGenSource (GPU model) | LibrarySource (files) |
    (future) StableAudioSource / ApiSource. All expose the same generate() ->
    (np.float32 mono, sr) contract, so the UI/mixer never care which is used.

The actual mixing/ducking lives in audio_bed.py (pure CPU). This file ONLY
produces a raw music/SFX waveform.
"""

import os
import numpy as np


# Length (seconds) of the raw bed we actually generate on the GPU. The mixer
# (audio_bed.fit_duration) loops this seamlessly up to the voice length, so a
# short seed = big speedup with no audible loss for a repetitive background bed.
# Raise it if you want a longer non-repeating musical bed.
SEED_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Emotion -> music-prompt mapping (this is what makes the bed "match" the voice)
# ---------------------------------------------------------------------------
# IndexTTS2 emotion vector order:
#   [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
EMO_ORDER = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

_EMO_TO_PROMPT = {
    "happy":       "modern uplifting cinematic score, bright acoustic guitar and warm piano, "
                   "soft claps and light hand percussion, lush strings, major key, 110 BPM, "
                   "hopeful and heartwarming, polished studio production, wide stereo, "
                   "underscore bed with plenty of headroom for a voiceover",
    "angry":       "intense dark cinematic trailer underscore, low distorted synth drone and "
                   "sub bass, pounding cinematic taiko and braams, tense staccato strings, "
                   "minor key, driving 90 BPM, aggressive and foreboding, modern hybrid "
                   "orchestral production, controlled dynamics under a voiceover",
    "sad":         "emotional cinematic piano ballad, intimate felt piano and sustained warm "
                   "strings, subtle cello, minor key, slow 60 BPM, melancholic and tender, "
                   "spacious reverb, delicate modern film-score production, soft dynamics "
                   "sitting under a narrator",
    "afraid":      "suspenseful horror underscore, eerie sustained string clusters and detuned "
                   "pads, sparse ticking pulse, low unsettling drone, dissonant minor tonality, "
                   "very slow tempo, tense and creeping, cinematic sound-design production, "
                   "quiet enough to speak over",
    "disgusted":   "dark brooding cinematic texture, dissonant low strings and uneasy metallic "
                   "drones, sparse prepared piano, atonal, slow and heavy, grim and unsettling, "
                   "modern experimental film-score production, low-level underscore",
    "melancholic": "wistful reflective neo-classical piece, nostalgic felt piano and airy synth "
                   "pads, distant strings, minor key, gentle 70 BPM, bittersweet and introspective, "
                   "warm analog reverb, cinematic ambient production, soft underscore for narration",
    "surprised":   "light playful modern score, curious pizzicato strings and gentle marimba, "
                   "airy plucks and soft mallets, major key, bouncy 120 BPM, whimsical and bright, "
                   "clean contemporary production, quirky but subtle underscore",
    "calm":        "serene ambient soundscape, soft evolving synth pads and gentle piano, "
                   "warm sustained textures, no drums, spacious and minimal, relaxing and "
                   "meditative, lush modern ambient production, smooth low-level bed under a voice",
}

# Neutral default when no emotion signal is available.
_DEFAULT_PROMPT = (
    "neutral modern cinematic underscore, soft evolving synth pad and light piano, "
    "warm strings, gentle and unobtrusive, no strong melody, spacious contemporary "
    "production, low-level background bed designed to sit under a voiceover"
)


# ---------------------------------------------------------------------------
# Emotion -> SFX/ambience-prompt mapping (AFX layer)
# ---------------------------------------------------------------------------
# Same bridge as music: the emotion vector you already set for TTS selects an
# ambience. The SFX model never hears the speech and never reads Amharic/Oromo/
# Tigrigna — it only receives this English scene prompt.
_EMO_TO_SFX = {
    "happy":       "bright cheerful outdoor ambience, gentle birdsong and rustling leaves, "
                   "distant playful chatter, warm sunny afternoon park atmosphere, "
                   "soft natural reverb, clean field-recording quality, low-level background bed",
    "angry":       "tense ominous atmosphere, distant low thunder and heavy gusting wind, "
                   "faint rumbling and creaking, oppressive storm-front ambience, "
                   "dark cinematic sound design, subtle enough to speak over",
    "sad":         "quiet melancholic interior, soft steady rain against a window, "
                   "faint distant traffic hum, still empty-room tone, gentle and muffled, "
                   "intimate field-recording ambience, low-level bed",
    "afraid":      "eerie unsettling night ambience, cold wind through a hollow space, "
                   "sparse creaks and distant indistinct sounds, faint high tension whine, "
                   "suspenseful horror sound design, quiet and sparse under narration",
    "disgusted":   "grim damp environment, slow echoing water drips in a cavern, "
                   "low uneasy air-conditioning drone, faint metallic groans, "
                   "claustrophobic industrial ambience, dark sound design, low-level",
    "melancholic": "reflective rainy-day atmosphere, light rain and occasional distant thunder, "
                   "soft wind, quiet contemplative room tone, warm and nostalgic, "
                   "cinematic field recording, gentle background bed",
    "surprised":   "light airy open-air ambience, soft whooshes and gentle wind chimes, "
                   "distant bright activity, clean and spacious, whimsical and curious, "
                   "modern sound design, subtle background layer",
    "calm":        "serene natural ambience, soft breeze through trees and faint distant water, "
                   "occasional gentle birdsong, peaceful and spacious countryside tone, "
                   "smooth field recording, relaxing low-level bed under a voice",
}

_DEFAULT_SFX_PROMPT = (
    "subtle neutral room tone, soft steady air presence and faint distant ambience, "
    "clean and unobtrusive, professional field-recording quality, "
    "low-level background bed designed to sit under a voiceover"
)


def emotion_to_sfx_prompt(emo_vector=None, emo_text=None, extra=None):
    """Build an ambience/SFX prompt mirroring the voice emotion. Same contract
    and fallbacks as emotion_to_prompt(), but maps to environmental sound."""
    parts = []
    if emo_vector is not None and len(emo_vector) == len(EMO_ORDER):
        idx = int(np.argmax(emo_vector))
        dom = EMO_ORDER[idx]
        if emo_vector[idx] > 1e-3:
            parts.append(_EMO_TO_SFX.get(dom, _DEFAULT_SFX_PROMPT))
    if not parts and emo_text:
        parts.append(f"{emo_text} scene, layered environmental ambience and sound design, "
                     "clean field-recording quality, low-level background bed under a voiceover")
    if not parts:
        parts.append(_DEFAULT_SFX_PROMPT)
    if extra:
        parts.append(extra.strip())
    return ", ".join(parts)


# Short mood adjective per emotion, used to blend in a secondary emotion when
# the vector is mixed (e.g. mostly-sad-but-a-bit-hopeful). Keeps the primary
# template intact and just adds a light colour phrase.
_EMO_MOOD_WORD = {
    "happy":       "hopeful and uplifting",
    "angry":       "tense and driving",
    "sad":         "sorrowful and tender",
    "afraid":      "anxious and suspenseful",
    "disgusted":   "uneasy and brooding",
    "melancholic": "wistful and nostalgic",
    "surprised":   "bright and curious",
    "calm":        "serene and gentle",
}


def _secondary_blend(emo_vector, primary_idx, rel_threshold=0.5):
    """If a second emotion is at least `rel_threshold` of the dominant one,
    return a short 'with an undercurrent of ...' colour phrase; else ''.
    Makes mixed emotions sound layered instead of snapping to one bucket."""
    dom = float(emo_vector[primary_idx])
    if dom <= 1e-6:
        return ""
    best_j, best_v = -1, 0.0
    for j, v in enumerate(emo_vector):
        if j == primary_idx:
            continue
        if v > best_v:
            best_j, best_v = j, float(v)
    if best_j >= 0 and best_v >= rel_threshold * dom and best_v > 1e-3:
        word = _EMO_MOOD_WORD.get(EMO_ORDER[best_j])
        if word:
            return f"with an undercurrent of {word}"
    return ""


def emotion_to_prompt(emo_vector=None, emo_text=None, extra=None):
    """Build a music-generation prompt that mirrors the voice's emotion.

    emo_vector : list[float] length 8 in EMO_ORDER (optional)
    emo_text   : free-text emotion description already used for TTS (optional)
    extra      : user-supplied override/append string (optional)

    Returns a prompt string.
    """
    parts = []
    if emo_vector is not None and len(emo_vector) == len(EMO_ORDER):
        idx = int(np.argmax(emo_vector))
        dom = EMO_ORDER[idx]
        if emo_vector[idx] > 1e-3:
            parts.append(_EMO_TO_PROMPT.get(dom, _DEFAULT_PROMPT))
            blend = _secondary_blend(emo_vector, idx)
            if blend:
                parts.append(blend)
    if not parts and emo_text:
        parts.append(f"{emo_text} mood, modern cinematic underscore, tasteful instrumentation, "
                     "polished studio production, low-level background bed under a voiceover")
    if not parts:
        parts.append(_DEFAULT_PROMPT)
    if extra:
        parts.append(extra.strip())
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Source base + implementations
# ---------------------------------------------------------------------------
class MusicSource:
    def generate(self, prompt, duration_sec, seed=None):
        """Return (np.float32 mono in [-1,1], sample_rate). Must be implemented."""
        raise NotImplementedError


class LibrarySource(MusicSource):
    """Zero-compute fallback: pick a file from a mood-tagged library folder.

    Expects a directory layout like:
        library/happy/*.mp3
        library/sad/*.wav
        ...
    `mood` selects the subfolder; falls back to any file in `library/`.
    """
    def __init__(self, library_dir):
        self.library_dir = library_dir

    def generate(self, prompt, duration_sec, seed=None, mood=None):
        import glob
        import random as _random
        search_dirs = []
        if mood:
            search_dirs.append(os.path.join(self.library_dir, mood))
        search_dirs.append(self.library_dir)
        candidates = []
        for d in search_dirs:
            for ext in ("*.wav", "*.mp3", "*.ogg", "*.flac"):
                candidates.extend(glob.glob(os.path.join(d, ext)))
            if candidates:
                break
        if not candidates:
            raise FileNotFoundError(
                f"LibrarySource: no audio files found under {self.library_dir}"
            )
        rng = _random.Random(seed)
        path = rng.choice(candidates)
        from audio_bed import load_wav
        audio, sr = load_wav(path)
        return audio, sr


class MusicGenSource(MusicSource):
    """MusicGen-small on GPU. Lazy-loads on first use; can offload after.

    ~300M params, ~2-3GB VRAM — sits alongside IndexTTS2 without crowding it.
    Requires: pip install audiocraft  (pulls torch/torchaudio, already present).
    """
    def __init__(self, model_name="facebook/musicgen-small", device="cuda",
                 offload_after=True):
        self.model_name = model_name
        self.device = device
        self.offload_after = offload_after
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from audiocraft.models import MusicGen
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "MusicGenSource requires 'audiocraft'. Install with "
                "`pip install audiocraft`. Original error: " + str(e)
            )
        self._model = MusicGen.get_pretrained(self.model_name, device=self.device)

    def _unload(self):
        if self._model is None:
            return
        try:
            import torch
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            self._model = None

    def _generate_wav(self, prompt):
        """SPEED LEVER 2 — run the autoregressive generation under mixed
        precision. AudioCraft loads the LM in fp32; bf16/fp16 roughly halves
        compute AND VRAM with no audible difference in a background bed.

        SAFETY: this is wrapped so that ANY failure (unsupported dtype, kernel
        issue, CPU device) falls back to the exact original fp32 path. Worst
        case == today's behaviour; it can never be worse. bf16 is preferred
        over fp16 (numerically safer, no GradScaler needed) on Ampere+.
        """
        import torch
        use_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        if use_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            try:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    wav = self._model.generate([prompt], progress=False)
                # autocast can return bf16/fp16; numpy can't convert bf16, so
                # ALWAYS bring it back to fp32 before it leaves this method.
                return wav.float()
            except Exception as e:
                print(f">> music_source: mixed-precision gen failed ({e}); "
                      f"retrying in fp32 (safe fallback).")
        # Full-precision path (original behaviour / universal fallback).
        return self._model.generate([prompt], progress=False).float()

    def generate(self, prompt, duration_sec, seed=None):
        self._ensure_loaded()
        import torch
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        # SPEED LEVER 1 — short-seed generation. A background/ambience bed is
        # repetitive by nature, and audio_bed.mix_voice_with_bed() ALWAYS loops
        # (with crossfade, via fit_duration) any short bed up to the exact voice
        # length. So there is no reason to pay for a full-length autoregressive
        # generation (~50 tokens/s, strictly sequential): generate a short seed
        # and let the mixer stretch it. Length correctness is unchanged because
        # the mixer, not this call, sets the final duration. Set SEED_SECONDS
        # (module constant) higher if you ever want a longer non-repeating bed.
        dur = float(max(1.0, min(duration_sec, SEED_SECONDS)))
        self._model.set_generation_params(duration=dur)
        wav = self._generate_wav(prompt)  # (1, C, T)
        sr = self._model.sample_rate
        audio = wav[0].detach().cpu().numpy()
        if audio.ndim == 2:      # (channels, T) -> mono
            audio = audio.mean(axis=0)
        audio = audio.astype(np.float32)
        # normalize to a safe headroom before it goes to the mixer
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-6:
            audio = audio / peak * 0.97
        if self.offload_after:
            self._unload()
        return audio, int(sr)


class AudioGenSource(MusicGenSource):
    """AudioGen for environmental SOUND EFFECTS / ambience (rain, wind, crowd,
    birds, whoosh...). Ships in the SAME 'audiocraft' package as MusicGen, so
    no extra install beyond what MusicGenSource already needs.

    Default is audiogen-medium (~1.5B). Generation is short (ambience beds or
    one-shots) and it lazy-loads + offloads exactly like MusicGenSource, so it
    never sits in VRAM alongside IndexTTS2 at peak. On a tight GPU you can point
    the UI at kind='musicgen' instead and still get a passable ambience bed.
    """
    def __init__(self, model_name="facebook/audiogen-medium", device="cuda",
                 offload_after=True):
        super().__init__(model_name=model_name, device=device,
                         offload_after=offload_after)

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from audiocraft.models import AudioGen
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "AudioGenSource requires 'audiocraft'. Install with "
                "`pip install audiocraft`. Original error: " + str(e)
            )
        self._model = AudioGen.get_pretrained(self.model_name, device=self.device)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_source(kind="musicgen", **kwargs):
    kind = (kind or "musicgen").lower()
    if kind in ("musicgen", "gpu", "model"):
        return MusicGenSource(**kwargs)
    if kind in ("audiogen", "sfx", "afx", "ambience"):
        return AudioGenSource(**kwargs)
    if kind in ("library", "files", "folder"):
        return LibrarySource(**kwargs)
    raise ValueError(f"Unknown music source kind: {kind!r}")


if __name__ == "__main__":
    # Prompt-mapper self-test (no model, no GPU).
    v = [0, 0, 0.9, 0, 0, 0.2, 0, 0]   # dominant = sad
    print("sad   ->", emotion_to_prompt(v))
    print("calm  ->", emotion_to_prompt([0, 0, 0, 0, 0, 0, 0, 1.0]))
    print("text  ->", emotion_to_prompt(None, emo_text="joyful excited"))
    print("none  ->", emotion_to_prompt())
    print("extra ->", emotion_to_prompt([1, 0, 0, 0, 0, 0, 0, 0], extra="lo-fi hip hop, vinyl crackle"))
    print("--- SFX/AFX mapper ---")
    print("sfx sad  ->", emotion_to_sfx_prompt([0, 0, 0.9, 0, 0, 0.2, 0, 0]))
    print("sfx calm ->", emotion_to_sfx_prompt([0, 0, 0, 0, 0, 0, 0, 1.0]))
    print("sfx text ->", emotion_to_sfx_prompt(None, emo_text="busy market"))
    print("sfx none ->", emotion_to_sfx_prompt())
