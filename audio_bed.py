"""
audio_bed.py — CPU-only DSP mix engine for layering background music / SFX
under IndexTTS2-generated speech.

Design goals:
  * ZERO GPU. Pure numpy + librosa/soundfile. Fast on CPU.
  * Model-agnostic: takes a voice wav + a music wav (from ANY source — local
    GPU model, library file, or API) and produces a professionally mixed track.
  * The "matching" that makes music sit under a voice is DUCKING + level match,
    both implemented here. Music generation is a separate, swappable step.

Public API:
  mix_voice_with_bed(voice, sr_voice, music, sr_music, ...) -> (mix, sr)
  load_wav(path) -> (np.float32 mono, sr)
  save_wav(path, audio, sr)

All internal processing is float32 mono in [-1, 1]. Callers convert as needed.
"""

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover - soundfile is expected in this env
    sf = None

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_wav(path, target_sr=None, mono=True):
    """Load an audio file to float32. Uses librosa (handles mp3/ogg/wav and
    resampling). Returns (audio[np.float32], sr)."""
    if librosa is None:
        raise RuntimeError("librosa is required to load audio in audio_bed.py")
    audio, sr = librosa.load(path, sr=target_sr, mono=mono)
    return audio.astype(np.float32), int(sr)


def save_wav(path, audio, sr):
    """Write float32/-1..1 (or int16) audio to disk as 16-bit PCM WAV."""
    audio = np.asarray(audio)
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    if sf is not None:
        sf.write(path, audio, sr, subtype="PCM_16")
    else:  # fallback via stdlib wave
        import wave
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(audio.tobytes())
    return path


# ---------------------------------------------------------------------------
# Small DSP primitives
# ---------------------------------------------------------------------------
def _to_mono(audio):
    if audio.ndim == 2:
        # (channels, n) or (n, channels) -> average to mono
        if audio.shape[0] < audio.shape[1]:
            return audio.mean(axis=0)
        return audio.mean(axis=1)
    return audio


def _resample(audio, sr_from, sr_to):
    if sr_from == sr_to:
        return audio
    if librosa is None:
        raise RuntimeError("librosa required for resampling")
    return librosa.resample(audio, orig_sr=sr_from, target_sr=sr_to).astype(np.float32)


def _rms(audio):
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _db_to_gain(db):
    return float(10.0 ** (db / 20.0))


def fit_duration(music, target_len, sr, crossfade_sec=0.5):
    """Loop (with crossfade) or trim music to exactly target_len samples."""
    if music.size == 0:
        return np.zeros(target_len, dtype=np.float32)

    if music.shape[0] >= target_len:
        return music[:target_len].copy()

    # Need to loop. Crossfade the seam to avoid clicks.
    xf = min(int(crossfade_sec * sr), music.shape[0] // 2)
    if xf <= 0:
        reps = int(np.ceil(target_len / music.shape[0]))
        looped = np.tile(music, reps)[:target_len]
        return looped.astype(np.float32)

    fade_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    result = music.copy()
    while result.shape[0] < target_len:
        tail = result[-xf:] * fade_out
        head = music[:xf] * fade_in
        seam = tail + head
        result = np.concatenate([result[:-xf], seam, music[xf:]]).astype(np.float32)
    return result[:target_len].astype(np.float32)


def apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=1.5):
    """In-place-ish linear fade in/out for a bed."""
    audio = audio.copy()
    n = audio.shape[0]
    fi = min(int(fade_in_sec * sr), n)
    fo = min(int(fade_out_sec * sr), n)
    if fi > 0:
        audio[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        audio[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return audio


# ---------------------------------------------------------------------------
# The core: sidechain ducking
# ---------------------------------------------------------------------------
def compute_speech_envelope(voice, sr, frame_ms=20.0, smooth_ms=150.0):
    """Return a per-sample 0..1 envelope that is high where speech is present.
    Cheap RMS-in-frames + smoothing. No GPU, no model."""
    frame = max(1, int(sr * frame_ms / 1000.0))
    # frame-wise RMS
    n_frames = int(np.ceil(voice.shape[0] / frame))
    padded = np.pad(voice, (0, n_frames * frame - voice.shape[0]))
    frames = padded.reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)

    # normalize to 0..1 relative to this clip's speech peak
    peak = np.percentile(rms, 95) or 1e-6
    env_frames = np.clip(rms / peak, 0.0, 1.0)

    # upsample frames -> samples
    env = np.repeat(env_frames, frame)[: voice.shape[0]].astype(np.float32)

    # smooth (moving average) to get natural attack/release, no clicks
    win = max(1, int(sr * smooth_ms / 1000.0))
    if win > 1:
        kernel = np.ones(win, dtype=np.float32) / win
        env = np.convolve(env, kernel, mode="same").astype(np.float32)
    return env


def sidechain_duck(music, voice_env, duck_db=-12.0, threshold=0.15):
    """Attenuate `music` toward duck_db wherever the speech envelope is active.

    music     : bed, same length as voice_env
    voice_env : 0..1 speech presence (from compute_speech_envelope)
    duck_db   : how far to pull the music down under speech (e.g. -12 dB)
    threshold : env level above which ducking begins to engage
    """
    duck_gain = _db_to_gain(duck_db)  # e.g. -12 dB -> ~0.25
    # map env: below threshold -> gain 1.0 (no duck); above -> toward duck_gain
    act = np.clip((voice_env - threshold) / max(1e-6, (1.0 - threshold)), 0.0, 1.0)
    gain_curve = 1.0 - act * (1.0 - duck_gain)  # 1.0 (no speech) .. duck_gain (full speech)
    return (music * gain_curve).astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level mixer
# ---------------------------------------------------------------------------
def mix_voice_with_bed(
    voice,
    sr_voice,
    music,
    sr_music,
    music_db=-18.0,        # bed level relative to voice, in dB
    duck_db=-12.0,         # extra duck under speech
    duck_threshold=0.15,
    fade_in_sec=1.0,
    fade_out_sec=1.5,
    tail_sec=0.0,          # extra music tail after voice ends
    peak_ceiling=0.99,
):
    """Layer a music bed under a voice track. All-CPU. Returns (mix, sr).

    Steps: mono -> match sr -> level-match bed to voice -> fit duration ->
    fade -> sidechain-duck under speech -> sum -> peak-guard.
    """
    voice = _to_mono(np.asarray(voice, dtype=np.float32))
    music = _to_mono(np.asarray(music, dtype=np.float32))

    sr = sr_voice
    music = _resample(music, sr_music, sr)

    target_len = voice.shape[0] + int(tail_sec * sr)

    # 1) level-match: scale music so its RMS sits music_db below the voice RMS
    v_rms = _rms(voice) or 1e-6
    m_rms = _rms(music) or 1e-6
    target_m_rms = v_rms * _db_to_gain(music_db)
    music = (music * (target_m_rms / m_rms)).astype(np.float32)

    # 2) fit duration (loop/trim) then fade
    music = fit_duration(music, target_len, sr)
    music = apply_fades(music, sr, fade_in_sec, fade_out_sec)

    # 3) sidechain duck under speech
    voice_padded = np.pad(voice, (0, target_len - voice.shape[0])) if target_len > voice.shape[0] else voice
    env = compute_speech_envelope(voice_padded, sr)
    music = sidechain_duck(music, env, duck_db=duck_db, threshold=duck_threshold)

    # 4) sum
    mix = voice_padded + music

    # 5) peak-guard against clipping
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > peak_ceiling:
        mix = mix * (peak_ceiling / peak)

    return mix.astype(np.float32), sr


def mix_files(voice_path, music_path, output_path, **kwargs):
    """Convenience: load two files, mix, save. Returns output_path."""
    voice, sr_v = load_wav(voice_path)
    music, sr_m = load_wav(music_path)
    mix, sr = mix_voice_with_bed(voice, sr_v, music, sr_m, **kwargs)
    save_wav(output_path, mix, sr)
    return output_path


def overlay_oneshot(track, sr, sfx, sr_sfx, at_sec=0.0, gain_db=-12.0,
                    fade_ms=15.0, peak_ceiling=0.99):
    """Overlay a one-shot SFX (whoosh, impact, sting) onto `track` at a point
    in time. Unlike a continuous ambience bed, this is a single placed sound
    and is NOT ducked (it's meant to punctuate, not underscore).

    track   : base mono float32 (already-mixed voice[+music]) — NOT modified in place
    sfx     : mono float32 one-shot
    at_sec  : where to place the SFX start, in seconds
    gain_db : level of the SFX relative to full-scale (negative = quieter)

    Returns (out[np.float32], sr). Extends the track if the SFX runs past its end.
    """
    track = _to_mono(np.asarray(track, dtype=np.float32)).copy()
    sfx = _to_mono(np.asarray(sfx, dtype=np.float32))
    sfx = _resample(sfx, sr_sfx, sr)

    # normalize sfx then apply target gain
    peak = float(np.max(np.abs(sfx))) if sfx.size else 0.0
    if peak > 1e-6:
        sfx = sfx / peak
    sfx = (sfx * _db_to_gain(gain_db)).astype(np.float32)
    sfx = apply_fades(sfx, sr, fade_ms / 1000.0, fade_ms / 1000.0)

    start = max(0, int(round(at_sec * sr)))
    end = start + len(sfx)
    if end > len(track):
        track = np.concatenate([track, np.zeros(end - len(track), dtype=np.float32)])
    track[start:end] += sfx

    p = float(np.max(np.abs(track))) if track.size else 0.0
    if p > peak_ceiling:
        track = track * (peak_ceiling / p)
    return track.astype(np.float32), sr


if __name__ == "__main__":
    # Tiny self-test with synthetic signals (no files, no model needed).
    sr = 22050
    t = np.linspace(0, 3, sr * 3, dtype=np.float32)
    # "voice": bursts of tone with gaps (so ducking has something to react to)
    voice = 0.4 * np.sin(2 * np.pi * 200 * t)
    gate = ((t % 1.0) < 0.6).astype(np.float32)   # on 0.6s, off 0.4s
    voice = voice * gate
    # "music": steady chord
    music = 0.3 * (np.sin(2 * np.pi * 130.81 * t) + np.sin(2 * np.pi * 164.81 * t))
    mix, out_sr = mix_voice_with_bed(voice, sr, music, sr,
                                     music_db=-16.0, duck_db=-14.0)
    print(f"self-test OK: mix len={mix.shape[0]} sr={out_sr} peak={np.max(np.abs(mix)):.3f}")
    # one-shot SFX overlay self-test
    sfx = 0.9 * np.sin(2 * np.pi * 800 * np.linspace(0, 0.25, int(sr * 0.25), dtype=np.float32))
    out, _ = overlay_oneshot(mix, out_sr, sfx, sr, at_sec=1.0, gain_db=-10.0)
    print(f"oneshot OK: len={out.shape[0]} peak={np.max(np.abs(out)):.3f} fresh={out is not mix}")
