"""ref_prep.py — put the reference's PITCH into range before it conditions the model.

THE PROBLEM (measured on this project)
--------------------------------------
The reference clip's fundamental frequency drives the OUTPUT's register, causally:

  * a female reference measured at 408-433 Hz produced near-silence;
  * the same clip retuned to ~200 Hz synthesised normally;
  * refs in the normal adult band (male 85-180, female 165-255 Hz) work as-is
    - e.g. the ear-approved KimW reference measures 217.6 Hz.

So an out-of-band reference is not a "bad sample" the model can ignore: it is the
hard ceiling on the take. `infer_v2._load_and_cut_audio` already trims silence and
normalises LOUDNESS (-23 dBFS) but nothing touches pitch, so a 420 Hz upload still
reaches feature extraction unchanged.

THE FIX
-------
Wrap the instance's `_load_and_cut_audio` (the exact function the model already
uses to preprocess every reference, spk and emo) and, only when the measured
median voiced f0 is clearly outside a generous band, retune the clip toward a
target pitch. Duration is preserved either way.

  * engine=pyworld   -> WORLD vocoder: f0 scaled, spectral envelope (formants)
                        kept, so speaker identity survives a large shift.
  * engine=librosa   -> phase-vocoder pitch_shift: formants move WITH the pitch
                        (tape-speed character). Used when pyworld is unavailable.
                        It is logged, because it is a lower-fidelity fallback.

SAFETY
------
  * never raises: any failure logs once and returns the original audio untouched;
  * a no-op for anything inside the band, so references that already work are
    bit-identical (the band default 80-330 Hz covers all normal adult speech);
  * opt-out, dry-run and per-field tuning via environment variables below;
  * runs only when the model's own reference cache misses, so the hot path
    (repeat segments of one speaker) costs nothing.

ENV
---
  INDEXTTS_REF_F0_NORM      auto (default) | force | dry | 0/off
  INDEXTTS_REF_F0_TARGET    target pitch for a retune, Hz       (default 200)
  INDEXTTS_REF_F0_HI        act when median f0 > this, Hz       (default 330)
  INDEXTTS_REF_F0_LO        act when median f0 < this, Hz       (default 80)
  INDEXTTS_REF_F0_MAX_SHIFT clamp on the shift factor           (default 3.0)
  INDEXTTS_REF_F0_EMO       also retune the emotion reference   (default 1)
  INDEXTTS_REF_F0_ENGINE    auto | pyworld | librosa           (default auto)

CLI (no GPU needed) — measure and optionally write a retuned copy:
  python ref_prep.py REF.wav [REF2.wav ...] [--apply OUTDIR]
"""
import logging
import math
import os

import numpy as np

log = logging.getLogger("ref_prep")

_MODES = {"off", "dry", "auto", "force"}
_OFF = {"0", "off", "false", "no", "disable", "disabled", "none"}


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def config():
    """Read the knobs from the environment (called per install, not per audio)."""
    mode = str(os.environ.get("INDEXTTS_REF_F0_NORM", "auto")).strip().lower()
    if mode in _OFF:
        mode = "off"
    elif mode not in _MODES:
        mode = "auto"
    eng = str(os.environ.get("INDEXTTS_REF_F0_ENGINE", "auto")).strip().lower()
    if eng not in ("auto", "pyworld", "librosa"):
        eng = "auto"
    emo = str(os.environ.get("INDEXTTS_REF_F0_EMO", "1")).strip().lower() not in _OFF
    return {
        "mode": mode,
        "target": _f("INDEXTTS_REF_F0_TARGET", 200.0),
        "hi": _f("INDEXTTS_REF_F0_HI", 330.0),
        "lo": _f("INDEXTTS_REF_F0_LO", 80.0),
        "max_shift": max(1.0, _f("INDEXTTS_REF_F0_MAX_SHIFT", 3.0)),
        "emo": emo,
        "engine": eng,
    }


# --------------------------------------------------------------------------- f0
def measure_f0(y, sr, max_seconds=12.0):
    """Median / p10 / p90 of the VOICED f0, or None if it cannot be measured.

    yin is used instead of pyin: same ordering for a register decision, far
    cheaper. Frames below 1% of the clip's peak RMS are excluded so trailing
    noise or a breath does not drag the median around.
    """
    import librosa

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr <= 0 or y.size < int(sr * 0.2):
        return None
    if y.size > int(max_seconds * sr):
        y = y[: int(max_seconds * sr)]
    if not np.isfinite(y).all():
        y = np.nan_to_num(y)
    if float(np.max(np.abs(y))) < 1e-4:
        return None
    try:
        # frame_length must hold >=2 periods of fmin or librosa warns and the
        # estimate degrades: at 48 kHz, 1024 samples is only one period of 50 Hz
        # (librosa wants strictly more than two, hence the +2).
        fl = min(4096, int(max(1024, math.ceil(2.0 * sr / 50.0) + 2)))
        hop = max(128, fl // 4)
        f0 = librosa.yin(y, fmin=50, fmax=600, sr=sr, frame_length=fl, hop_length=hop)
        rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=hop)[0]
        n = min(len(f0), len(rms))
        f0, rms = f0[:n], rms[:n]
        # 5% of the loudest frame (~-26 dB): keeps quiet but real speech, drops
        # near-silence, where yin's output is noise and octave errors live.
        keep = (f0 > 0) & np.isfinite(f0) & (rms > 0.05 * float(rms.max()))
        v = f0[keep]
    except Exception as e:
        log.debug(f"[ref_prep] f0 measurement failed: {e}")
        return None
    if v.size < 8:
        return None
    return (float(np.median(v)), float(np.percentile(v, 10)),
            float(np.percentile(v, 90)), int(v.size))


# ------------------------------------------------------------------------ shift
def _shift_pyworld(y, sr, factor):
    """Scale f0 while KEEPING the spectral envelope -> identity survives."""
    import pyworld as pw

    y64 = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    f0, t = pw.harvest(y64, float(sr), frame_period=5.0)
    f0 = pw.stonemask(y64, f0, t, float(sr))
    sp = pw.cheaptrick(y64, f0, t, float(sr))
    ap = pw.d4c(y64, f0, t, float(sr))
    out = pw.synthesize(f0 * float(factor), sp, ap, float(sr), frame_period=5.0)
    return np.asarray(out, dtype=np.float32)


def _shift_librosa(y, sr, factor):
    """Phase-vocoder shift: pitch and formants move together (lower fidelity)."""
    import librosa

    steps = 12.0 * math.log2(float(factor))
    return np.asarray(librosa.effects.pitch_shift(np.asarray(y, dtype=np.float32),
                                                  sr=int(sr), n_steps=steps),
                      dtype=np.float32)


def _fit(y2, n):
    """Force the shifted signal to the original length (pad or trim)."""
    y2 = np.asarray(y2, dtype=np.float32).reshape(-1)
    if y2.size == n:
        return y2
    if y2.size > n:
        return y2[:n]
    return np.pad(y2, (0, n - y2.size))


def retune(y, sr, factor, engine="auto"):
    """Return (audio, engine_name). engine_name is 'none' if nothing worked."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    order = [("pyworld", _shift_pyworld), ("librosa", _shift_librosa)]
    if engine in ("pyworld", "librosa"):
        order.sort(key=lambda kv: kv[0] != engine)
    for name, fn in order:
        try:
            out = _fit(fn(y, sr, factor), y.size)
            if out.size and np.isfinite(out).all() and float(np.max(np.abs(out))) > 1e-6:
                return np.clip(out, -1.0, 1.0), name
        except Exception as e:
            log.debug(f"[ref_prep] engine {name} failed: {e}")
    return y, "none"


# ------------------------------------------------------------------------ apply
def process(y, sr, cfg, label="spk", dry=False):
    """Decide and (unless dry) retune one reference array. Returns (audio, note)."""
    m = measure_f0(y, sr)
    if m is None:
        return y, "f0 unmeasurable -> unchanged"
    med, p10, p90, n = m
    tag = f"{label} ref f0 {med:.0f} Hz (p10-p90 {p10:.0f}-{p90:.0f}, n={n})"
    factor = 1.0
    if med > cfg["hi"]:
        factor = cfg["target"] / med
    elif med < cfg["lo"]:
        factor = cfg["target"] / med
    if med > cfg["hi"] or med < cfg["lo"]:
        factor = float(np.clip(factor, 1.0 / cfg["max_shift"], cfg["max_shift"]))
    if abs(factor - 1.0) < 0.02:
        return y, f"{tag} in band -> unchanged"
    if dry:
        return y, f"{tag} -> x{factor:.3f} to {med * factor:.0f} Hz [dry run: not applied]"
    out, engine = retune(y, sr, factor, cfg["engine"])
    if engine == "none":
        return y, f"{tag} -> retune wanted x{factor:.3f} but no engine available -> unchanged"
    return out, f"{tag} -> x{factor:.3f} to {med * factor:.0f} Hz via {engine}"


def install(tts, verbose=True):
    """Wrap `tts._load_and_cut_audio` so references are register-checked.

    Safe to call more than once; returns True if the hook is now active.
    """
    if getattr(tts, "_ref_prep_applied", False):
        return True
    cfg = config()
    if cfg["mode"] == "off":
        if verbose:
            log.info("[ref_prep] disabled (INDEXTTS_REF_F0_NORM=0)")
        tts._ref_prep_applied = True
        return False

    import torch

    orig = tts._load_and_cut_audio
    state = {"warned": False}
    dry = cfg["mode"] == "dry"

    def wrapped(audio_path, max_audio_length_seconds, verbose=False, sr=None,
                preprocess=True, target_dbfs=-23.0, trim_db=30.0):
        audio, out_sr = orig(audio_path, max_audio_length_seconds, verbose, sr,
                             preprocess, target_dbfs, trim_db)
        try:
            # the two call sites are the speaker ref (no sr) and the emotion ref
            # (sr=16000); only the latter passes sr.
            is_emo = sr is not None and int(sr) == 16000
            if is_emo and not cfg["emo"]:
                return audio, out_sr
            if torch.is_tensor(audio):
                y = audio.reshape(-1).detach().cpu().numpy().astype(np.float32)
            else:
                y = np.asarray(audio, dtype=np.float32).reshape(-1)
            y2, note = process(y, out_sr, cfg, "emo" if is_emo else "spk", dry=dry)
            if y2 is y or np.array_equal(y2, y):
                log.info(f"[ref_prep] {note}")
                return audio, out_sr
            new = torch.tensor(np.ascontiguousarray(y2), dtype=torch.float32).unsqueeze(0)
            log.info(f"[ref_prep] {note}")
            return new, out_sr
        except Exception as e:
            if not state["warned"]:
                state["warned"] = True
                log.warning(f"[ref_prep] skipped ({type(e).__name__}: {e}); "
                            f"reference used unchanged")
            return audio, out_sr

    tts._load_and_cut_audio = wrapped
    tts._ref_prep_applied = True
    if verbose:
        log.info(f"[ref_prep] installed: mode={cfg['mode']} band "
                 f"{cfg['lo']:.0f}-{cfg['hi']:.0f} Hz target {cfg['target']:.0f} Hz "
                 f"engine={cfg['engine']} emo={'on' if cfg['emo'] else 'off'}")
    return True


def report(paths):
    """Print the measured register of one or more reference files."""
    import librosa

    cfg = config()
    print(f"[ref_prep] band {cfg['lo']:.0f}-{cfg['hi']:.0f} Hz, target "
          f"{cfg['target']:.0f} Hz, mode={cfg['mode']}")
    print(f"{'file':44s} {'median':>7s} {'p10':>6s} {'p90':>6s} {'frames':>7s}  verdict")
    for p in paths:
        try:
            y, sr = librosa.load(p, sr=None, mono=True)
            m = measure_f0(y, sr)
            if m is None:
                print(f"{os.path.basename(p)[:44]:44s}      -      -      -       -  unmeasurable")
                continue
            med, p10, p90, n = m
            if med > cfg["hi"]:
                v = f"TOO HIGH -> x{cfg['target'] / med:.3f} (retune on upload)"
            elif med < cfg["lo"]:
                v = f"TOO LOW  -> x{cfg['target'] / med:.3f} (retune on upload)"
            else:
                v = "in band -> passthrough"
            print(f"{os.path.basename(p)[:44]:44s} {med:7.1f} {p10:6.1f} {p90:6.1f} {n:7d}  {v}")
        except Exception as e:
            print(f"{os.path.basename(p)[:44]:44s}  ERROR {e}")


def _main(argv):
    import soundfile as sf

    outdir = None
    if "--apply" in argv:
        i = argv.index("--apply")
        outdir = argv[i + 1] if i + 1 < len(argv) else "."
        argv = argv[:i] + argv[i + 2:]
    paths = [a for a in argv[1:] if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 1
    report(paths)
    if outdir:
        import librosa

        cfg = config()
        os.makedirs(outdir, exist_ok=True)
        for p in paths:
            y, sr = librosa.load(p, sr=None, mono=True)
            y2, note = process(y, sr, cfg, "cli")
            dst = os.path.join(outdir, os.path.basename(p))
            sf.write(dst, y2, sr)
            print(f"  -> {dst}   ({note})")
    return 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(_main(sys.argv))
