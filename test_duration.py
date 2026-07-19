#!/usr/bin/env python3
"""
test_duration.py — Does *this* IndexTTS2 checkpoint really support duration control?

Background (verified against IndexTeam/IndexTTS-2 weights + fork source):
  * IndexTTS2's duration/speed control conditions generation on a target token
    count. Seconds are converted at 50 tokens/second: num_codes = round(sec * 50)
    (the semantic codec runs at 50 Hz).
  * The GPT wraps this in two tiny modules:
        speed_emb            : nn.Embedding(2, model_dim)   # normal vs half-speed switch
        mel_pos_embedding.emb : positional table reused as the duration embedding
    In code these are *initialized to zero* (speed_emb via normal_(std=0.0)).
    -> If a checkpoint never trained them, speed_emb stays all-zero and the
       mechanism is inert. If it trained them, the weights are non-zero.
  * The OFFICIAL inference (infer_v2.py) hard-codes use_speed=0 and dropped the
    num_codes arg, so it *refuses to send* the signal even when weights support it.
    The fork (iszhanjiawei/IndexTTS2, indextts/infer_indextts2.py) sends it.

This script answers two things for YOUR finetuned ckpt:
  [STATIC]  Are the duration params present AND trained (non-zero)?  -> torch only
  [LIVE]    (optional) Does target_dur actually change output length? -> needs your
            inference module + a reference wav.

Usage
-----
  # Static check only (fast, no generation):
  python test_duration.py --ckpt /path/to/checkpoints

  # Static + live generation test (measures real durations):
  python test_duration.py --ckpt /path/to/checkpoints \
      --ref /path/to/reference_voice.wav \
      --text "አንድ ሁለት ሦስት አራት አምስት ስድስት ሰባት"      # Amharic sample, any language ok

Exit code is 0 if duration control looks supported, 1 otherwise.
"""

import argparse
import os
import sys
import wave
import contextlib

# ----------------------------- pretty printing ------------------------------ #
def _c(txt, code):
    # colour only when attached to a real terminal
    return f"\033[{code}m{txt}\033[0m" if sys.stdout.isatty() else txt

OK   = lambda s: _c("PASS " + s, "32")
BAD  = lambda s: _c("FAIL " + s, "31")
WARN = lambda s: _c("WARN " + s, "33")
HEAD = lambda s: _c(s, "1;36")


# --------------------------- checkpoint discovery --------------------------- #
def find_gpt_checkpoint(ckpt_dir):
    """Return path to the GPT weights file inside a checkpoints dir."""
    candidates = ["gpt.pth", "gpt.safetensors", "gpt_v2.pth"]
    for name in candidates:
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return p
    # fall back: any file with 'gpt' in the name
    for f in sorted(os.listdir(ckpt_dir)):
        if "gpt" in f.lower() and f.lower().endswith((".pth", ".pt", ".safetensors")):
            return os.path.join(ckpt_dir, f)
    return None


def load_state_dict(path):
    """Load a state_dict from .pth/.pt or .safetensors, weights only, on CPU."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")

    import torch
    # weights_only=True is safe + fast; fall back for older torch.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    # unwrap common containers
    for key in ("model", "state_dict", "gpt", "net"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return obj


# ------------------------------ static check -------------------------------- #
# tensors that carry the duration/speed mechanism, matched by suffix so we are
# agnostic to any 'gpt.' / 'module.' prefix your finetune may have added.
SPEED_KEYS = ("speed_emb.weight",)
POS_KEYS   = ("mel_pos_embedding.emb.weight",)


def _find(sd, suffixes):
    for k in sd:
        if any(k.endswith(sfx) for sfx in suffixes):
            return k
    return None


def static_check(ckpt_dir):
    print(HEAD("\n=== [STATIC] checkpoint introspection ==="))
    gpt_path = find_gpt_checkpoint(ckpt_dir)
    if not gpt_path:
        print(BAD(f"no gpt weights found in {ckpt_dir}"))
        return False
    print(f"  gpt weights : {gpt_path}")

    try:
        sd = load_state_dict(gpt_path)
    except Exception as e:
        print(BAD(f"could not load state_dict: {e}"))
        return False
    print(f"  tensors     : {len(sd)}")

    import torch

    speed_k = _find(sd, SPEED_KEYS)
    pos_k   = _find(sd, POS_KEYS)

    supported = True

    # 1) speed_emb present?
    if speed_k is None:
        print(BAD("speed_emb.weight NOT in checkpoint -> duration head absent"))
        return False
    w = sd[speed_k].float()
    print(f"  found       : {speed_k}  shape={list(w.shape)}")

    # 2) speed_emb trained (non-zero)?  <-- the decisive test
    nonzero = int((w != 0).sum())
    total   = w.numel()
    maxabs  = float(w.abs().max())
    if nonzero == 0 or maxabs == 0.0:
        print(BAD(f"speed_emb is ALL-ZERO ({nonzero}/{total} non-zero) "
                  "-> mechanism UNTRAINED / disabled in these weights"))
        supported = False
    else:
        print(OK(f"speed_emb TRAINED  ({nonzero}/{total} non-zero, "
                 f"max|w|={maxabs:.5f}, mean|w|={float(w.abs().mean()):.6f})"))

    # 3) positional/duration table present + trained?
    if pos_k is None:
        print(WARN("mel_pos_embedding.emb.weight not found (unusual) "
                   "-> duration embedding table missing"))
    else:
        pw = sd[pos_k].float()
        pnz = int((pw != 0).sum())
        if pnz == 0:
            print(BAD(f"{pos_k} is ALL-ZERO -> duration table untrained"))
            supported = False
        else:
            print(OK(f"{pos_k} present & trained  shape={list(pw.shape)} "
                     f"(rows={pw.shape[0]} => max ~{pw.shape[0] // 50}s @50tok/s)"))

    print()
    if supported:
        print(OK("STATIC VERDICT: duration control is TRAINED into this checkpoint."))
    else:
        print(BAD("STATIC VERDICT: duration params are inert (zero) — "
                  "this ckpt will NOT respond to target_dur."))
    return supported


# ------------------------------- live check --------------------------------- #
def wav_duration_seconds(path):
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getnframes() / float(w.getframerate())


def _patch_tokenizer_compat(tts):
    """Bridge a tokenizer API drift: the fork's infer_indextts2 calls
    tokenizer.split_sentences(), but newer repos renamed it to split_segments().
    Alias it so FREE mode and long-text (multi-segment) generation don't crash.
    """
    tok = getattr(tts, "tokenizer", None)
    if tok is not None and not hasattr(tok, "split_sentences") and hasattr(tok, "split_segments"):
        try:
            tok.split_sentences = tok.split_segments
            print("  [compat] aliased tokenizer.split_sentences -> split_segments")
        except Exception as e:  # pragma: no cover - defensive
            print(WARN(f"could not alias split_sentences: {e}"))
    return tts


def _load_tts(ckpt_dir):
    """
    Try the known IndexTTS2 entry points. Returns an object with .infer(...).
    Adjust the import here if your project wraps IndexTTS2 differently.
    """
    cfg = os.path.join(ckpt_dir, "config.yaml")
    errors = []
    # fork layout (has use_speed/target_dur)
    try:
        from indextts.infer_indextts2 import IndexTTS2
        return _patch_tokenizer_compat(
            IndexTTS2(cfg_path=cfg, model_dir=ckpt_dir, is_fp16=False, use_cuda_kernel=False)
        ), "infer_indextts2"
    except Exception as e:
        errors.append(f"infer_indextts2: {e}")
    # official layout (v2) — may ignore target_dur, we still test empirically
    try:
        from indextts.infer_v2 import IndexTTS2
        return _patch_tokenizer_compat(
            IndexTTS2(cfg_path=cfg, model_dir=ckpt_dir, is_fp16=False, use_cuda_kernel=False)
        ), "infer_v2"
    except Exception as e:
        errors.append(f"infer_v2: {e}")
    raise ImportError("could not import IndexTTS2:\n  " + "\n  ".join(errors))


# ------------------------- token-rate measurement --------------------------- #
# The mapping seconds<->tokens is num_codes = round(seconds * TOKENS_PER_SECOND).
# Theoretically ~50 (semantic 50Hz; audio_sec = tokens * 1.72 * hop/sr).
# A finetune that changed the s2mel/vocoder config could shift this. We MEASURE it.
TOKENS_PER_SECOND = 50


def _install_token_capture(tts):
    """Monkeypatch tts.gpt.inference_speech to record how many semantic tokens the
    GPT actually generates per call. Returns (capture, restore).

    capture['per_call'] is a list appended once per inference_speech call; clear it
    before each infer() and sum after to get that utterance's generated-token count.
    """
    capture = {"per_call": [], "raw": [], "errors": []}
    gpt = getattr(tts, "gpt", None)
    if gpt is None or not hasattr(gpt, "inference_speech"):
        capture["errors"].append("tts.gpt.inference_speech not found")
        return capture, (lambda: None)

    orig = gpt.inference_speech
    stop = getattr(tts, "stop_mel_token", getattr(gpt, "stop_mel_token", 8193))

    def wrapped(*a, **k):
        out = orig(*a, **k)
        try:
            codes = out.sequences if hasattr(out, "sequences") else out
            row = codes[0]
            hit = (row == stop).nonzero(as_tuple=False)
            n = int(hit[0].item()) if hit.numel() > 0 else int(row.shape[-1])
            capture["per_call"].append(n)
            capture["raw"].append(int(row.shape[-1]))
        except Exception as e:  # pragma: no cover - defensive
            capture["errors"].append(repr(e))
        return out

    gpt.inference_speech = wrapped
    return capture, (lambda: setattr(gpt, "inference_speech", orig))


def _rate_verdict(rate: float) -> None:
    """Print whether a measured tokens/sec matches the assumed 50."""
    if rate != rate:  # NaN
        print(WARN("could not measure tokens/sec (token capture unavailable)"))
        return
    off = (rate - TOKENS_PER_SECOND) / TOKENS_PER_SECOND * 100.0
    if abs(rate - TOKENS_PER_SECOND) <= 1.5:
        print(OK(f"MEASURED rate = {rate:.2f} tok/s  (~{TOKENS_PER_SECOND}, {off:+.1f}%). "
                 f"num_codes = round(seconds * {TOKENS_PER_SECOND}) is CORRECT for this model."))
    else:
        rec = round(rate)
        print(WARN(f"MEASURED rate = {rate:.2f} tok/s  ({off:+.1f}% vs {TOKENS_PER_SECOND}). "
                   f"This model is NOT 50/s -> use num_codes = round(seconds * {rec})."))


def calibrate_rate(ckpt_dir, ref_wav, text, out_dir, preloaded=None):
    """Measure the pure architectural tokens/sec via a FREE-mode generation
    (no duration control): generated_tokens / audio_seconds. Use a single-sentence
    --text so no inter-segment silence biases the ratio."""
    print(HEAD("\n=== [CALIBRATE] pure tokens/sec (free-mode generation) ==="))
    if not os.path.isfile(ref_wav):
        print(BAD(f"reference wav not found: {ref_wav}"))
        return None
    try:
        tts, mod = preloaded if preloaded else _load_tts(ckpt_dir)
    except Exception as e:
        print(BAD(str(e)))
        return None
    os.makedirs(out_dir, exist_ok=True)

    capture, restore = _install_token_capture(tts)
    out = os.path.join(out_dir, "calib_free.wav")
    try:
        capture["per_call"].clear()
        # NOTE: no interval_silence kwarg — this fork forwards unknown kwargs to HF
        # .generate() which rejects them. Single-sentence text => no inter-segment gap anyway.
        tts.infer(spk_audio_prompt=ref_wav, text=text, output_path=out, verbose=False)
    except Exception as e:
        print(WARN(f"free-mode generation failed ({e}); skipping calibration. "
                   "The LIVE table's tok/s column still measures the rate under duration control."))
        return None
    finally:
        restore()

    tok = sum(capture["per_call"])
    sec = wav_duration_seconds(out)
    if not tok:
        print(WARN("could not capture generated token count (model layout differs). "
                   "Fall back to the tok/s column in the LIVE table below."))
        return None
    rate = (tok / sec) if sec > 0 else float("nan")
    print(f"  free-mode: {tok} tokens -> {sec:.2f}s audio")
    _rate_verdict(rate)
    return rate


def live_check(ckpt_dir, ref_wav, text, targets, out_dir, preloaded=None):
    print(HEAD("\n=== [LIVE] real generation @ target durations ==="))
    if not os.path.isfile(ref_wav):
        print(BAD(f"reference wav not found: {ref_wav}"))
        return False
    try:
        tts, mod = preloaded if preloaded else _load_tts(ckpt_dir)
    except Exception as e:
        print(BAD(str(e)))
        return False
    print(f"  loaded via  : {mod}")
    os.makedirs(out_dir, exist_ok=True)

    capture, restore = _install_token_capture(tts)
    rows, ok_count = [], 0
    try:
        for tgt in targets:
            out = os.path.join(out_dir, f"dur_{tgt:.1f}s.wav")
            capture["per_call"].clear()
            try:
                tts.infer(spk_audio_prompt=ref_wav, text=text, output_path=out,
                          use_speed=True, target_dur=float(tgt), verbose=False)
            except TypeError:
                print(WARN("infer() rejected use_speed/target_dur -> "
                           "this inference module does not expose duration control "
                           "(official build). Weights may still support it via the fork."))
                return False
            except Exception as e:
                print(BAD(f"generation failed at {tgt}s: {e}"))
                return False

            actual = wav_duration_seconds(out)
            gen_tok = sum(capture["per_call"])
            rate = (gen_tok / actual) if (actual > 0 and gen_tok) else float("nan")
            err = actual - tgt
            req_tokens = round(tgt * TOKENS_PER_SECOND)
            hit = abs(err) <= max(0.25, 0.1 * tgt)   # within 250ms or 10%
            ok_count += hit
            rows.append((tgt, req_tokens, gen_tok, actual, err, rate, hit))
    finally:
        restore()

    # req_tok = seconds*50 we asked for; gen_tok = what the model actually emitted;
    # tok/s = gen_tok/actual = the model's TRUE rate (this is the "is it 50?" answer).
    print(f"\n  {'target':>7} {'req_tok':>7} {'gen_tok':>7} {'actual':>7} {'error':>7} {'tok/s':>7}   result")
    print("  " + "-" * 64)
    measured_rates = []
    for tgt, rtk, gtk, act, err, rate, hit in rows:
        tag = OK("") if hit else BAD("")
        if rate == rate:
            measured_rates.append(rate)
            rate_s = f"{rate:6.1f}"
        else:
            rate_s = "   n/a"
        print(f"  {tgt:7.2f} {rtk:7d} {gtk:7d} {act:7.2f} {err:+7.2f} {rate_s:>7}   {tag}")

    if measured_rates:
        avg_rate = sum(measured_rates) / len(measured_rates)
        print()
        _rate_verdict(avg_rate)

    # decisive: does output length actually TRACK the target? (indices: tgt=0, actual=3)
    spread_target = max(r[0] for r in rows) - min(r[0] for r in rows)
    spread_actual = max(r[3] for r in rows) - min(r[3] for r in rows)
    tracks = spread_actual >= 0.5 * spread_target

    # Below-floor detection: every request undershoots what the text needs, so the
    # model overshoots the target (actual > target) AND emits more tokens than asked.
    all_overshoot = all(r[4] > 0.25 for r in rows)          # err = actual - target
    all_gen_exceeds = all(r[2] > r[1] for r in rows if r[2]) # gen_tok > req_tok
    est_natural = max(r[3] for r in rows)                    # lower bound on natural length

    print()
    if ok_count == len(rows):
        print(OK(f"LIVE VERDICT: durations hit target ({ok_count}/{len(rows)}). "
                 "Duration control WORKS on this checkpoint."))
        return True
    if all_overshoot and all_gen_exceeds:
        lo, hi = est_natural, est_natural * 1.25
        print(WARN(
            f"LIVE VERDICT: TARGETS BELOW NATURAL FLOOR — every target is shorter than the "
            f"text can be spoken, so the model floored near its natural length (~{est_natural:.1f}s "
            f"or more) and overshot. This is NOT a failure of duration control; you asked it to "
            f"compress past what the content allows.\n"
            f"       -> Retest with targets >= the natural length, e.g. --targets "
            f"{lo:.0f},{(lo+hi)/2:.0f},{hi:.0f}. For genuinely tight cues, compress the text or "
            f"time-stretch the audio afterward."))
        return True
    if tracks:
        print(WARN(f"LIVE VERDICT: output length tracks target "
                   f"(Δactual={spread_actual:.2f}s vs Δtarget={spread_target:.2f}s) "
                   f"but accuracy is loose ({ok_count}/{len(rows)} within tolerance). "
                   "Mechanism is active. Check the tok/s column above: if it is ~50 the "
                   "looseness is the text's natural floor/ceiling, not a rate mismatch."))
        return True
    print(BAD(f"LIVE VERDICT: output length does NOT track target "
              f"(Δactual={spread_actual:.2f}s). target_dur is being ignored."))
    return False


# ---------------------------------- main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Check IndexTTS2 ckpt duration-control support.")
    ap.add_argument("--ckpt", default="checkpoints",
                    help="checkpoints dir (contains gpt.pth, config.yaml, s2mel.pth ...)")
    ap.add_argument("--ref", default=None,
                    help="reference speaker wav -> enables the LIVE generation test")
    ap.add_argument("--text", default="አንድ ሁለት ሦስት አራት አምስት ስድስት ሰባት ስምንት ዘጠኝ አስር",
                    help="text to synthesize in the live test (default: Amharic counting)")
    ap.add_argument("--targets", default="3,5,8",
                    help="comma-separated target seconds for the live test")
    ap.add_argument("--out", default="duration_test_out",
                    help="output dir for generated wavs")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure this model's true tokens/sec via a free-mode generation "
                         "(answers 'is it exactly 50/s?'). Requires --ref.")
    args = ap.parse_args()

    if not os.path.isdir(args.ckpt):
        print(BAD(f"--ckpt dir not found: {args.ckpt}"))
        sys.exit(2)

    static_ok = static_check(args.ckpt)

    # Load the model once and share it across calibrate + live to avoid a double load.
    preloaded = None
    if args.ref or args.calibrate:
        if args.calibrate and not args.ref:
            print(WARN("\n--calibrate needs --ref (a reference wav) to generate; skipping calibration."))
        try:
            preloaded = _load_tts(args.ckpt)
        except Exception as e:
            print(BAD(f"\ncould not load model for live/calibrate: {e}"))
            preloaded = None

    if args.calibrate and args.ref and preloaded:
        calibrate_rate(args.ckpt, args.ref, args.text, args.out, preloaded=preloaded)

    live_ok = None
    if args.ref and preloaded:
        targets = [float(x) for x in args.targets.split(",") if x.strip()]
        live_ok = live_check(args.ckpt, args.ref, args.text, targets, args.out, preloaded=preloaded)
    elif not args.ref:
        print(WARN("\nskipping LIVE test (no --ref wav given). "
                   "Static result is definitive for whether the weights are trained; "
                   "add --ref to measure real output durations and tokens/sec."))

    # final rollup
    print(HEAD("\n=== SUMMARY ==="))
    print(f"  static (weights trained?) : {'YES' if static_ok else 'NO'}")
    print(f"  live   (target respected?) : "
          f"{'—' if live_ok is None else ('YES' if live_ok else 'NO')}")

    final = static_ok if live_ok is None else (static_ok and live_ok)
    print(OK("\nDuration control SUPPORTED on this checkpoint.") if final
          else BAD("\nDuration control NOT usable on this checkpoint as-is."))
    sys.exit(0 if final else 1)


if __name__ == "__main__":
    main()
