# Plan: Background Music + SFX Generator for TTS Voices (`webui_enhanced.py`)

**Goal:** Add an optional, mood-matched background music (and SFX) layer under IndexTTS2-generated
voices, mixed and ducked professionally. Must run WITHOUT heavy GPU/CPU.

**Status:** PLAN ONLY — no code written yet. Awaiting confirmation of the two design choices below.

---

## 0. The core insight (why this is cheap)

"Matching" the voice = **mixing (DSP)**, not generation:
- **Mood match** → reuse the existing `emo_vector` / `emo_text` / QwenEmotion pipeline to auto-write the music prompt.
- **Duration match** → loop/trim the bed to the TTS length (pure numpy/pydub).
- **Level match / ducking** → sidechain-duck music under speech (music dips ~-12 dB when voice present, rises in gaps). **This is the single biggest "sounds professional" factor. Zero GPU.**

Music GENERATION is the only optional heavy part → make it **pluggable** so the Lightning box can do
zero music compute if desired.

---

## 1. Architecture — pluggable source, one shared mixer

```
                         ┌─────────────────────────────┐
  Voice emotion  ──────► │  Mood→Prompt mapper          │  (reuses emo_vector/emo_text)
  (emo_vector)           └──────────────┬──────────────┘
                                        ▼
                          ┌──────────── Music SOURCE (pick one) ───────────┐
                          │ Tier 0: library folder  (0 compute)            │
                          │ Tier 1: local light model (CPU-tolerable)      │
                          │ Tier 2: hosted API      (0 local compute)      │
                          └──────────────┬─────────────────────────────────┘
                                         ▼  raw music wav
  TTS voice wav ────────►  ┌─────────────────────────────┐
                           │  MIX ENGINE (pure DSP, CPU)  │
                           │  • duration-fit (loop/trim)  │
                           │  • sidechain ducking         │
                           │  • level/LUFS match          │
                           │  • fade in/out               │
                           └──────────────┬──────────────┘
                                          ▼
                                    final mixed wav
```

The **MIX ENGINE is identical for all tiers.** Only the "raw music source" swaps. Build the mixer once;
sources are plugins.

---

## 2. Music source — DECISION: local GPU model (mid-size), Lightning GPU

User runs on a Lightning GPU and WANTS to use it — just not a "very huge" model. So the DEFAULT source
is a **local GPU model** sitting alongside IndexTTS2, not the zero-compute path.

### Model choice (sizing for a ~16-24GB Lightning GPU with IndexTTS2 already resident)

| Model | Params | ~VRAM | Verdict |
|-------|--------|-------|---------|
| **Stable Audio Open Small** ← PRIMARY | ~340M | ~3-4GB | music **+ SFX in one model** — covers both asks |
| **MusicGen-small** ← FALLBACK | 300M | ~2-3GB | most battle-tested, ~10-line integration |
| MusicGen-medium | 1.5B | ~6-8GB | one step up if small musicality too thin — still fine |
| MusicGen-large | 3.3B | ~12-16GB | "huge" by user's bar — SKIP |
| 8B+ diffusion | — | 24GB+ | rejected |

**Primary = Stable Audio Open Small** because it does music AND sound-effects (the "afx" ask) in ONE
model → no second download for SFX. Built for fast/low-VRAM generation → short beds in seconds on GPU.
**Fallback = MusicGen-small** if license/deps misbehave on Lightning.
Escalation ceiling = MusicGen-medium (1.5B) if musicality is too thin. Never larger.

> ACTION before install: confirm live (web search was down at plan time) current VRAM/latency + license
> for Stable Audio Open Small as of 2026. MusicGen-small/medium numbers are stable fallback.

Still keep source as a swappable plugin (library folder / API) for flexibility, but the model runs on GPU
by default now.

**Build order: local GPU model first (primary), library + API as optional plugins.**
- Local GPU model = default (user wants to use the Lightning GPU).
- Library folder = zero-compute fallback / offline safety.
- API = optional, for when the box is busy.

### VRAM budgeting note
IndexTTS2 is already resident on the Lightning GPU. Load the music model lazily (only when
`enable_bg_music` is ON) and offload/`del`+`torch.cuda.empty_cache()` after generation if VRAM is tight,
so the two models don't need to co-reside at peak. Small model (~3-4GB) likely co-resides fine anyway.

---

## 3. Mood→prompt mapping (what makes it "match the voice")

Reuse the 8-emotion vector already in the codebase:
`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`

Map dominant emotion(s) → a music prompt template, e.g.:
- sad / melancholic → `"slow melancholic ambient piano, soft strings, low energy, minor key"`
- happy → `"uplifting warm acoustic, gentle percussion, major key, moderate tempo"`
- calm → `"peaceful ambient pad, soft, minimal, spacious"`
- angry / afraid → `"tense low drone, dark cinematic underscore"`

For Tier 0 (library), the same map picks a **mood folder** instead of writing a prompt.
Optional manual override box so the user can type their own prompt/genre.

---

## 4. DSP mix engine (the real work, all CPU)

Implement in a new module `audio_bed.py`, using `pydub` + `numpy` (add `pydub`, `soundfile`; ffmpeg
likely already present):

1. **duration-fit(music, target_len)** — loop with crossfade if too short, trim + fade if too long.
2. **sidechain_duck(voice, music, threshold, duck_db=-12, attack, release)** — envelope-follow the voice,
   attenuate music where speech is present. The professional-sounding core.
3. **level_match(voice, music, music_lufs=-23 or user gain)** — normalize bed relative to voice.
4. **mix(voice, ducked_music)** — sum, peak-guard against clipping, final fade in/out.

All deterministic, all fast on CPU. No model needed for this layer.

---

## 5. UI additions in `webui_enhanced.py` (wired like the diffusion_steps slider)

New accordion **"🎵 Background Music / SFX"** in both Single and SRT tabs:
- `enable_bg_music` (Checkbox, default OFF — fully backward compatible)
- `music_source` (Dropdown: Library / Local model / API)
- `music_mood` (Dropdown: Auto-from-emotion / manual list) + optional `music_prompt` textbox
- `music_volume` (Slider, e.g. -30..0 dB, default -18)
- `ducking_amount` (Slider, 0..-24 dB, default -12)
- `enable_sfx` (Checkbox) + SFX source (library / AudioGen)

Threading: same pattern already used — add components to the `advanced_params` list (appended LAST to
stay position-safe), extend `gen_single` / `gen_srt` unpacking, do the mix as a POST-step after the voice
wav is produced (music never touches the TTS inference path → zero risk to the consistency/quality work
already done).

---

## 6. Scope decisions (defaults chosen; change anytime)

- **Music source:** Tiered/pluggable (Tier 0 + Tier 2 first, Tier 1 optional). ← default
- **Scope:** music bed + SFX + auto-mood-from-emotion. ← default
- SFX (AFX) is the most complex to time/place; propose shipping **music bed first**, SFX as phase 2.

---

## 7. Proposed build order

1. `audio_bed.py` — DSP mix + duck engine (Tier 0 library source). *Testable immediately, no model.*
2. UI accordion + wiring in `webui_enhanced.py` (music bed only, Single tab).
3. Extend to SRT tab (per-segment or whole-track bed).
4. Mood→prompt/library mapping from `emo_vector`.
5. Tier 2 (API) source plugin.
6. Tier 1 (local small model) source plugin — after verifying best current model.
7. SFX (AFX) layer — phase 2.

---

## 8. Dependencies (all light)
- `pydub`, `soundfile`, `numpy` (have numpy) — mixing. ffmpeg for pydub (usually present).
- Tier 1 only: `audiocraft` (MusicGen) OR `stable-audio-tools` — added behind a flag, not required.
- Tier 2 only: `requests` + an API key.

No new heavy GPU dependency for the default (Tier 0 + Tier 2) path.
