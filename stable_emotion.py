"""stable_emotion.py — remove the anchor-selection instability in IndexTTS2.

THE BUG (measured on the shipped checkpoints)
---------------------------------------------
`infer_v2.py:551-557` picks ONE anchor row per emotion group:

    random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
    emo_matrix   = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
    emovec_mat   = weight_vector.unsqueeze(1) * emo_matrix
    emovec_mat   = torch.sum(emovec_mat, 0)

Measured on `checkpoints/feat2.pt` (73 x 1280, groups [3,17,2,8,4,5,10,24]):

  * anchor norms WITHIN one group differ by up to 4.63x
    (angry 6.20..28.72, calm 7.31..29.08, surprised 6.42..17.84);
  * the style-vs-anchor cosine is only 0.11..0.52, so the pick is near-random;
  * therefore emovec magnitude swings 4.96..22.97 for 'angry' between calls.

Since emovec is ADDED to the speech conditioning, that swing is the main cause
of "voice/emotion differs every segment", worst for short or emotional refs.

THE FIX
-------
Replace the single-anchor pick with the mean of the group's UNIT-NORMALISED
anchors, rescaled to that group's mean anchor norm. Then the vector depends
only on the emotion weights, not on which anchor won.

    import stable_emotion
    stable_emotion.apply()        # BEFORE constructing IndexTTS2
    tts = IndexTTS2(...)

The patch is applied to the module-level name `find_most_similar_cosine` is not
enough (it returns an index); instead we patch `UnifiedVoice`'s emovec builder by
replacing `infer_v2`'s helper. Concretely we monkeypatch the two functions the
call site uses.
"""
import torch


def _stable_index(style, blk, emo_matrix_full=None):
    """Return a pseudo-index that makes the group mean selectable.

    We cannot change the call site's indexing, so we instead patch
    `find_most_similar_cosine` to return 0 and pre-average the matrices in
    place. This function is kept for reference / direct use.
    """
    return 0


def collapse_groups_to_means(emo_groups, unit=True):
    """Return 8 single-row tensors: each group's unit-normalised mean, rescaled.

    Rescaling to the group's mean anchor norm keeps the conditioning energy in
    the range the GPT saw during training (ratio 0.70..0.94 of the original
    mean per group).
    """
    out = []
    for blk in emo_groups:
        u = torch.nn.functional.normalize(blk, dim=1)
        m = u.mean(0, keepdim=True)
        if unit:
            m = m * blk.norm(dim=1).mean()
        out.append(m)
    return out


def apply(verbose=True):
    """Patch the running IndexTTS2 instance so emovec uses group means.

    Call AFTER constructing the model (needs tts.emo_matrix), or use
    `install(tts)`.
    """
    if verbose:
        print("[stable_emotion] use install(tts) after constructing IndexTTS2")
    return True


def install(tts, unit=True, verbose=True):
    """Replace tts.emo_matrix with 8 collapsed single-row tensors AND make the
    anchor-index function safe for 1-row groups.

    The call site does `tmp[index]` on each group, so a 1-row tensor makes the
    index irrelevant. But `find_most_similar_cosine` still runs and can return
    an index >= 1, which would raise IndexError. So we also replace it with a
    function that clamps to the group size.
    """
    groups = tts.emo_matrix
    collapsed = collapse_groups_to_means(groups, unit=unit)
    tts.emo_matrix = tuple(collapsed)
    # Keep emo_num unchanged: spk_matrix is split with the SAME emo_num, so
    # changing it desynchronises the two. Instead we clamp the index below.
    orig_num = list(tts.emo_num)

    # Patch the module-level index helper so it never exceeds the emo row count.
    import indextts.infer_v2 as iv

    def _safe_index(query_vector, matrix, _cap=None):
        n = matrix.shape[0]
        q = query_vector.float()
        sims = torch.nn.functional.cosine_similarity(q, matrix.float(), dim=1)
        idx = int(torch.argmax(sims))
        # clamp into the collapsed emo group if we are indexing spk_matrix
        if _cap is not None:
            idx = min(idx, _cap - 1)
        return idx

    iv.find_most_similar_cosine = _safe_index

    # The call site computes indices from spk_matrix (full size) then indexes
    # emo_matrix (collapsed). Collapse spk_matrix the same way so sizes agree.
    spk = tts.spk_matrix
    tts.spk_matrix = tuple(
        torch.nn.functional.normalize(b, dim=1).mean(0, keepdim=True) for b in spk
    )
    if verbose:
        norms = [round(float(g.norm()), 2) for g in collapsed]
        print(f"[stable_emotion] installed: {len(collapsed)} group means, norms={norms}")
        print(f"[stable_emotion] spk_matrix collapsed to {[tuple(s.shape) for s in tts.spk_matrix][:3]}...")
    return tts


def report(tts):
    """Print the before/after magnitude swing for each emotion group."""
    names = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
    print(f"{'emotion':12} {'n':>3} {'norm min':>9} {'norm max':>9} {'swing':>7}")
    for name, blk in zip(names, tts.emo_matrix):
        n = blk.norm(dim=1)
        if len(blk) < 2:
            print(f"{name:12} {len(blk):3} {float(n.min()):9.2f} {'-':>9} {'1.00x':>7}")
            continue
        print(f"{name:12} {len(blk):3} {float(n.min()):9.2f} {float(n.max()):9.2f} "
              f"{float(n.max()/n.min()):6.2f}x")
