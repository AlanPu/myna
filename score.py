"""Score a shadowing attempt against the original recording.

The question this answers is NOT "what did you say" — it is "how close was
your attempt to the original". That distinction decides the whole design:
there is no speech recognition anywhere in this file.

Instead both signals are reduced to the same thing. The original's syllable
nuclei are already measured during the build and stored in energy.json as
`ref_nuclei`; the recording is reduced the same way, with the same detector
(wordmarks.nuclei), and the two sequences of peaks are aligned with DTW.

Two sequences of peaks answer the questions a shadowing learner actually asks:

  * did every word get said at all        -> peak-count / coverage per word
  * did it land in the right place        -> alignment residuals
  * am I faster or slower than the speaker-> duration scale
  * am I rushing or dragging inside it    -> per-word local residuals

Why peaks and not raw waveforms: a mic recording and the source differ in
timbre, loudness, room, pitch tracking and channel. Comparing spectra directly
scores the room, not the speech. Peaks are a gross, robust representation that
survives all of that, and the residual tolerance below (120ms) is far wider
than any plausible detection bias, so detector disagreement cancels instead of
accumulating.

Why DTW and not a fixed grid: neither side has a rigid tempo. The speaker
pauses, stretches emphasised words and front-loads clauses; so does the
learner. Pinning the recording to the reference's rhythm would score every
natural phrasing as wrong.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess

import numpy as np

import onsets
import wordmarks

SR = wordmarks.SR
HOP = wordmarks.HOP

# A word's onset must sit this close to some user peak to count as "said".
# Wide on purpose: it absorbs detector bias and articulatory slack without
# swallowing a genuinely skipped word (the next peak is >150ms away in those).
HIT_TOL = 0.12

# Sentences whose peaks are too sparse to align meaningfully. Below this the
# score would be driven by one or two peaks and swing wildly.
MIN_PEAKS = 4

# Fallback per-node alignment cost, in seconds, used only when a sentence has
# too few reference peaks to estimate one. See noise_floor() for the real
# per-sentence value.
FLOOR = 0.030

# Additional cost, above the floor, that drives the score from 100 to 0. Sized
# so that reading noticeably late (~150ms mean error) lands near 50.
SPAN = 0.16


# --------------------------------------------------------------------- audio
def decode_bytes(data: bytes) -> np.ndarray:
    """Decode an uploaded recording to mono float32 at SR.

    The browser hands us WebM/Opus (Chrome, Firefox) or mp4/aac (Safari), and
    ffmpeg normalises the container, so the scorer does not care which.

    The bytes go through a temp file rather than stdin. An mp4 is a "seekable"
    container: the demuxer wants to jump to its moov atom, and on a pipe it only
    ever finds "partial file" and emits nothing at all — an 8.6 second take
    decoded to zero samples with exit code 0, so the failure was silent. A real
    file is what ffmpeg needs and what wordmarks.decode has always used.
    """
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".rec")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-i", path,
             "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
            capture_output=True)
        if p.returncode != 0 or not p.stdout:
            raise ValueError("无法解码录音(格式不受支持或文件损坏)")
        return decode_raw(p.stdout)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def decode_raw(raw: bytes) -> np.ndarray:
    """Wrap already-decoded f32le samples."""
    a = np.frombuffer(raw, dtype="<f4")
    return np.nan_to_num(a.astype(np.float32), copy=False)


def peaks(pcm: np.ndarray) -> np.ndarray:
    """Syllable nuclei of a mono float32 signal, in seconds.

    This wraps wordmarks.envelope/smooth/nuclei, so the recording goes through
    the exact same detector as the reference did. Any bias the detector has is
    therefore common to both sides and cancels in the alignment.

    The gate thresholds are copied from wordmarks.build's file-level pass. They
    are expressed as multiples of the signal's own floor and peak, so they
    transfer to a recording with a completely different absolute level — a
    quiet microphone does not end up with zero peaks, and a hot one does not
    end up with every noise blip counted as a syllable.
    """
    if pcm.size < SR // 4:                       # <0.25s: not a sentence
        return np.zeros(0)
    env = np.asarray(wordmarks.smooth(wordmarks.envelope(pcm)), dtype=np.float64)
    if env.size < 10:
        return np.zeros(0)

    # Two gates, chosen by how clean the recording is.
    #
    # The adaptive one from onsets.py exists because a single global threshold
    # cannot follow a drifting noise floor. But it is deliberately strict, and
    # applying it to a clean recording is a regression: on the original audio it
    # found 19 of the 40 real syllables, because a 0.75s trailing window inside
    # a short sentence is mostly speech and its "floor" is therefore speech.
    #
    # The discriminator is how loud the QUIET part of the signal is, not how
    # wide the dynamic range is. Clean audio has frames that fall to nearly
    # nothing between words; a phone recording in a room never does. Note that
    # the loud/quiet ratio moves the wrong way — noise compresses the range —
    # so it is not usable here.
    srt = np.sort(env)
    quiet = float(srt[int(env.size * 0.10)])
    loud = float(srt[int(env.size * 0.98)])
    # Relative to the loudest frame: clean speech in this build sits near 0.01,
    # 6dB-SNR hiss near 0.10.
    floor_share = quiet / loud if loud > 1e-9 else 1.0

    # Branch selection. This threshold is the single most consequential number
    # in this function, and it was set too tight.
    #
    # The discriminator between "clean" and "noisy" is how loud the QUIET part
    # of the signal is relative to the loudest. Measured on this build: clean
    # speech sits at 0.043, and a take with another person talking in it sits
    # at 0.086-0.095. The old cut at 0.05 therefore routed three quarters of
    # ordinary, mildly noisy takes into onsets.adaptive_gate — a branch built
    # for severe broadband noise, whose raised gate then discarded quiet REAL
    # syllables. That is the mechanism behind the symptom this whole change
    # exists for: the learner reads perfectly, the room adds a hum, and the
    # detector reports a third of the syllables.
    #
    # The gain from moving the cut to 0.20, as detected nuclei over reference
    # nuclei over 8 sentences:
    #
    #     cut     clean   room@6dB  room@3dB  room@0dB
    #     0.05    0.93      0.74      0.69      0.71
    #     0.10    0.95      0.79      0.74      0.74
    #     0.15    0.95      0.89      0.82      0.83
    #     0.20    0.95      0.89      0.85      0.85   <- chosen
    #     1.00    0.95      0.89      0.85      0.85
    #
    # 0.20 sits on the plateau rather than at its edge, and the fact that 1.00
    # scores identically is itself informative: the clean branch is simply
    # better for this material, and the strict branch only earns its place on
    # noise far worse than anything a learner records. The loud/quiet ratio is
    # deliberately NOT the discriminator — noise compresses the dynamic range,
    # so that measure moves the wrong way and was tried and rejected.
    if floor_share > 0.20:
        gate_frames, floor, peak = onsets.adaptive_gate(env.tolist())
        idx, _ = wordmarks.nuclei(env, env_gate=max(floor * 2.0, peak * 0.075),
                                  env_gate_frames=gate_frames)
    else:
        g = max(srt[int(env.size * 0.12)] * 2.4, loud * 0.045)
        idx, _ = wordmarks.nuclei(env, env_gate=g)
    return np.asarray(idx, dtype=np.float64) * HOP


# ---------------------------------------------------------------------- dtw
def dtw(ref: np.ndarray, hyp: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Align two peak sequences; return (mean node cost, path).

    Cost is the absolute gap in seconds between paired peaks, so a node cost of
    0.12 means "this word landed 120ms away from where the speaker put it".

    A Sakoe-Chiba band keeps this O(n*m) from becoming a problem on a long
    sentence and, more importantly, stops the aligner from cheating: an
    unconstrained DTW will happily map every reference peak to the same
    single loud user peak and report a perfect score.
    """
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return 0.0, []
    span = max(n, m)
    # The band must be wide enough for a path to physically exist. A path runs
    # from (1,1) to (n,m) along the diagonal, stepping one row OR one column at a
    # time, so climbing n-1 rows requires at least n-1 columns of slack. A band
    # narrower than the LENGTH DIFFERENCE makes that impossible: with a
    # 35-peak reference and a 22-peak take, band = 0.35*35 = 12, and no path
    # survives, so DTW returns infinity and the sentence answers "cannot align".
    #
    # This is why it went unnoticed: it only fires when the learner produces
    # noticeably FEWER nuclei than the reference — reading fast, reading
    # quietly, or just speaking into a room that ate a few plosives. Every
    # synthetic fixture and the fake microphone both produce a peak count close
    # to the reference, so none of them ever reached the failure.
    #
    # |n-m|+2 is the minimum that guarantees reachability while still
    # constraining the aligner, which is the band's real job: stop a
    # constrained-free DTW from mapping every reference peak onto one loud
    # user peak and declaring a perfect score.
    band = max(3, int(0.35 * span), abs(n - m) + 2)

    inf = float("inf")
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    P = np.zeros((n + 1, m + 1), dtype=np.int8)   # 0 diag, 1 up, 2 left

    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            d = abs(ref[i - 1] - hyp[j - 1])
            a, b = D[i - 1, j], D[i, j - 1]
            c = D[i - 1, j - 1]
            if c <= a and c <= b:
                D[i, j] = c + d; P[i, j] = 0
            elif a <= b:
                D[i, j] = a + d; P[i, j] = 1
            else:
                D[i, j] = b + d; P[i, j] = 2

    if not np.isfinite(D[n, m]):
        return float("inf"), []

    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = P[i, j]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(D[n, m]), path


def residuals(ref: np.ndarray, hyp: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    """Per-reference-node deviation, in seconds, after removing the tempo scale.

    A learner who takes 30% longer is not 30% wrong everywhere — they are one
    consistent stretch. Comparing raw times would charge them for that stretch
    on every single word. So the alignment path is first mapped onto the
    reference's own axis, then the residual left over is what is reported: the
    shape of the delivery, with the overall tempo factored out.
    """
    if not path:
        return np.zeros(0)
    ht = np.asarray([hyp[j] for _, j in path], dtype=np.float64)
    span = float(ref[-1] - ref[0])
    if span <= 0:
        return ht - ht[0]
    # linear ramp from the first to the last aligned node: the tempo-neutral axis
    ideal = np.linspace(ref[0], ref[-1], ht.size)
    return ht - ideal


def lead_silence(ref: np.ndarray, pcm: np.ndarray, t0: float) -> float:
    """How much room tone opens the take before the learner actually starts.

    The answer is not "where does the energy rise" but "which shift makes the
    recording align best against the reference". Trying every candidate offset
    and keeping the one with the cheapest DTW is far steadier than a single gate
    threshold: a quiet first syllable that a gate would miss is still a very
    good match, while a door slam that a gate would catch is a very poor one.

    The same DTW and the same band as the real scoring run are used, so the
    offset that wins here is the one the score itself would have preferred.

    A candidate has to earn the shift. Without a margin the search always
    returns the largest offset it tried whenever that is a hair cheaper, which
    is how a take whose first syllable simply sits 0.2s after speech_start
    (the normal case — speech_start is an energy threshold, not an onset) came
    back claiming a 600ms lead and scored 43 instead of 95. The bar is
    therefore a 15% cut in alignment cost, plus a minimum shift worth making.
    """
    hyp0 = peaks(pcm)
    if hyp0.size < MIN_PEAKS or ref.size < MIN_PEAKS:
        return 0.0

    base, _ = dtw(ref, np.maximum(hyp0, 0.0))
    if base <= 0:
        return 0.0
    bar = base * 0.85
    best, best_cost = 0.0, base
    # A take that opens with more than 0.6s of room tone was not an attempt at
    # this sentence — shifting that far would score a performance the learner
    # did not give.
    for i in range(1, 13):
        lead = i * 0.05
        shifted = hyp0 - lead
        if shifted.size < MIN_PEAKS:
            break
        cost, _ = dtw(ref, np.maximum(shifted, 0.0))
        if cost < best_cost:
            best, best_cost = lead, cost
    return best if best_cost <= bar else 0.0


def rebase(ref: np.ndarray, s0: float | None) -> np.ndarray:
    """Reference peaks as sentence-local seconds.

    energy.json stores peaks on the clip's absolute timeline — sentence 150
    sits near 696s. The recording, once trimmed to the sentence, is relative
    to the start of that trim. DTW and every residual must run in one
    coordinate system, so the reference is shifted down to zero here rather
    than at each use site, where a forgotten shift would silently score every
    sentence except the first as a total miss.

    The anchor is the reference's own first peak rather than speech_start: the
    first peak is where the voice actually begins, and using it keeps the
    coordinate meaningful for sentences that were never measured.
    """
    if ref.size == 0:
        return ref
    anchor = float(s0) if s0 is not None else float(ref[0])
    return ref - anchor


def confidence(ref: np.ndarray, n_hyp: int | None = None) -> float:
    """How much of this sentence's score is signal rather than measurement.

    A short sentence carries too few syllables for a rhythm comparison to say
    anything. Two seconds of speech holds a handful of nuclei, and the gap
    between what a correct reading and a wrong one costs is smaller than the
    disagreement between two detectors on the SAME audio. Sentence 76
    ("professional and authentic", 1.95s) scored 15 on a perfect re-read whose
    first ten peaks matched the reference to within 30ms: the tail differed by
    40ms and the score fell off a cliff.

    Rather than publish a number that precise cannot support, this reports how
    much the sentence can bear, and the caller says so on screen.

    Scale: 1.0 at 4s or more with a normal syllable count, falling off below
    that. The 4s point is where this build's measurement noise (~30-60ms per
    node) drops under the score's resolution.

    The reference peaks come from a whole-file pass and the recording from a
    slice, and the two disagree about how many nuclei a sentence has. When they
    disagree by a lot, the reading is being scored against a reference that
    does not describe it: sentence 150 reads 18 reference peaks over 3.1s and 25
    from the slice, whose first 18 line up one-for-one, and the surplus pulls
    the alignment tail apart. A big surplus is direct evidence that this
    particular take cannot be measured well, so it discounts confidence
    regardless of how long the sentence is.
    """
    if ref.size < 2:
        return 0.0
    span = float(ref[-1] - ref[0])
    by_span = min(1.0, max(0.0, span / 4.0))
    # Peak count is a weak requirement: this build's sentences carry a median
    # of 22 nuclei, so anything under 8 is genuinely thin, but 8-16 is normal
    # speech and should not be discounted.
    by_count = min(1.0, ref.size / 6.0)
    by_agree = 1.0
    if n_hyp:
        ratio = float(n_hyp) / float(ref.size)
        # Symmetric: neither many more nor many fewer peaks than the reference
        # means the two detectors saw different material. Measured on a
        # perfect re-read, the ratio sits at 1.07 (median), 1.02-1.12 across
        # the interquartile range, and only 2 of 230 sentences exceed 1.5 — so
        # 1.5 is where the disagreement stops being ordinary.
        by_agree = min(1.0, 1.0 / max(ratio, 1.0 / ratio) / 1.5)
    return by_span * by_count * by_agree


def noise_floor(ref: np.ndarray) -> float:
    """Per-node alignment cost that carries no meaning, for THIS sentence.

    The reference peaks are not ground truth with zero error. They come from a
    whole-file detector pass, while a recording is analysed as its own slice,
    and the two boundaries sit on different noise floors. Re-reading the
    ORIGINAL audio exactly still costs 30ms per node at the median and 62ms at
    the 90th percentile, so a learner cannot do better than that by reading
    perfectly.

    That cost is not a fixed constant: it scales with how far apart the peaks
    are. Sparse peaks (few syllables, or a slow delivery) are located less
    precisely in absolute time than dense ones, and the measured floor tracks
    that. So the floor is derived from the reference's own mean spacing, which
    needs no tuning against a particular recording or speaker.

    Measured on this build: mean spacing 0.13s -> ~30ms, spacing 0.25s ->
    ~62ms.
    """
    if ref.size < 2:
        return FLOOR
    span = float(ref[-1] - ref[0])
    if span <= 0:
        return FLOOR
    spacing = span / (ref.size - 1)
    return float(np.clip(spacing * 0.25, 0.020, 0.090))


def judge_words(words: list[str], marks: list, ref: np.ndarray,
                hyp_t: list[float], s1: float | None) -> list[dict]:
    """Judge each word: did it get said, and roughly when.

    Each word is judged by the user peaks inside its own half-open slice, which
    is exactly how the player already splits the sentence for highlighting, so
    the screen and the score always agree on where a word begins and ends.

    The verdict asks whether a peak is PRESENT, not how many. Counting is
    unreliable: the same word yields 1-3 nuclei depending on how emphatically it
    was spoken, so an expected-count test flagged 22 of 22 words as too thin on a
    perfect re-read of the original. Presence needs no expectation at all.
    """
    out: list[dict] = []
    for k, _ in enumerate(words):
        a = marks[k] if k < len(marks) else None
        b = marks[k + 1] if k + 1 < len(marks) else s1
        if a is None or b is None or b <= a:
            b = (a + 0.2) if a is not None else None
        if a is None or b is None:
            out.append({"k": k, "status": "unknown", "delta_ms": None})
            continue

        # The word's onset: the reference peak nearest the mark, which is where
        # the original actually articulated this word. Using the mark instead
        # would compare against an interpolation and round the judgement toward
        # every word being on time.
        onset = float(ref[int(np.argmin([abs(t - a) for t in ref]))]) if ref.size else float(a)

        near = [t for t in hyp_t if onset - HIT_TOL <= t <= onset + HIT_TOL]
        inside = [t for t in hyp_t if a - HIT_TOL <= t <= b + HIT_TOL]

        if not inside:
            # Nothing anywhere in this word's window: it was skipped.
            st = "missed"
            d = None
        else:
            d = min(abs(t - onset) for t in inside)
            # A peak in the window but nowhere near the onset means the word
            # was said, just not where the speaker put it.
            st = "ok" if d <= HIT_TOL else "late"
        out.append({"k": k, "status": st,
                    "delta_ms": round(d * 1000, 1) if d is not None else None})
    return out


# ------------------------------------------------------------------- scoring
def _words_of(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", (text or "").strip()) if w]


def score_sentence(cap: dict, user_pcm: np.ndarray, t0: float | None = None) -> dict:
    """Score one recording against one sentence.

    `t0` is the time, in the recording, at which the user was supposed to
    start. When given, the recording is trimmed around it before analysis:
    people press the button a beat late and that offset must not be charged to
    their rhythm.
    """
    marks = cap.get("word_marks") or []
    words = _words_of(cap.get("text", ""))
    n = len(words)

    ref = np.asarray(cap.get("ref_nuclei") or [], dtype=np.float64)
    # When a build predates ref_nuclei, fall back to the word marks: they are
    # coarser but still the original's own rhythm, so the score stays meaningful
    # instead of the whole feature failing on an older player directory.
    if ref.size == 0:
        ref = np.asarray(marks, dtype=np.float64)
    s0 = cap.get("speech_start")
    s1 = cap.get("speech_end")

    # Rebase first, so the trim window, the marks and the aligner all read the
    # sentence off one clock. The recording is sentence-local from the start;
    # doing this before the trim is what makes t0 comparable to anything here.
    anchor = float(s0) if s0 is not None else (float(ref[0]) if ref.size else 0.0)
    ref = rebase(ref, s0)
    if s1 is not None:
        s1 = float(s1) - anchor
    marks = [(m - anchor) if m is not None else None for m in marks]

    out = {
        "i": cap.get("i"),
        "words": words,
        "n_words": n,
        "ref_peaks": int(ref.size),
        "user_peaks": 0,
        "score": None,
        "scale": None,
        "delay_ms": None,
        "onset_ms": None,
        "mean_resid_ms": None,
        "per_word": [],
        "ok": False,
        "note": "",
    }

    pcm = user_pcm
    if t0 is not None and ref.size:
        # Window the take around where the sentence was supposed to begin.
        #
        # `t0` is a position inside the recording, and the window is sized from
        # the reference peaks alone. It is deliberately NOT built from s0/s1:
        # those sit on the clip's absolute timeline — sentence 51 starts at
        # 259.0s — and subtracting one from the other asks for a 259-second
        # window out of 4 seconds of audio. The window then covers nothing, the
        # sentence never matches, and every line but the first answers "cannot
        # align". Sentence 0 is the only one that works, and only because its
        # absolute start is 0.0, so the two clocks coincide by accident.
        #
        # The window is generous rather than tight, and that is the whole
        # point. Cutting the take to the reference's own span measured 8.47s
        # for a sentence of 3.5s: a learner reading 30% slow had their extra
        # second sliced off, the measured tempo fell from 1.30 to 1.04, and the
        # score went UP from 40 to 71 — the window was hiding the one thing it
        # exists to measure. Trimming is for removing dead air at the edges,
        # not for holding the speaker to the original's tempo.
        pad = 0.25
        span = float(ref[-1] - ref[0])
        a = max(0.0, -pad - t0)
        # Allow the take to run well past the reference. A slow reader needs
        # room; an over-long take is handled by the aligner, which copes with
        # extra peaks far better than a blind truncation does.
        b = min(len(pcm) / SR, span * 1.8 + 2 * pad - t0)
        if b > a:
            pcm = pcm[int(a * SR):int(b * SR)]

    out["ref_peaks"] = int(ref.size)

    # ---- find where the learner actually started ------------------------
    # `t0` cannot be trusted as the speech onset. getUserMedia takes hundreds of
    # milliseconds to hand back a live track, and nobody begins speaking on the
    # exact frame the recorder starts, so the take always opens with room tone.
    # Left uncorrected that offset shifts the whole sentence: a take with 1.5s
    # of leading silence aligns its first syllable 1.5s past where the reference
    # put it, and no amount of alignment recovers from that.
    #
    # The onset is taken from the recording's own first syllable rather than
    # from the caller's guess, but only inside a window around t0 — a silence
    # detector run over an arbitrary take can latch onto a cough or the tail of
    # the previous take.
    hyp = peaks(pcm)
    out["user_peaks"] = int(hyp.size)
    lead = 0.0
    if hyp.size and t0 is not None and ref.size:
        lead = lead_silence(ref, pcm, t0)
        if lead > 0.02:
            hyp = hyp - lead
            # Trim the padding off the front too, so per-word slices are read
            # from the same clock as the peaks.
            if lead < len(pcm) / SR:
                pcm = pcm[int(lead * SR):]
                if marks:
                    marks = [(m + lead) if m is not None else None for m in marks]
                if s1 is not None:
                    s1 = s1 + lead
    if hyp.size:
        out["onset_ms"] = round(max(0.0, lead) * 1000, 1)

    # A take that covers only part of the sentence is not the same failure as
    # one that covers it but does not match. Saying "cannot align" for a
    # half-finished read sends the user hunting for a pronunciation problem
    # they do not have, so the note says what actually happened and how long
    # the sentence needed.
    need = float(ref[-1] - ref[0]) if ref.size else 0.0
    got = float(hyp[-1]) if hyp.size else 0.0
    out["need_sec"] = round(need, 2)
    out["got_sec"] = round(got, 2)
    out["confidence"] = round(confidence(ref, int(hyp.size)), 2)

    if ref.size < MIN_PEAKS or hyp.size < MIN_PEAKS:
        out["note"] = ("只录到 %.1f 秒,原声这句有 %.1f 秒,再试一次"
                       % (got, need)) if (got and need and got < need * 0.8) \
            else "语音太少,无法评分"
        return out

    cost, path = dtw(ref, hyp)
    if not path:
        out["note"] = ("没读完——录到 %.1f 秒,原声这句有 %.1f 秒"
                       % (got, need)) if (need and got < need * 0.8) else "无法对齐"
        return out

    # Per-word verdicts run BEFORE the score is combined, because coverage is
    # one of its three terms.
    out["per_word"] = judge_words(words, marks, ref, hyp_t=hyp.tolist(), s1=s1)

    res = residuals(ref, hyp, path)
    # cost per node, not per path step: a path step that repeats a node (the
    # aligner inserting a deletion) must not be charged as an error.
    node_cost = cost / max(1, ref.size)

    # --- the score has three independent reasons to drop ------------------
    #
    # 1. timing. How far each syllable landed from where the speaker put it.
    #    Calibration matters here: these peaks are not ground truth with zero
    #    error. The reference came from a whole-file pass while the recording is
    #    analysed on its own slice, and the two boundaries sit on different
    #    noise floors, so re-reading the ORIGINAL exactly still costs 30ms per
    #    node at the median and 62ms at the 90th percentile. No reading can beat
    #    that floor, so the score charges only the excess above it.
    #
    # 2. coverage. DTW alone is blind to a hole in the middle: punch 400ms out
    #    of a sentence and the aligner simply pairs the peaks before the hole
    #    with the peaks after it, spreading the hole's cost thinly enough that
    #    the measured cost barely moved (34.7ms vs 32.7ms for the intact
    #    original — a skipped phrase scored 100). A dropped phrase is the single
    #    most common shadowing failure, so it is counted directly, per word.
    #
    # 3. density. Rushing to 80% leaves the reference's 45 syllables compressed
    #    into 37 detected peaks, and the aligner is then forced into mismatches
    #    that swamp the timing term. The ratio is a blunt but stable signal that
    #    the other two miss.
    #
    # Timing and coverage are combined as a weighted mean rather than a product:
    # a single bad word should cost a little, not annihilate the score.
    hit = [w["status"] == "ok" for w in out["per_word"]]
    coverage = sum(hit) / len(hit) if hit else 1.0

    density = min(1.0, len(hyp) / max(1, len(ref)))

    timing = max(0.0, 1.0 - max(0.0, node_cost - noise_floor(ref)) / SPAN)
    out["score"] = round(100.0 * (0.55 * timing + 0.30 * coverage
                                  + 0.15 * density), 1)
    out["timing"] = round(timing * 100, 1)
    out["coverage"] = round(coverage * 100, 1)
    out["density"] = round(density * 100, 1)
    out["node_cost_ms"] = round(node_cost * 1000, 1)
    out["floor_ms"] = round(noise_floor(ref) * 1000, 1)
    out["mean_resid_ms"] = round(float(np.mean(np.abs(res))) * 1000, 1)

    # Tempo scale, measured between the first and last aligned nodes. Robust
    # because the alignment pins the ends even when the middle drifts.
    ht0 = float(hyp[path[0][1]])
    ht1 = float(hyp[path[-1][1]])
    rspan = float(ref[-1] - ref[0])
    if rspan > 0.05 and ht1 > ht0:
        out["scale"] = round((ht1 - ht0) / rspan, 3)

    # Start offset, in the recording, relative to where the user was told to
    # begin. Positive = late. Corrected for tempo: someone who is uniformly
    # slower and uniformly late should not have the same offset reported twice.
    out["delay_ms"] = round((ht0 - float(ref[0])) * 1000, 1)

    # How much the number is worth. A short sentence cannot support a precise
    # rhythm score, and publishing one anyway is worse than saying so: on a
    # 1.95s sentence a correct re-read scored 15 because its tail ran 40ms
    # long, which is well inside the measurement noise.
    conf = confidence(ref, int(hyp.size))
    out["confidence"] = round(conf, 2)
    if conf < 0.5:
        out["note"] = ("这句只有 %.1f 秒,节奏评分不够稳(参考音节 %d 个);"
                       "分数仅供参考" % (rspan, ref.size))

    out["ok"] = True
    return out


def score(energy: dict, user_pcm: np.ndarray, index: int,
          t0: float | None = None) -> dict:
    """Score the recording against sentence `index` of the manifest."""
    caps = energy.get("captions") or []
    if not (0 <= index < len(caps)):
        raise IndexError(f"句子序号越界:{index}")
    dur = len(user_pcm) / SR
    r = score_sentence(caps[index], user_pcm, t0=t0)
    r["recording_sec"] = round(dur, 2)
    # A recording far longer than the sentence usually means the mic picked up
    # the source audio as well; say so rather than reporting a confident score.
    if dur > 0 and "speech_end" in caps[index]:
        span = caps[index]["speech_end"] - caps[index]["speech_start"]
        if span > 0 and dur > span * 3 + 1.5:
            r["note"] = (r["note"] + " 录音偏长,可能有环境音串入").strip()
    return r


def ref_stats(energy: dict, index: int) -> dict:
    """What the scorer needs for one sentence, for the browser to show."""
    caps = energy.get("captions") or []
    if not (0 <= index < len(caps)):
        raise IndexError(f"句子序号越界:{index}")
    c = caps[index]
    return {
        "i": index,
        "words": _words_of(c.get("text", "")),
        "marks": c.get("word_marks") or [],
        "ref_peaks": len(c.get("ref_nuclei") or []),
        "start": c.get("speech_start"),
        "end": c.get("speech_end"),
    }


if __name__ == "__main__":
    import sys

    # Self-test: score the original audio against itself. A perfect reading must
    # come out near 100, or the metric is measuring something else.
    en = json.loads(open("player/energy.json", encoding="utf-8").read())
    man = json.loads(open("player/player_captions.json", encoding="utf-8").read())
    import glob
    # energy.json carries no video id; the manifest names the clip.
    native = sorted(glob.glob(f"audio2/native_{man['videoId']}.*"))
    if not native:
        raise SystemExit(f"找不到 audio2/native_{man['videoId']}.*")
    src = native[0]
    which = [int(x) for x in sys.argv[1:]] or [3, 5, 12, 40, 88, 150]
    tot = []
    for i in which:
        cap = en["captions"][i]
        s0, s1 = cap["speech_start"], cap["speech_end"]
        pcm = decode_raw(subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-ss", str(s0),
             "-t", str(s1 - s0), "-i", src, "-f", "f32le", "-ac", "1",
             "-ar", str(SR), "-"], capture_output=True, check=True).stdout)
        r = score(en, pcm, i, t0=s0)
        tot.append(r["score"] or 0)
        stat = {}
        for w in r["per_word"]:
            stat[w["status"]] = stat.get(w["status"], 0) + 1
        print(f"[{i:3}] score {r['score']}  scale {r['scale']}  "
              f"delay {r['delay_ms']}ms  resid {r['mean_resid_ms']}ms  "
              f"peaks ref/user {r['ref_peaks']}/{r['user_peaks']}  {stat}")
    if tot:
        print(f"self-test mean {sum(tot)/len(tot):.1f} (expect ~100)")