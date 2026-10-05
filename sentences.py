"""Turn YouTube's rolling caption cues into shadowable sentences.

Why this exists
---------------
Short, manually-captioned clips (the first test video) came back as clean
sentences: 6 cues, 6 sentences. A typical auto-captioned upload is nothing
like that. On this 18-minute video only 18% of the 479 cues end in sentence
punctuation, because YouTube emits a *rolling* window that re-shows the tail
of the previous line while the new line starts:

    [ 0.00- 4.44] What if I told you that people who speak
    [ 2.12- 6.40] with confidence weren't born like that?   <- starts 2.3s
    [ 4.44- 8.36] They just figured out how to improve        <- before #1 ends

Shadowing a half clause teaches the wrong unit, so cues are regrouped into
sentences first, and each sentence inherits a time span that covers the
audio of all its constituent cues.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict


# A cue ends a sentence when it carries terminal punctuation.
SENT_END = re.compile(r"[.!?…]['\"\)\]]*\s*$")
# Abbreviations that would otherwise split a sentence too early.
ABBREV = re.compile(
    r"\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|e\.g|i\.e|fig|approx|inc|ltd|co)\.$",
    re.I,
)
# Long clause without punctuation: still a usable shadowing unit.
MAX_CUE_RUN = 8
MAX_SENT_SEC = 9.0
MIN_SENT_SEC = 1.2

# A sentence can end *inside* a cue, because the rolling window appends the
# first words of the next line:
#     "know how to command their attention. By"
# Splitting only on cues that END with punctuation silently merged three real
# sentences into one 11.7s shadowing unit.
BOUNDARY = re.compile(r"[.!?…](?=\s|$)")


@dataclass
class Sentence:
    i: int
    start: float
    end: float
    text: str
    cues: list          # indices of the source cues
    overlap: bool = False   # its cues overlapped the neighbours
    ends_on_punct: bool = True


def _ends_sentence(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if ABBREV.search(t):
        return False
    return bool(SENT_END.search(t))


def _internal_boundaries(text: str) -> list:
    """Positions (char offsets) of sentence breaks *inside* one cue.

    Abbreviations and decimal points are skipped, and a break right at the
    end of the cue is not reported (that case is handled by _ends_sentence).
    """
    t = text
    hits = []
    for m in BOUNDARY.finditer(t):
        head = t[:m.start()]
        if ABBREV.search(head.rstrip() + "."):
            continue
        # skip decimals like "3.5" and initials like "J. R."
        if head and head[-1].isalnum() and t[m.end():m.end() + 1].isdigit():
            continue
        if m.start() >= len(t.rstrip()) - 1:
            break
        hits.append(m.start())
    return hits


def group_sentences(cues: list) -> list:
    """Regroup rolling cues into sentences.

    `cues` is a list of objects with .start, .duration, .text.
    """
    out: list[Sentence] = []
    run: list = []          # (text, start, end) fragments
    run_idx: list = []

    def flush(ends_on_punct: bool):
        if not run:
            return
        start = min(f[1] for f in run)
        end = max(f[2] for f in run)
        text = re.sub(r"\s+", " ", " ".join(f[0] for f in run)).strip()
        if text:
            out.append(Sentence(
                i=len(out), start=round(start, 3), end=round(end, 3),
                text=text, cues=list(run_idx), ends_on_punct=ends_on_punct,
            ))
        run.clear()
        run_idx.clear()

    for idx, c in enumerate(cues):
        c_start, c_end = c.start, c.start + c.duration
        text = c.text.replace("\n", " ").strip()

        # split this cue at any sentence break inside it
        cuts = _internal_boundaries(text)
        if cuts:
            pieces, prev = [], 0
            for p in cuts:
                pieces.append((text[prev:p + 1], prev, p + 1))
                prev = p + 1
            pieces.append((text[prev:], prev, len(text)))
            pieces = [p for p in pieces if p[0].strip()]
        else:
            pieces = [(text, 0, len(text))]

        n_frag = len([p for p in pieces if p[0].strip()])
        for k, (piece, a, b) in enumerate([p for p in pieces if p[0].strip()]):
            # time-share the cue proportionally to the piece's character
            # count; rolling cues overlap, so this is the best linear guess
            n_chars = max(1, len(text))
            fa, fb = a / n_chars, b / n_chars
            frag_start = c_start + (c_end - c_start) * fa
            frag_end = c_start + (c_end - c_start) * fb
            run.append((piece, frag_start, frag_end))
            run_idx.append(idx)

            dur = max(f[2] for f in run) - min(f[1] for f in run)
            # A fragment that itself ends a sentence must close the run even
            # when it is not the cue's last piece — that is exactly the case
            # "…their attention." | " By" in a rolling window.
            piece_ends = _ends_sentence(piece)
            if piece_ends:
                if dur < MIN_SENT_SEC and idx + 1 < len(cues) \
                        and not _ends_sentence(cues[idx + 1].text):
                    continue
                flush(True)
            elif dur >= MAX_SENT_SEC or len(run) >= MAX_CUE_RUN:
                flush(False)

    flush(False)

    # merge sentences that ended up too short to shadow
    merged: list[Sentence] = []
    for s in out:
        if merged and (s.end - s.start) < MIN_SENT_SEC:
            prev = merged[-1]
            prev.end = s.end
            prev.text = prev.text + " " + s.text
            prev.cues = prev.cues + s.cues
            prev.ends_on_punct = s.ends_on_punct
        else:
            merged.append(s)
    for n, s in enumerate(merged):
        s.i = n

    # flag overlap for the UI
    for a, b in zip(merged, merged[1:]):
        if b.start < a.end - 0.05:
            a.overlap = b.overlap = True
    return merged


def build(cues: list, max_len: int | None = None) -> dict:
    sents = group_sentences(cues)
    if max_len:
        sents = sents[:max_len]
    data = [asdict(s) for s in sents]
    lens = [s.end - s.start for s in sents]
    return {
        "sentences": data,
        "stats": {
            "source_cues": len(cues),
            "sentences": len(sents),
            "cues_per_sentence": round(len(cues) / max(1, len(sents)), 2),
            "mean_sec": round(sum(lens) / max(1, len(lens)), 2),
            "median_sec": round(sorted(lens)[len(lens) // 2], 2) if lens else 0,
            "max_sec": round(max(lens), 2) if lens else 0,
            "overlapping": sum(1 for s in sents if s.overlap),
            "ended_by_punct": sum(1 for s in sents if s.ends_on_punct),
        },
    }


if __name__ == "__main__":
    from youtube_transcript_api import YouTubeTranscriptApi
    import sys

    vid = sys.argv[1] if len(sys.argv) > 1 else "nxgF9NfYjRc"
    cues = list(YouTubeTranscriptApi().fetch(vid, languages=["en"]))
    res = build(cues)
    st = res["stats"]
    print(json.dumps(st, indent=2))
    print("\n--- first 12 sentences ---")
    for s in res["sentences"][:12]:
        print(f"  [{s['start']:7.2f}-{s['end']:7.2f}] "
              f"({s['end']-s['start']:4.1f}s) {s['text'][:66]}")
