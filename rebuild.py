"""Rebuild the shadowing player for a different video, end to end.

Pipeline
  1. fetch captions            (youtube-transcript-api, proxy-aware)
  2. regroup into sentences   (sentences.py)
  3. slice audio to a window  (ffmpeg)
  4. pre-render speed variants (atempo, pitch preserved)
  5. measure speech + syllable timing (onsets.py, wordmarks.py)
  6. emit the player manifest
  7. verify the highlight against the audio (check.py)

Usage:
  .venv/bin/python rebuild.py <video-id-or-url> [start_sec] [length_sec] [title]
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
import sentences as S

PLAYER = Path("player")
AUDIO_DIR = Path("audio2")
SPEEDS = [1, 0.75, 0.6, 0.5]


def video_id(arg: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", arg)
    return m.group(1) if m else arg.strip()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


def probe_dur(p) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries",
                      "format=duration", "-of", "csv=p=0", str(p)]).stdout.strip())


def fetch_audio(vid: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Name the download by video id. The two names used to disagree — the
    # fetcher wrote native.webm while the caller looked for native_<vid>.webm
    # — so the cache never hit and every build re-downloaded 15 MB.
    raw = out_dir / f"native_{vid}.%(ext)s"
    yt = sys.executable.rsplit("/", 1)[0] + "/yt-dlp"
    if not os.path.exists(yt):
        yt = shutil.which("yt-dlp") or "yt-dlp"
    print(f"[1/7] downloading audio for {vid} …")
    subprocess.run([yt, "--no-warnings", "-f", "251", "-o", str(raw),
                    f"https://www.youtube.com/watch?v={vid}"], check=True)
    # Match this video's file specifically. A bare native.* glob would happily
    # return a different video's cached audio, producing a clip whose words do
    # not match its captions.
    hits = sorted(out_dir.glob(f"native_{vid}.*"))
    if not hits:
        raise SystemExit(f"下载后找不到音频文件:native_{vid}.*")
    got = hits[0]
    print(f"      -> {got} ({got.stat().st_size/1048576:.1f} MB)")
    return got


def slice_audio(src: Path, start: float, length: float, dst: Path):
    print(f"[3/7] slicing {start:.0f}s +{length:.0f}s -> {dst.name}")
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(length),
         "-i", str(src), "-vn", "-ac", "1", "-ar", "24000", "-c:a", "aac",
         "-b:a", "32k", str(dst)])
    return probe_dur(dst)


def speed_variants(base: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("speech_*.m4a"):
        f.unlink()
    print("[4/7] rendering speed variants (atempo, pitch preserved)")
    rows = []
    for sp in SPEEDS:
        name = f"speech_{sp:.2f}".replace(".", "p") + ".m4a"
        dst = out_dir / name
        stages, rem = [], sp
        while rem < 0.5:
            stages.append("atempo=0.5")
            rem /= 0.5
        stages.append(f"atempo={rem:g}")
        run(["ffmpeg", "-y", "-v", "error", "-i", str(base),
             "-af", ",".join(stages), "-c:a", "aac", "-b:a", "64k",
             "-ar", "24000", "-ac", "1", str(dst)])
        rows.append((sp, name, probe_dur(dst)))
        print(f"      {sp}x -> {rows[-1][2]:7.2f}s  "
              f"{dst.stat().st_size/1024:6.0f} KB")
    return rows


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "nxgF9NfYjRc"
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    length = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
    title = sys.argv[4] if len(sys.argv) > 4 else "Shadowing clip"
    vid = video_id(arg)

    # 1. audio
    native = AUDIO_DIR / f"native_{vid}.webm"
    if not native.exists():
        try:
            native = fetch_audio(vid, AUDIO_DIR)
        except subprocess.CalledProcessError:
            # yt-dlp already printed the reason (private video, no captions,
            # network blocked). The browser shows this log, so say it in the
            # user's terms and stop here rather than dumping a traceback.
            print(f"\n下载失败:{vid} 无法获取音频。")
            print("可能原因:视频不存在/为私有、年龄限制、地区限制,或网络/代理不通。")
            return 1
    else:
        print(f"[1/7] reusing {native}")

    # 2. captions -> sentences
    print("[2/7] fetching captions and regrouping into sentences …")
    # Priority order, not a single language: an English-only request fails on
    # the many videos that publish zh-Hans/zh-Hant, which is the common case
    # for the material this player is meant for.
    fetched, cues, used = None, [], None
    for langs in (["en", "en-US", "en-GB"],
                  ["zh-CN", "zh-TW", "zh-Hans", "zh-Hant", "zh"],
                  ["en", "zh-CN", "zh-TW", "zh", "ja", "ko", "es", "fr",
                   "de", "pt", "ru"]):
        try:
            fetched = YouTubeTranscriptApi().fetch(vid, languages=langs)
        except Exception:
            continue
        cues = list(fetched)
        if cues:
            used = "+".join(langs[:3])
            break
    if not cues:
        print(f"\n获取字幕失败:{vid} 没有可用的字幕轨。")
        print("可能原因:视频未开启字幕、字幕仅限特定地区,或网络/代理不通。")
        return 1
    print(f"      字幕轨: {used}  ({len(cues)} cues)")
    res = S.build(cues)
    st = res["stats"]
    print(f"      {st['source_cues']} cues -> {st['sentences']} sentences "
          f"({st['cues_per_sentence']} cues/sentence)")
    print(f"      median {st['median_sec']}s   ended by punctuation: "
          f"{st['ended_by_punct']}/{st['sentences']}")

    # keep sentences fully inside the audio window
    # length 0 means "the whole video": resolve it against the real duration
    # now that the audio has been fetched, so every later comparison sees a
    # concrete number instead of a 0 that would filter out every sentence.
    if length <= 0:
        full = probe_dur(native)
        start_at = min(start, max(0.0, full - 1.0))
        length = max(1.0, full - start_at)
        start = start_at
        print(f"      length=0 -> 截取完整音频 {length:.1f}s "
              f"(从 {start:.1f}s 到视频结尾)")

    sents = [s for s in res["sentences"]
             if s["start"] >= start and s["end"] <= start + length]
    if len(sents) < 3:
        print("      not enough sentences in the window; widening to 300s")
        length = 300.0
        sents = [s for s in res["sentences"] if s["start"] >= start
                 and s["end"] <= start + length]
    print(f"      using {len(sents)} sentences "
          f"({sents[0]['start']:.1f}s -> {sents[-1]['end']:.1f}s)")

    # rebase to the clip's own timeline
    rel = [
        {"i": n, "start": round(s["start"] - start, 3),
         "end": round(s["end"] - start, 3),
         "text": s["text"], "overlap": s["overlap"],
         "ends_on_punct": s["ends_on_punct"]}
        for n, s in enumerate(sents)
    ]
    clip_dur = max(x["end"] for x in rel)

    # 3. audio slice
    base = PLAYER / "speech.m4a"
    if start > 0 or length < probe_dur(native) - 1:
        slice_audio(native, start, clip_dur, base)
    else:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(native), "-vn",
             "-ac", "1", "-ar", "24000", "-c:a", "aac", "-b:a", "32k", str(base)])
    print(f"      base clip: {probe_dur(base):.2f}s "
          f"({base.stat().st_size/1024:.0f} KB)")

    # 4. speed variants
    variants = speed_variants(base, PLAYER / "slow")

    # 5. measure real speech per sentence, so the highlight tracks the voice
    print("[5/7] measuring speech boundaries (energy map) …")
    import onsets
    import wordmarks
    caps = onsets.measure(native, rel, "player/energy.json", start=0.0)
    # Word-level timing must come from the detected syllable peaks, otherwise
    # the highlight drifts: measured against the waveform, evenly spread words
    # lagged the voice by 0.46s on average.
    wordmarks.build(native, caps, "player/energy.json", start=0.0)

    # 6. manifest
    print("[6/7] writing player manifest")
    manifest = {
        "videoId": vid,
        "title": title,
        "audio": "speech.m4a",
        "video": None,
        "duration": round(clip_dur, 3),
        "captions": rel,
        "speeds": [{"rate": s, "file": f"slow/{n}", "duration": round(d, 3)}
                   for s, n, d in variants],
        "source": {
            "kind": "auto" if fetched.is_generated else "human",
            "cues": st["source_cues"],
            "sentences": st["sentences"],
            "median_sentence_sec": st["median_sec"],
        },
    }
    (PLAYER / "player_captions.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # this clip has no video track worth shipping
    for stale in ["player_video.mp4"]:
        p = PLAYER / stale
        if p.exists():
            p.unlink()
            print(f"      removed stale {stale} (audio-only clip)")

    total = sum((PLAYER / "slow" / n).stat().st_size for _, n, _ in variants)
    print(f"\nready: {len(rel)} sentences, {clip_dur:.1f}s, "
          f"4 speeds = {total/1024:.0f} KB")

    # 7. verify against the audio, not against ourselves
    #
    # Every check the project began with compared the player to its own data,
    # which cannot see a highlight that lags or skips words. The two that caught
    # the real defects measure the built page against the waveform, so a new
    # video cannot silently inherit the same problems.
    print("\n[7/7] verifying the highlight against the audio …", flush=True)
    # Stream the checks' output straight through: their verdicts are the whole
    # point of this step, and a build that silently swallows them is worse
    # than no build.
    r = subprocess.run([sys.executable, "check.py"], env=dict(os.environ))
    sys.stdout.flush()
    if r.returncode != 0:
        print("\nbuild finished, but verification FAILED — "
              "the highlight would read as lagging. Player is not ready.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
