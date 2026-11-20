"""可选:产出逐词时间,让换气斜杠挂到正确的词上。

    .venv/bin/python align_words.py <video-id> [model]

写出 audio2/words_<videoId>.json,格式 [{w, s, e}, …],由 wordmarks.load_spans
读取。**不是构建的必要步骤** —— 没有它 rebuild.py 照跑,只是斜杠的锚词会退回
到 word_marks。

为什么要这一步
-------------
斜杠的位置是波形测出来的(哪一段真的静了),这件事本项目自己能算。难点在
"这处停顿挂到哪个词":word_marks 是"词亮起来"的时刻,快句子上会整体后移,
全片 767 处停顿里只有 16% 和真实词界一致,37% 错两个词以上。逐词时间在句子
里就守着停顿,用它定锚点才能让屏幕上的斜杠落在真词后面。

为什么不做成依赖
---------------
它需要 faster-whisper(以及一次模型下载,base.en 约 74MB)。整片跑一次要几
分钟,和"改一行 CSS 就 rebuild"的节奏不搭。所以它单独跑、结果落盘,构建只
读落盘的结果。缺了也不影响出片。

如果用的是别的 ASR,只要写出同样格式的文件,后续步骤不用改。
"""

from __future__ import annotations

import glob
import json
import struct
import subprocess
import sys

from pathlib import Path

SR = 16000


def video_id(arg: str) -> str:
    p = Path("player/player_captions.json")
    if not arg and p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["videoId"]
    return arg.strip()


def decode_native(path: Path) -> "list":
    """整片解成 16k 单声道 float32。

    不用 faster-whisper 自己的解码:它走 av,而 av 的版本和 faster-whisper
    需要的 kwargs 对不上会抛 `metadata_errors`。直接喂 numpy 数组绕开。
    """
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=True).stdout
    n = len(out) // 4
    import numpy as np
    return np.frombuffer(out[: n * 4], dtype="<f4").copy()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    vid = video_id(sys.argv[1] if len(sys.argv) > 1 else "")
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base.en"

    hits = glob.glob(f"audio2/native_{vid}.*")
    if not hits:
        print(f"找不到 audio2/native_{vid}.* —— 先跑 rebuild.py 抓一次音频")
        return 1

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("没装 faster-whisper。装一个再跑:\n"
              f"    .venv/bin/pip install faster-whisper\n"
              "或者跳过这一步 —— 出片不依赖它,只是斜杠可能挂到隔壁的词上。")
        return 1

    pcm = decode_native(Path(hits[0]))
    print(f"{hits[0]}  {len(pcm)/SR:.0f}s  模型 {model_size}(int8)")

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, info = model.transcribe(pcm, language="en", beam_size=5,
                                  word_timestamps=True, vad_filter=False)

    words, last = [], 0.0
    for s in segs:
        for w in (s.words or []):
            t = w.word.strip()
            if not t:
                continue
            # whisper 偶尔给出 s == e 或倒序,给一个最小宽度,免得后面按词尾
            # 排序时把两个词并到一起。
            a, b = float(w.start), max(float(w.end), float(w.start) + 0.04)
            a = max(a, last)
            words.append({"w": t, "s": round(a, 3), "e": round(b, 3)})
            last = a
        if len(words) % 500 < 12:
            print(f"  … {len(words)} 词 / {s.end:.0f}s", flush=True)

    out = f"audio2/words_{vid}.json"
    json.dump(words, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(words)} 词 → {out}")
    print("再跑一次 wordmarks.py(或 rebuild.py)把锚点写进 energy.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
