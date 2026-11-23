"""대본 → 오디오 조립 파이프라인 (엔진과 무관)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio
from .engine import Engine
from .script import Segment, Silence, Speech


@dataclass
class Cue:
    """자막/더빙용 타이밍 정보."""

    index: int
    start: float
    end: float
    text: str
    line_no: int


@dataclass
class Result:
    wav: np.ndarray
    sample_rate: int
    cues: list[Cue]
    clips: list[np.ndarray]  # 문장별 개별 오디오

    @property
    def duration(self) -> float:
        return len(self.wav) / self.sample_rate


ProgressFn = Callable[[int, int, str], None]


def render(
    segments: list[Segment],
    engine: Engine,
    *,
    progress: ProgressFn | None = None,
    trim: bool = True,
) -> Result:
    sr = engine.sample_rate
    speeches = [s for s in segments if isinstance(s, Speech)]
    total = len(speeches)
    parts: list[np.ndarray] = []
    cues: list[Cue] = []
    clips: list[np.ndarray] = []
    t = 0.0
    done = 0

    for seg in segments:
        if isinstance(seg, Silence):
            chunk = audio.silence(seg.seconds, sr)
            parts.append(chunk)
            t += len(chunk) / sr
            continue

        done += 1
        if progress:
            progress(done, total, seg.text)
        wav = engine.synthesize(seg.text)
        if trim:
            trimmed = audio.trim_silence(wav, sr, threshold_db=-45.0, pad_ms=40)
            wav = trimmed if trimmed.size else wav
        wav = audio.fade(wav, sr)
        clips.append(wav)
        start = t
        parts.append(wav)
        t += len(wav) / sr
        cues.append(Cue(len(cues) + 1, start, t, seg.text, seg.line_no))

    full = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    peak = float(np.max(np.abs(full))) if full.size else 0.0
    if peak > 0.99:  # 클리핑 방지
        full = full / peak * 0.95
    return Result(full.astype(np.float32), sr, cues, clips)


def _srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [f"{c.index}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n{c.text}\n" for c in cues]
    return "\n".join(blocks)


def save_clips(result: Result, out_dir: str | Path, fmt: str = "wav") -> list[Path]:
    """문장별 개별 파일 저장 (영상 편집기에서 하나씩 배치할 때 유용)."""
    out_dir = Path(out_dir)
    paths = []
    for cue, clip in zip(result.cues, result.clips):
        p = out_dir / f"{cue.index:03d}.{fmt}"
        audio.save_audio(clip, result.sample_rate, p)
        paths.append(p)
    return paths
