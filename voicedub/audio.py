"""오디오 입출력·전처리 유틸리티 (numpy + soundfile, 필요 시 ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

FFMPEG = shutil.which("ffmpeg")


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """오디오 파일을 mono float32 로 읽는다. wav/flac/ogg 는 직접, 그 외(mp3, m4a 등)는 ffmpeg 로."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        if not FFMPEG:
            raise RuntimeError(
                f"{path.name}: 이 형식을 읽으려면 ffmpeg 가 필요합니다. "
                "ffmpeg 를 설치하거나 WAV 로 변환해 주세요."
            ) from None
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "converted.wav"
            subprocess.run(
                [FFMPEG, "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", str(out)],
                check=True,
            )
            data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def trim_silence(wav: np.ndarray, sr: int, threshold_db: float = -40.0, pad_ms: int = 60) -> np.ndarray:
    """앞뒤 무음을 잘라낸다(프레임 RMS 기준)."""
    if wav.size == 0:
        return wav
    frame = max(1, int(sr * 0.02))
    n = len(wav) // frame
    if n == 0:
        return wav
    frames = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    peak = rms.max()
    if peak <= 0:
        return wav[:0]
    loud = np.where(20 * np.log10(rms / peak) > threshold_db)[0]
    if loud.size == 0:
        return wav[:0]
    pad = int(sr * pad_ms / 1000)
    start = max(0, loud[0] * frame - pad)
    end = min(len(wav), (loud[-1] + 1) * frame + pad)
    return wav[start:end]


def normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav if m == 0 else (wav / m * peak).astype(np.float32)


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(seconds * sr)), dtype=np.float32)


def fade(wav: np.ndarray, sr: int, ms: int = 10) -> np.ndarray:
    """조각을 이어 붙일 때 '딱' 소리가 나지 않도록 짧게 페이드 인/아웃."""
    n = min(len(wav) // 2, int(sr * ms / 1000))
    if n <= 0:
        return wav
    wav = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    return wav


def prepare_reference(paths: list[str | Path], out_dir: str | Path) -> list[Path]:
    """참조 음성(내 목소리 샘플)을 mono WAV 로 변환하고 앞뒤 무음을 정리해 저장한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[Path] = []
    for i, p in enumerate(paths):
        wav, sr = load_audio(p)
        wav = normalize(trim_silence(wav, sr))
        if len(wav) < sr * 2:
            raise ValueError(f"{Path(p).name}: 목소리 샘플이 너무 짧습니다(2초 미만). 6~30초 분량을 권장합니다.")
        out = out_dir / f"ref_{i:02d}.wav"
        sf.write(str(out), wav, sr)
        prepared.append(out)
    return prepared


def save_audio(wav: np.ndarray, sr: int, path: str | Path) -> Path:
    """확장자에 따라 저장. .wav/.flac/.ogg 는 직접, .mp3/.m4a 등은 ffmpeg 로 변환."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        sf.write(str(path), wav, sr)
        return path
    if not FFMPEG:
        raise RuntimeError(f"{path.suffix} 로 저장하려면 ffmpeg 가 필요합니다. .wav 로 저장해 주세요.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = Path(tmp) / "out.wav"
        sf.write(str(tmp_wav), wav, sr)
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(tmp_wav), str(path)],
            check=True,
        )
    return path
