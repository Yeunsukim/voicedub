"""음성 합성 엔진.

기본 엔진은 Coqui XTTS-v2 (오픈소스, 한국어 포함 17개 언어, 6초 이상의 목소리 샘플로
제로샷 보이스 클로닝). 모델은 처음 실행할 때 자동으로 내려받는다(약 1.8GB).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGES = (
    "ko en ja zh-cn es fr de it pt pl tr ru nl cs ar hu hi".split()
)


class Engine(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class XTTSEngine:
    """XTTS-v2 로 '내 목소리' 를 복제해 텍스트를 읽는다."""

    def __init__(
        self,
        *,
        language: str = "ko",
        device: str = "auto",
        temperature: float = 0.65,
        speed: float = 1.0,
        repetition_penalty: float = 5.0,
        top_p: float = 0.85,
    ) -> None:
        if language not in XTTS_LANGUAGES:
            raise ValueError(f"지원하지 않는 언어: {language} (지원: {', '.join(XTTS_LANGUAGES)})")
        try:
            from TTS.api import TTS
        except ImportError as e:  # pragma: no cover - 설치 안내
            raise RuntimeError(
                "coqui-tts 가 설치되어 있지 않습니다:  pip install -e .   (또는 pip install coqui-tts)"
            ) from e

        self.language = language
        self.device = pick_device(device)
        self.params = dict(
            temperature=temperature,
            speed=speed,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
        )
        self._tts = TTS(XTTS_MODEL).to(self.device)
        self._model = self._tts.synthesizer.tts_model
        self.sample_rate = int(self._model.config.audio.output_sample_rate)
        self._latents = None

    # --- 목소리 등록 -------------------------------------------------------
    def load_voice(self, reference_wavs: list[str | Path]) -> None:
        """내 목소리 샘플(WAV 여러 개 가능)에서 화자 특징을 추출한다."""
        gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
            audio_path=[str(p) for p in reference_wavs]
        )
        self._latents = (gpt_cond_latent, speaker_embedding)

    def save_profile(self, path: str | Path) -> Path:
        """추출한 목소리 특징을 파일로 저장 → 다음부터 샘플 없이 바로 사용."""
        import torch

        if self._latents is None:
            raise RuntimeError("먼저 load_voice() 로 목소리를 등록하세요.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        gpt, spk = self._latents
        torch.save({"gpt_cond_latent": gpt.cpu(), "speaker_embedding": spk.cpu(), "model": XTTS_MODEL}, path)
        return path

    def load_profile(self, path: str | Path) -> None:
        import torch

        data = torch.load(str(path), map_location=self.device)
        self._latents = (
            data["gpt_cond_latent"].to(self.device),
            data["speaker_embedding"].to(self.device),
        )

    # --- 합성 -------------------------------------------------------------
    def synthesize(self, text: str) -> np.ndarray:
        if self._latents is None:
            raise RuntimeError("목소리가 등록되지 않았습니다. --voice 또는 --profile 을 지정하세요.")
        gpt, spk = self._latents
        out = self._model.inference(
            text,
            self.language,
            gpt,
            spk,
            enable_text_splitting=False,  # 문장 분할은 voicedub 가 직접 한다
            **self.params,
        )
        wav = out["wav"]
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        return np.asarray(wav, dtype=np.float32).reshape(-1)
