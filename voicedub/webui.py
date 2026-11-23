"""Gradio 웹 화면: 목소리 녹음/업로드 → 대본 입력 → 음성 생성."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from .audio import prepare_reference, save_audio
from .engine import XTTS_LANGUAGES, XTTSEngine
from .pipeline import render, to_srt
from .script import Speech, parse_script

_engines: dict[str, XTTSEngine] = {}
_voice_key: dict[str, str] = {}

SAMPLE_SCRIPT = """# '#' 으로 시작하는 줄은 읽지 않습니다.
안녕하세요. 오늘은 제 목소리로 만든 음성을 들려드리겠습니다.
이 프로그램은 대본을 문장 단위로 나눠서 자연스럽게 읽어 줍니다.

빈 줄을 넣으면 문단이 바뀌면서 조금 더 쉬어 갑니다. [쉼 1.5] 이렇게 원하는 만큼 쉴 수도 있어요.
"""


def _hash_files(paths: list[str]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def _get_engine(lang: str, device: str) -> XTTSEngine:
    # 모델 가중치는 언어와 무관 → 하나만 로드하고 언어만 바꿔 쓴다.
    eng = _engines.get("xtts")
    if eng is None:
        eng = _engines["xtts"] = XTTSEngine(language=lang, device=device)
    eng.language = lang
    return eng


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False, device: str = "auto") -> None:
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("웹 화면에는 gradio 가 필요합니다:  pip install -e \".[ui]\"") from None

    def generate(voice_files, mic, script_text, script_file, lang, speed, temperature, sent_pause, para_pause,
                 progress=gr.Progress()):
        paths = [f if isinstance(f, str) else f.name for f in (voice_files or [])]
        if mic:
            paths.append(mic)
        if not paths:
            raise gr.Error("목소리 샘플을 업로드하거나 녹음해 주세요.")
        if script_file:
            script_text = Path(script_file if isinstance(script_file, str) else script_file.name).read_text(
                encoding="utf-8-sig")
        segments = parse_script(script_text or "", language=lang,
                                sentence_pause=sent_pause, paragraph_pause=para_pause)
        if not any(isinstance(s, Speech) for s in segments):
            raise gr.Error("대본에 읽을 문장이 없습니다.")

        progress(0, desc="모델 준비 중 (처음엔 다운로드로 오래 걸립니다)")
        eng = _get_engine(lang, device)
        eng.params.update(speed=speed, temperature=temperature)

        key = _hash_files(paths)
        if _voice_key.get("current") != key:
            progress(0, desc="목소리 분석 중")
            with tempfile.TemporaryDirectory() as tmp:
                eng.load_voice(prepare_reference(paths, tmp))
            _voice_key["current"] = key

        def cb(i, total, text):
            progress(i / total, desc=f"{i}/{total}: {text[:30]}")

        result = render(segments, eng, progress=cb)
        out_dir = Path(tempfile.mkdtemp(prefix="voicedub_"))
        wav_path = save_audio(result.wav, result.sample_rate, out_dir / "voicedub.wav")
        srt_path = out_dir / "voicedub.srt"
        srt_path.write_text(to_srt(result.cues), encoding="utf-8")
        return str(wav_path), [str(wav_path), str(srt_path)]

    with gr.Blocks(title="VoiceDub - 내 목소리로 대본 읽기") as demo:
        gr.Markdown(
            "# 🎙️ VoiceDub\n대본을 **내 목소리**로 읽어 줍니다. "
            "조용한 곳에서 녹음한 **6~30초** 분량의 깨끗한 목소리 샘플이 가장 좋습니다."
        )
        with gr.Row():
            with gr.Column():
                voice_files = gr.File(label="① 내 목소리 샘플 (여러 개 가능: wav/mp3/m4a)",
                                      file_count="multiple", type="filepath")
                mic = gr.Audio(label="또는 여기서 바로 녹음", sources=["microphone"], type="filepath")
                script_text = gr.Textbox(label="② 대본", lines=12, value=SAMPLE_SCRIPT)
                script_file = gr.File(label="또는 대본 파일(.txt) 업로드", file_types=[".txt"], type="filepath")
            with gr.Column():
                lang = gr.Dropdown(list(XTTS_LANGUAGES), value="ko", label="언어")
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="속도")
                temperature = gr.Slider(0.1, 1.0, value=0.65, step=0.05,
                                        label="표현력 (낮음=안정적, 높음=풍부하지만 불안정)")
                sent_pause = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="문장 사이 쉼(초)")
                para_pause = gr.Slider(0.0, 3.0, value=0.8, step=0.05, label="문단 사이 쉼(초)")
                btn = gr.Button("③ 음성 만들기", variant="primary")
                out_audio = gr.Audio(label="결과", type="filepath")
                out_files = gr.File(label="다운로드 (WAV + SRT 자막)")
        btn.click(generate,
                  [voice_files, mic, script_text, script_file, lang, speed, temperature, sent_pause, para_pause],
                  [out_audio, out_files])

    demo.queue().launch(server_name=host, server_port=port, share=share)
