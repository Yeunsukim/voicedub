"""명령줄 인터페이스.

    voicedub enroll 내목소리.wav -o voices/me.pt        # 목소리 등록(선택)
    voicedub speak 대본.txt --profile voices/me.pt -o out.wav --srt
    voicedub speak 대본.txt --voice 내목소리.wav -o out.mp3
    voicedub preview 대본.txt                           # 문장 분할 확인(모델 불필요)
    voicedub ui                                         # 웹 화면 실행
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from . import __version__
from .script import Silence, Speech, char_limit_for, parse_script


def _read_script(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    p = Path(path)
    for enc in ("utf-8-sig", "cp949"):  # 윈도우 메모장(ANSI) 저장 파일도 읽을 수 있게
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"대본 파일 인코딩을 읽을 수 없습니다: {path} (UTF-8 로 저장해 주세요)")


def _add_script_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("script", help="대본 텍스트 파일 경로 ('-' 는 표준입력)")
    p.add_argument("--lang", default="ko", help="대본 언어 (기본: ko)")
    p.add_argument("--sentence-pause", type=float, default=0.3, help="문장 사이 쉼(초), 기본 0.3")
    p.add_argument("--paragraph-pause", type=float, default=0.8, help="빈 줄(문단) 쉼(초), 기본 0.8")
    p.add_argument("--max-chars", type=int, default=None, help="한 번에 합성할 최대 글자 수")


def _add_engine_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--speed", type=float, default=1.0, help="말하기 속도 (0.5~2.0, 기본 1.0)")
    p.add_argument("--temperature", type=float, default=0.65,
                   help="낮을수록 안정적·단조롭고, 높을수록 표현이 풍부하나 불안정 (기본 0.65)")


def _segments(args) -> list:
    return parse_script(
        _read_script(args.script),
        language=args.lang,
        sentence_pause=args.sentence_pause,
        paragraph_pause=args.paragraph_pause,
        max_chars=args.max_chars,
    )


def _make_engine(args):
    from .engine import XTTSEngine

    print(f"모델 불러오는 중... (처음에는 약 1.8GB 다운로드)", file=sys.stderr)
    return XTTSEngine(
        language=getattr(args, "lang", "ko"),
        device=args.device,
        speed=args.speed,
        temperature=args.temperature,
    )


def _load_voice(engine, voices: list[str] | None, profile: str | None) -> None:
    from .audio import prepare_reference

    if profile:
        engine.load_profile(profile)
        return
    if not voices:
        raise SystemExit("목소리를 지정하세요: --voice 내목소리.wav  또는  --profile voices/me.pt")
    with tempfile.TemporaryDirectory() as tmp:
        refs = prepare_reference(voices, tmp)
        engine.load_voice(refs)


def cmd_preview(args) -> int:
    segments = _segments(args)
    limit = args.max_chars or char_limit_for(args.lang)
    n = 0
    for seg in segments:
        if isinstance(seg, Silence):
            print(f"      ··· 쉼 {seg.seconds:.2f}초")
        else:
            n += 1
            flag = "  ⚠ 길이 초과" if len(seg.text) > limit else ""
            print(f"[{n:03d}] (줄 {seg.line_no}) {seg.text}{flag}")
    print(f"\n총 {n}개 문장, 문장당 최대 {limit}자", file=sys.stderr)
    return 0


def cmd_enroll(args) -> int:
    engine = _make_engine(args)
    _load_voice(engine, args.voices, None)
    path = engine.save_profile(args.output)
    print(f"목소리 프로필 저장: {path}")
    return 0


def cmd_speak(args) -> int:
    from .audio import save_audio
    from .pipeline import render, save_clips, to_srt

    segments = _segments(args)
    if not any(isinstance(s, Speech) for s in segments):
        raise SystemExit("대본에 읽을 문장이 없습니다.")

    engine = _make_engine(args)
    _load_voice(engine, args.voice, args.profile)

    started = time.time()

    def progress(i: int, total: int, text: str) -> None:
        short = text if len(text) <= 40 else text[:39] + "…"
        print(f"\r[{i}/{total}] {short:<42}", end="", file=sys.stderr, flush=True)

    result = render(segments, engine, progress=progress)
    print(file=sys.stderr)

    out = save_audio(result.wav, result.sample_rate, args.output)
    print(f"저장 완료: {out}  (길이 {result.duration:.1f}초, 소요 {time.time() - started:.0f}초)")

    if args.srt:
        srt_path = out.with_suffix(".srt")
        srt_path.write_text(to_srt(result.cues), encoding="utf-8")
        print(f"자막 저장: {srt_path}")
    if args.clips:
        paths = save_clips(result, args.clips)
        print(f"문장별 파일 {len(paths)}개 저장: {args.clips}")
    return 0


def cmd_ui(args) -> int:
    from .webui import launch

    launch(host=args.host, port=args.port, share=args.share, device=args.device)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicedub", description="대본을 내 목소리로 읽어주는 프로그램")
    parser.add_argument("--version", action="version", version=f"voicedub {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("speak", help="대본을 내 목소리 음성 파일로 변환")
    _add_script_opts(p)
    _add_engine_opts(p)
    p.add_argument("-v", "--voice", action="append", help="내 목소리 샘플 파일 (여러 번 지정 가능)")
    p.add_argument("-p", "--profile", help="enroll 로 만든 목소리 프로필(.pt)")
    p.add_argument("-o", "--output", default="output.wav", help="출력 파일 (.wav/.mp3/.m4a/.flac)")
    p.add_argument("--srt", action="store_true", help="같은 이름의 .srt 자막(타이밍) 파일도 생성")
    p.add_argument("--clips", metavar="DIR", help="문장별 개별 WAV 파일을 이 폴더에 저장")
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("enroll", help="목소리 샘플로 프로필(.pt)을 만들어 저장")
    p.add_argument("voices", nargs="+", help="내 목소리 샘플 파일들 (6~30초씩 권장)")
    p.add_argument("-o", "--output", default="voices/me.pt")
    _add_engine_opts(p)
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("preview", help="대본이 어떻게 나뉘어 읽힐지 미리 보기 (모델 불필요)")
    _add_script_opts(p)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("ui", help="브라우저용 웹 화면 실행")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="외부 접속용 공개 링크 생성")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n중단되었습니다.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
