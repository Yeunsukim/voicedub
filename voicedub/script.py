"""대본(스크립트) 파싱: 텍스트를 합성 가능한 문장 단위 + 쉼(무음) 구간으로 나눈다.

지원하는 대본 문법
------------------
- 한 줄에 한 문장 또는 여러 문장. 문장부호(. ! ? … 。 등) 기준으로 자동 분할.
- 빈 줄           : 문단 구분 → 조금 더 긴 쉼(기본 0.8초)
- ``# ...``        : 주석 줄 (읽지 않음)
- ``[pause 1.5]``  : 1.5초 쉼. ``[쉼 1.5]`` 도 같다. 줄 중간에 써도 된다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

# XTTS-v2 의 언어별 권장 최대 글자 수(이를 넘으면 품질이 떨어지거나 잘림).
DEFAULT_CHAR_LIMITS = {"ko": 95, "ja": 71, "zh-cn": 82, "en": 250}
FALLBACK_CHAR_LIMIT = 200

_PAUSE_RE = re.compile(r"\[\s*(?:pause|쉼)\s*([0-9]*\.?[0-9]+)\s*(?:s|초)?\s*\]", re.IGNORECASE)
# 문장 끝: 마침표/물음표/느낌표/말줄임표(+닫는 따옴표·괄호) 뒤의 공백
_SENTENCE_END_RE = re.compile(r"(?:(?<=[.!?…。！？])|(?<=[.!?…。！？][\"'”’)\]]))\s+")
_CLAUSE_END_RE = re.compile(r"(?<=[,;:，、])\s+")


@dataclass(frozen=True)
class Speech:
    text: str
    line_no: int  # 원본 대본의 줄 번호 (1부터)


@dataclass(frozen=True)
class Silence:
    seconds: float


Segment = Speech | Silence


def char_limit_for(language: str) -> int:
    return DEFAULT_CHAR_LIMITS.get(language, FALLBACK_CHAR_LIMIT)


def split_long(text: str, max_chars: int) -> list[str]:
    """max_chars 를 넘는 문장을 쉼표 → 공백 순으로 잘라 나눈다."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    for splitter in (_CLAUSE_END_RE, re.compile(r"\s+")):
        parts = [p for p in splitter.split(text) if p]
        if len(parts) > 1:
            return _pack(parts, max_chars)

    # 공백조차 없는 아주 긴 덩어리: 강제로 자른다.
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _pack(parts: list[str], max_chars: int) -> list[str]:
    """작은 조각들을 max_chars 이하가 되도록 다시 묶는다."""
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(part) > max_chars:
            sub = split_long(part, max_chars)
            chunks.extend(sub[:-1])
            current = sub[-1]
        else:
            current = part
    if current:
        chunks.append(current)
    return chunks


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]


def parse_script(
    script: str,
    *,
    language: str = "ko",
    sentence_pause: float = 0.3,
    paragraph_pause: float = 0.8,
    max_chars: int | None = None,
) -> list[Segment]:
    """대본 문자열을 Speech / Silence 목록으로 변환한다."""
    limit = max_chars or char_limit_for(language)
    segments: list[Segment] = []

    def add_silence(seconds: float) -> None:
        if seconds <= 0:
            return
        if segments and isinstance(segments[-1], Silence):
            # 연속된 쉼은 더 긴 쪽으로 합친다(명시적 [pause]는 합산).
            segments[-1] = Silence(max(segments[-1].seconds, seconds))
        else:
            segments.append(Silence(seconds))

    blank_pending = False
    for line_no, raw in enumerate(script.splitlines(), start=1):
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not line:
            blank_pending = True
            continue
        if blank_pending and segments:
            add_silence(paragraph_pause)
        blank_pending = False

        for kind, value in _tokenize_line(line):
            if kind == "pause":
                # 명시적 쉼은 자동 쉼을 대체한다.
                if segments and isinstance(segments[-1], Silence):
                    segments[-1] = Silence(float(value))
                else:
                    segments.append(Silence(float(value)))
                continue
            for sentence in split_sentences(value):
                for chunk in split_long(sentence, limit):
                    if segments and isinstance(segments[-1], Speech):
                        add_silence(sentence_pause)
                    segments.append(Speech(chunk, line_no))

        # 줄바꿈도 문장 경계로 본다.
        if segments and isinstance(segments[-1], Speech):
            add_silence(sentence_pause)

    while segments and isinstance(segments[-1], Silence):
        segments.pop()
    return segments


def _tokenize_line(line: str) -> Iterator[tuple[str, str]]:
    pos = 0
    for m in _PAUSE_RE.finditer(line):
        before = line[pos : m.start()].strip()
        if before:
            yield "text", before
        yield "pause", m.group(1)
        pos = m.end()
    rest = line[pos:].strip()
    if rest:
        yield "text", rest
