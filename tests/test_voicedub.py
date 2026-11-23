import numpy as np
import soundfile as sf

from voicedub import audio
from voicedub.cli import main
from voicedub.pipeline import render, save_clips, to_srt
from voicedub.script import Silence, Speech, parse_script, split_long, split_sentences


class FakeEngine:
    """모델 없이 테스트: 글자 수에 비례하는 길이의 사인파 + 앞뒤 무음."""

    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        sr = self.sample_rate
        n = int(sr * 0.05 * len(text))
        tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(n) / sr).astype(np.float32)
        pad = np.zeros(int(sr * 0.3), dtype=np.float32)
        return np.concatenate([pad, tone, pad])


def test_split_sentences_korean():
    s = '안녕하세요. 반갑습니다! 정말요? 네… 그렇습니다. "좋아요." 끝'
    assert split_sentences(s) == ["안녕하세요.", "반갑습니다!", "정말요?", "네…", "그렇습니다.", '"좋아요."', "끝"]


def test_split_long_respects_limit():
    text = "가나다라마바사 " * 30
    chunks = split_long(text, 40)
    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_split_long_prefers_commas():
    text = "첫 번째 부분은 여기까지이고, 두 번째 부분은 여기서부터 시작합니다"
    assert split_long(text, 30) == ["첫 번째 부분은 여기까지이고,", "두 번째 부분은 여기서부터 시작합니다"]


def test_split_long_no_spaces():
    assert split_long("가" * 25, 10) == ["가" * 10, "가" * 10, "가" * 5]


def test_parse_script_pauses_and_comments():
    script = "# 주석\n첫 문장. 둘째 문장.\n\n셋째 [쉼 2] 넷째\n[pause 0.5]\n다섯째\n\n\n"
    segs = parse_script(script, sentence_pause=0.3, paragraph_pause=0.8)
    assert segs == [
        Speech("첫 문장.", 2),
        Silence(0.3),
        Speech("둘째 문장.", 2),
        Silence(0.8),
        Speech("셋째", 4),
        Silence(2.0),
        Speech("넷째", 4),
        Silence(0.5),
        Speech("다섯째", 6),
    ]


def test_parse_script_empty():
    assert parse_script("# only comment\n\n   \n") == []


def test_trim_silence():
    sr = 16000
    wav = np.concatenate([np.zeros(sr), 0.5 * np.ones(sr // 2), np.zeros(sr)]).astype(np.float32)
    out = audio.trim_silence(wav, sr, pad_ms=0)
    assert abs(len(out) - sr // 2) <= sr * 0.02


def test_render_timing_and_srt(tmp_path):
    eng = FakeEngine()
    segs = parse_script("하나. 둘.\n\n셋.", sentence_pause=0.5, paragraph_pause=1.0)
    res = render(segs, eng)
    assert eng.calls == ["하나.", "둘.", "셋."]
    assert len(res.cues) == 3
    # 앞뒤 무음이 잘려서 각 클립은 원래 길이(0.6초+)보다 짧아야 함
    assert all(len(c) < 24000 * 0.6 for c in res.clips)
    # 큐 사이 간격 = 지정한 쉼
    assert abs((res.cues[1].start - res.cues[0].end) - 0.5) < 1e-3
    assert abs((res.cues[2].start - res.cues[1].end) - 1.0) < 1e-3
    assert abs(res.duration - res.cues[-1].end) < 1e-3

    srt = to_srt(res.cues)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,")
    assert "3\n" in srt and "셋." in srt

    paths = save_clips(res, tmp_path / "clips")
    assert [p.name for p in paths] == ["001.wav", "002.wav", "003.wav"]


def test_prepare_reference(tmp_path):
    sr = 22050
    t = np.arange(sr * 4) / sr
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 200 * t)] * 2, axis=1)
    stereo = np.concatenate([np.zeros((sr, 2)), stereo, np.zeros((sr, 2))])
    src = tmp_path / "me.wav"
    sf.write(src, stereo, sr)
    [out] = audio.prepare_reference([src], tmp_path / "prep")
    data, out_sr = sf.read(out)
    assert data.ndim == 1 and out_sr == sr
    assert 3.9 < len(data) / sr < 4.3
    assert abs(np.max(np.abs(data)) - 0.95) < 0.01


def test_cli_preview(tmp_path, capsys):
    script = tmp_path / "s.txt"
    script.write_text("안녕하세요. 반가워요.\n\n[쉼 1]\n끝", encoding="utf-8")
    assert main(["preview", str(script)]) == 0
    out = capsys.readouterr().out
    assert "[001] (줄 1) 안녕하세요." in out
    assert "쉼 1.00초" in out
    assert "[003] (줄 4) 끝" in out


def test_cli_reads_cp949(tmp_path, capsys):
    script = tmp_path / "s.txt"
    script.write_bytes("윈도우 메모장 대본입니다.".encode("cp949"))
    assert main(["preview", str(script)]) == 0
    assert "윈도우 메모장 대본입니다." in capsys.readouterr().out
