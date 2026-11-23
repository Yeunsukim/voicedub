# 🎙️ VoiceDub — 대본을 내 목소리로

텍스트 대본을 **내 목소리**로 읽어 주는 음성 파일(WAV/MP3)을 만들어 주는 프로그램입니다.
6~30초 분량의 내 목소리 녹음만 있으면 별도 학습 없이 바로 사용할 수 있습니다
(오픈소스 보이스 클로닝 모델 [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) 사용, 한국어 지원).

- 명령줄(CLI)과 브라우저 웹 화면 둘 다 제공
- 대본을 문장 단위로 자동 분할 → 긴 대본도 끝까지 또렷하게
- 쉼 조절: 빈 줄 = 문단 쉼, `[쉼 1.5]` = 원하는 만큼 쉼
- **SRT 자막(타이밍) 파일**, **문장별 개별 오디오** 출력 → 영상 더빙 편집에 바로 활용
- 목소리 프로필 저장(`.pt`) → 다음부터는 샘플 없이 바로 생성

## 1. 설치

Python 3.10 ~ 3.14, [ffmpeg](https://ffmpeg.org/) (mp3/m4a 입출력 시 필요)

```bash
git clone <이 저장소> && cd voicedub
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[ui]"             # 웹 화면이 필요 없으면 pip install -e .
```

> NVIDIA GPU가 있으면 [PyTorch 공식 안내](https://pytorch.org/get-started/locally/)대로 CUDA 버전 torch를 먼저 설치하세요.
> CPU로도 동작하지만 문장당 수 초~수십 초가 걸립니다. (Apple Silicon은 `--device mps` 시도 가능)

처음 실행 시 모델(약 1.8GB)을 자동으로 내려받으며, **Coqui Public Model License(비상업적 이용)** 동의를 묻습니다.
미리 동의하려면 환경변수 `COQUI_TOS_AGREED=1` 을 설정하세요. 상업적 용도로 쓰려면 라이선스를 별도로 확인해야 합니다.

## 2. 내 목소리 녹음 팁

- 조용한 방에서, 마이크와 일정한 거리를 두고 **평소 말투로** 6~30초 읽기
- 배경음악·잡음·에코가 없을수록 결과가 좋아집니다
- 휴대폰 녹음(m4a)도 OK. 여러 파일을 주면 함께 분석해서 더 안정적입니다
- 원하는 톤(차분한 내레이션, 밝은 진행 등)으로 녹음하면 그 톤을 따라갑니다

## 3. 사용법

### 웹 화면 (가장 쉬움)

```bash
voicedub ui
```
브라우저에서 http://127.0.0.1:7860 접속 → 목소리 업로드/녹음 → 대본 입력 → **음성 만들기**.
결과 WAV와 SRT 자막을 내려받을 수 있습니다.

### 명령줄

```bash
# 대본이 어떻게 나뉘어 읽힐지 미리 확인 (모델 불필요, 즉시 실행)
voicedub preview examples/sample_script.txt

# 바로 생성
voicedub speak examples/sample_script.txt --voice 내목소리.m4a -o 결과.mp3 --srt

# 목소리를 프로필로 한 번 등록해 두고 반복 사용
voicedub enroll 샘플1.wav 샘플2.wav -o voices/me.pt
voicedub speak 대본.txt --profile voices/me.pt -o 결과.wav --srt --clips 문장별/
```

| 옵션 | 설명 |
|---|---|
| `-v, --voice` | 목소리 샘플 (여러 번 지정 가능) |
| `-p, --profile` | `enroll`로 만든 프로필 |
| `-o, --output` | 출력 파일 (`.wav` `.mp3` `.m4a` `.flac`) |
| `--srt` | 문장별 타이밍 자막 파일 생성 |
| `--clips DIR` | 문장별 개별 WAV 저장 (001.wav, 002.wav…) |
| `--lang` | 언어 (기본 `ko`; en, ja, zh-cn 등 17개 언어) |
| `--speed` | 말하기 속도 (기본 1.0) |
| `--temperature` | 낮을수록 안정적, 높을수록 표현이 풍부 (기본 0.65) |
| `--sentence-pause` / `--paragraph-pause` | 문장/문단 사이 쉼(초) |
| `--device` | `auto` `cuda` `mps` `cpu` |

## 4. 대본 작성법

```text
# '#'으로 시작하는 줄은 읽지 않습니다 (메모용)
안녕하세요. 문장부호(. ! ? …)를 기준으로 문장이 나뉩니다.
줄바꿈도 문장 구분으로 처리됩니다.

빈 줄을 넣으면 문단이 바뀌며 더 길게 쉽니다.
여기서 잠깐 [쉼 2] 2초 쉬고 이어 읽습니다.   ← [pause 2] 도 가능
```

- 숫자·영어 약어는 읽는 방식이 어색할 수 있으니 **읽는 그대로 한글로** 적는 것이 가장 정확합니다 (예: `2024년` → `이천이십사 년`, `AI` → `에이아이`).
- 마음에 안 드는 문장이 있으면 같은 명령을 다시 실행하거나 `--temperature`를 낮춰 보세요(매번 약간씩 다르게 생성됩니다).

## 5. 주의

본인의 목소리 또는 **명시적 동의를 받은 사람의 목소리**만 사용하세요.
타인의 목소리를 무단으로 복제하거나 사칭·사기에 이용하는 것은 법적 책임이 따를 수 있습니다.

## 개발

```bash
pip install -e ".[dev]"
pytest
```
테스트는 가짜 엔진을 사용하므로 모델 다운로드 없이 실행됩니다.
