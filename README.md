# ser

여러 사전학습 음성 감정 모델을 같은 데이터셋 구조에서 비교 실행하기 위한 벤치마크 스크립트 모음입니다.

현재 포함된 스크립트:

- `benchmark_emotion2vec.py`: `emotion2vec_plus_seed/base/large` 3개 모델 비교
- `benchmark_audeering_msp_dim.py`: audEERING MSP-DIM 회귀 모델 실행
- `benchmark_speechbrain_iemocap.py`: SpeechBrain IEMOCAP 4클래스 분류 모델 실행

## 설치

```bash
uv sync
```

## 데이터 구조

스크립트는 아래 구조를 가정합니다.

```text
data/
  wav/
    1.기쁨/
    2.슬픔/
    3.분노/
    4.불안/
    5.상처/
    6.당황/
    7.중립/
  label/
    1.기쁨/
    2.슬픔/
    3.분노/
    4.불안/
    5.상처/
    6.당황/
    7.중립/
```

각 `wav` 파일에는 같은 상대 경로의 `json` 라벨 파일이 있어야 하며, 라벨에서 `화자정보.Emotion`과 `화자정보.Sensitivity`를 읽습니다.

## 실행 예시

emotion2vec 3개 모델 비교:

```bash
uv run python benchmark_emotion2vec.py --samples-per-emotion 50 --results-dir results_50
```

audEERING MSP-DIM:

```bash
uv run python benchmark_audeering_msp_dim.py --samples-per-emotion 10 --results-dir results_audeering_10
```

SpeechBrain IEMOCAP:

```bash
uv run python benchmark_speechbrain_iemocap.py --samples-per-emotion 10 --results-dir results_speechbrain_10
```

전체 데이터를 쓰고 싶으면 `--samples-per-emotion`을 생략하거나 `0`으로 주면 됩니다.

## 결과물

각 스크립트는 지정한 결과 디렉터리에 아래 파일들을 생성합니다.

- 샘플별 예측 CSV
- `summary_overall.csv`
- `summary_by_emotion.csv`
- `meta.json`

## 모델별 주의사항

`benchmark_emotion2vec.py`

- 3개 emotion2vec 분류 모델을 한 번에 돌립니다.
- 데이터셋 감정 라벨과 모델 라벨 간 매핑을 사용해 정확도를 계산합니다.

`benchmark_audeering_msp_dim.py`

- 이 모델은 분류가 아니라 `arousal`, `dominance`, `valence` 연속값을 출력합니다.
- 그래서 정확도 대신 차원값 통계가 `summary_overall.csv`와 `summary_by_emotion.csv`에 저장됩니다.

`benchmark_speechbrain_iemocap.py`

- 이 모델은 `neu`, `ang`, `hap`, `sad` 4개 클래스만 지원합니다.
- 따라서 데이터셋의 `Happy`, `Sad`, `Angry`, `Neutrality`만 직접 평가하고, `Anxious`, `Hurt`, `Embarrassed`는 `unsupported`로 기록합니다.

## Git 관리 기준

이 저장소에서는 아래 항목을 기본적으로 커밋하지 않도록 `.gitignore`에 넣었습니다.

- 로컬 데이터셋 (`data/`, `example/`)
- 실행 결과 (`results*/`, `ser_*_smoke/`)
- 로컬 가상환경과 캐시 (`.venv/`, `wav2vec2_checkpoints/`, `__pycache__/`)
- 로컬 도구 파일 (`.codex`)

벤치마크 결과를 같이 공유하고 싶다면 별도 아카이브나 릴리스 산출물로 다루는 편이 안전합니다.
