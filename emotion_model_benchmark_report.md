# 음성 감정 분석 모델 테스트 결과 보고서

말씀하신 음성 감정 분석 모델 테스트를 완료하여 결과 공유드립니다.

이번 보고서는 `results_emotion2vec`, `results_audeering`, `results_speechbrain` 폴더의 결과를 기준으로 작성했습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. 테스트 개요

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| 항목 | 내용 |
|---|---|
| 데이터 | AI Hub "감성 및 발화 스타일별 음성합성 데이터" Validation 셋 |
| 샘플 | 감정별 50개씩 총 350개 |
| 샘플 목록 | `results_0408/plus_base.csv`의 `wav_relpath` 순서를 재사용 |
| 테스트 대상 1 | emotion2vec 계열 3종: `plus_large`, `plus_base`, `plus_seed` |
| 테스트 대상 2 | SpeechBrain IEMOCAP 감정 분류 모델 |
| 테스트 대상 3 | audEERING MSP-DIM 차원형 감정 회귀 모델 |
| 결과 폴더 | `results_emotion2vec`, `results_speechbrain`, `results_audeering` |

이번 테스트는 동일한 350개 음성 샘플을 사용했습니다. 따라서 모델 간 차이는 샘플링 차이가 아니라 모델 구조, 학습 데이터, 출력 레이블 체계의 차이로 해석하는 것이 적절합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 2. 데이터셋 감정 레이블

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

데이터셋은 한국어 감정 기반의 7개 감정 레이블을 사용합니다.

| 번호 | 감정 | 영어 레이블 | 세부 감성 예시 |
|---:|---|---|---|
| 1 | 기쁨 | Happy | 즐겁다, 반갑다, 자유롭다 등 |
| 2 | 슬픔 | Sad | 울적하다, 서럽다, 처량하다 등 |
| 3 | 분노 | Angry | 화나다, 짜증나다 등 |
| 4 | 불안 | Anxious | 걱정되다, 초조하다, 두렵다 등 |
| 5 | 상처 | Hurt | 배신감, 억울함 등 |
| 6 | 당황 | Embarrassed | 놀라다, 당황하다, 난감하다 등 |
| 7 | 중립 | Neutrality | 담백하다, 뚜렷하다 등 |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 3. 모델별 출력 체계 및 매핑

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 3.1 emotion2vec

emotion2vec는 감정을 범주형 레이블로 예측합니다. 본 테스트에서 사용한 모델 출력 레이블은 아래와 같습니다.

| 모델 출력 | 한국어 의미 | 비고 |
|---|---|---|
| happy | 행복 |
| sad | 슬픔 |
| angry | 분노 |
| fearful | 공포/두려움 |
| surprised | 놀람 |
| neutral | 중립 |
| disgusted | 혐오 | 데이터셋에는 직접 대응 감정 없음 |

평가를 위해 아래와 같이 매핑했습니다.

| 데이터셋 레이블 | 모델 레이블 | 매핑 유형 | 비고 |
|---|---|---|---|
| 기쁨(Happy) | happy | 직접 대응 | 동일 개념 |
| 슬픔(Sad) | sad | 직접 대응 | 동일 개념 |
| 분노(Angry) | angry | 직접 대응 | 동일 개념 |
| 중립(Neutrality) | neutral | 직접 대응 | 동일 개념 |
| 불안(Anxious) | fearful | 근사 매핑 | 불안은 걱정/초조를 포함하나 fearful은 공포/두려움에 가까움 |
| 상처(Hurt) | sad | 근사 매핑 | 상처는 슬픔과 분노가 섞인 복합 감정 |
| 당황(Embarrassed) | surprised | 근사 매핑 | 당황을 놀람으로 근사 |
| 해당 없음 | disgusted | 미사용 | 데이터셋에 대응 감정 없음 |

### 3.2 SpeechBrain IEMOCAP

SpeechBrain IEMOCAP 모델은 IEMOCAP 데이터셋 기준의 4개 감정 레이블을 출력합니다.

| 모델 출력 | 한국어 의미 | 평가 사용 여부 |
|---|---|---|
| hap | 기쁨 | 사용 |
| sad | 슬픔 | 사용 |
| ang | 분노 | 사용 |
| neu | 중립 | 사용 |

데이터셋의 7개 감정 중 `기쁨`, `슬픔`, `분노`, `중립`만 직접 평가할 수 있습니다. `불안`, `상처`, `당황`은 모델 출력 체계에 해당 레이블이 없어 정확도 계산에서 제외했습니다.

### 3.3 audEERING MSP-DIM

audEERING MSP-DIM 모델은 감정을 특정 레이블로 분류하지 않고, 3개의 연속값으로 예측합니다.

| 차원 | 의미 | 쉽게 말하면 |
|---|---|---|
| arousal | 각성도 | 목소리가 얼마나 에너지 있고 흥분되어 들리는가 |
| dominance | 지배감/통제감 | 말하는 사람이 얼마나 자신감 있고 주도적으로 들리는가 |
| valence | 긍정도 | 감정이 얼마나 긍정적으로 들리는가 |

따라서 audEERING 모델은 `기쁨`, `슬픔` 같은 정답 레이블과 바로 비교하는 정확도 모델이 아닙니다. 대신 감정별 평균 차원값을 비교하여 각 감정이 어떤 음성적 성격으로 나타나는지 분석하는 데 적합합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 4. 전체 결과 요약

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 4.1 범주형 감정 분류 모델

| 모델 | 평가 방식 | 정확도 | 정답/평가수 | 추론 시간 |
|---|---|---:|---:|---:|
| emotion2vec plus_large | 7개 감정 매핑 | 46.29% | 162/350 | 50.7초 |
| emotion2vec plus_base | 7개 감정 매핑 | 40.29% | 141/350 | 82.0초 |
| emotion2vec plus_seed | 7개 감정 매핑 | 37.43% | 131/350 | 86.3초 |
| SpeechBrain IEMOCAP | 직접 대응 4개 감정만 평가 | 30.00% | 60/200 | 479.4초 |

주의: SpeechBrain 모델은 `불안`, `상처`, `당황`을 출력할 수 없기 때문에 전체 350건 중 직접 대응 가능한 200건만 정확도 평가에 사용했습니다.

### 4.2 차원형 감정 회귀 모델

| 모델 | 평가 방식 | 샘플 수 | arousal 평균 | dominance 평균 | valence 평균 | 추론 시간 |
|---|---|---:|---:|---:|---:|---:|
| audEERING MSP-DIM | 3차원 감정 점수 | 350 | 0.4890 | 0.5436 | 0.4565 | 509.8초 |

audEERING 모델은 정확도보다 감정의 강도/긍정도/통제감 분포를 보는 모델입니다. 따라서 emotion2vec, SpeechBrain과 같은 방식의 정확도 비교 대상은 아닙니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 5. emotion2vec 상세 결과

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| 감정 | 매핑 대상 | 매핑 유형 | plus_large | plus_base | plus_seed |
|---|---|---|---:|---:|---:|
| 기쁨(Happy) | happy | 직접 | 56.00% (28/50) | 28.00% (14/50) | 54.00% (27/50) |
| 슬픔(Sad) | sad | 직접 | 80.00% (40/50) | 60.00% (30/50) | 38.00% (19/50) |
| 분노(Angry) | angry | 직접 | 48.00% (24/50) | 30.00% (15/50) | 26.00% (13/50) |
| 불안(Anxious) | fearful | 근사 | 10.00% (5/50) | 8.00% (4/50) | 18.00% (9/50) |
| 상처(Hurt) | sad | 근사 | 40.00% (20/50) | 24.00% (12/50) | 22.00% (11/50) |
| 당황(Embarrassed) | surprised | 근사 | 62.00% (31/50) | 74.00% (37/50) | 62.00% (31/50) |
| 중립(Neutrality) | neutral | 직접 | 28.00% (14/50) | 58.00% (29/50) | 42.00% (21/50) |

plus_large 기준:

| 구분 | 평균 정확도 |
|---|---:|
| 직접 대응 감정 평균 | 53.00% |
| 근사 매핑 감정 평균 | 37.33% |

emotion2vec 예측 분포는 아래와 같습니다.

| 모델 | 가장 많이 나온 예측 | 비중 |
|---|---|---:|
| plus_large | surprised | 28.86% (101/350) |
| plus_base | surprised | 35.71% (125/350) |
| plus_seed | surprised | 30.57% (107/350) |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 6. SpeechBrain IEMOCAP 상세 결과

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| 감정 | 매핑 대상 | 평가 유형 | 정확도 |
|---|---|---|---:|
| 기쁨(Happy) | hap | 직접 | 62.00% (31/50) |
| 슬픔(Sad) | sad | 직접 | 0.00% (0/50) |
| 분노(Angry) | ang | 직접 | 28.00% (14/50) |
| 불안(Anxious) | 해당 없음 | 미지원 | 미평가 |
| 상처(Hurt) | 해당 없음 | 미지원 | 미평가 |
| 당황(Embarrassed) | 해당 없음 | 미지원 | 미평가 |
| 중립(Neutrality) | neu | 직접 | 30.00% (15/50) |

SpeechBrain IEMOCAP 모델의 예측 분포는 아래와 같습니다.

| 예측 레이블 | 개수 | 비중 |
|---|---:|---:|
| hap | 241 | 68.86% |
| ang | 70 | 20.00% |
| neu | 39 | 11.14% |
| sad | 0 | 0.00% |

이 모델은 이번 한국어 데이터에서 `hap`으로 예측이 크게 쏠렸고, `sad`는 한 건도 예측하지 않았습니다. 따라서 현재 데이터셋에 그대로 적용하기에는 한계가 큽니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 7. audEERING MSP-DIM 상세 결과

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

audEERING 모델은 정확도 대신 감정별 평균 차원값을 확인했습니다.

| 감정 | arousal 평균 | dominance 평균 | valence 평균 |
|---|---:|---:|---:|
| 기쁨(Happy) | 0.6301 | 0.6374 | 0.5267 |
| 슬픔(Sad) | 0.3317 | 0.4233 | 0.4033 |
| 분노(Angry) | 0.6000 | 0.6354 | 0.4624 |
| 불안(Anxious) | 0.4684 | 0.5239 | 0.4264 |
| 상처(Hurt) | 0.4196 | 0.4997 | 0.4305 |
| 당황(Embarrassed) | 0.5546 | 0.5803 | 0.4804 |
| 중립(Neutrality) | 0.4186 | 0.5051 | 0.4659 |

해석:

- 기쁨과 분노는 arousal이 높게 나타났습니다. 두 감정 모두 음성 에너지가 비교적 크게 잡힌 것으로 볼 수 있습니다.
- 슬픔은 arousal, dominance, valence가 모두 낮은 편입니다. 낮은 에너지와 낮은 긍정도로 표현된 것으로 해석됩니다.
- 당황은 arousal과 dominance가 중간 이상으로 나타나, 놀람/긴장감이 섞인 음성 특성이 반영된 것으로 보입니다.
- 중립은 완전히 낮은 값으로만 나타나지는 않았고, dominance와 valence가 중간 수준에 머물렀습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 8. 모델별 특징, 사용 용도, 아키텍처 차이

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 8.1 emotion2vec

| 항목 | 내용 |
|---|---|
| 모델 성격 | 범용 음성 감정 표현(representation)을 학습한 뒤, 감정 분류에 활용하는 모델 |
| 기반 구조 | CNN feature extractor + Transformer backbone |
| 사전학습 방식 | self-supervised online distillation |
| 학습 목표 | utterance-level loss + frame-level loss를 함께 사용 |
| 후속 모델 | emotion2vec+ seed/base/large는 emotion2vec 기반 SER 모델로, 공개 모델 카드 기준 각각 201h/4,788h/42,526h 규모 데이터로 fine-tuning |
| 출력 | happy, sad, angry, fearful, surprised, neutral, disgusted |
| 적합한 용도 | 감정 레이블 자동 분류, 감정 임베딩 추출, downstream 감정 태스크의 feature extractor |
| 이번 테스트 결과 | plus_large가 46.29%로 가장 높음 |

emotion2vec의 핵심은 "감정 이름을 바로 외우는 모델"이라기보다, 음성 안에 있는 감정 표현을 잘 담는 표현 공간을 먼저 만드는 데 있습니다. 논문에서는 emotion2vec를 self-supervised online distillation 방식으로 학습한다고 설명합니다. 여기서 self-supervised는 사람이 모든 음성에 정답 감정을 붙이지 않아도 학습한다는 뜻이고, online distillation은 teacher-student 구조가 학습 중 함께 갱신되면서 student가 teacher의 표현을 따라가도록 학습하는 방식입니다.

일반적인 knowledge distillation은 이미 학습된 큰 teacher 모델의 지식을 작은 student 모델로 옮기는 "압축" 용도로 많이 쓰입니다. emotion2vec의 distillation은 그와 조금 다릅니다. 별도의 고정된 teacher를 먼저 만들어두는 방식이 아니라, teacher와 student가 같은 구조를 가지고 학습 과정에서 같이 움직이며 bootstrap 형태로 표현을 개선합니다. 그래서 "knowledge distillation을 쓴다"는 표현은 넓은 의미에서는 맞지만, 정확히는 "self-supervised online distillation 기반 사전학습"이라고 쓰는 것이 더 안전합니다.

학습 흐름을 단순화하면 다음과 같습니다.

1. 원본 음성을 feature extractor가 짧은 음향 단위의 특징으로 변환합니다.
2. Transformer backbone이 시간 흐름에 따른 음성 표현을 학습합니다.
3. teacher network는 비교적 안정적인 목표 표현을 만들고, student network는 일부 정보가 가려진 입력에서도 그 표현을 맞히도록 학습합니다.
4. utterance-level loss는 발화 전체의 감정 분위기를 잡도록 하고, frame-level loss는 음성의 짧은 구간마다 나타나는 감정 단서를 잡도록 돕습니다.
5. 이후 emotion2vec+ 모델은 이 표현을 기반으로 실제 감정 분류 태스크에 맞게 fine-tuning됩니다.

비전공자 관점에서는, emotion2vec는 먼저 "목소리에서 감정이 드러나는 패턴"을 폭넓게 배우고, 그다음 "이 패턴은 happy/sad/angry 중 무엇인가"를 붙이는 방식에 가깝습니다. 이번 테스트에서 plus_large가 가장 좋은 성능을 보인 것은 더 큰 모델과 더 많은 fine-tuning 데이터가 감정 표현을 더 잘 포착했기 때문으로 해석할 수 있습니다.

### 8.2 SpeechBrain IEMOCAP

| 항목 | 내용 |
|---|---|
| 모델 성격 | Wav2Vec2 base를 IEMOCAP 감정 분류에 fine-tuning한 모델 |
| 기반 구조 | Wav2Vec2 encoder + pooling/classification head |
| 학습 데이터 | IEMOCAP training data |
| 학습 방식 | 16kHz 음성을 입력으로 받아 wav2vec2 표현을 추출하고, 감정 레이블 분류 손실로 fine-tuning |
| 출력 | hap, sad, ang, neu |
| 장점 | IEMOCAP과 유사한 조건의 4개 감정 분류에는 사용하기 쉬움 |
| 한계 | 출력 감정 수가 4개로 제한됨. `불안`, `상처`, `당황` 평가 불가 |
| 적합한 용도 | IEMOCAP과 유사한 영어 음성 감정 분류, 4개 감정 중심의 간단한 분류 |
| 이번 테스트 결과 | 직접 대응 가능한 200건 기준 30.00% |

Wav2Vec2는 원래 음성 인식을 위해 많이 쓰이는 self-supervised speech encoder입니다. 원본 파형을 convolutional feature encoder로 처리한 뒤, Transformer가 긴 시간 문맥을 반영한 음성 표현을 만듭니다. SpeechBrain IEMOCAP 모델은 이 wav2vec2 표현 위에 감정 분류용 head를 붙이고, IEMOCAP의 감정 레이블에 맞게 fine-tuning한 모델입니다.

학습 흐름을 단순화하면 다음과 같습니다.

1. 16kHz 단일 채널 음성이 입력됩니다.
2. Wav2Vec2가 음성의 발음, 억양, 리듬, 음색 정보를 포함한 고차원 표현을 만듭니다.
3. pooling 단계에서 시간축 전체의 정보를 하나의 발화 단위 표현으로 요약합니다.
4. classification head가 `hap`, `sad`, `ang`, `neu` 중 하나로 분류합니다.
5. 학습 중에는 정답 감정 레이블과 예측값의 차이가 줄어들도록 모델이 조정됩니다.

이 모델의 강점은 구조가 명확하고 IEMOCAP 같은 4감정 영어 데이터셋에서는 성능이 검증되어 있다는 점입니다. 다만 이번 테스트 데이터는 한국어 7감정이고, `불안`, `상처`, `당황`처럼 모델 출력에 없는 감정이 포함되어 있습니다. 따라서 모델의 아키텍처가 나쁘다기보다, 학습 데이터와 출력 체계가 이번 문제와 잘 맞지 않는 것이 핵심 한계입니다.

### 8.3 audEERING MSP-DIM

| 항목 | 내용 |
|---|---|
| 모델 성격 | Wav2Vec2 기반의 차원형 감정 회귀 모델 |
| 기반 구조 | Wav2Vec2-Large-Robust 기반, Transformer layer를 24개에서 12개로 줄인 뒤 fine-tuning |
| 학습 데이터 | MSP-Podcast v1.7 |
| 학습 방식 | 발화 전체 hidden state를 pooling한 뒤 arousal/dominance/valence 3개 값을 회귀 |
| 출력 | arousal, dominance, valence |
| 장점 | 감정을 하나의 이름으로 강제하지 않고 연속적인 감정 상태로 표현 |
| 한계 | `기쁨`, `슬픔` 같은 레이블 정확도를 바로 계산하기 어려움 |
| 적합한 용도 | 감정 강도, 긍정도, 에너지 변화 분석, 음성 UX/상담/콘텐츠 분석 |
| 이번 테스트 결과 | 감정별 차원 평균 분석에 적합 |

audEERING 모델도 Wav2Vec2 계열을 기반으로 하지만, emotion2vec나 SpeechBrain처럼 감정 이름을 고르는 방식이 아닙니다. 모델은 음성을 듣고 arousal, dominance, valence 세 값을 예측합니다. 이는 감정을 범주가 아니라 연속적인 좌표로 보는 접근입니다.

학습 흐름을 단순화하면 다음과 같습니다.

1. 원본 음성이 Wav2Vec2 processor를 거쳐 모델 입력으로 정규화됩니다.
2. Wav2Vec2-Large-Robust encoder가 프레임별 음성 hidden state를 생성합니다.
3. 마지막 Transformer layer의 hidden state들을 평균 pooling하여 발화 전체 표현으로 만듭니다.
4. regression head가 이 표현을 받아 arousal, dominance, valence 세 값을 예측합니다.
5. 학습 중에는 실제 사람이 평가한 차원 점수와 모델 예측값의 차이가 줄어들도록 조정됩니다.

이 접근의 장점은 복합 감정을 다루기 쉽다는 점입니다. 예를 들어 `상처`는 슬픔과 분노가 섞인 감정이라 하나의 레이블로 고르기 어렵지만, arousal은 중간, valence는 낮음처럼 수치 조합으로 표현할 수 있습니다. 반대로 단점은 "정답 감정이 맞았는가?"를 바로 계산하기 어렵다는 점입니다.

### 8.4 세 모델의 핵심 차이

| 구분 | emotion2vec | SpeechBrain IEMOCAP | audEERING MSP-DIM |
|---|---|---|---|
| 모델 계열 | 감정 표현 사전학습 + 감정 분류 | Wav2Vec2 fine-tuning 분류 | Wav2Vec2 fine-tuning 회귀 |
| 주요 학습 아이디어 | self-supervised online distillation | labeled IEMOCAP 감정 분류 fine-tuning | MSP-Podcast 차원 점수 회귀 fine-tuning |
| 아키텍처 핵심 | CNN feature extractor + Transformer backbone + 분류 head | Wav2Vec2 encoder + pooling/classification head | Wav2Vec2-Large-Robust encoder + pooling/regression head |
| 출력 방식 | 감정 레이블 | 감정 레이블 | 감정 차원 점수 |
| 출력 개수 | 7개 감정 | 4개 감정 | 3개 연속값 |
| 평가 방식 | 매핑 후 정확도 | 직접 대응 감정만 정확도 | 평균/분포 분석 |
| 한국어 7감정 적용성 | 보통 | 낮음 | 직접 분류는 불가, 분석용 가능 |
| 설명 용이성 | 높음 | 높음 | 중간 |
| 추천 용도 | 감정 자동 분류/임베딩 추출 | 제한된 4감정 분류 | 감정 상태/강도 분석 |

정리하면, emotion2vec는 감정 표현 자체를 잘 배우는 데 초점을 둔 모델이고, SpeechBrain IEMOCAP은 특정 데이터셋의 감정 이름을 맞히도록 fine-tuning된 모델이며, audEERING MSP-DIM은 감정 이름 대신 감정의 성질을 숫자로 예측하는 모델입니다. 같은 "음성 감정 분석"이라는 이름 아래에 있지만, 실제 학습 목표와 출력 형식은 꽤 다릅니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 9. 주요 발견사항

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 범주형 분류 기준으로는 emotion2vec plus_large가 가장 높은 성능을 보였습니다.

   plus_large는 전체 350건 기준 46.29%의 정확도를 보였고, 특히 슬픔에서 80.00%, 당황에서 62.00%를 기록했습니다.

2. emotion2vec는 직접 대응 감정과 근사 매핑 감정의 차이가 큽니다.

   plus_large 기준 직접 대응 감정 평균은 53.00%, 근사 매핑 감정 평균은 37.33%였습니다. 이는 모델 자체의 성능 문제뿐 아니라, 한국어 데이터셋의 감정 체계와 모델 출력 레이블 체계가 다르기 때문입니다.

3. 불안 감정은 emotion2vec에서도 낮게 나타났습니다.

   불안은 fearful로 근사 매핑했지만 plus_large 기준 10.00%에 그쳤습니다. 한국어의 불안은 걱정, 초조, 긴장 등을 포함하는 넓은 개념이고, 모델의 fearful은 공포/두려움에 가까운 좁은 개념이라 매핑 한계가 큽니다.

4. SpeechBrain IEMOCAP은 이번 한국어 7감정 데이터셋에 그대로 쓰기 어렵습니다.

   이 모델은 4개 감정만 출력하고, 이번 테스트에서는 `hap` 예측이 68.86%로 크게 쏠렸습니다. 또한 슬픔을 한 건도 `sad`로 예측하지 않아 실사용 전 재학습 또는 데이터셋 적합성 검토가 필요합니다.

5. audEERING MSP-DIM은 분류 모델이 아니라 감정 상태 분석 모델로 보는 것이 적절합니다.

   예를 들어 기쁨과 분노는 arousal이 높고, 슬픔은 arousal과 valence가 낮게 나타났습니다. 이런 결과는 "정답/오답"보다는 음성의 감정적 성향을 수치화하는 데 의미가 있습니다.

6. 동일 샘플을 사용했기 때문에 이번 결과는 모델 간 비교가 가능합니다.

   모든 모델은 `results_0408/plus_base.csv`의 샘플 목록을 재사용했으며, 감정별 50개씩 총 350개를 대상으로 평가했습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 10. 비전공자용 설명

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

이번 테스트는 쉽게 말해, 같은 음성 350개를 세 종류의 감정 분석 도구에 들려주고 각각이 어떻게 판단하는지 비교한 것입니다.

### 10.1 세 모델은 서로 보는 방식이 다릅니다

emotion2vec와 SpeechBrain은 음성을 듣고 "이건 기쁨", "이건 슬픔"처럼 감정 이름을 고르는 모델입니다. 반면 audEERING은 감정 이름을 고르지 않고, "얼마나 에너지가 있는지", "얼마나 긍정적인지", "얼마나 자신감 있게 들리는지"를 점수로 알려주는 모델입니다.

비유하면 다음과 같습니다.

| 모델 | 비유 |
|---|---|
| emotion2vec | 사진을 보고 "고양이/강아지/새"처럼 이름표를 붙이는 모델 |
| SpeechBrain | 선택지가 4개뿐인 감정 이름표 모델 |
| audEERING | 이름표 대신 "밝기/강도/분위기" 같은 수치를 매기는 모델 |

### 10.2 왜 정확도가 아주 높지 않은가?

가장 큰 이유는 모델이 사용하는 감정 이름과 한국어 데이터셋의 감정 이름이 정확히 같지 않기 때문입니다.

예를 들어 한국어 데이터셋의 `불안`은 걱정, 초조, 두려움 같은 여러 느낌을 포함합니다. 그런데 emotion2vec의 `fearful`은 공포에 더 가깝습니다. 그래서 사람이 보기에는 불안한 음성이라도 모델은 놀람이나 중립으로 판단할 수 있습니다.

또 `상처`는 슬픔과 분노가 섞인 복합 감정입니다. 하지만 모델은 이런 복합 감정을 하나의 레이블로 갖고 있지 않습니다. 그래서 `sad`로 근사해서 평가했지만, 이 평가 자체에 한계가 있습니다.

### 10.3 어떤 모델을 쓰는 것이 좋은가?

목적에 따라 다릅니다.

| 목적 | 추천 |
|---|---|
| 감정 이름을 바로 붙이고 싶다 | emotion2vec plus_large |
| 기쁨/슬픔/분노/중립 4개만 간단히 보고 싶다 | SpeechBrain은 가능하지만 이번 한국어 데이터에는 부적합 |
| 감정의 강도나 분위기를 수치로 보고 싶다 | audEERING MSP-DIM |
| 한국어 7감정에 맞춘 실사용 모델이 필요하다 | 한국어 데이터로 추가 학습/보정 필요 |

현재 결과만 놓고 보면, 한국어 7감정 분류에는 emotion2vec plus_large가 가장 현실적인 출발점입니다. 다만 실사용 수준으로 안정화하려면 한국어 감정 데이터로 추가 학습하거나, 모델 출력 레이블을 한국어 7감정 체계에 맞게 보정하는 과정이 필요합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 11. 첨부 결과 파일 설명

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| 폴더 | 파일 | 설명 |
|---|---|---|
| `results_emotion2vec` | `plus_large.csv`, `plus_base.csv`, `plus_seed.csv` | emotion2vec 모델별 예측 결과 |
| `results_emotion2vec` | `summary_overall.csv` | emotion2vec 전체 정확도 요약 |
| `results_emotion2vec` | `summary_by_emotion.csv` | emotion2vec 감정별 정확도 요약 |
| `results_emotion2vec` | `meta.json` | 샘플 수, 모델 정보, 예측 분포, 평가 메타 정보 |
| `results_speechbrain` | `speechbrain_iemocap.csv` | SpeechBrain 모델 예측 결과 |
| `results_speechbrain` | `summary_overall.csv` | SpeechBrain 전체 정확도 요약 |
| `results_speechbrain` | `summary_by_emotion.csv` | SpeechBrain 감정별 정확도 요약 |
| `results_speechbrain` | `meta.json` | 샘플 수, 모델 정보, 예측 분포, 평가 메타 정보 |
| `results_audeering` | `audeering_msp_dim.csv` | audEERING 모델의 3차원 감정 점수 |
| `results_audeering` | `summary_overall.csv` | audEERING 전체 차원 통계 |
| `results_audeering` | `summary_by_emotion.csv` | audEERING 감정별 차원 통계 |
| `results_audeering` | `meta.json` | 샘플 수, 모델 정보, 차원별 통계 |

CSV 주요 컬럼 설명:

| 컬럼 | 설명 |
|---|---|
| `wav_relpath` | 원본 음성 파일의 상대경로 |
| `gt_emotion` | 데이터셋의 정답 감정 레이블 |
| `gt_sensitivity` | 데이터셋의 세부 감성 태그 |
| `pred` | 분류 모델이 예측한 감정 레이블 |
| `score` | 분류 모델의 예측 신뢰도 |
| `arousal` | audEERING의 각성도 점수 |
| `dominance` | audEERING의 지배감/통제감 점수 |
| `valence` | audEERING의 긍정도 점수 |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 12. 종합 의견

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

이번 테스트에서 한국어 7감정 데이터셋에 가장 직접적으로 적용 가능한 모델은 emotion2vec plus_large였습니다. 다만 전체 정확도는 46.29%로, 즉시 실사용하기에는 충분히 높다고 보기 어렵습니다. 특히 불안, 상처, 당황처럼 한국어 감정 체계에서 복합적이거나 세밀한 감정은 모델 출력 레이블과 정확히 맞지 않아 성능 해석에 주의가 필요합니다.

SpeechBrain IEMOCAP은 모델 출력 체계가 4개 감정으로 제한되어 이번 데이터셋과 맞지 않았고, 예측도 `hap`에 크게 쏠렸습니다. 반면 audEERING MSP-DIM은 감정 이름을 맞히는 모델은 아니지만, 음성의 에너지와 긍정도 같은 감정적 특성을 수치화하는 데 유용했습니다.

따라서 향후 방향은 다음과 같습니다.

1. 한국어 7감정 자동 분류가 목적이면 emotion2vec plus_large를 기준 모델로 두고 한국어 데이터 기반 보정 또는 추가 학습을 검토합니다.
2. 감정 이름보다 음성의 정서적 상태를 분석하려면 audEERING MSP-DIM의 arousal/dominance/valence 값을 함께 활용합니다.
3. SpeechBrain IEMOCAP은 현재 형태로는 이번 한국어 7감정 분류 목적에 적합하지 않으므로 참고 모델 수준으로 보는 것이 적절합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 13. 참고 출처

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

모델 구조와 학습 방식 설명은 아래 논문 및 공개 모델 카드를 기준으로 정리했습니다.

### 13.1 논문 Reference

| 구분 | 논문 |
|---|---|
| emotion2vec | Ma, Z., Zheng, Z., Ye, J., Li, J., Gao, Z., Zhang, S., & Chen, X. (2024). [emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation](https://aclanthology.org/2024.findings-acl.931/). Findings of the Association for Computational Linguistics: ACL 2024, 15747-15760. |
| SpeechBrain toolkit | Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., et al. (2021). [SpeechBrain: A General-Purpose Speech Toolkit](https://arxiv.org/abs/2106.04624). arXiv:2106.04624. |
| audEERING MSP-DIM / dimensional SER | Wagner, J., Triantafyllopoulos, A., Wierstorf, H., Schmitt, M., Burkhardt, F., Eyben, F., & Schuller, B. W. (2022). [Dawn of the transformer era in speech emotion recognition: closing the valence gap](https://arxiv.org/abs/2203.07378). arXiv:2203.07378. |
| Wav2Vec2 기반 구조 | Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477). Advances in Neural Information Processing Systems 33. |

### 13.2 모델 카드 및 구현 출처

| 모델 | 참고 자료 |
|---|---|
| emotion2vec | [emotion2vec 논문: Self-Supervised Pre-Training for Speech Emotion Representation](https://aclanthology.org/2024.findings-acl.931.pdf), [emotion2vec GitHub](https://github.com/ddlBoJack/emotion2vec), [Hugging Face paper page](https://huggingface.co/papers/2312.15185) |
| SpeechBrain IEMOCAP | [speechbrain/emotion-recognition-wav2vec2-IEMOCAP model card](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) |
| audEERING MSP-DIM | [audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim model card](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) |
