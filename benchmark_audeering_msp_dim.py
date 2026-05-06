import argparse
import json
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
import soundfile as sf
import soxr
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from benchmark_common import (
    EMOTION_SUMMARY_ROWS,
    LABEL_DIR,
    WAV_DIR,
    build_samples,
    compute_basic_stats,
    norm_gt,
    write_csv,
)

MODEL_KEY = "audeering_msp_dim"
MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DIMENSIONS = ["arousal", "dominance", "valence"]


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionRegressionModel(Wav2Vec2PreTrainedModel):
    all_tied_weights_keys: dict[str, str] = {}

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


def load_audio_16k_mono(path: Path) -> np.ndarray:
    signal, sample_rate = sf.read(path, dtype="float32")
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if sample_rate != 16000:
        signal = soxr.resample(signal, sample_rate, 16000)
    return np.asarray(signal, dtype=np.float32)


def load_model_config() -> Wav2Vec2Config:
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
    with open(config_path, encoding="utf-8") as file:
        config_dict = json.load(file)
    if config_dict.get("vocab_size") is None:
        config_dict["vocab_size"] = 32
    return Wav2Vec2Config(**config_dict)


def run_model(samples: list[dict]) -> tuple[dict[str, dict], str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    config = load_model_config()
    model = EmotionRegressionModel.from_pretrained(MODEL_ID, config=config).to(device)
    model.eval()

    predictions: dict[str, dict] = {}
    with torch.no_grad():
        for row in tqdm(samples, desc=f"  {MODEL_KEY}"):
            signal = load_audio_16k_mono(row["wav_path"])
            inputs = feature_extractor(
                signal,
                sampling_rate=16000,
                return_tensors="pt",
            )
            _, logits = model(inputs["input_values"].to(device))
            values = logits.squeeze(0).detach().cpu().tolist()
            predictions[row["wav_relpath"]] = {
                dimension: round(float(value), 4)
                for dimension, value in zip(DIMENSIONS, values)
            }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, device


def write_prediction_csv(results_dir: Path, samples: list[dict], preds: dict[str, dict]):
    rows = []
    for row in samples:
        pred = preds[row["wav_relpath"]]
        rows.append(
            {
                "wav_relpath": row["wav_relpath"],
                "gt_emotion": row["gt_emotion"],
                "gt_sensitivity": row["gt_sensitivity"],
                "arousal": pred["arousal"],
                "dominance": pred["dominance"],
                "valence": pred["valence"],
            }
        )

    write_csv(
        results_dir / f"{MODEL_KEY}.csv",
        [
            "wav_relpath",
            "gt_emotion",
            "gt_sensitivity",
            "arousal",
            "dominance",
            "valence",
        ],
        rows,
    )


def write_summary_tables(
    results_dir: Path,
    samples: list[dict],
    preds: dict[str, dict],
    elapsed_seconds: float,
) -> dict:
    overall_row = {
        "model": MODEL_KEY,
        "task_type": "dimensional_regression",
        "count": len(samples),
        "elapsed_seconds": round(elapsed_seconds, 1),
    }
    overall_stats: dict[str, dict] = {}
    for dimension in DIMENSIONS:
        stats = compute_basic_stats([preds[row["wav_relpath"]][dimension] for row in samples])
        overall_stats[dimension] = stats
        overall_row[f"{dimension}_mean"] = stats["mean"]
        overall_row[f"{dimension}_std"] = stats["std"]
        overall_row[f"{dimension}_min"] = stats["min"]
        overall_row[f"{dimension}_max"] = stats["max"]

    write_csv(
        results_dir / "summary_overall.csv",
        [
            "model",
            "task_type",
            "count",
            "elapsed_seconds",
            "arousal_mean",
            "arousal_std",
            "arousal_min",
            "arousal_max",
            "dominance_mean",
            "dominance_std",
            "dominance_min",
            "dominance_max",
            "valence_mean",
            "valence_std",
            "valence_min",
            "valence_max",
        ],
        [overall_row],
    )

    by_emotion_rows: list[dict] = []
    by_emotion_stats: dict[str, dict] = {}
    for gt_norm, display in EMOTION_SUMMARY_ROWS:
        emotion_rows = [
            row for row in samples if norm_gt(row["gt_emotion"]) == gt_norm
        ]
        summary_row = {
            "emotion_display": display,
            "count": len(emotion_rows),
        }
        by_emotion_stats[gt_norm] = {}
        for dimension in DIMENSIONS:
            stats = compute_basic_stats(
                [preds[row["wav_relpath"]][dimension] for row in emotion_rows]
            )
            by_emotion_stats[gt_norm][dimension] = stats
            summary_row[f"{dimension}_mean"] = stats["mean"]
            summary_row[f"{dimension}_std"] = stats["std"]
            summary_row[f"{dimension}_min"] = stats["min"]
            summary_row[f"{dimension}_max"] = stats["max"]
        by_emotion_rows.append(summary_row)

    write_csv(
        results_dir / "summary_by_emotion.csv",
        [
            "emotion_display",
            "count",
            "arousal_mean",
            "arousal_std",
            "arousal_min",
            "arousal_max",
            "dominance_mean",
            "dominance_std",
            "dominance_min",
            "dominance_max",
            "valence_mean",
            "valence_std",
            "valence_min",
            "valence_max",
        ],
        by_emotion_rows,
    )

    return {
        "task_type": "dimensional_regression",
        "dimensions": list(DIMENSIONS),
        "overall": overall_stats,
        "by_gt_emotion": by_emotion_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="audEERING MSP-DIM 모델 감정별 WAV 추론 및 통계 저장",
    )
    parser.add_argument(
        "--samples-per-emotion",
        type=int,
        default=None,
        metavar="N",
        help="감정 폴더당 샘플 개수. 생략하거나 0이면 해당 폴더 내 전체 WAV 사용.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="샘플링 난수 시드 (--samples-per-emotion > 0일 때만 적용).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_audeering_msp_dim"),
        help="결과 저장 디렉터리 (기본: results_audeering_msp_dim)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_arg = args.samples_per_emotion
    use_all = n_arg is None or n_arg == 0
    if not use_all and n_arg < 0:
        raise SystemExit("--samples-per-emotion은 0 이상이어야 합니다.")
    n_per_emotion: int | None = None if use_all else n_arg

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    samples, per_emotion = build_samples(
        WAV_DIR, LABEL_DIR, n_per_emotion, args.seed
    )
    if use_all:
        mode_desc = "감정당 전체"
        seed_desc = "(샘플링 없음)"
    else:
        mode_desc = f"감정당 {n_arg}개"
        seed_desc = f"seed={args.seed}"
    print(
        f"감정 폴더 {len(per_emotion)}개 / 샘플 총 {len(samples)}개 "
        f"({mode_desc}, {seed_desc})\n"
    )

    print(f"[{MODEL_KEY}] 추론 시작 ({MODEL_ID})")
    started_at = time.time()
    predictions, device = run_model(samples)
    elapsed_seconds = time.time() - started_at
    print(f"  완료: {elapsed_seconds:.1f}s\n")

    write_prediction_csv(results_dir, samples, predictions)
    evaluation = write_summary_tables(
        results_dir, samples, predictions, elapsed_seconds
    )

    meta = {
        "total_files": len(samples),
        "per_emotion_sampled": per_emotion,
        "random_seed": None if use_all else args.seed,
        "samples_per_emotion": None if use_all else n_arg,
        "full_emotion_pools": use_all,
        "models": {
            MODEL_KEY: {
                "model_id": MODEL_ID,
                "task_type": "dimensional_regression",
                "device": device,
                "elapsed_seconds": round(elapsed_seconds, 1),
                "dimensions": list(DIMENSIONS),
            }
        },
        "evaluation": evaluation,
    }

    meta_path = results_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    print(f"메타 저장: {meta_path}")

    print("\n===== 전체 평균 차원값 =====")
    for dimension in DIMENSIONS:
        stats = evaluation["overall"][dimension]
        print(
            f"  {dimension:10s}: mean={stats['mean']:.4f} "
            f"std={stats['std']:.4f}"
        )


if __name__ == "__main__":
    main()
