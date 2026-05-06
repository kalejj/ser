import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from speechbrain.inference.interfaces import foreign_class
from tqdm import tqdm

from benchmark_common import (
    EMOTION_SUMMARY_ROWS,
    LABEL_DIR,
    WAV_DIR,
    build_samples,
    norm_gt,
    write_csv,
)

MODEL_KEY = "speechbrain_iemocap"
MODEL_ID = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

GT_TO_EXPECTED_PRED: dict[str, dict[str, str | None]] = {
    "happy": {"pred": "hap", "mapping_type": "direct"},
    "sad": {"pred": "sad", "mapping_type": "direct"},
    "angry": {"pred": "ang", "mapping_type": "direct"},
    "neutrality": {"pred": "neu", "mapping_type": "direct"},
    "anxious": {"pred": None, "mapping_type": "unsupported"},
    "hurt": {"pred": None, "mapping_type": "unsupported"},
    "embarrassed": {"pred": None, "mapping_type": "unsupported"},
}


def expected_pred(gt_emotion: str) -> str | None:
    return GT_TO_EXPECTED_PRED.get(norm_gt(gt_emotion), {}).get("pred")


def run_model(samples: list[dict]) -> tuple[dict[str, dict], str]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier = foreign_class(
        source=MODEL_ID,
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": device},
    )

    predictions: dict[str, dict] = {}
    for row in tqdm(samples, desc=f"  {MODEL_KEY}"):
        out_prob, score, index, text_lab = classifier.classify_file(
            str(row["wav_path"])
        )
        label = text_lab[0] if isinstance(text_lab, list) else text_lab
        predictions[row["wav_relpath"]] = {
            "pred": str(label),
            "score": round(float(score.squeeze().detach().cpu().item()), 4),
            "index": int(index.squeeze().detach().cpu().item()),
            "probabilities": [
                round(float(value), 4)
                for value in out_prob.squeeze(0).detach().cpu().tolist()
            ],
        }

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, device


def eval_overall(samples: list[dict], preds: dict[str, dict]) -> tuple[int, int]:
    correct = 0
    total = 0
    for row in samples:
        expected = expected_pred(row["gt_emotion"])
        if expected is None:
            continue
        total += 1
        if preds[row["wav_relpath"]]["pred"] == expected:
            correct += 1
    return correct, total


def eval_per_emotion(
    samples: list[dict],
    preds: dict[str, dict],
    gt_norm: str,
    expected: str | None,
) -> tuple[int, int]:
    if expected is None:
        return 0, 0

    correct = 0
    total = 0
    for row in samples:
        if norm_gt(row["gt_emotion"]) != gt_norm:
            continue
        total += 1
        if preds[row["wav_relpath"]]["pred"] == expected:
            correct += 1
    return correct, total


def write_prediction_csv(results_dir: Path, samples: list[dict], preds: dict[str, dict]):
    rows = []
    for row in samples:
        pred = preds[row["wav_relpath"]]
        rows.append(
            {
                "wav_relpath": row["wav_relpath"],
                "gt_emotion": row["gt_emotion"],
                "gt_sensitivity": row["gt_sensitivity"],
                "pred": pred["pred"],
                "score": pred["score"],
                "index": pred["index"],
                "probabilities": json.dumps(pred["probabilities"]),
            }
        )

    write_csv(
        results_dir / f"{MODEL_KEY}.csv",
        [
            "wav_relpath",
            "gt_emotion",
            "gt_sensitivity",
            "pred",
            "score",
            "index",
            "probabilities",
        ],
        rows,
    )


def write_evaluation_tables(
    results_dir: Path,
    samples: list[dict],
    preds: dict[str, dict],
    elapsed_seconds: float,
) -> dict:
    overall_correct, overall_total = eval_overall(samples, preds)
    overall_accuracy_pct = (
        round(100.0 * overall_correct / overall_total, 2)
        if overall_total
        else 0.0
    )

    write_csv(
        results_dir / "summary_overall.csv",
        ["model", "accuracy_pct", "correct", "total", "elapsed_seconds"],
        [
            {
                "model": MODEL_KEY,
                "accuracy_pct": overall_accuracy_pct,
                "correct": overall_correct,
                "total": overall_total,
                "elapsed_seconds": round(elapsed_seconds, 1),
            }
        ],
    )

    by_emotion_rows: list[dict] = []
    by_emotion_meta: dict[str, dict] = {}
    for gt_norm, display in EMOTION_SUMMARY_ROWS:
        mapping = GT_TO_EXPECTED_PRED[gt_norm]
        expected = mapping["pred"]
        correct, total = eval_per_emotion(samples, preds, gt_norm, expected)
        accuracy_pct = round(100.0 * correct / total, 2) if total else None
        report = (
            f"{accuracy_pct:.2f}% ({correct}/{total})"
            if expected is not None and total
            else "unsupported"
            if expected is None
            else "0.00% (0/0)"
        )
        row = {
            "emotion_display": display,
            "mapped_target": expected or "",
            "mapping_type": mapping["mapping_type"],
            f"{MODEL_KEY}_pct": accuracy_pct if accuracy_pct is not None else "",
            f"{MODEL_KEY}_correct": correct if expected is not None else "",
            f"{MODEL_KEY}_total": total if expected is not None else "",
            f"{MODEL_KEY}_report": report,
        }
        by_emotion_rows.append(row)
        by_emotion_meta[gt_norm] = {
            "emotion_display": display,
            "mapped_target": expected,
            "mapping_type": mapping["mapping_type"],
            "accuracy_pct": accuracy_pct,
            "correct": correct if expected is not None else None,
            "total": total if expected is not None else None,
        }

    write_csv(
        results_dir / "summary_by_emotion.csv",
        [
            "emotion_display",
            "mapped_target",
            "mapping_type",
            f"{MODEL_KEY}_pct",
            f"{MODEL_KEY}_correct",
            f"{MODEL_KEY}_total",
            f"{MODEL_KEY}_report",
        ],
        by_emotion_rows,
    )

    return {
        "gt_to_pred_mapping": {
            key: value["pred"] for key, value in GT_TO_EXPECTED_PRED.items()
        },
        "unsupported_gt_emotions": [
            key for key, value in GT_TO_EXPECTED_PRED.items() if value["pred"] is None
        ],
        "overall": {
            MODEL_KEY: {
                "accuracy_pct": overall_accuracy_pct,
                "correct": overall_correct,
                "total": overall_total,
                "elapsed_seconds": round(elapsed_seconds, 1),
            }
        },
        "by_gt_emotion": by_emotion_meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpeechBrain IEMOCAP 모델 감정별 WAV 추론 및 CSV/meta 저장",
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
        default=Path("results_speechbrain_iemocap"),
        help="결과 저장 디렉터리 (기본: results_speechbrain_iemocap)",
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
    evaluation = write_evaluation_tables(
        results_dir, samples, predictions, elapsed_seconds
    )

    distribution = Counter(pred["pred"] for pred in predictions.values())
    total_predictions = len(predictions)
    meta = {
        "total_files": len(samples),
        "per_emotion_sampled": per_emotion,
        "random_seed": None if use_all else args.seed,
        "samples_per_emotion": None if use_all else n_arg,
        "full_emotion_pools": use_all,
        "models": {
            MODEL_KEY: {
                "model_id": MODEL_ID,
                "device": device,
                "elapsed_seconds": round(elapsed_seconds, 1),
                "prediction_distribution": {
                    label: {
                        "count": count,
                        "ratio": round(count / total_predictions, 4),
                    }
                    for label, count in sorted(
                        distribution.items(), key=lambda item: -item[1]
                    )
                },
            }
        },
        "evaluation": evaluation,
    }

    meta_path = results_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    print(f"메타 저장: {meta_path}")

    print("\n===== 예측 분포 =====")
    for label, info in meta["models"][MODEL_KEY]["prediction_distribution"].items():
        print(f"  {label:8s}: {info['count']:5d} ({info['ratio'] * 100:.1f}%)")


if __name__ == "__main__":
    main()
