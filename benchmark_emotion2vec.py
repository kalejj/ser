import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from funasr import AutoModel
from tqdm import tqdm

from benchmark_common import load_samples_from_csv

MODELS = {
    "plus_seed": "iic/emotion2vec_plus_seed",
    "plus_base": "iic/emotion2vec_plus_base",
    "plus_large": "iic/emotion2vec_plus_large",
}

DATA_DIR = Path("data")
WAV_DIR = DATA_DIR / "wav"
LABEL_DIR = DATA_DIR / "label"

# 데이터셋 gt_emotion(JSON) → emotion2vec 예측 라벨(소문자) 매핑 (정규화 키: 소문자)
_GT_NORM_TO_EXPECTED_PRED: dict[str, str] = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "neutrality": "neutral",
    "anxious": "fearful",
    "hurt": "sad",
    "embarrassed": "surprised",
}

# 감정별 요약 표 행 순서: (gt 정규화 키, 표시명, 기대 pred, 매핑 유형)
_EMOTION_SUMMARY_ROWS: list[tuple[str, str, str, str]] = [
    ("happy", "기쁨(Happy)", "happy", "direct"),
    ("sad", "슬픔(Sad)", "sad", "direct"),
    ("angry", "분노(Angry)", "angry", "direct"),
    ("anxious", "불안(Anxious)", "fearful", "approximate"),
    ("hurt", "상처(Hurt)", "sad", "approximate"),
    ("embarrassed", "당황(Embarrassed)", "surprised", "approximate"),
    ("neutrality", "중립(Neutrality)", "neutral", "direct"),
]

# 전체 요약 표에서 모델 표시 순서 (보고서와 동일)
_SUMMARY_MODEL_ORDER = ["plus_large", "plus_base", "plus_seed"]


def _norm_gt(emotion: str) -> str:
    return emotion.strip().lower()


def _expected_pred(gt_emotion: str) -> str | None:
    return _GT_NORM_TO_EXPECTED_PRED.get(_norm_gt(gt_emotion))


def _eval_overall(
    samples: list[dict], preds: dict[str, dict]
) -> tuple[int, int]:
    correct = 0
    total = 0
    for row in samples:
        key = row["wav_relpath"]
        exp = _expected_pred(row["gt_emotion"])
        if exp is None:
            continue
        total += 1
        if preds[key]["pred"] == exp:
            correct += 1
    return correct, total


def _eval_per_emotion(
    samples: list[dict], preds: dict[str, dict], gt_norm: str, expected: str
) -> tuple[int, int]:
    correct = 0
    total = 0
    for row in samples:
        if _norm_gt(row["gt_emotion"]) != gt_norm:
            continue
        key = row["wav_relpath"]
        total += 1
        if preds[key]["pred"] == expected:
            correct += 1
    return correct, total


def write_evaluation_tables(
    results_dir: Path,
    samples: list[dict],
    all_predictions: dict[str, dict[str, dict]],
    elapsed: dict[str, float],
) -> dict:
    """summary_overall.csv, summary_by_emotion.csv 작성 및 meta용 evaluation dict 반환."""
    overall_rows = []
    for mk in _SUMMARY_MODEL_ORDER:
        if mk not in all_predictions:
            continue
        c, t = _eval_overall(samples, all_predictions[mk])
        pct = round(100.0 * c / t, 2) if t else 0.0
        overall_rows.append(
            {
                "model": mk,
                "accuracy_pct": pct,
                "correct": c,
                "total": t,
                "elapsed_seconds": round(elapsed[mk], 1),
            }
        )

    path_overall = results_dir / "summary_overall.csv"
    with open(path_overall, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "accuracy_pct",
                "correct",
                "total",
                "elapsed_seconds",
            ],
        )
        w.writeheader()
        w.writerows(overall_rows)
    print(f"CSV 저장: {path_overall}")

    by_emotion_fieldnames = [
        "emotion_display",
        "mapped_target",
        "mapping_type",
    ]
    for mk in _SUMMARY_MODEL_ORDER:
        by_emotion_fieldnames += [
            f"{mk}_pct",
            f"{mk}_correct",
            f"{mk}_total",
            f"{mk}_report",  # "56.00% (28/50)" 형태
        ]

    by_emotion_rows: list[dict] = []
    for gt_norm, disp, expected, mtype in _EMOTION_SUMMARY_ROWS:
        row: dict = {
            "emotion_display": disp,
            "mapped_target": expected,
            "mapping_type": mtype,
        }
        for mk in _SUMMARY_MODEL_ORDER:
            if mk not in all_predictions:
                continue
            c, t = _eval_per_emotion(
                samples, all_predictions[mk], gt_norm, expected
            )
            pct = round(100.0 * c / t, 2) if t else 0.0
            row[f"{mk}_pct"] = pct
            row[f"{mk}_correct"] = c
            row[f"{mk}_total"] = t
            row[f"{mk}_report"] = f"{pct:.2f}% ({c}/{t})" if t else "— (0/0)"
        by_emotion_rows.append(row)

    path_by = results_dir / "summary_by_emotion.csv"
    with open(path_by, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=by_emotion_fieldnames)
        w.writeheader()
        w.writerows(by_emotion_rows)
    print(f"CSV 저장: {path_by}")

    # plus_large 기준 직접/근사 평균 정확도 (보고서 각주)
    def _avg_pct_for_types(types: set[str]) -> float | None:
        if "plus_large" not in all_predictions:
            return None
        pcts: list[float] = []
        for gt_norm, _disp, expected, mtype in _EMOTION_SUMMARY_ROWS:
            if mtype not in types:
                continue
            c, t = _eval_per_emotion(
                samples, all_predictions["plus_large"], gt_norm, expected
            )
            if t:
                pcts.append(100.0 * c / t)
        return round(sum(pcts) / len(pcts), 2) if pcts else None

    direct_avg = _avg_pct_for_types({"direct"})
    approx_avg = _avg_pct_for_types({"approximate"})

    evaluation_meta = {
        "gt_to_pred_mapping": dict(_GT_NORM_TO_EXPECTED_PRED),
        "overall": {
            r["model"]: {
                "accuracy_pct": r["accuracy_pct"],
                "correct": r["correct"],
                "total": r["total"],
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for r in overall_rows
        },
        "per_emotion_plus_large_avg": {
            "direct_mapping_emotions_mean_pct": direct_avg,
            "approximate_mapping_emotions_mean_pct": approx_avg,
        },
    }
    return evaluation_meta


def discover_emotion_dirs(wav_dir: Path) -> list[Path]:
    return sorted(p for p in wav_dir.iterdir() if p.is_dir())


def build_samples(
    wav_dir: Path,
    label_dir: Path,
    n_per_emotion: int | None,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    """n_per_emotion이 None이면 감정 폴더 내 전체 WAV를 경로 순으로 사용."""
    samples: list[dict] = []
    per_emotion: dict[str, int] = {}

    if n_per_emotion is not None:
        random.seed(seed)

    for emo_dir in discover_emotion_dirs(wav_dir):
        pool = sorted(emo_dir.rglob("*.wav"), key=lambda p: p.as_posix())
        if not pool:
            raise ValueError(f"감정 폴더 '{emo_dir.name}'에 WAV가 없습니다.")

        if n_per_emotion is None:
            chosen = sorted(pool, key=lambda p: p.as_posix())
        else:
            if len(pool) < n_per_emotion:
                raise ValueError(
                    f"감정 폴더 '{emo_dir.name}'에 WAV가 {len(pool)}개뿐입니다. "
                    f"{n_per_emotion}개 이상 필요합니다."
                )
            chosen = random.sample(pool, n_per_emotion)
        for wav_path in chosen:
            rel = wav_path.relative_to(wav_dir)
            label_path = label_dir / rel.with_suffix(".json")
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"라벨이 없습니다: {label_path} (wav: {wav_path})"
                )
            with open(label_path, encoding="utf-8") as fp:
                data = json.load(fp)
            speaker = data["화자정보"]
            samples.append(
                {
                    "wav_path": wav_path,
                    "wav_relpath": rel.as_posix(),
                    "gt_emotion": speaker["Emotion"],
                    "gt_sensitivity": speaker["Sensitivity"],
                }
            )
        per_emotion[emo_dir.name] = len(chosen)

    return samples, per_emotion


def run_model(model_id: str, samples: list[dict]) -> dict[str, dict]:
    model = AutoModel(model=model_id, hub="hf")

    predictions: dict[str, dict] = {}
    for row in tqdm(samples, desc=f"  {model_id}"):
        wav_path = row["wav_path"]
        key = row["wav_relpath"]
        result = model.generate(
            str(wav_path),
            granularity="utterance",
            extract_embedding=False,
        )
        entry = result[0]
        top_idx = max(range(len(entry["scores"])), key=lambda i: entry["scores"][i])
        predictions[key] = {
            "pred": entry["labels"][top_idx].split("/")[-1],
            "score": round(entry["scores"][top_idx], 4),
        }

    del model
    torch.cuda.empty_cache()
    return predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="emotion2vec+ (FunASR) 감정별 WAV 추론 및 CSV/meta 저장",
    )
    p.add_argument(
        "--samples-per-emotion",
        type=int,
        default=None,
        metavar="N",
        help="감정 폴더당 샘플 개수. 생략하거나 0이면 해당 폴더 내 전체 WAV 사용.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="샘플링 난수 시드 (--samples-per-emotion > 0일 때만 적용).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="결과 저장 디렉터리 (기본: results)",
    )
    p.add_argument(
        "--sample-list-csv",
        type=Path,
        default=None,
        help="기존 결과 CSV의 wav_relpath 순서를 그대로 재사용합니다.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    n_arg = args.samples_per_emotion
    use_all = n_arg is None or n_arg == 0
    if not use_all and n_arg < 0:
        raise SystemExit("--samples-per-emotion은 0 이상이어야 합니다.")
    n_per_emotion: int | None = None if use_all else n_arg

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_list_csv is not None:
        samples, per_emotion = load_samples_from_csv(
            args.sample_list_csv, WAV_DIR, LABEL_DIR
        )
        mode_desc = f"샘플 목록 CSV 사용: {args.sample_list_csv}"
        seed_desc = "(샘플링 없음)"
    else:
        samples, per_emotion = build_samples(
            WAV_DIR, LABEL_DIR, n_per_emotion, args.seed
        )
        if use_all:
            mode_desc = "감정당 전체"
            seed_desc = "(샘플링 없음)"
        else:
            mode_desc = f"감정당 {n_arg}개"
            seed_desc = f"seed={args.seed}"
    if use_all and args.sample_list_csv is None:
        mode_desc = "감정당 전체"
        seed_desc = "(샘플링 없음)"
    print(
        f"감정 폴더 {len(per_emotion)}개 / 샘플 총 {len(samples)}개 "
        f"({mode_desc}, {seed_desc})\n"
    )

    all_predictions: dict[str, dict[str, dict]] = {}
    elapsed: dict[str, float] = {}

    for model_key, model_id in MODELS.items():
        print(f"[{model_key}] 추론 시작 ({model_id})")
        t0 = time.time()
        all_predictions[model_key] = run_model(model_id, samples)
        elapsed[model_key] = time.time() - t0
        print(f"  완료: {elapsed[model_key]:.1f}s\n")

    fieldnames = [
        "wav_relpath",
        "gt_emotion",
        "gt_sensitivity",
        "pred",
        "score",
    ]

    for model_key in MODELS:
        csv_path = results_dir / f"{model_key}.csv"
        preds = all_predictions[model_key]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in samples:
                key = row["wav_relpath"]
                p = preds[key]
                writer.writerow(
                    {
                        "wav_relpath": row["wav_relpath"],
                        "gt_emotion": row["gt_emotion"],
                        "gt_sensitivity": row["gt_sensitivity"],
                        "pred": p["pred"],
                        "score": p["score"],
                    }
                )
        print(f"CSV 저장: {csv_path}")

    n = len(samples)
    meta: dict = {
        "total_files": n,
        "per_emotion_sampled": per_emotion,
        "random_seed": None if use_all or args.sample_list_csv else args.seed,
        "samples_per_emotion": None if use_all or args.sample_list_csv else n_arg,
        "sample_list_csv": (
            args.sample_list_csv.as_posix() if args.sample_list_csv else None
        ),
        "full_emotion_pools": use_all and args.sample_list_csv is None,
        "models": {},
    }
    for mk, preds in all_predictions.items():
        dist = Counter(p["pred"] for p in preds.values())
        meta["models"][mk] = {
            "model_id": MODELS[mk],
            "elapsed_seconds": round(elapsed[mk], 1),
            "prediction_distribution": {
                k: {"count": v, "ratio": round(v / n, 4)}
                for k, v in sorted(dist.items(), key=lambda x: -x[1])
            },
        }

    meta["evaluation"] = write_evaluation_tables(
        results_dir, samples, all_predictions, elapsed
    )

    meta_path = results_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"메타 저장: {meta_path}")

    print("\n===== 모델별 예측 분포 =====")
    for mk in MODELS:
        print(f"\n[{mk}] ({elapsed[mk]:.1f}s)")
        for emotion, info in meta["models"][mk]["prediction_distribution"].items():
            print(f"  {emotion:12s}: {info['count']:5d} ({info['ratio']*100:.1f}%)")


if __name__ == "__main__":
    main()
