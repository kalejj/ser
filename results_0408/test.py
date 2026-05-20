import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from funasr import AutoModel
from tqdm import tqdm

SAMPLES_PER_EMOTION = 50
RANDOM_SEED = 42

MODELS = {
    "plus_seed": "iic/emotion2vec_plus_seed",
    "plus_base": "iic/emotion2vec_plus_base",
    "plus_large": "iic/emotion2vec_plus_large",
}

DATA_DIR = Path("data")
WAV_DIR = DATA_DIR / "wav"
LABEL_DIR = DATA_DIR / "label"
RESULTS_DIR = Path("results")


def discover_emotion_dirs(wav_dir: Path) -> list[Path]:
    return sorted(p for p in wav_dir.iterdir() if p.is_dir())


def build_samples(
    wav_dir: Path,
    label_dir: Path,
    n_per_emotion: int,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    random.seed(seed)
    samples: list[dict] = []
    per_emotion: dict[str, int] = {}

    for emo_dir in discover_emotion_dirs(wav_dir):
        pool = list(emo_dir.rglob("*.wav"))
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
        per_emotion[emo_dir.name] = n_per_emotion

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


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    samples, per_emotion = build_samples(
        WAV_DIR, LABEL_DIR, SAMPLES_PER_EMOTION, RANDOM_SEED
    )
    print(
        f"감정 폴더 {len(per_emotion)}개 / 샘플 총 {len(samples)}개 "
        f"(감정당 {SAMPLES_PER_EMOTION}개, seed={RANDOM_SEED})\n"
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
        csv_path = RESULTS_DIR / f"{model_key}.csv"
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
        "random_seed": RANDOM_SEED,
        "samples_per_emotion": SAMPLES_PER_EMOTION,
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

    meta_path = RESULTS_DIR / "meta.json"
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
