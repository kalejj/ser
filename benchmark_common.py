import csv
import json
import math
import random
from pathlib import Path

DATA_DIR = Path("data")
WAV_DIR = DATA_DIR / "wav"
LABEL_DIR = DATA_DIR / "label"

EMOTION_SUMMARY_ROWS: list[tuple[str, str]] = [
    ("happy", "기쁨(Happy)"),
    ("sad", "슬픔(Sad)"),
    ("angry", "분노(Angry)"),
    ("anxious", "불안(Anxious)"),
    ("hurt", "상처(Hurt)"),
    ("embarrassed", "당황(Embarrassed)"),
    ("neutrality", "중립(Neutrality)"),
]


def norm_gt(emotion: str) -> str:
    return emotion.strip().lower()


def discover_emotion_dirs(wav_dir: Path) -> list[Path]:
    return sorted(p for p in wav_dir.iterdir() if p.is_dir())


def build_samples(
    wav_dir: Path,
    label_dir: Path,
    n_per_emotion: int | None,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
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


def load_samples_from_csv(
    csv_path: Path,
    wav_dir: Path,
    label_dir: Path,
) -> tuple[list[dict], dict[str, int]]:
    samples: list[dict] = []
    per_emotion: dict[str, int] = {}

    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if "wav_relpath" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path}에 wav_relpath 컬럼이 없습니다.")

        seen: set[str] = set()
        for row in reader:
            rel_text = row["wav_relpath"]
            if rel_text in seen:
                raise ValueError(f"샘플 목록에 중복 WAV가 있습니다: {rel_text}")
            seen.add(rel_text)

            rel = Path(rel_text)
            wav_path = wav_dir / rel
            if not wav_path.is_file():
                raise FileNotFoundError(f"WAV가 없습니다: {wav_path}")

            label_path = label_dir / rel.with_suffix(".json")
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"라벨이 없습니다: {label_path} (wav: {wav_path})"
                )
            with open(label_path, encoding="utf-8") as label_fp:
                data = json.load(label_fp)
            speaker = data["화자정보"]
            samples.append(
                {
                    "wav_path": wav_path,
                    "wav_relpath": rel.as_posix(),
                    "gt_emotion": speaker["Emotion"],
                    "gt_sensitivity": speaker["Sensitivity"],
                }
            )
            per_emotion[rel.parts[0]] = per_emotion.get(rel.parts[0], 0) + 1

    if not samples:
        raise ValueError(f"샘플 목록이 비어 있습니다: {csv_path}")

    return samples, per_emotion


def compute_basic_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return {
        "count": len(values),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 저장: {path}")
