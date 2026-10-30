"""Evaluate held-out vessel type and named-exhibit retrieval performance."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vision import GalleryEncoder, VesselClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=Path, default=Path("data/exhibit_gallery"))
    parser.add_argument("--metadata", type=Path, default=Path("data/exhibits.json"))
    parser.add_argument("--classifier", type=Path, default=Path("models/vessel_classifier/best.pt"))
    parser.add_argument("--feature-model", type=Path, default=Path("models/resnet-18"))
    parser.add_argument("--test-per-exhibit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/vessel_classifier/exhibit_evaluation.json"))
    parser.add_argument("--split-output", type=Path, default=Path("data/exhibit_gallery_split.json"))
    args = parser.parse_args()
    exhibits = json.loads(args.metadata.read_text(encoding="utf-8"))
    classifier = VesselClassifier(args.classifier)
    encoder = GalleryEncoder(args.feature_model)
    random.seed(args.seed)
    references: dict[str, list[np.ndarray]] = {}
    tests: list[tuple[dict, Path]] = []
    type_rows: list[dict] = []
    split_manifest = {"seed": args.seed, "test_per_exhibit": args.test_per_exhibit, "exhibits": {}}
    for exhibit in exhibits:
        paths = sorted(
            path for path in (args.gallery / exhibit["gallery_dir"]).rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        random.shuffle(paths)
        test_paths = paths[: args.test_per_exhibit]
        reference_paths = paths[args.test_per_exhibit :]
        split_manifest["exhibits"][exhibit["id"]] = {
            "reference": [path.relative_to(args.gallery).as_posix() for path in reference_paths],
            "test": [path.relative_to(args.gallery).as_posix() for path in test_paths],
        }
        references[exhibit["id"]] = [encoder.encode(Image.open(path)) for path in reference_paths]
        tests.extend((exhibit, path) for path in test_paths)
    exhibit_correct = 0
    exhibit_top3 = 0
    type_correct = 0
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    rows: list[dict] = []
    names = {item["id"]: item["name"] for item in exhibits}
    for exhibit, path in tests:
        image = Image.open(path).convert("RGB")
        type_candidates = classifier.predict(image, top_k=3)
        predicted_type = type_candidates[0]["label"]
        type_correct += int(predicted_type == exhibit["vessel_type"])
        query = encoder.encode(image)
        scores = []
        for candidate in exhibits:
            if candidate["vessel_type"] != predicted_type:
                continue
            candidate_scores = [float(np.dot(query, vector)) for vector in references[candidate["id"]]]
            scores.append((max(candidate_scores), candidate["id"]))
        scores.sort(reverse=True)
        predicted_id = scores[0][1] if scores else None
        ranked_ids = [item[1] for item in scores[:3]]
        exhibit_correct += int(predicted_id == exhibit["id"])
        exhibit_top3 += int(exhibit["id"] in ranked_ids)
        own_score = next((score for score, item_id in scores if item_id == exhibit["id"]), 0.0)
        positive_scores.append(own_score)
        negative_scores.extend(score for score, item_id in scores if item_id != exhibit["id"])
        rows.append(
            {
                "image": str(path),
                "expected_type": exhibit["vessel_type"],
                "predicted_type": predicted_type,
                "type_confidence": type_candidates[0]["confidence"],
                "expected_exhibit": exhibit["id"],
                "predicted_exhibit": predicted_id,
                "predicted_name": names.get(predicted_id),
                "expected_similarity": round(own_score, 4),
                "candidates": [{"id": item_id, "name": names[item_id], "score": round(score, 4)} for score, item_id in scores[:3]],
            }
        )
    total = len(tests)
    result = {
        "samples": total,
        "reference_images_per_exhibit": 20 - args.test_per_exhibit,
        "type_accuracy": type_correct / total,
        "exhibit_top1_accuracy": exhibit_correct / total,
        "exhibit_top3_accuracy": exhibit_top3 / total,
        "positive_similarity": summarize(positive_scores),
        "negative_similarity": summarize(negative_scores),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.split_output.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": round(min(values), 4), "mean": round(float(np.mean(values)), 4), "max": round(max(values), 4)}


if __name__ == "__main__":
    main()
