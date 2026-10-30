"""Prepare a grouped vessel-type train/validation/test dataset."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/images"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw_dataset"))
    parser.add_argument("--metadata", type=Path, default=Path("data/exhibits.json"))
    parser.add_argument("--groups", type=Path, default=Path("data/raw_dataset/groups.json"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    items = json.loads(args.metadata.read_text(encoding="utf-8"))
    manual_groups = json.loads(args.groups.read_text(encoding="utf-8")) if args.groups.exists() else {}
    grouped_names: dict[str, dict[str, str]] = {}
    for vessel_type, groups in manual_groups.items():
        for index, filenames in enumerate(groups, start=1):
            group_id = f"manual_group_{index:03d}"
            for filename in filenames:
                grouped_names.setdefault(vessel_type, {})[filename] = group_id
    grouped: dict[str, dict[str, list[Path]]] = {}
    if args.source.exists():
        for path in args.source.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            item = next((entry for entry in items if path.stem == entry["id"] or path.stem.startswith(entry["id"] + "_")), None)
            if item and item.get("vessel_type"):
                grouped.setdefault(item["vessel_type"], {}).setdefault(item["id"], []).append(path)
    if args.raw.exists():
        for type_dir in args.raw.iterdir():
            if not type_dir.is_dir():
                continue
            direct_files = [path for path in type_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            for path in direct_files:
                object_id = grouped_names.get(type_dir.name, {}).get(path.name, path.stem)
                grouped.setdefault(type_dir.name, {}).setdefault(object_id, []).append(path)
            for object_dir in type_dir.iterdir():
                if not object_dir.is_dir():
                    continue
                paths = [path for path in object_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
                if paths:
                    grouped.setdefault(type_dir.name, {}).setdefault(object_dir.name, []).extend(paths)
    random.seed(args.seed)
    if args.output.exists():
        shutil.rmtree(args.output)
    can_evaluate = bool(grouped) and all(len(objects) >= 3 for objects in grouped.values())
    for vessel_type, objects in sorted(grouped.items()):
        object_ids = list(objects)
        random.shuffle(object_ids)
        splits = {"train": object_ids, "val": [], "test": []}
        if can_evaluate:
            test_count = max(1, round(len(object_ids) * args.test_ratio))
            val_count = max(1, round(len(object_ids) * args.val_ratio))
            if test_count + val_count >= len(object_ids):
                test_count = val_count = 1
            splits = {
                "test": object_ids[:test_count],
                "val": object_ids[test_count : test_count + val_count],
                "train": object_ids[test_count + val_count :],
            }
        for split, split_ids in splits.items():
            target = args.output / split / vessel_type
            target.mkdir(parents=True, exist_ok=True)
            for object_id in split_ids:
                for index, path in enumerate(objects[object_id]):
                    shutil.copy2(path, target / f"{object_id}_{index:03d}{path.suffix.lower()}")
        image_count = sum(len(paths) for paths in objects.values())
        print(f"{vessel_type}: {len(objects)} object(s), {image_count} image(s), train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"Dataset written to {args.output}")
    if not can_evaluate:
        print("WARNING: each class needs at least 3 independent objects before train/val/test evaluation is enabled.")


if __name__ == "__main__":
    main()
