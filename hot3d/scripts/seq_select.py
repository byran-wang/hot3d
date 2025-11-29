import argparse
import json
from pathlib import Path
from typing import List, Tuple


def load_metadata(seq_dir: Path) -> dict:
    metadata_path = seq_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def matches_objects(object_names: List[str], selected: List[str]) -> bool:
    if not selected:
        return True
    object_set = set(object_names)
    return all(obj in object_set for obj in selected)


def find_sequences(dataset_dir: Path, selected_obj: List[str]) -> List[Tuple[str, List[str]]]:
    matches: List[Tuple[str, List[str]]] = []
    for seq_dir in dataset_dir.iterdir():
        if not seq_dir.is_dir():
            continue
        metadata = load_metadata(seq_dir)
        if not metadata or metadata.get("headset") != "Aria":
            continue
        object_names = metadata.get("object_names", [])
        if matches_objects(object_names, selected_obj):
            matches.append((seq_dir.name, object_names))
    return sorted(matches, key=lambda item: item[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Aria sequences containing the specified objects."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="Root dataset directory containing sequence folders.",
    )
    parser.add_argument(
        "--object",
        "-o",
        dest="objects",
        action="append",
        nargs="+",
        default=[],
        help="Object name to require (can be passed multiple times, supports multiple names per flag).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_objects = [obj for group in args.objects for obj in group]
    sequences = find_sequences(args.dataset_dir, selected_objects)
    for seq, object_names in sequences:
        joined_objects = ", ".join(map(str, object_names))
        print(f"{seq}: {joined_objects}")


if __name__ == "__main__":
    main()
