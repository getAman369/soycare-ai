"""
Phase 0 dataset audit.

Checks that data/raw actually holds a trustworthy real-image corpus before any
retraining is allowed to proceed. It reports, per class:

  * image count against config.MIN_IMAGES_PER_CLASS
  * files that do not decode (truncated or non-image payloads)
  * exact duplicates by SHA-256
  * near-duplicate clusters by perceptual hash, which is also how capture
    sessions are approximated so a burst of frames from one plant cannot be
    split across train and test

Exit status is non-zero when a gate fails, so it can be wired into CI or a
pre-training check.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from config import (
    DATA_DIR,
    MIN_IMAGES_PER_CLASS,
    OUTPUTS_DIR,
    TARGET_CLASSES,
    ALLOWED_EXTENSIONS,
    logger,
)
from src.dataset_groups import (
    NEAR_DUPLICATE_DISTANCE,
    cluster_by_near_duplicate,
    file_sha256,
    perceptual_hash,
)
from scripts.fetch_datasets import read_manifest

REPORT_NAME = "dataset_audit.json"
DEFAULT_WORKERS = 8


# ---------------------------------------------------------------------------
# Image inspection
# ---------------------------------------------------------------------------
@dataclass
class ImageFacts:
    path: Path
    class_name: str
    sha256: str = ""
    dhash: int = 0
    width: int = 0
    height: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def perceptual_hash_of(image: Image.Image) -> int:
    return perceptual_hash(image)


def inspect_image(path: Path, class_name: str) -> ImageFacts:
    facts = ImageFacts(path=path, class_name=class_name)
    try:
        facts.sha256 = file_sha256(path)
        with Image.open(path) as image:
            facts.dhash = perceptual_hash(ImageOps.exif_transpose(image))
            facts.width, facts.height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as err:
        facts.error = str(err)
    return facts


def collect_images(raw_dir: Path, workers: int = DEFAULT_WORKERS) -> List[ImageFacts]:
    targets = [
        (path, class_dir.name)
        for class_dir in sorted(raw_dir.iterdir())
        if class_dir.is_dir()
        for path in sorted(class_dir.iterdir())
        if path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    if not targets:
        return []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda item: inspect_image(*item), targets))


# ---------------------------------------------------------------------------
# Near-duplicate clustering
# ---------------------------------------------------------------------------
def cluster_near_duplicates(
    facts: Sequence[ImageFacts],
    max_distance: int = NEAR_DUPLICATE_DISTANCE,
) -> Dict[int, List[str]]:
    """
    Groups perceptually near-identical images.

    The groups act as a capture-session proxy: a plant photographed in a burst
    produces many near-identical frames, and DATASET_GUIDE requires those to stay
    in one split. Only groups of more than one image are returned.
    """
    usable = [(fact.path, fact.dhash) for fact in facts if fact.ok]
    return {
        root: [str(path) for path in members]
        for root, members in cluster_by_near_duplicate(usable, max_distance).items()
    }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def build_report(
    raw_dir: Path,
    facts: Sequence[ImageFacts],
    groups: Dict[int, List[str]],
    min_images: int,
) -> Dict[str, Any]:
    manifest = read_manifest(raw_dir)
    source_by_class: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in manifest:
        source_by_class[record.get("target_class", "")][record.get("source_id", "unknown")] += 1

    by_class: Dict[str, Dict[str, Any]] = {}
    for class_name in TARGET_CLASSES:
        class_facts = [fact for fact in facts if fact.class_name == class_name]
        broken = [str(fact.path) for fact in class_facts if not fact.ok]

        digests: Dict[str, List[str]] = defaultdict(list)
        for fact in class_facts:
            if fact.ok:
                digests[fact.sha256].append(str(fact.path))
        exact_duplicates = {digest: paths for digest, paths in digests.items() if len(paths) > 1}

        readable = [fact for fact in class_facts if fact.ok]
        resolutions = {(fact.width, fact.height) for fact in readable}
        near_duplicate_groups = sum(
            1 for members in groups.values()
            if any(Path(member).parent.name == class_name for member in members)
        )

        by_class[class_name] = {
            "images": len(class_facts),
            "readable": len(readable),
            "unreadable": len(broken),
            "meets_minimum": len(readable) >= min_images,
            "sources": dict(source_by_class.get(class_name, {})),
            "exact_duplicate_groups": len(exact_duplicates),
            "exact_duplicate_images": sum(len(paths) for paths in exact_duplicates.values()),
            "near_duplicate_groups": near_duplicate_groups,
            "distinct_resolutions": len(resolutions),
            "unreadable_files": broken[:20],
        }

    classes_present = [name for name, entry in by_class.items() if entry["images"] > 0]
    classes_ready = [name for name, entry in by_class.items() if entry["meets_minimum"]]
    classes_short = [name for name, entry in by_class.items() if not entry["meets_minimum"]]

    gate_failures: List[str] = []
    if not classes_present:
        gate_failures.append("No images found under the raw dataset directory.")
    for name in classes_short:
        # A completely absent class must fail too: silently passing an empty
        # class is how a six-class model ends up trained on three.
        found = by_class[name]["readable"]
        detail = "no images at all" if found == 0 else f"{found} readable images"
        gate_failures.append(
            f"Class '{name}' has {detail}, below the minimum of {min_images}."
        )
    for name in TARGET_CLASSES:
        if by_class[name]["unreadable"]:
            gate_failures.append(
                f"Class '{name}' has {by_class[name]['unreadable']} unreadable files."
            )

    return {
        "raw_dir": str(raw_dir),
        "min_images_per_class": min_images,
        "total_images": len(facts),
        "total_readable": sum(1 for fact in facts if fact.ok),
        "classes": by_class,
        "classes_present": classes_present,
        "classes_meeting_minimum": classes_ready,
        "classes_below_minimum": classes_short,
        "near_duplicate_group_count": len(groups),
        "largest_near_duplicate_group": max((len(m) for m in groups.values()), default=0),
        "gate_passed": not gate_failures,
        "gate_failures": gate_failures,
    }


def run_audit(
    raw_dir: Path,
    min_images: int = MIN_IMAGES_PER_CLASS,
    workers: int = DEFAULT_WORKERS,
    write_report: bool = True,
) -> Dict[str, Any]:
    if not raw_dir.exists():
        logger.error("Raw dataset directory %s does not exist.", raw_dir)
        return {"gate_passed": False, "gate_failures": [f"Missing raw directory {raw_dir}"]}

    logger.info("Inspecting images under %s", raw_dir)
    facts = collect_images(raw_dir, workers)
    groups = cluster_near_duplicates(facts)
    report = build_report(raw_dir, facts, groups, min_images)

    if write_report:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        destination = OUTPUTS_DIR / REPORT_NAME
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        logger.info("Audit report written to %s", destination)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the real soybean image dataset")
    parser.add_argument("--raw-dir", default=str(DATA_DIR / "raw"))
    parser.add_argument("--min-images", type=int, default=MIN_IMAGES_PER_CLASS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-report", action="store_true", help="Skip writing outputs/dataset_audit.json")
    args = parser.parse_args(argv)

    report = run_audit(
        Path(args.raw_dir),
        min_images=args.min_images,
        workers=args.workers,
        write_report=not args.no_report,
    )

    print(f"\nImages found: {report.get('total_images', 0)} ({report.get('total_readable', 0)} readable)")
    print(f"Minimum per class: {report.get('min_images_per_class', 0)}\n")
    print(f"{'Class':<22}{'Images':>8}{'Readable':>10}{'DupGroups':>11}  Ready")
    for class_name, entry in report.get("classes", {}).items():
        print(
            f"{class_name:<22}{entry['images']:>8}{entry['readable']:>10}"
            f"{entry['exact_duplicate_groups']:>11}  {'yes' if entry['meets_minimum'] else 'no'}"
        )

    print(f"\nNear-duplicate groups: {report.get('near_duplicate_group_count', 0)}"
          f" (largest {report.get('largest_near_duplicate_group', 0)})")

    if report.get("gate_passed"):
        print("\nGate: PASSED. The dataset meets the per-class minimums.")
        return 0

    print("\nGate: FAILED")
    for failure in report.get("gate_failures", []):
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
