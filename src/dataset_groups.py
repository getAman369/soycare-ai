"""
Capture-session grouping for real soybean leaf images.

DATASET_GUIDE requires that photographs from the same plant or capture session
stay in a single split. Neither the public dataset filenames nor the Mendeley
file lists carry session metadata, so sessions are approximated by clustering
perceptually near-identical images: a burst of frames of one leaf collapses into
one group, and the splitter then assigns whole groups to one split.

Shared by scripts.audit_dataset.py and src.prepare_dataset.py so the audit and
the split can never disagree about what a group is.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

HASH_SIZE = 8
HASH_BITS = HASH_SIZE * HASH_SIZE
NEAR_DUPLICATE_DISTANCE = 5
BLOCK_BITS = 16


def perceptual_hash(image: Image.Image) -> int:
    """
    Difference hash: one bit per pixel comparing it with its right neighbour.

    Preferred over average hash because lesion texture drives the difference
    hash, and texture is what distinguishes two frames of the same leaf from two
    genuinely different leaves.
    """
    grayscale = image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    bits = 0
    for bit in (pixels[:, 1:] > pixels[:, :-1]).flatten():
        bits = (bits << 1) | int(bit)
    return bits


def image_dhash(path: Path) -> Optional[int]:
    """Returns the perceptual hash of an image file, or None if it cannot be read."""
    try:
        with Image.open(path) as image:
            return perceptual_hash(ImageOps.exif_transpose(image))
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hamming_distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_by_near_duplicate(
    items: Sequence[Tuple[Hashable, int]],
    max_distance: int = NEAR_DUPLICATE_DISTANCE,
) -> Dict[Hashable, List[Hashable]]:
    """
    Clusters (key, perceptual hash) pairs whose images are near-identical.

    Blocking on the leading hash bits keeps the comparison near-linear rather
    than quadratic over a six-figure image count. Only clusters of more than one
    member are returned; singletons need no grouping.
    """
    union_find = _UnionFind(len(items))
    blocks: Dict[int, List[int]] = defaultdict(list)
    for index, (_, digest) in enumerate(items):
        blocks[digest >> (HASH_BITS - BLOCK_BITS)].append(index)

    for bucket in blocks.values():
        for offset, left in enumerate(bucket):
            for right in bucket[offset + 1:]:
                if hamming_distance(items[left][1], items[right][1]) <= max_distance:
                    union_find.union(left, right)

    clusters: Dict[Hashable, List[Hashable]] = defaultdict(list)
    for index, (key, _) in enumerate(items):
        clusters[union_find.find(index)].append(key)
    return {root: members for root, members in clusters.items() if len(members) > 1}


def group_paths(
    paths: Iterable[Path],
    max_distance: int = NEAR_DUPLICATE_DISTANCE,
) -> Dict[str, List[Path]]:
    """
    Assigns every path to a group id, near-duplicates sharing one id.

    Each image is its own group unless it was clustered with others, so the
    result is always a complete partition of the input.
    """
    paths = list(paths)
    hashed: List[Tuple[Path, int]] = []
    for path in paths:
        digest = image_dhash(path)
        if digest is not None:
            hashed.append((path, digest))

    groups: Dict[str, List[Path]] = {str(path): [path] for path in paths}
    clusters = cluster_by_near_duplicate([(path, digest) for path, digest in hashed], max_distance)
    for index, members in enumerate(clusters.values()):
        group_id = f"nd{index:05d}"
        for member in members:
            groups[str(member)] = members
            groups.pop(str(member), None)
        groups[group_id] = list(members)
    return groups


def assign_groups_to_splits(
    groups: Dict[str, Sequence[Hashable]],
    ratios: Dict[str, float],
    seed: int = 0,
) -> Dict[str, str]:
    """
    Splits group ids across named splits in the given proportions.

    Groups are shuffled, packed largest-first, and each is given to the split
    with the largest remaining deficit against its target image count. Choosing
    by remaining deficit rather than by "fraction filled so far" is what keeps
    the ratios honest: comparing filled fractions makes every split look equally
    empty at the start and degenerates into a round robin.

    No group is ever split across two outputs, which is the property
    DATASET_GUIDE requires.
    """
    import random

    order = list(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda group_id: len(groups[group_id]), reverse=True)

    total_images = sum(len(groups[group_id]) for group_id in order)
    targets = {split: ratio * total_images for split, ratio in ratios.items()}
    assigned_counts = {split: 0 for split in ratios}
    assignment: Dict[str, str] = {}

    for group_id in order:
        size = len(groups[group_id])
        chosen = max(ratios, key=lambda split: targets[split] - assigned_counts[split])
        assignment[group_id] = chosen
        assigned_counts[chosen] += size

    return assignment
