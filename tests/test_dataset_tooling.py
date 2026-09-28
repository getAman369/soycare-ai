"""
Tests for Phase 0 dataset tooling: remote zip parsing, source registry,
capture-session grouping, split assignment, and the dataset audit gate.
"""
from pathlib import Path
import struct
import zipfile
import zlib
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

import config
from scripts import fetch_datasets
from scripts import audit_dataset
from src import dataset_groups
from src import prepare_dataset


class BytesReader:
    """In-memory stand-in for HttpReader so zip parsing is testable offline."""

    def __init__(self, raw: bytes):
        self.raw = raw
        self.size = len(raw)

    def read(self, offset: int, length: int) -> bytes:
        return self.raw[offset:offset + length]


class FakeResponse:
    """Minimal urlopen response that can deliver fewer bytes than promised."""

    def __init__(self, payload: bytes, status: int = 200):
        self._buffer = BytesIO(payload)
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_download_rejects_a_truncated_transfer(monkeypatch, tmp_path):
    """
    A short read looks like end-of-file, so the transfer must be verified against
    Content-Length instead of trusted.
    """
    payload = b"x" * 4096
    calls = []

    def fake_urlopen(request, timeout=None, context=None):
        calls.append(request)
        # The first attempt delivers a short prefix and reports the full length.
        if len(calls) == 1:
            return FakeResponse(payload[:1024], status=200)
        return FakeResponse(payload, status=200)

    monkeypatch.setattr(fetch_datasets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fetch_datasets, "probe_size", lambda url, timeout=60: len(payload))

    destination = tmp_path / "shard.parquet"
    fetch_datasets.download_to_file("https://example.invalid/shard.parquet", destination)

    assert destination.read_bytes() == payload
    assert len(calls) >= 2, "a truncated transfer must be retried"


def test_download_raises_when_never_complete(monkeypatch, tmp_path):
    payload = b"y" * 2048

    def fake_urlopen(request, timeout=None, context=None):
        return FakeResponse(payload[:512], status=200)

    monkeypatch.setattr(fetch_datasets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fetch_datasets, "probe_size", lambda url, timeout=60: len(payload))
    monkeypatch.setattr(fetch_datasets.time, "sleep", lambda seconds: None)

    with pytest.raises(IOError, match="completely"):
        fetch_datasets.download_to_file("https://example.invalid/shard.parquet", tmp_path / "s.parquet", attempts=2)


def test_download_accepts_a_server_that_reports_no_length(monkeypatch, tmp_path):
    """Without a Content-Length the transfer cannot be verified, so it must not fail."""
    payload = b"q" * 1500

    monkeypatch.setattr(fetch_datasets.urllib.request, "urlopen",
                        lambda request, timeout=None, context=None: FakeResponse(payload, status=200))
    monkeypatch.setattr(fetch_datasets, "probe_size", lambda url, timeout=60: None)

    destination = tmp_path / "nolength.bin"
    fetch_datasets.download_to_file("https://example.invalid/f", destination, attempts=2)
    assert destination.read_bytes() == payload


def test_download_discards_a_short_cached_file(monkeypatch, tmp_path):
    payload = b"z" * 3000
    destination = tmp_path / "cached.bin"
    destination.write_bytes(payload[:100])  # truncated by an earlier run

    monkeypatch.setattr(fetch_datasets.urllib.request, "urlopen",
                        lambda request, timeout=None, context=None: FakeResponse(payload, status=200))
    monkeypatch.setattr(fetch_datasets, "probe_size", lambda url, timeout=60: len(payload))

    fetch_datasets.download_to_file("https://example.invalid/f", destination)
    assert destination.read_bytes() == payload


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
def make_leaf_image(path: Path, seed: int = 0, lesions: int = 5, size=(320, 320), tone: int = 0) -> Path:
    """
    Writes a synthetic leaf photo with a deterministic lesion pattern.

    `tone` shifts the whole colour scheme so two images can be made genuinely
    different in structure, not just in lesion count: a difference hash keys on
    overall texture, so fixtures that only vary lesion count still collide.
    """
    base = [(46, 125, 50), (34, 139, 34), (76, 175, 80), (104, 159, 56)][tone % 4]
    image = Image.new("RGB", size, color=base)
    draw = ImageDraw.Draw(image)
    centre = size[0] // 2
    draw.line([(centre, 8), (centre, size[1] - 8)], fill=(20, 90, 20), width=3)
    for offset in range(30, size[1] - 20, 40):
        draw.line([(centre, offset), (centre - 60, offset + 25)], fill=(25, 100, 25), width=2)
        draw.line([(centre, offset), (centre + 60, offset + 25)], fill=(25, 100, 25), width=2)
    for index in range(lesions):
        x = 40 + ((index * 37 + seed * 11) % (size[0] - 80))
        y = 40 + ((index * 53 + seed * 17) % (size[1] - 80))
        draw.ellipse([x, y, x + 12, y + 12], fill=(139, 69, 19))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=90)
    return path


def build_zip_bytes(entries, compression=zipfile.ZIP_DEFLATED) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Remote zip parsing
# ---------------------------------------------------------------------------
def test_central_directory_round_trip_with_real_bytes():
    """Parses a locally built zip through the same code path used for remote zips."""
    payloads = {f"bacterial_blight/img_{i}.jpg": bytes([i]) * (500 + i * 7) for i in range(5)}
    raw = build_zip_bytes(payloads)

    parsed = fetch_datasets.read_central_directory(
        "http://example.invalid/x.zip", len(raw), reader=BytesReader(raw)
    )
    assert len(parsed) == len(payloads)
    for member in parsed:
        assert member.file_size == len(payloads[member.name])
        assert member.compress_type == zipfile.ZIP_DEFLATED

    recovered = b""
    for member in parsed:
        header = raw[member.header_offset:member.header_offset + 30]
        assert header[:4] == b"PK\x03\x04"
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        start = member.header_offset + 30 + name_len + extra_len
        payload = raw[start:start + member.compress_size]
        recovered += zlib.decompress(payload, -zlib.MAX_WBITS)
    assert recovered == b"".join(payloads[member.name] for member in parsed)


def test_central_directory_handles_stored_entries():
    raw = build_zip_bytes({"a/plain.jpg": b"z" * 64}, compression=zipfile.ZIP_STORED)
    parsed = fetch_datasets.read_central_directory(
        "http://example.invalid/x.zip", len(raw), reader=BytesReader(raw)
    )
    assert [m.compress_type for m in parsed] == [zipfile.ZIP_STORED]
    assert parsed[0].file_size == 64


def test_find_eocd_reports_entry_count():
    raw = build_zip_bytes({f"a/{i}.jpg": b"x" * 10 for i in range(9)})
    eocd_offset, count, directory_size, directory_offset = fetch_datasets.find_eocd(BytesReader(raw))
    assert count == 9
    assert directory_offset + directory_size <= len(raw)


def test_central_directory_rejects_garbage():
    with pytest.raises(IOError):
        fetch_datasets.find_eocd(BytesReader(b"not a zip file at all" * 100))


def test_list_archive_filters_non_images():
    payloads = {"a/one.jpg": b"x" * 10, "a/two.JPG": b"y" * 10, "a/notes.txt": b"z" * 10, "a/": b""}
    raw = build_zip_bytes(payloads)
    members = fetch_datasets.list_archive("http://example.invalid/x.zip", len(raw), reader=BytesReader(raw))
    names = {member.name for member in members}
    assert "a/one.jpg" in names and "a/two.JPG" in names
    assert "a/notes.txt" not in names


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
def test_every_registry_class_maps_to_a_target_class():
    registry = fetch_datasets.load_sources()
    assert registry["sources"], "registry declares no sources"
    for source in registry["sources"]:
        assert source["license"], f"{source['id']} has no license recorded"
        assert source["doi"], f"{source['id']} has no DOI recorded"
        for entry in fetch_datasets.source_classes(source).values():
            assert entry["target_class"] in config.TARGET_CLASSES


def test_registry_records_why_sources_were_rejected():
    registry = fetch_datasets.load_sources()
    assert registry.get("rejected_sources"), "rejected sources should record their rationale"
    for rejected in registry["rejected_sources"]:
        assert rejected.get("reason")


def test_target_classes_are_sorted_and_unique():
    assert config.TARGET_CLASSES == sorted(config.TARGET_CLASSES)
    assert len(set(config.TARGET_CLASSES)) == len(config.TARGET_CLASSES)


def test_known_coverage_gap_is_declared():
    """Septoria brown spot is the documented blocker for the six-class retrain."""
    registry = fetch_datasets.load_sources()
    coverage = registry["coverage"]["classes"]
    assert coverage["Septoria brown spot"]["meets_minimum"] is False
    assert registry["coverage"]["known_gaps"]


# ---------------------------------------------------------------------------
# Perceptual hashing and grouping
# ---------------------------------------------------------------------------
def test_perceptual_hash_is_stable_and_discriminating(tmp_path):
    first = make_leaf_image(tmp_path / "a.jpg", seed=1)
    same = make_leaf_image(tmp_path / "b.jpg", seed=1)
    different = make_leaf_image(tmp_path / "c.jpg", seed=5, lesions=14)

    assert dataset_groups.perceptual_hash(Image.open(first)) == dataset_groups.perceptual_hash(Image.open(same))
    assert dataset_groups.hamming_distance(
        dataset_groups.image_dhash(first), dataset_groups.image_dhash(different)
    ) > 0


def test_near_identical_frames_share_a_group(tmp_path):
    frame_one = make_leaf_image(tmp_path / "session" / "leaf_a_1.jpg", seed=3, lesions=6, tone=0)
    frame_two = make_leaf_image(tmp_path / "session" / "leaf_a_2.jpg", seed=3, lesions=6, tone=0)
    other = make_leaf_image(tmp_path / "other" / "leaf_b_1.jpg", seed=9, lesions=16, tone=3)

    groups = dataset_groups.group_paths([frame_one, frame_two, other])
    grouping = {str(path): group_id for group_id, members in groups.items() for path in members}

    assert grouping[str(frame_one)] == grouping[str(frame_two)]
    assert grouping[str(frame_one)] != grouping[str(other)]


def test_group_paths_partitions_every_input(tmp_path):
    paths = [make_leaf_image(tmp_path / f"img_{i}.jpg", seed=i, lesions=3 + i) for i in range(6)]
    groups = dataset_groups.group_paths(paths)

    assigned = [str(path) for members in groups.values() for path in members]
    assert sorted(assigned) == sorted(str(path) for path in paths)
    assert len(assigned) == len(set(assigned)), "an image landed in more than one group"


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------
def test_no_group_is_split_across_partitions():
    groups = {f"g{i}": [f"img_{i}_{j}" for j in range(3 + (i % 4))] for i in range(40)}
    assignment = dataset_groups.assign_groups_to_splits(
        groups, {"train": 0.8, "validation": 0.1, "test": 0.1}, seed=7
    )
    assert set(assignment) == set(groups)
    assert all(split in {"train", "validation", "test"} for split in assignment.values())


def test_split_assignment_respects_target_ratios():
    groups = {f"g{i}": [f"img_{i}_{j}" for j in range(2)] for i in range(200)}
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assignment = dataset_groups.assign_groups_to_splits(groups, ratios, seed=11)

    counts = {split: 0 for split in ratios}
    for group_id, split in assignment.items():
        counts[split] += len(groups[group_id])
    total = sum(counts.values())
    for split, ratio in ratios.items():
        assert abs(counts[split] / total - ratio) < 0.05


def test_split_assignment_is_deterministic_for_a_seed():
    groups = {f"g{i}": [f"img_{i}_{j}" for j in range(i % 5 + 1)] for i in range(30)}
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert dataset_groups.assign_groups_to_splits(groups, ratios, seed=3) == \
        dataset_groups.assign_groups_to_splits(groups, ratios, seed=3)


# ---------------------------------------------------------------------------
# Split integration
# ---------------------------------------------------------------------------
def test_split_raw_dataset_keeps_duplicate_frames_in_one_split(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"

    class_name = "Frogeye leaf spot"
    # Eight near-identical frames of one leaf plus eight distinct images.
    burst = [make_leaf_image(raw / class_name / f"burst_{i}.jpg", seed=2, lesions=6) for i in range(8)]
    distinct = [make_leaf_image(raw / class_name / f"solo_{i}.jpg", seed=20 + i, lesions=2 + i) for i in range(8)]

    monkeypatch.setattr(prepare_dataset, "DATA_DIR", tmp_path)
    summary = prepare_dataset.split_raw_dataset(
        raw, classes=[class_name], train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )

    assert summary is not None and class_name in summary
    placed = {}
    for split in ("train", "validation", "test"):
        for path in (processed / split / class_name).iterdir():
            placed[path.name] = split

    assert set(placed) == {path.name for path in burst + distinct}
    burst_splits = {placed[path.name] for path in burst}
    assert len(burst_splits) == 1, f"duplicate frames were split across {burst_splits}"


# ---------------------------------------------------------------------------
# Audit gate
# ---------------------------------------------------------------------------
def test_audit_fails_when_classes_are_missing(tmp_path):
    raw = tmp_path / "raw"
    make_leaf_image(raw / "Healthy" / "h1.jpg", seed=1)
    report = audit_dataset.run_audit(raw, min_images=2, write_report=False)

    assert report["gate_passed"] is False
    assert any("Downy mildew" in failure for failure in report["gate_failures"])
    assert report["classes"]["Healthy"]["readable"] == 1


def test_audit_passes_when_all_classes_are_populated(tmp_path):
    raw = tmp_path / "raw"
    for index, class_name in enumerate(config.TARGET_CLASSES):
        for item in range(3):
            make_leaf_image(raw / class_name / f"{class_name}_{item}.jpg", seed=index * 10 + item, lesions=3 + item)

    report = audit_dataset.run_audit(raw, min_images=3, write_report=False)
    assert report["gate_passed"] is True, report["gate_failures"]
    assert len(report["classes_meeting_minimum"]) == len(config.TARGET_CLASSES)


def test_audit_flags_unreadable_files(tmp_path):
    raw = tmp_path / "raw"
    for class_name in config.TARGET_CLASSES:
        make_leaf_image(raw / class_name / "ok.jpg", seed=1)
    (raw / "Healthy" / "corrupt.jpg").write_bytes(b"this is not an image")

    report = audit_dataset.run_audit(raw, min_images=1, write_report=False)
    assert report["gate_passed"] is False
    assert report["classes"]["Healthy"]["unreadable"] == 1
    assert any("unreadable" in failure for failure in report["gate_failures"])


def test_audit_detects_exact_duplicates(tmp_path):
    raw = tmp_path / "raw"
    for class_name in config.TARGET_CLASSES:
        make_leaf_image(raw / class_name / "a.jpg", seed=4)
    payload = (raw / "Healthy" / "a.jpg").read_bytes()
    (raw / "Healthy" / "b.jpg").write_bytes(payload)

    report = audit_dataset.run_audit(raw, min_images=1, write_report=False)
    assert report["classes"]["Healthy"]["exact_duplicate_groups"] == 1
    assert report["classes"]["Healthy"]["exact_duplicate_images"] == 2


def test_audit_reports_near_duplicate_groups(tmp_path):
    raw = tmp_path / "raw"
    for index, class_name in enumerate(config.TARGET_CLASSES):
        make_leaf_image(raw / class_name / "frame_1.jpg", seed=7, lesions=6, tone=index % 4)
        make_leaf_image(raw / class_name / "frame_2.jpg", seed=7, lesions=6, tone=index % 4)
        make_leaf_image(raw / class_name / "unique.jpg", seed=31, lesions=17, tone=(index + 1) % 4)

    report = audit_dataset.run_audit(raw, min_images=1, write_report=False)
    assert report["near_duplicate_group_count"] >= 1
    assert report["largest_near_duplicate_group"] >= 2


def test_burst_frames_group_together_within_each_class(tmp_path):
    """The property the splitter depends on: a burst shares one group id."""
    raw = tmp_path / "raw"
    for index, class_name in enumerate(config.TARGET_CLASSES):
        tone = index % 4
        make_leaf_image(raw / class_name / "frame_1.jpg", seed=7, lesions=6, tone=tone)
        make_leaf_image(raw / class_name / "frame_2.jpg", seed=7, lesions=6, tone=tone)
        make_leaf_image(raw / class_name / "unique.jpg", seed=29, lesions=17, tone=(tone + 2) % 4)

        groups = dataset_groups.group_paths(sorted((raw / class_name).iterdir()))
        mapping = {str(path): group for group, members in groups.items() for path in members}
        assert mapping[str(raw / class_name / "frame_1.jpg")] == mapping[str(raw / class_name / "frame_2.jpg")]
        assert mapping[str(raw / class_name / "unique.jpg")] != mapping[str(raw / class_name / "frame_1.jpg")]


def test_audit_fails_when_a_class_is_entirely_absent(tmp_path):
    """An empty class directory must not pass the gate."""
    raw = tmp_path / "raw"
    for class_name in config.TARGET_CLASSES:
        if class_name != "Downy mildew":
            make_leaf_image(raw / class_name / "a.jpg", seed=2)

    report = audit_dataset.run_audit(raw, min_images=1, write_report=False)
    assert report["gate_passed"] is False
    assert report["classes"]["Downy mildew"]["images"] == 0
    assert any("Downy mildew" in failure and "no images" in failure for failure in report["gate_failures"])


def test_cli_builds_fetcher_kwargs_each_signature_accepts(monkeypatch, tmp_path):
    """
    Guards a real regression: passing workers to every fetcher broke the
    non-zip sources with an unexpected-keyword TypeError.
    """
    import inspect

    registry = fetch_datasets.load_sources()
    for source in registry["sources"]:
        access_type = source["access"]["type"]
        fetcher = fetch_datasets.FETCHERS[access_type]
        accepted = set(inspect.signature(fetcher).parameters)

        kwargs = {
            "classes": [],
            "raw_dir": tmp_path,
            "max_per_class": None,
            "limit": None,
            "max_dimension": None,
        }
        if access_type == "range_request_zip":
            kwargs["workers"] = 4
        if access_type == "rest_file_list":
            kwargs["folder_map"] = None

        unexpected = set(kwargs) - accepted
        assert not unexpected, f"{source['id']} ({access_type}) rejects {unexpected}"


def test_cli_list_mode_runs(capsys):
    assert fetch_datasets.main(["--list"]) == 0
    assert "asdid" in capsys.readouterr().out


def test_cli_rejects_unknown_class():
    with pytest.raises(SystemExit):
        fetch_datasets.main(["--source", "asdid", "--classes", "Not A Disease"])


def test_mendeley_fetcher_requires_folder_map(tmp_path):
    source = fetch_datasets.get_source("india_soyabean_leaf")
    with pytest.raises(SystemExit, match="folder-map"):
        fetch_datasets.fetch_mendeley(
            source, classes=[], raw_dir=tmp_path, max_per_class=None, limit=None, folder_map=None
        )


# ---------------------------------------------------------------------------
# Model class alignment guard
# ---------------------------------------------------------------------------
class _FakeModel:
    def __init__(self, output_shape):
        self.output_shape = output_shape


def test_predictor_rejects_mismatched_class_count(tmp_path, monkeypatch):
    from src.predict import DiseasePredictor
    import src.predict as predict_module

    monkeypatch.setattr(
        predict_module, "DiseasePredictor._load_model", lambda self: setattr(self, "model", None), raising=False
    )
    predictor = DiseasePredictor(model_path=tmp_path / "absent.keras")
    predictor.model = _FakeModel((None, 224, 224, 3, 6))

    with pytest.raises(ValueError, match="outputs 6 classes"):
        predictor._verify_class_alignment()


def test_predictor_accepts_aligned_class_count(tmp_path, monkeypatch):
    from src.predict import DiseasePredictor
    import src.predict as predict_module

    monkeypatch.setattr(
        predict_module, "DiseasePredictor._load_model", lambda self: setattr(self, "model", None), raising=False
    )
    predictor = DiseasePredictor(model_path=tmp_path / "absent.keras")
    predictor.model = _FakeModel((None, 224, 224, 3, len(config.CLASSES)))
    predictor._verify_class_alignment()


def test_predictor_handles_multi_output_model_shape(tmp_path, monkeypatch):
    from src.predict import DiseasePredictor
    import src.predict as predict_module

    monkeypatch.setattr(
        predict_module, "DiseasePredictor._load_model", lambda self: setattr(self, "model", None), raising=False
    )
    predictor = DiseasePredictor(model_path=tmp_path / "absent.keras")
    predictor.model = _FakeModel([(None, 224, 224, 3, 6)])

    with pytest.raises(ValueError):
        predictor._verify_class_alignment()
