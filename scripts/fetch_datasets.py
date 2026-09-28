"""
Phase 0 dataset acquisition.

Fetches real labelled soybean leaf photographs declared in data/sources.json into
data/raw/<class>/ together with a provenance manifest.

The ASDID archives are 2.8-8.4 GB each but are served by Zenodo with HTTP range
support, so this script reads each zip's central directory over ranged requests and
pulls only the image members it needs, rather than downloading whole archives.

Usage:
    python -m scripts.fetch_datasets --source asdid --classes "Frogeye leaf spot,Healthy"
    python -m scripts.fetch_datasets --source asdid --all --max-per-class 400
    python -m scripts.fetch_datasets --source agml_india_soyabean
    python -m scripts.fetch_datasets --list
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import ssl
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DATA_DIR, DATASET_SOURCES_PATH, TARGET_CLASSES, logger

MANIFEST_NAME = "_provenance.jsonl"
CACHE_DIR = DATA_DIR / "download_cache"
USER_AGENT = "SoyCareAI/0.1 (dataset acquisition; contact: repository maintainer)"
CHUNK = 1 << 20
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEFAULT_WORKERS = 8


def _ssl_context() -> ssl.SSLContext:
    """
    Builds a verifying TLS context. Some Python builds (notably the macOS
    framework build) ship without a usable CA bundle, so prefer certifi's
    certificates when it is installed rather than silently trusting anything.
    """
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return context
    try:
        context.load_verify_locations(cafile=certifi.where())
    except (OSError, ssl.SSLError) as err:
        logger.warning("Could not load certifi CA bundle (%s); using system trust store.", err)
    return context


SSL_CONTEXT = _ssl_context()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class HttpReader:
    """
    Read-only ranged HTTP reader over a keep-alive connection.

    Reusing one TLS connection matters a lot here: the archives are served as
    many small member ranges, and a fresh handshake per member dominated
    runtime. Connections are kept per thread so extraction can be parallelised.
    """

    _local = threading.local()

    def __init__(self, url: str, size: int | None = None, retries: int = 4, timeout: int = 60):
        self.url = url
        self.size = size
        self.retries = retries
        self.timeout = timeout
        self._bytes_read = 0
        parsed = urllib.parse.urlsplit(url)
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._secure = parsed.scheme == "https"

    def _connection(self) -> http.client.HTTPConnection:
        pool = getattr(HttpReader._local, "connections", None)
        if pool is None:
            pool = {}
            HttpReader._local.connections = pool
        key = (self._host, self._port, self._secure)
        connection = pool.get(key)
        if connection is None:
            if self._secure:
                connection = http.client.HTTPSConnection(
                    self._host, self._port, timeout=self.timeout, context=SSL_CONTEXT
                )
            else:
                connection = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)
            pool[key] = connection
        return connection

    @staticmethod
    def close_thread_connections() -> None:
        pool = getattr(HttpReader._local, "connections", None)
        if not pool:
            return
        for connection in pool.values():
            try:
                connection.close()
            except OSError:
                pass
        pool.clear()

    def read(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        last_error: Exception | None = None
        for attempt in range(self.retries):
            connection = self._connection()
            try:
                connection.request(
                    "GET",
                    self.url,
                    headers={"User-Agent": USER_AGENT, "Range": f"bytes={offset}-{offset + length - 1}"},
                )
                response = connection.getresponse()
                data = response.read()
                if response.status not in (200, 206):
                    raise IOError(f"HTTP {response.status} for {self.url} at offset {offset}")
                if not data:
                    raise IOError(f"Empty response for {self.url} at offset {offset}")
                self._bytes_read += len(data)
                return data
            except (OSError, http.client.HTTPException) as err:
                # A reused connection can be closed by the server at any time,
                # so drop it and reconnect on the next attempt.
                self.close_thread_connections()
                last_error = err
                time.sleep(min(2 ** attempt, 15))
        raise IOError(f"Failed reading {self.url} at offset {offset}: {last_error}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence != os.SEEK_SET:
            raise ValueError("HttpReader only supports absolute seeks")
        return offset

    def tell(self) -> int:
        raise OSError("HttpReader position is tracked by callers")

    @property
    def bytes_read(self) -> int:
        return self._bytes_read


def fetch_json(url: str, timeout: int = 60) -> Dict[str, Any]:
    """Fetches and decodes a JSON document over the verifying TLS context."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
        return json.loads(response.read().decode("utf-8"))


def probe_size(url: str, timeout: int = 60) -> Optional[int]:
    """Returns the resource size in bytes, or None if the server does not say."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
            length = response.headers.get("Content-Length")
            if length:
                return int(length)
            return int(response.headers["Content-Range"].split("/")[-1])
    except (urllib.error.URLError, KeyError, ValueError, TypeError) as err:
        logger.warning("Could not determine remote size for %s: %s", url, err)
        return None


def download_to_file(
    url: str,
    destination: Path,
    timeout: int = 120,
    attempts: int = 4,
) -> Path:
    """
    Streams a URL to disk, resuming and verifying until the full size arrives.

    A short read is indistinguishable from end-of-file unless the expected length
    is checked, and a silently truncated parquet or zip is worse than no file at
    all: it fails much later with a confusing parse error. The transfer is
    therefore resumed with a Range request and rejected unless it matches
    Content-Length exactly.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = probe_size(url, timeout=timeout)

    if destination.exists() and destination.stat().st_size > 0:
        if expected is None or destination.stat().st_size == expected:
            logger.info("Using cached download %s", destination.name)
            return destination
        logger.warning(
            "Cached %s is %d bytes but the server reports %d; re-downloading.",
            destination.name, destination.stat().st_size, expected,
        )
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(attempts):
        have = partial.stat().st_size if partial.exists() else 0
        if expected is not None and have == expected:
            break

        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"

        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
                # A resumed transfer answers 206; a fresh one answers 200 and
                # restarts, so truncate before appending.
                mode = "ab" if have and response.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(partial, mode) as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            last_error = err
            logger.warning("Transfer of %s interrupted: %s", destination.name, err)

        received = partial.stat().st_size if partial.exists() else 0
        if expected is None:
            # No advertised length, so the transfer cannot be verified. Accept a
            # single completed pass rather than looping into a false failure.
            logger.info("Server reported no length for %s; accepting %d bytes.", destination.name, received)
            break
        if received == expected:
            break
        logger.warning(
            "%s: got %d of %d bytes; retrying.",
            destination.name, received, expected,
        )
        time.sleep(min(2 ** attempt, 15))
    else:
        received = partial.stat().st_size if partial.exists() else 0
        raise IOError(
            f"Failed to download {url} completely: got {received} of {expected} bytes"
            + (f" ({last_error})" if last_error else "")
        )

    partial.replace(destination)
    logger.info("Downloaded %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
    return destination


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
def load_sources(path: Path | None = None) -> Dict[str, Any]:
    path = path or DATASET_SOURCES_PATH
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def get_source(source_id: str, path: Path | None = None) -> Dict[str, Any]:
    registry = load_sources(path)
    for source in registry["sources"]:
        if source["id"] == source_id:
            return source
    known = ", ".join(s["id"] for s in registry["sources"])
    raise SystemExit(f"Unknown source '{source_id}'. Known sources: {known}")


def source_classes(source: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Maps native class key -> entry for the target classes this source supplies."""
    return {
        key: entry
        for key, entry in source.get("classes", {}).items()
        if entry.get("target_class") in TARGET_CLASSES
    }


# ---------------------------------------------------------------------------
# Remote zip extraction
# ---------------------------------------------------------------------------
EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"
ZIP64_EOCD_LOCATOR = b"PK\x06\x07"
EOCD_TAIL_BYTES = 66_000  # comment field is 16-bit, so the tail always holds the EOCD


@dataclass(frozen=True)
class RemoteMember:
    """One file entry in a remote zip, located by its local-header offset."""

    name: str
    compress_size: int
    file_size: int
    compress_type: int
    header_offset: int


def find_eocd(reader: HttpReader) -> Tuple[int, int, int, int]:
    """Returns (eocd_offset, entry_count, central_dir_size, central_dir_offset)."""
    size = reader.size
    if size is None:
        raise IOError("Resource size is required to read a remote zip")
    tail_start = max(0, size - EOCD_TAIL_BYTES)
    tail = reader.read(tail_start, size - tail_start)
    position = tail.rfind(EOCD_SIGNATURE)
    if position < 0:
        raise IOError("End of central directory record not found")
    eocd_offset = tail_start + position
    # signature, disk numbers, entries-on-disk, total entries, dir size, dir offset, comment length
    fields = struct.unpack_from("<4s4H2LH", tail, position)
    entry_count, directory_size, directory_offset = fields[4], fields[5], fields[6]
    return eocd_offset, entry_count, directory_size, directory_offset


def read_central_directory(
    url: str,
    size: int,
    retries: int = 4,
    reader: Optional[HttpReader] = None,
) -> List[RemoteMember]:
    """
    Parses a remote zip's central directory, including the zip64 variant.

    `reader` is injectable so tests can parse locally built zips through the same
    code path without touching the network.
    """
    reader = reader or HttpReader(url, size, retries=retries)
    eocd_offset, entry_count, directory_size, directory_offset = find_eocd(reader)

    if directory_offset == 0xFFFFFFFF or entry_count == 0xFFFF:
        locator_position = eocd_offset - 20
        locator = reader.read(locator_position, 20)
        if locator[:4] != ZIP64_EOCD_LOCATOR:
            raise IOError("zip64 locator not found")
        zip64_eocd_offset = struct.unpack_from("<Q", locator, 8)[0]
        zip64_record = reader.read(zip64_eocd_offset, 56)
        if zip64_record[:4] != b"PK\x06\x06":
            raise IOError("zip64 end of central directory record not found")
        entry_count, directory_size, directory_offset = struct.unpack_from("<QQQ", zip64_record, 24)

    directory = reader.read(directory_offset, directory_size)

    members: List[RemoteMember] = []
    position = 0
    for _ in range(entry_count):
        if directory[position:position + 4] != CENTRAL_SIGNATURE:
            break
        # signature, versions, flags, method, mod time/date, crc, sizes, name
        # length, extra length, comment length, disk start, attributes, offset
        fields = struct.unpack_from("<4s6H3L5H2L", directory, position)
        compress_type = fields[4]
        name_len, extra_len, comment_len = fields[10], fields[11], fields[12]
        compress_size, file_size = fields[8], fields[9]
        header_offset = fields[16]
        name = directory[position + 46:position + 46 + name_len].decode("utf-8", "replace")

        if file_size == 0xFFFFFFFF or compress_size == 0xFFFFFFFF or header_offset == 0xFFFFFFFF:
            file_size, compress_size, header_offset = _read_zip64_extra(
                directory[position + 46 + name_len:position + 46 + name_len + extra_len],
                file_size,
                compress_size,
                header_offset,
            )

        members.append(RemoteMember(
            name=name,
            compress_size=compress_size,
            file_size=file_size,
            compress_type=compress_type,
            header_offset=header_offset,
        ))
        position += 46 + name_len + extra_len + comment_len

    return members


def _read_zip64_extra(extra: bytes, file_size: int, compress_size: int, header_offset: int) -> Tuple[int, int, int]:
    """Extracts the real sizes from a zip64 extended information extra field."""
    position = 0
    while position + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, position)
        payload = extra[position + 4:position + 4 + size]
        if header_id == 0x0001:
            cursor = 0
            if file_size == 0xFFFFFFFF:
                file_size = struct.unpack_from("<Q", payload, cursor)[0]
                cursor += 8
            if compress_size == 0xFFFFFFFF:
                compress_size = struct.unpack_from("<Q", payload, cursor)[0]
                cursor += 8
            if header_offset == 0xFFFFFFFF:
                header_offset = struct.unpack_from("<Q", payload, cursor)[0]
            break
        position += 4 + size
    return file_size, compress_size, header_offset


def fetch_member(
    reader: HttpReader,
    member: RemoteMember,
    destination: Path,
    max_dimension: Optional[int] = None,
) -> Path:
    """
    Downloads and inflates one member of a remote zip to destination.

    When max_dimension is set, the image is downscaled on the way to disk. The
    source photographs are 5472x3648 at roughly 6 MB each, so storing them
    untouched would cost about 30 GB for the six target classes while the model
    only ever sees 224x224. Downscaling at ingest keeps full framing and lesion
    detail while making the corpus cheap to store and audit.
    """
    if destination.exists() and destination.stat().st_size:
        if max_dimension is None or destination.stat().st_size != member.file_size:
            return destination

    header = reader.read(member.header_offset, 30)
    if header[:4] != LOCAL_SIGNATURE:
        raise IOError(f"Bad local header signature for {member.name}")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    data_offset = member.header_offset + 30 + name_len + extra_len

    payload = reader.read(data_offset, member.compress_size)
    if member.compress_type == zipfile.ZIP_STORED:
        data = payload
    elif member.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(payload, -zlib.MAX_WBITS)
    else:
        raise IOError(f"Unsupported zip compression method {member.compress_type} for {member.name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if max_dimension:
        temporary.write_bytes(_downscale(data, max_dimension))
    else:
        temporary.write_bytes(data)
    temporary.replace(destination)
    return destination


def _downscale(data: bytes, max_dimension: int) -> bytes:
    """Resizes an encoded image so its longest edge is max_dimension, as JPEG."""
    from PIL import Image, ImageOps

    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            target = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
            image = image.resize(target, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
        return buffer.getvalue()


def list_archive(
    url: str,
    size: int,
    reader: Optional[HttpReader] = None,
) -> List[RemoteMember]:
    """Lists image members of a remote zip without downloading its payload."""
    return [
        member for member in read_central_directory(url, size, reader=reader)
        if member.name.lower().endswith(IMAGE_SUFFIXES) and not member.name.endswith("/")
    ]


def extract_members(
    url: str,
    size: int,
    members: List[RemoteMember],
    destination: Path,
    max_dimension: Optional[int] = None,
    workers: int = DEFAULT_WORKERS,
) -> List[Path]:
    """
    Extracts the given members from a remote zip into destination, in parallel.

    Each worker keeps its own TLS connection, so throughput scales with the
    worker count rather than with sequential handshakes.
    """
    destination.mkdir(parents=True, exist_ok=True)

    def pull(member: RemoteMember) -> Path:
        reader = HttpReader(url, size)
        try:
            return fetch_member(reader, member, destination / Path(member.name).name, max_dimension)
        finally:
            HttpReader.close_thread_connections()

    if workers <= 1 or len(members) <= 1:
        return [pull(member) for member in members]

    written: List[Path] = []
    failures: List[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(pull, member): member for member in members}
        for future in as_completed(futures):
            member = futures[future]
            try:
                written.append(future.result())
            except Exception as err:  # noqa: BLE001 - one bad member must not abort the run
                failures.append(f"{member.name}: {err}")
                logger.warning("Failed to extract %s: %s", member.name, err)
            completed += 1
            if completed % 50 == 0 or completed == len(members):
                logger.info("Extracted %d/%d", completed, len(members))

    if failures:
        logger.error("%d of %d members failed to extract", len(failures), len(members))
    return written


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def manifest_path(raw_dir: Path) -> Path:
    return raw_dir / MANIFEST_NAME


def append_manifest(raw_dir: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path(raw_dir), "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_manifest(raw_dir: Path) -> List[Dict[str, Any]]:
    path = manifest_path(raw_dir)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def fetch_asdid(
    source: Dict[str, Any],
    classes: List[str],
    raw_dir: Path,
    max_per_class: Optional[int],
    limit: Optional[int],
    max_dimension: Optional[int] = None,
    workers: int = DEFAULT_WORKERS,
) -> Dict[str, int]:
    template = source["access"]["file_url_template"]
    fetched: Dict[str, int] = {}

    for native_key, entry in source_classes(source).items():
        target_class = entry["target_class"]
        if classes and target_class not in classes:
            continue

        archive_name = entry["archive"]
        url = template.format(file=archive_name)
        size = probe_size(url)
        if size is None:
            logger.error("Server did not report a size for %s; skipping", archive_name)
            continue

        logger.info("Listing %s (%.1f MB archive) over HTTP range requests", archive_name, size / 1e6)
        members = list_archive(url, size)
        members.sort(key=lambda info: info.name)

        budget = limit or max_per_class or len(members)
        selected = members[:budget]

        destination = raw_dir / target_class
        written = extract_members(url, size, selected, destination, max_dimension, workers)
        logger.info("%s: extracted %d/%d images", target_class, len(written), len(members))
        fetched[target_class] = fetched.get(target_class, 0) + len(written)

        append_manifest(raw_dir, [
            {
                "filename": path.name,
                "target_class": target_class,
                "native_class": native_key,
                "source_id": source["id"],
                "source_doi": source["doi"],
                "license": source["license"],
                "archive": archive_name,
                "bytes": path.stat().st_size,
            }
            for path in written
        ])
    return fetched


def fetch_hf_parquet(
    source: Dict[str, Any],
    classes: List[str],
    raw_dir: Path,
    max_per_class: Optional[int],
    limit: Optional[int],
    max_dimension: Optional[int] = None,
) -> Dict[str, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("pyarrow is required for parquet sources: pip install pyarrow")

    template = source["access"]["file_url_template"]
    label_map = {key: entry for key, entry in source_classes(source).items()}

    cache = CACHE_DIR / source["id"]
    cache.mkdir(parents=True, exist_ok=True)
    listing = fetch_json(f"{source['access']['listing_url']}?full=true", timeout=60)

    shards = [
        sibling["rfilename"]
        for sibling in listing.get("siblings", [])
        if sibling["rfilename"].endswith(".parquet")
    ]
    if not shards:
        raise SystemExit(f"No parquet shards listed for source '{source['id']}'")

    fetched: Dict[str, int] = {entry["target_class"]: 0 for entry in label_map.values()}

    for shard in sorted(shards):
        budget = limit or max_per_class
        if budget and all(count >= budget for count in fetched.values()):
            break

        local = cache / Path(shard).name
        download_to_file(template.format(file=shard), local)

        table = pq.read_table(local)
        labels = table.column("label").to_pylist() if "label" in table.column_names else []
        images = table.column("image").to_pylist()

        for label, image in zip(labels, images):
            target_class = label_map.get(label, {}).get("target_class")
            if target_class is None:
                continue
            if classes and target_class not in classes:
                continue
            budget = limit or max_per_class
            if budget and fetched.get(target_class, 0) >= budget:
                continue

            destination = raw_dir / target_class
            destination.mkdir(parents=True, exist_ok=True)
            name = Path(image.get("path") or f"{source['id']}_{hashlib.sha1(image['bytes']).hexdigest()[:12]}.jpg").name
            target = destination / name
            if not target.exists():
                payload = image["bytes"]
                if max_dimension:
                    payload = _downscale(payload, max_dimension)
                target.write_bytes(payload)
            fetched[target_class] = fetched.get(target_class, 0) + 1
            append_manifest(raw_dir, [{
                "filename": name,
                "target_class": target_class,
                "native_class": label,
                "source_id": source["id"],
                "source_doi": source["doi"],
                "license": source["license"],
                "archive": shard,
                "bytes": target.stat().st_size,
            }])

        logger.info("Processed shard %s: %d target images so far", local.name, sum(fetched.values()))

    logger.info("%s: %s", source["id"], fetched)
    return fetched


def fetch_mendeley(
    source: Dict[str, Any],
    classes: List[str],
    raw_dir: Path,
    max_per_class: Optional[int],
    limit: Optional[int],
    folder_map: Optional[Path],
    max_dimension: Optional[int] = None,
) -> Dict[str, int]:
    """
    Downloads the Mendeley India dataset, which needs an operator-supplied
    folder-id to native-class map because the public API does not expose folder names.
    """
    if not folder_map:
        raise SystemExit(
            "Source 'india_soyabean_leaf' requires --folder-map: a JSON file of "
            '{"<folder_id>": "<native class name>"} built from the dataset web UI.'
        )
    with open(folder_map, encoding="utf-8") as handle:
        mapping = json.load(handle)

    template = source["access"]["file_url_template"]
    listing = fetch_json(source["access"]["listing_url"], timeout=90)

    fetched: Dict[str, int] = {}
    for entry in listing.get("files", []):
        native = mapping.get(entry.get("folder_id"))
        if native is None:
            continue
        target_class = source.get("classes", {}).get(native, {}).get("target_class")
        if target_class is None or (classes and target_class not in classes):
            continue
        budget = limit or max_per_class
        if budget and fetched.get(target_class, 0) >= budget:
            continue

        destination = raw_dir / target_class
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / entry["filename"]
        if not target.exists():
            url = template.format(file_id=entry["id"])
            download_to_file(url, target)
        fetched[target_class] = fetched.get(target_class, 0) + 1
        append_manifest(raw_dir, [{
            "filename": entry["filename"],
            "target_class": target_class,
            "native_class": native,
            "source_id": source["id"],
            "source_doi": source["doi"],
            "license": source["license"],
            "bytes": target.stat().st_size,
        }])
    return fetched


FETCHERS = {
    "range_request_zip": fetch_asdid,
    "hf_parquet": fetch_hf_parquet,
    "rest_file_list": fetch_mendeley,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_classes(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real soybean leaf datasets declared in data/sources.json")
    parser.add_argument("--source", help="Source id from the registry")
    parser.add_argument("--all", action="store_true", help="Fetch every registry source")
    parser.add_argument("--classes", help="Comma-separated target class names to limit the fetch")
    parser.add_argument("--max-per-class", type=int, help="Stop after N images per class")
    parser.add_argument("--limit", type=int, help="Alias for --max-per-class")
    parser.add_argument("--raw-dir", default=str(DATA_DIR / "raw"), help="Destination raw dataset directory")
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
        help="Downscale ingested images so the longest edge is at most this many pixels (0 keeps originals)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel range-request connections used for zip extraction",
    )
    parser.add_argument("--folder-map", help="folder-id to native-class JSON for Mendeley sources")
    parser.add_argument("--list", action="store_true", help="List registry sources and exit")
    args = parser.parse_args(argv)

    if args.list:
        registry = load_sources()
        for source in registry["sources"]:
            print(f"{source['id']:<22} {source['license']:<12} {source['name']}")
            for native, entry in source_classes(source).items():
                print(f"    - {entry['target_class']:<22} (native: {native})")
        print("\nRejected sources:")
        for rejected in registry.get("rejected_sources", []):
            print(f"    - {rejected['name']}: {rejected['reason']}")
        return 0

    if not args.source and not args.all:
        parser.error("provide --source, --all, or --list")

    source_ids = [s["id"] for s in load_sources()["sources"]] if args.all else [args.source]
    classes = parse_classes(args.classes)
    if classes:
        unknown = [c for c in classes if c not in TARGET_CLASSES]
        if unknown:
            raise SystemExit(f"Unknown target class(es): {unknown}. Known: {TARGET_CLASSES}")

    raw_dir = Path(args.raw_dir)
    totals: Dict[str, int] = {}
    for source_id in source_ids:
        source = get_source(source_id)
        fetcher = FETCHERS.get(source["access"]["type"])
        if fetcher is None:
            logger.error("No fetcher for access type %s; skipping %s", source["access"]["type"], source_id)
            continue
        logger.info("Fetching %s (%s)", source["name"], source["license"])
        kwargs: Dict[str, Any] = {
            "classes": classes,
            "raw_dir": raw_dir,
            "max_per_class": args.max_per_class,
            "limit": args.limit,
            "max_dimension": args.max_dimension or None,
        }
        access_type = source["access"]["type"]
        if access_type == "range_request_zip":
            kwargs["workers"] = args.workers
        if access_type == "rest_file_list":
            kwargs["folder_map"] = Path(args.folder_map) if args.folder_map else None
        for target_class, count in fetcher(source, **kwargs).items():
            totals[target_class] = totals.get(target_class, 0) + count

    print("\nFetched image counts per class:")
    for target_class in TARGET_CLASSES:
        print(f"  {target_class:<22} {totals.get(target_class, 0)}")
    print(f"\nProvenance manifest: {manifest_path(raw_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
