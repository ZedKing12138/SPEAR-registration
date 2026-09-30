"""Official-source dataset acquisition, separate from registration and results.

IXI distributes raw T1 scans without regional labels. OASIS-1 requires its
current access/usage process; the historical archive host is not a substitute
for authorization. IBSR18 on NITRC may require an authenticated session.
This module never logs cookies or signed URL query strings. Local archives
and a private URL manifest support changed or authenticated provider links.

An upstream SHA-256 can be supplied in a URL manifest. When none is published,
we record the downloaded hash for provenance, but do not call it an upstream
integrity guarantee. Archive extraction rejects traversal and linked files.
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile


DATASETS = {
    "ixi": {
        "name": "IXI T1",
        "homepage": "https://brain-development.org/ixi-dataset/",
        "access": "CC BY-SA 3.0; acknowledge IXI; raw T1 archive has no regional labels",
        "urls": [
            {
                "url": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
                "filename": "IXI-T1.tar",
            }
        ],
    },
    "oasis": {
        "name": "OASIS-1 cross-sectional (one MR1 session per subject)",
        "homepage": "https://sites.wustl.edu/oasisbrains/",
        "access_url": "https://sites.wustl.edu/oasisbrains/request-access/",
        "access": "Complete the current OASIS request/usage process before downloading; --terms-accepted declares that you have already done so",
        # The official directory lists 12 raw discs. FreeSurfer discs are a
        # separate derivative and are not substituted for these raw releases.
        "urls": [
            {
                "url": f"https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc{i}.tar.gz",
                "filename": f"oasis_cross-sectional_disc{i}.tar.gz",
            }
            for i in range(1, 13)
        ],
    },
    "ibsr18": {
        "name": "IBSR v2.0: 18 subjects, skull-stripped NIfTI with segmentations",
        "homepage": "https://www.nitrc.org/projects/ibsr/",
        "access_url": "https://www.nitrc.org/frs/?group_id=48",
        "access": "Follow IBSR/NITRC terms; a NITRC login may be required",
        "urls": [
            {
                "url": "https://www.nitrc.org/frs/download.php/5731/IBSR_V2.0_nifti_stripped.tgz",
                "filename": "IBSR_V2.0_nifti_stripped.tgz",
            }
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_url(url: str) -> str:
    """Strip credentials, fragments and signed query parameters from provenance."""
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))


def load_url_manifest(path: Path) -> list[dict]:
    """Read private JSON: [{url, filename, sha256?}, ...] (or {archives: [...]}).

    Keep this file under data/ or name it *.urls.json so it is Git-ignored.
    Use --cookie-file for a Netscape cookie jar instead of embedding passwords.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("archives") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("URL manifest must contain a non-empty list of archive records")
    names = set()
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            raise ValueError("Each archive needs a URL and an explicit local filename")
        name = item.get("filename", "")
        if not name or Path(name).name != name or "\\" in name or name in (".", ".."):
            raise ValueError("Archive filename must be a single safe basename")
        if name in names:
            raise ValueError("Archive filenames must be unique")
        names.add(name)
        checksum = item.get("sha256")
        if checksum and (
            len(checksum) != 64 or any(c not in "0123456789abcdefABCDEF" for c in checksum)
        ):
            raise ValueError("sha256 must be 64 hexadecimal characters")
        parsed = urllib.parse.urlsplit(item["url"])
        if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username:
            raise ValueError("Use HTTP(S) URLs without embedded login credentials")
    return entries


def make_opener(cookie_file: Path | None = None) -> urllib.request.OpenerDirector:
    """Reuse user-authorized cookies without persisting or displaying them."""
    handlers = []
    if cookie_file is not None:
        jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
        jar.load(ignore_discard=True, ignore_expires=False)
        handlers.append(urllib.request.HTTPCookieProcessor(jar))
    return urllib.request.build_opener(*handlers)


def fetch_archive(entry: dict, destination: Path, opener=None) -> dict:
    """Download atomically and reuse validated cached archives.

    Partial downloads are kept separate and removed after a failed transfer.
    We intentionally do not append HTTP ranges without validating ETags,
    because mixing two remote archive versions corrupts large MRI downloads.
    """
    destination.mkdir(parents=True, exist_ok=True)
    name = entry["filename"]
    if not isinstance(name, str) or not name or name in (".", ".."):
        raise ValueError("Unsafe archive filename")
    path = destination / name
    if path.name != name or "\\" in name:
        raise ValueError("Unsafe archive filename")
    if path.is_symlink():
        raise ValueError("Cached archive must not be a symbolic link")
    opener = opener or make_opener()
    expected = entry.get("sha256", "").lower()
    if not path.exists():
        partial = None
        request = urllib.request.Request(
            entry["url"], headers={"User-Agent": "SPEAR-dataset-tool/0.1"}
        )
        try:
            with opener.open(request, timeout=120) as response:
                ctype = response.headers.get("Content-Type", "").lower()
                initial = response.read(4096)
                if "text/html" in ctype or initial.lstrip().lower().startswith(
                    (b"<!doctype html", b"<html")
                ):
                    raise RuntimeError(
                        "The provider returned an HTML/login page. Complete its access process; pass an authorized cookie file, URL manifest or local archive."
                    )
                length = response.headers.get("Content-Length")
                with tempfile.NamedTemporaryFile(
                    dir=destination, prefix=name + ".", suffix=".download", delete=False
                ) as stream:
                    partial = Path(stream.name)
                    stream.write(initial)
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                if length is not None and partial.stat().st_size != int(length):
                    raise RuntimeError("Incomplete download; rerun to restart the transfer")
            checksum = sha256(partial)
            if expected and checksum != expected:
                partial.unlink()
                raise RuntimeError("Archive SHA-256 does not match the provided upstream hash")
            # Test the container before promoting it. Rejects downloaded error
            # pages even when a server incorrectly labels them as binary data.
            if not (zipfile.is_zipfile(partial) or tarfile.is_tarfile(partial)):
                partial.unlink()
                raise RuntimeError("Downloaded file is not a supported tar/zip archive")
            partial.replace(path)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Dataset download returned HTTP {exc.code}; check provider access or use --archive/--url-list"
            ) from None
        except urllib.error.URLError:
            raise RuntimeError(
                "Dataset host could not be reached; retry or supply a local --archive"
            ) from None
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
    checksum = sha256(path)
    if expected and checksum != expected:
        raise RuntimeError(f"Cached archive {path.name} does not match the supplied SHA-256")
    return {
        "filename": path.name,
        "source": _public_url(entry["url"]),
        "sha256": checksum,
        "upstream_hash_verified": bool(expected),
        "bytes": path.stat().st_size,
    }


def _target(root: Path, member: str) -> Path:
    name = PurePosixPath(member)
    if name.is_absolute() or ".." in name.parts or "\\" in member:
        raise ValueError("Unsafe path in dataset archive")
    target = root.joinpath(*name.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Archive entry would escape the destination directory")
    return target


def extract_archive(archive: Path, output: Path, max_bytes: int = 128 * 1024**3) -> int:
    """Extract regular tar/zip files only; never materialize archive links.

    A per-file temporary file prevents a terminated extraction from leaving
    half a NIfTI volume. Re-running extraction is idempotent. Archive metadata
    is preserved as content, not executed or used to set file permissions.
    """
    output.mkdir(parents=True, exist_ok=True)
    total, count = 0, 0

    def copy(source, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        # A unique temporary file cannot follow a pre-existing *.extracting
        # symlink supplied by another archive or left in the data directory.
        partial = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".extract-", delete=False
            ) as stream:
                partial = Path(stream.name)
                shutil.copyfileobj(source, stream, length=1024 * 1024)
            os.replace(partial, target)
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as container:
            for item in container.infolist():
                target = _target(output, item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Linked archive entries are unsupported")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                total += item.file_size
                if total > max_bytes:
                    raise ValueError("Archive exceeds extraction size limit")
                with container.open(item) as source:
                    copy(source, target)
                count += 1
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r|*") as container:
            for item in container:
                target = _target(output, item.name)
                if item.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not item.isfile():
                    raise ValueError("Non-regular/linked archive entries are unsupported")
                total += item.size
                if total > max_bytes:
                    raise ValueError("Archive exceeds extraction size limit")
                with container.extractfile(item) as source:
                    copy(source, target)
                count += 1
    else:
        raise ValueError("Expected .tar, .tar.gz/.tgz or .zip dataset archive")
    return count


def acquire_dataset(
    dataset: str,
    output: Path,
    *,
    archives: list[Path] | None = None,
    url_manifest: Path | None = None,
    cookie_file: Path | None = None,
    terms_accepted: bool = False,
) -> Path:
    """Return extracted raw/ directory and write local-only download provenance.

    --terms-accepted is a declaration made by the script's human operator; it
    cannot grant access or replace any approval required by the data provider.
    Already obtained local archives do not trigger a new download/acceptance.
    """
    info = DATASETS[dataset]
    output = Path(output).resolve()
    records, paths = [], []
    if archives and url_manifest:
        raise ValueError("Choose local --archive OR --url-list")
    if archives:
        for path in archives:
            path = Path(path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
            records.append(
                {
                    "filename": path.name,
                    "source": "operator-provided local archive",
                    "sha256": sha256(path),
                    "upstream_hash_verified": False,
                    "bytes": path.stat().st_size,
                }
            )
    else:
        if not terms_accepted:
            raise ValueError(
                f"Read/follow {info.get('access_url', info['homepage'])}; after authorization, pass --terms-accepted"
            )
        entries = load_url_manifest(url_manifest) if url_manifest else info["urls"]
        if not entries:
            raise ValueError(
                f"Supply --url-list with the OASIS-1 archives authorized for your account, or --archive. Access: {info['access_url']}"
            )
        opener = make_opener(cookie_file)
        for entry in entries:
            print(f"Downloading/checking {entry['filename']}", flush=True)
            records.append(fetch_archive(entry, output / "archives", opener))
            paths.append(output / "archives" / entry["filename"])
    raw = output / "raw"
    for path, record in zip(paths, records):
        record["extracted_files"] = extract_archive(path, raw)
    (output / "download_provenance.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "provider": info["homepage"],
                "access": info["access"],
                "archives": records,
                "raw_directory": "raw",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return raw
