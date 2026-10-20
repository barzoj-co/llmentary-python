#!/usr/bin/env python3
"""Fail if a built wheel or sdist has a PEP 639 License-File that is not present.

PyPI rejects Metadata 2.4+ uploads when a License-File path is missing. For
wheels the file must exist at ``<name>.dist-info/licenses/<License-File>``.
For sdists it must exist at ``<name>-<version>/<License-File>``.

Also tests each wheel as a zip and checks that RECORD hashes still match after
any retag/repack.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

LICENSE_FILE_PREFIX = "License-File:"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist",
        nargs="?",
        default="dist",
        type=Path,
        help="directory of wheels and sdists (default: dist)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run fixture checks and exit",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    return 0 if check_dist(args.dist) else 1


def check_dist(dist: Path) -> bool:
    if not dist.is_dir():
        print(f"{dist} is not a directory", file=sys.stderr)
        return False
    artifacts = sorted(
        path
        for path in dist.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if not artifacts:
        print(f"no wheels or sdists in {dist}", file=sys.stderr)
        return False
    ok = True
    for path in artifacts:
        try:
            if path.suffix == ".whl":
                check_wheel(path)
            else:
                check_sdist(path)
        except SystemExit as exc:
            print(f"{path.name}: {exc}", file=sys.stderr)
            ok = False
            continue
        print(f"{path.name}: License-File paths ok")
    return ok


def check_wheel(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"not a valid zip: {exc}") from exc
    if bad is not None:
        raise SystemExit(f"corrupt zip member {bad}")

    with zipfile.ZipFile(path) as zf:
        names = {_norm(name) for name in zf.namelist() if not name.endswith("/")}
        contents = {
            _norm(info.filename): zf.read(info.filename)
            for info in zf.infolist()
            if not info.is_dir()
        }
    metadata_name = _one(name for name in names if name.endswith(".dist-info/METADATA"))
    dist_info = metadata_name.rsplit("/", 1)[0]
    metadata = contents[metadata_name].decode("utf-8")
    license_files = _license_files(metadata)
    if not license_files:
        raise SystemExit("METADATA has no License-File entries")
    for license_file in license_files:
        expected = f"{dist_info}/licenses/{license_file}"
        if expected not in names:
            raise SystemExit(
                f"License-File {license_file} does not exist at {expected} "
                f"(have: {_license_members(names, dist_info)})"
            )
    _check_record(dist_info, contents)


def check_sdist(path: Path) -> None:
    try:
        with tarfile.open(path) as tf:
            names = {_norm(name) for name in tf.getnames() if name and not name.endswith("/")}
            pkg_info = next((name for name in names if name.endswith("/PKG-INFO")), None)
            if pkg_info is None:
                raise SystemExit("sdist has no PKG-INFO")
            extracted = tf.extractfile(pkg_info)
            if extracted is None:
                raise SystemExit("sdist PKG-INFO is not a regular file")
            metadata = extracted.read().decode("utf-8")
    except tarfile.TarError as exc:
        raise SystemExit(f"not a valid sdist: {exc}") from exc
    root = pkg_info.rsplit("/", 1)[0]
    license_files = _license_files(metadata)
    if not license_files:
        raise SystemExit("PKG-INFO has no License-File entries")
    for license_file in license_files:
        expected = f"{root}/{license_file}"
        if expected not in names:
            raise SystemExit(
                f"License-File {license_file} does not exist at {expected}"
            )


def _check_record(dist_info: str, contents: dict[str, bytes]) -> None:
    record_name = f"{dist_info}/RECORD"
    if record_name not in contents:
        raise SystemExit("wheel has no RECORD")
    recorded: dict[str, tuple[str, str]] = {}
    for raw in contents[record_name].decode("utf-8").splitlines():
        if not raw:
            continue
        parts = raw.split(",")
        if len(parts) != 3:
            raise SystemExit(f"malformed RECORD line: {raw}")
        recorded[_norm(parts[0])] = (parts[1], parts[2])
    expected_files = set(contents)
    if set(recorded) != expected_files:
        missing = sorted(expected_files - set(recorded))
        extra = sorted(set(recorded) - expected_files)
        raise SystemExit(f"RECORD does not match wheel files: missing={missing} extra={extra}")
    for name, data in contents.items():
        digest, size = recorded[name]
        if name == record_name:
            if digest or size:
                raise SystemExit("RECORD must have an empty hash and size")
            continue
        want = _record_digest(data)
        if digest != want or size != str(len(data)):
            raise SystemExit(f"RECORD hash mismatch for {name}")


def _license_files(metadata: str) -> list[str]:
    files: list[str] = []
    header = metadata.split("\n\n", 1)[0]
    for line in header.splitlines():
        if line.startswith(LICENSE_FILE_PREFIX):
            value = line[len(LICENSE_FILE_PREFIX) :].strip().replace("\\", "/")
            if not value:
                raise SystemExit("empty License-File entry")
            if value.startswith("../") or "/../" in f"/{value}/":
                raise SystemExit(f"unsafe License-File path: {value}")
            files.append(value)
    return files


def _license_members(names: set[str], dist_info: str) -> str:
    prefix = f"{dist_info}/licenses/"
    found = sorted(name[len(prefix) :] for name in names if name.startswith(prefix))
    return ", ".join(found) if found else "(none)"


def _record_digest(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _record_line(name: str, data: bytes) -> str:
    return f"{name},{_record_digest(data)},{len(data)}"


def _norm(name: str) -> str:
    return name.replace("\\", "/")


def _one(names: object) -> str:
    items = list(names)
    if len(items) != 1:
        raise SystemExit(f"expected exactly one METADATA file, found {len(items)}")
    return items[0]


def self_test() -> None:
    dist_info = "llmentary-0.1.1.dist-info"
    metadata = (
        b"Metadata-Version: 2.4\nName: llmentary\nVersion: 0.1.1\n"
        b"License-File: LICENSE\nLicense-File: licenses/AGPL-3.0.txt\n"
    )
    license_text = b"MIT license\n"
    agpl_text = b"AGPL license\n"
    payload = b"x = 1\n"

    def write_wheel(path: Path, files: dict[str, bytes]) -> None:
        record = "\n".join(_record_line(name, data) for name, data in sorted(files.items()))
        record += f"\n{dist_info}/RECORD,,\n"
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
            zf.writestr(f"{dist_info}/RECORD", record)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        good_files = {
            f"{dist_info}/METADATA": metadata,
            "llmentary/__init__.py": payload,
            f"{dist_info}/licenses/LICENSE": license_text,
            f"{dist_info}/licenses/licenses/AGPL-3.0.txt": agpl_text,
        }
        good = tmp_path / "good.whl"
        write_wheel(good, good_files)
        check_wheel(good)

        bad_files = {
            f"{dist_info}/METADATA": metadata,
            "llmentary/__init__.py": payload,
            f"{dist_info}/licenses/LICENSE": license_text,
            f"{dist_info}/licenses/AGPL-3.0.txt": agpl_text,
        }
        bad = tmp_path / "bad.whl"
        write_wheel(bad, bad_files)
        try:
            check_wheel(bad)
        except SystemExit as exc:
            message = str(exc)
            expected = f"{dist_info}/licenses/licenses/AGPL-3.0.txt"
            if expected not in message:
                raise SystemExit(
                    f"self-test: bad wheel error missing {expected}: {message}"
                ) from exc
        else:
            raise SystemExit("self-test: expected the mismatched License-File wheel to fail")

        stale = tmp_path / "stale-record.whl"
        write_wheel(stale, good_files)
        with zipfile.ZipFile(stale, "a") as zf:
            zf.writestr("llmentary/extra.py", b"y = 2\n")
        try:
            check_wheel(stale)
        except SystemExit as exc:
            if "RECORD" not in str(exc):
                raise SystemExit(f"self-test: expected RECORD failure, got {exc}") from exc
        else:
            raise SystemExit("self-test: expected a stale RECORD to fail")

        sdist_root = "llmentary-0.1.1"
        good_sdist = tmp_path / "good.tar.gz"
        with tarfile.open(good_sdist, "w:gz") as tf:
            for name, data in {
                f"{sdist_root}/PKG-INFO": metadata,
                f"{sdist_root}/LICENSE": license_text,
                f"{sdist_root}/licenses/AGPL-3.0.txt": agpl_text,
            }.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        check_sdist(good_sdist)

        bad_sdist = tmp_path / "bad.tar.gz"
        with tarfile.open(bad_sdist, "w:gz") as tf:
            for name, data in {
                f"{sdist_root}/PKG-INFO": metadata,
                f"{sdist_root}/LICENSE": license_text,
            }.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        try:
            check_sdist(bad_sdist)
        except SystemExit as exc:
            if "licenses/AGPL-3.0.txt" not in str(exc):
                raise SystemExit(f"self-test: bad sdist error missing AGPL path: {exc}") from exc
        else:
            raise SystemExit("self-test: expected the sdist missing AGPL to fail")

    print("self-test: ok")


if __name__ == "__main__":
    sys.exit(main())
