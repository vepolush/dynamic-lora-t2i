# src/dynamic_lora_t2i/utils/entity_zip.py

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

try:
    from src.dynamic_lora_t2i.config import ENTITIES_DIR, ensure_project_directories, setup_logging
except Exception:
    from src.dynamic_lora_t2i.utils.config import ENTITIES_DIR, ensure_project_directories, setup_logging  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".txt",
    ".caption",
    ".json",
    ".yaml",
    ".yml",
}

DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES = 800 * 1024 * 1024
DEFAULT_MAX_FILES = 20_000

DEFAULT_TMP_BASE_DIR = Path(os.getenv("DYNAMIC_LORA_T2I_TMP_DIR", "")).expanduser()
if not str(DEFAULT_TMP_BASE_DIR):
    DEFAULT_TMP_BASE_DIR = Path("/workspace/tmp") if Path("/workspace").exists() else Path(tempfile.gettempdir())


def sanitize_entity_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-]+", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise ValueError("Entity name is empty after sanitization.")
    return name


def _is_path_within_base(base_dir: Path, target_path: Path) -> bool:
    base_dir = base_dir.resolve()
    target_path = target_path.resolve()
    try:
        target_path.relative_to(base_dir)
        return True
    except ValueError:
        return False


def _iter_zip_members(zf: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for info in zf.infolist():
        name = info.filename
        if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
            continue
        yield info


def _detect_single_root_dir(member_names: list[str]) -> Optional[str]:
    roots = set()
    for n in member_names:
        n = n.lstrip("/\\")
        if not n or n.endswith("/"):
            continue
        parts = n.split("/")
        if parts:
            roots.add(parts[0])
        if len(roots) > 1:
            return None
    return next(iter(roots)) if len(roots) == 1 else None


def _move_with_collision_handling(src: Path, dst: Path) -> None:
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return

    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return
        i += 1


def _safe_extract_to_dir(
    zf: zipfile.ZipFile,
    dest_dir: Path,
    *,
    allowed_exts: Optional[set[str]],
    max_total_uncompressed_bytes: int,
    max_files: int,
) -> tuple[list[Path], int, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)

    total_uncompressed = 0
    written_files = 0
    extracted_files: list[Path] = []

    for info in _iter_zip_members(zf):
        name = info.filename.replace("\\", "/").lstrip("/")
        if not name or name.endswith("/"):
            continue

        if written_files >= max_files:
            raise ValueError(f"Zip contains too many files (>{max_files}).")

        target_path = dest_dir / name
        if not _is_path_within_base(dest_dir, target_path):
            raise ValueError(f"Unsafe path in zip (path traversal): {info.filename}")

        total_uncompressed += int(info.file_size)
        if total_uncompressed > max_total_uncompressed_bytes:
            raise ValueError(
                f"Zip too large after decompression (> {max_total_uncompressed_bytes} bytes)."
            )

        if allowed_exts is not None:
            ext = target_path.suffix.lower()
            if ext not in allowed_exts:
                logger.warning("Skipping file with disallowed extension: %s", name)
                continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info, "r") as src, target_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

        extracted_files.append(target_path)
        written_files += 1

    return extracted_files, written_files, total_uncompressed


def unpack_entity_zip(
    zip_path: Path,
    entity_name: str,
    *,
    entities_dir: Path = ENTITIES_DIR,
    overwrite: bool = False,
    flatten_single_root: bool = True,
    allowed_exts: Optional[set[str]] = DEFAULT_ALLOWED_EXTS,
    max_total_uncompressed_bytes: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
) -> Path:
    ensure_project_directories()

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")

    safe_name = sanitize_entity_name(entity_name)
    entities_dir = Path(entities_dir)
    entity_dir = entities_dir / safe_name

    if entity_dir.exists():
        if overwrite:
            shutil.rmtree(entity_dir)
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
            entity_dir = entities_dir / f"{safe_name}__{ts}"

    entity_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Unpacking entity zip: %s -> %s", zip_path, entity_dir)

    DEFAULT_TMP_BASE_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        member_names = [
            i.filename.replace("\\", "/")
            for i in _iter_zip_members(zf)
            if i.filename and not i.filename.endswith("/")
        ]

        single_root = _detect_single_root_dir(member_names) if flatten_single_root else None

        with tempfile.TemporaryDirectory(prefix="entity_unpack_", dir=str(DEFAULT_TMP_BASE_DIR)) as tmp:
            tmp_dir = Path(tmp)

            _, total_files, total_bytes = _safe_extract_to_dir(
                zf,
                tmp_dir,
                allowed_exts=allowed_exts,
                max_total_uncompressed_bytes=max_total_uncompressed_bytes,
                max_files=max_files,
            )

            if single_root:
                content_root = tmp_dir / single_root
                if not content_root.exists() or not content_root.is_dir():
                    content_root = tmp_dir
            else:
                content_root = tmp_dir

            for item in content_root.iterdir():
                _move_with_collision_handling(item, entity_dir / item.name)

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_count = sum(
        1
        for p in entity_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    )
    if image_count == 0:
        raise ValueError(f"No images found after unpacking into {entity_dir}")

    meta = {
        "entity_name": safe_name,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_zip": zip_path.name,
        "image_count": image_count,
        "path": str(entity_dir),
        "flatten_single_root": flatten_single_root,
        "allowed_exts": sorted(list(allowed_exts)) if allowed_exts is not None else None,
        "max_total_uncompressed_bytes": int(max_total_uncompressed_bytes),
        "max_files": int(max_files),
    }
    (entity_dir / "entity_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Entity unpacked OK. images=%d, dir=%s", image_count, entity_dir)
    return entity_dir


def main() -> None:
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description="Unpack an entity zip into data/entities/<entity_name>/")
    parser.add_argument("--zip", dest="zip_path", required=True, help="Path to .zip file")
    parser.add_argument("--name", dest="entity_name", required=True, help="Entity name (folder name will be sanitized)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing entity folder if exists")
    parser.add_argument("--no-flatten", action="store_true", help="Do not flatten single root dir inside zip")

    parser.add_argument(
        "--max-mb",
        type=int,
        default=int(DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES / (1024 * 1024)),
        help="Max total uncompressed size (MB) to protect from zip bombs",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help="Max files to extract to protect from zip bombs",
    )

    args = parser.parse_args()

    out_dir = unpack_entity_zip(
        zip_path=Path(args.zip_path),
        entity_name=args.entity_name,
        overwrite=args.overwrite,
        flatten_single_root=not args.no_flatten,
        max_total_uncompressed_bytes=int(args.max_mb) * 1024 * 1024,
        max_files=int(args.max_files),
    )

    print(f"OK: unpacked to {out_dir}")


if __name__ == "__main__":
    main()
