from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_str: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path.resolve()
    return (base_dir / path).resolve()
