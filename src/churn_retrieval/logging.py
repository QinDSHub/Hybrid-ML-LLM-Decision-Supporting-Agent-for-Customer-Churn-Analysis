from __future__ import annotations

import logging
from pathlib import Path

from churn_retrieval.utils.io import ensure_dir


def setup_logging(log_dir: Path, log_name: str = "pipeline.log") -> None:
    ensure_dir(log_dir)
    log_path = log_dir / log_name

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
