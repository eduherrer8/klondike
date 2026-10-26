"""Shared file logger for diagnosing Monte Carlo runs.

Writes timestamped, structured lines to `logs/mc.log` so a hard run can be
reviewed after the fact (parallelism used? how many worlds sampled? racing
eliminations? timing? final verdict?). Safe to import anywhere; configured once.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "mc.log"
_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("mc")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if not lg.handlers:
        fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        lg.addHandler(fh)
    _logger = lg
    return lg


def log(msg: str) -> None:
    get_logger().info(msg)
