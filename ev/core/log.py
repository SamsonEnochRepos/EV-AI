"""File logging with redaction. Console output only with --debug."""
from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_REDACT = [
    (re.compile(r"(?i)(password|passwd|token|secret|api[_-]?key)\s*[=:]\s*\S+"), r"\1=<redacted>"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<email>"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<number>"),
]


def redact(text: str) -> str:
    for pat, rep in _REDACT:
        text = pat.sub(rep, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


def setup_logging(log_dir: Path, debug: bool = False, enabled: bool = True) -> logging.Logger:
    logger = logging.getLogger("ev")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if enabled:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_dir / "ev.log", maxBytes=1_000_000, backupCount=2,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(RedactingFilter())
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    if debug:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.addFilter(RedactingFilter())
        logger.addHandler(sh)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ev.{name}")
