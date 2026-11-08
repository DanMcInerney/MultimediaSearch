"""MultimediaSearch: a multimodal memory for coding agents (DESIGN.md). A program logs to it with `mms.log` (§8.1)."""

from mms.logger import Logger, close, flush, log, stats

__all__ = ["Logger", "close", "flush", "log", "stats"]
