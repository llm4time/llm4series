import logging
from colorlog import ColoredFormatter
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
IS_DEV = os.getenv("ENVIRONMENT", "production") == "development"

LOG_COLORS = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}

format_str = (
    "%(log_color)s[%(levelname)-8s]%(reset)s %(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    if IS_DEV else
    "%(log_color)s[%(levelname)s]%(reset)s %(message)s"
)

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(format_str, log_colors=LOG_COLORS))

logger = logging.getLogger("llm4series")
logger.setLevel(LOG_LEVEL)
logger.addHandler(handler)
logger.propagate = False
