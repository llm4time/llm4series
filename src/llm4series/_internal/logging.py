import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
IS_DEV = os.getenv("ENVIRONMENT", "production") == "development"

format_str = "[%(levelname)-8s] %(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s" if IS_DEV \
             else "[%(levelname)s] %(message)s"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=format_str,
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("llm4series")
