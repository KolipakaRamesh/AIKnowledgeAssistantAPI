import sys
from loguru import logger


def setup_logger() -> None:
    """Configure Loguru to emit JSON-structured logs to stdout."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            '{{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
            '"level":"{level}",'
            '"message":"{message}",'
            '"module":"{module}",'
            '"function":"{function}",'
            '"line":{line}}}'
        ),
        serialize=False,
        colorize=False,
    )


setup_logger()
