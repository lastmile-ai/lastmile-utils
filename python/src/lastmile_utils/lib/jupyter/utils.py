import logging


def set_log_level(level: int | str) -> None:
    logger = logging.getLogger()
    if len(logger.handlers) > 0:
        handler = logger.handlers[0]
        handler.setLevel(level)
