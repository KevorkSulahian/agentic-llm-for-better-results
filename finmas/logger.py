import logging
import sys

import panel as pn

FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@pn.cache
def get_logger(name, format_=FORMAT, level=logging.INFO):
    """
    Returns a logger with the given name, format and level.
    This is useful for loggers for specific modules
    """
    logger = logging.getLogger(name)

    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setStream(sys.stdout)
    formatter = logging.Formatter(format_)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    logger.setLevel(level)
    return logger
