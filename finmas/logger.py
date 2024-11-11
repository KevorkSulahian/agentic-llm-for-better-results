import functools
import logging
import sys
import time
from typing import Any, Callable, TypeVar, cast

import panel as pn

T = TypeVar("T", bound=Callable[..., Any])

FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


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
    formatter = logging.Formatter(fmt=format_, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    logger.setLevel(level)
    return logger


def log_execution_time(logger: logging.Logger) -> Callable[[T], T]:
    """
    Returns decorator that logs the time a function or method takes to execute.

    Args:
        logger: The logger instance to use for logging execution time
    """

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)

            # Check for method of a class or a function
            if args and hasattr(args[0].__class__, func.__name__):
                name = f"Method {args[0].__class__.__name__}.{func.__name__}"
            else:
                name = f"Function {func.__name__}"

            logger.info("%s spent %.1fs", name, time.time() - start_time)
            return result

        return cast(T, wrapper)

    return decorator
