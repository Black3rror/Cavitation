import logging
import logging.config
import os

DEFAULT_LOG_LEVEL = "INFO"


def get_logger(name, level=DEFAULT_LOG_LEVEL, log_path=None, use_rich=False):
    """Get logger with given name and level.

    Args:
        name (str): Name of the logger. Usually __name__.
        level (str): Level of the logger. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        log_path (str): Path to the log file. If None, no log file is created.
        use_rich (bool): Whether to use rich to beautify the log.

    Returns:
        logging.Logger: Logger.

    Examples:
        >>> logger = get_logger(__name__, level="DEBUG", log_path="logs/latest.log", use_rich=False)
        >>> logger.debug("Used for debugging your code.")
        >>> logger.info("Informative messages from your code.")
        >>> logger.warning("Everything works but there is something to be aware of.")
        >>> logger.error("There's been a mistake with the process.")
        >>> logger.critical("There is something terribly wrong and process may terminate.")
    """
    config = {
        "version": 1,
        "formatters": {
            "minimal": {
                "format": "%(message)s"
            },
            "detailed": {
                "format": "%(levelname)s - %(asctime)s - [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
            },
        },

        "handlers": {},         # To be filled below

        "root": {
            "handlers": [],     # To be filled below
            "level": level.upper(),
        },
    }

    if use_rich:
        config["handlers"]["rich"] = {
            "class": "rich.logging.RichHandler",
            "level": "DEBUG",
            "formatter": "minimal"
        }
        config["root"]["handlers"].append("rich")
    else:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "minimal",
            "stream": "ext://sys.stdout"
        }
        config["root"]["handlers"].append("console")

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": "WARNING",
            "formatter": "detailed",
            "filename": log_path,
            "mode": "w"
        }
        config["root"]["handlers"].append("file")

    logging.config.dictConfig(config)
    logger = logging.getLogger(name)

    return logger
