"""
Script to log information from functions
"""

import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import date
from pathlib import PurePosixPath, Path


# May need to create 2 .py files

# Get project root -> Here is 3 levels up
def get_project_root() -> Path:
    # 3 levels up to root
    return Path(__file__).parent.parent.parent


# Set the root logger to handle messages of INFO level and above
logging.getLogger("fragmentation").setLevel(logging.INFO)

# Format of logs
_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def log_info(title='fragmentation', **kwargs):
    """
    Generate a simple log.
    :param title: str
        Give the title of the fragmentation file. It will be appended with the current date of when the program is run.
    :param kwargs:
        Additional logger kwargs
    :return:
    """
    # File path
    filename = PurePosixPath(get_project_root() / "logs") / f"{title}-{date.today()}.log"
    file_handler = TimedRotatingFileHandler(filename, when="midnight")
    file_handler.setFormatter(_formatter)

    # Fragmenting Kinome
    logger = logging.getLogger("fragmentation")
    logger.setLevel(logging.DEBUG)

    # remove handlers to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)

    # custom error message
    logger.info(f"{kwargs['msg']}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
