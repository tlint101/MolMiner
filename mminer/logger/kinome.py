"""
Script to log information from functions
"""

import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import date
from pathlib import PurePosixPath, Path


# Function to get the project root -> Here is 3 levels up
def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


# Format of logs
_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# File path for the specific submodule log
filename = PurePosixPath(get_project_root() / "logs") / f"kinome-dataset-{date.today()}.log"
file_handler = TimedRotatingFileHandler(filename, when="midnight")
file_handler.setFormatter(_formatter)

# Create a logger specifically for this submodule
logger = logging.getLogger("kinome_dataset_logger")

# Ensure logger is only configured once (to avoid duplicate handlers)
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.propagate = False  # Disable propagation to root logger


# Function to log info messages
def log_info(**kwargs):
    logger.info(f"{kwargs['msg']}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
