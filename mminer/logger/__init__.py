import logging

# Create a logger
logger = logging.getLogger(__name__)

# Add a NullHandler to prevent any logging output
null_handler = logging.NullHandler()
logger.addHandler(null_handler)
