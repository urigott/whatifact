"""
Logger module 
"""
import logging
import sys

def get_logger():
    """
    Returns a logger
    """

    file_handler = logging.FileHandler(filename="confetti.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        encoding="utf-8",
        handlers=handlers,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    logger = logging.getLogger(__name__)
    return logger
