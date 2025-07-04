import os
import gzip
import shutil
import logging
from typing import Iterator


def compress_and_delete_file(uncompressed_path: str, compressed_path: str, logger: logging.Logger):
    # 1. Compress file
    try:
        with open(uncompressed_path, 'rb') as file_in:
            with gzip.open(compressed_path, 'wb', encoding='utf-8') as file_out:
                shutil.copyfileobj(file_in, file_out)
        logger.debug(f"Successfully compressed {uncompressed_path} to {compressed_path}")
    except Exception as e:
        logger.error(f"Failed to compress file {uncompressed_path}: {e}")
        return

    # 2. Delete the original uncompressed file
    try:
        os.remove(uncompressed_path)
    except OSError as e:
        logger.error(f"Failed to remove original file {uncompressed_path}: {e}")


def stream_lines_from_gzip_file(gzip_file_path: str) -> Iterator[str]:
    """
    A generator that can read a large gzip file line by line to save memory.
    """
    with gzip.open(filename=gzip_file_path, mode='rt', encoding='utf-8') as file:
        for line in file:
            yield line.strip()
