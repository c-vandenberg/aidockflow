import os
import gzip
import random
import shutil
import logging
from typing import Iterator, Optional


def compress_and_delete_file(uncompressed_path: str, compressed_path: str, logger: logging.Logger):
    # 1. Compress file
    try:
        with open(uncompressed_path, 'rb') as file_in:
            with gzip.open(compressed_path, 'wb') as file_out:
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


def create_random_sample_gzip_file(
    input_file: str,
    output_file: str,
    total_lines: int,
    sample_size: int,
    logger: Optional[logging.Logger] = None
):
    """
    Randomly samples a specific number of lines from a gzipped file
    and saves them to a new gzipped file.

    To ensure memory-efficiency, only the lines numbers to be sampled
    are stored in memory, not the file content itself.
    """
    intro_message = f'Sampling {sample_size:,} lines from {total_lines:,} total lines.'
    logger.info(intro_message) if logger else print(intro_message)

    # 1. Generate a set of unique random line numbers to keep
    indices_to_keep = set(random.sample(range(total_lines), sample_size))
    indices_message = 'Finished generating random line indices.'
    logger.info(indices_message) if logger else print(indices_message)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 2. Stream the input file and write the selected lines to the output
    lines_written = 0
    with gzip.open(input_file, 'rt', encoding='utf-8') as infile, \
            gzip.open(output_file, 'wt', encoding='utf-8') as outfile:

        for i, line in enumerate(infile):
            if i in indices_to_keep:
                outfile.write(line)
                lines_written += 1
                # 2.1. Optimization: Stop early if all lines are found
                if lines_written == sample_size:
                    break

    success_message = f'Successfully wrote {lines_written:,} lines to {output_file}.'
    logger.info(success_message) if logger else print (success_message)


def count_gzip_lines(gzip_file_path: str):
    with gzip.open(filename=gzip_file_path, mode='rt', encoding='utf-8') as file:
        return sum(1 for line in file)


def validate_file_extension(file_path: str, valid_file_ext: str, logger: Optional[logging.Logger]):
    file_ext: str = os.path.splitext(file_path)[1]

    if file_ext != valid_file_ext:
        message = f'Input file to process must be a {valid_file_ext} file, {file_path} found'
        logger.error(message) if logger else print(message)
