import os
import gzip
import json
import random
import shutil
import logging
from typing import List, Tuple, Iterator, Optional

import numpy as np


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
        return len(file.readlines())


def validate_file_extension(file_path: str, valid_file_ext: str, logger: Optional[logging.Logger]):
    file_ext: str = os.path.splitext(file_path)[1]

    if file_ext != valid_file_ext:
        message = f'Input file to process must be a {valid_file_ext} file, {file_path} found'
        logger.error(message) if logger else print(message)


def save_fp_cache(cache_base: str, smiles: List[str], fp_batches: List[np.ndarray]) -> None:
    """
    Save SMILES and fingerprint chunks to disk.
    Writes:
      - {base}.smiles.gz: Gzipped UTF-8 SMILES (one per line, same order as fps)
      - {base}.fp_uint8.npy: Stacked (N, nbytes) uint8 fingerprint matrix
      - {base}.batch_sizes.json: List of batch row counts to reconstruct `fp_batch`
    """
    os.makedirs(os.path.dirname(cache_base), exist_ok=True)

    # 1) SMILES (gz)
    smi_path = f"{cache_base}.smiles.gz"
    with gzip.open(smi_path, "wt", encoding="utf-8") as f:
        for s in smiles:
            f.write(s)
            f.write("\n")

    # 2) Save fingerprint batches as one large matrix
    if not fp_batches:
        raise ValueError("Failed to save fingerprints to cache (`save_fp_cache()`): `fp_batches` is empty")

    fp_uint8 = np.vstack(fp_batches)  # (N, nbytes) uint8
    np.save(f"{cache_base}.fp_uint8.npy", fp_uint8)

    # 3) Batch sizes
    sizes = [c.shape[0] for c in fp_batches]
    with open(f"{cache_base}.batch_sizes.json", "w") as f:
        json.dump(sizes, f)


def load_fp_cache(cache_base: str) -> Tuple[List[str], List[np.ndarray]]:
    """
    Load SMILES and fingerprint chunks previously saved with save_fp_cache().
    Returns (all_smiles, fp_chunks).
    Uses memmap for the big array to allow fast, low-RAM slicing.
    """
    smi_path = f"{cache_base}.smiles.gz"
    fp_path = f"{cache_base}.fp_uint8.npy"
    sz_path = f"{cache_base}.batch_sizes.json"

    if not (os.path.exists(smi_path) and os.path.exists(fp_path) and os.path.exists(sz_path)):
        raise FileNotFoundError(f"Missing cache files in '{cache_base}'")

    # 1) SMILES
    smiles: List[str] = []
    with gzip.open(smi_path, "rt", encoding="utf-8") as f:
        for line in f:
            smiles.append(line.rstrip("\n"))

    # 2) Fingerprint matrix (memmap)
    fp_uint8 = np.load(fp_path, mmap_mode="r")  # shape (N, nbytes), dtype=uint8

    # 3) Batch sizes -> Split back to list of arrays (views)
    with open(sz_path, "r") as f:
        sizes = json.load(f)

    if sum(sizes) != fp_uint8.shape[0]:
        raise ValueError("Cache inconsistency: `sum(batch_sizes)` != number of rows in `fp_uint8.npy`")

    splits = np.cumsum(sizes[:-1])
    fp_batches = np.split(fp_uint8, splits, axis=0)  # list of views over the memmap

    if len(smiles) != fp_uint8.shape[0]:
        raise ValueError("Cache inconsistency: number of SMILES != number of fingerprints")

    return smiles, fp_batches
