#!/usr/bin/env python3

import os
import argparse
import logging
import subprocess
import stat
import shlex
from typing import Dict, List, Any

import validators
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from ml_training_base import load_config, configure_multi_level_logger


def parse_config_argument() -> argparse.Namespace:
    """Parses the --config_file_path command-line argument."""
    parser = argparse.ArgumentParser(
        description='Downloads ZINC15 compound data using a configuration file.'
    )
    # This is now the ONLY argument the script takes.
    parser.add_argument(
        '--config_file_path',
        type=str,
        required=True,
        help='Path to the YAML configuration file for the download.'
    )
    return parser.parse_args()

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Executes a ZINC downloader script to fetch compound data.'
    )
    parser.add_argument(
        '--downloader_script_path',
        type=str,
        required=True,
        help='Path to the .wget or .csh downloader script from ZINC.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./zinc_downloaded_files',
        help='Path to the directory where the compound files will be saved.'
    )
    return parser.parse_args()


def parse_wget_script(script_path: str, logger: logging.Logger) -> List[Dict[str, str]]:
    """
    Parses a ZINC downloader script to extract download jobs.

    Parameters
    ----------
    script_path : str
        Path to the .wget or .csh script from ZINC.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries, where each dict has 'url' and 'output_path' keys.
    """
    jobs = []
    with open(script_path, 'r') as f:
        for line in f:
            # 1. Check if 'wget' exists anywhere in the line, if not, skip.
            if 'wget' not in line:
                continue

            # 2. Isolate the wget command string (it usually comes after '&&').
            try:
                wget_command_str = line.split('&&')[1].strip()
            except IndexError:
                # Fallback for lines that might only contain the wget command
                wget_command_str = line.strip()

            # 3. Parse the isolated command string to find the URL and Output Path.
            parts = shlex.split(wget_command_str)
            try:
                url = ''
                for part in parts:
                    if part.startswith('http') or part.startswith('https'):
                        url = part
                        break

                # The output path comes after the '-O' flag.
                output_flag_index = parts.index('-O')
                output_path = parts[output_flag_index + 1]

                # Check if the URL is a valid
                if validators.url(url):
                    jobs.append({'url': url, 'output_path': output_path})
                else:
                    logger.warning(f'Skipping malformed line (invalid URL): {line.strip()}')
            except (ValueError, IndexError):
                logger.warning(f'Could not parse line: {line.strip()}')
                continue
    return jobs


def download_worker(job: Dict[str, str], output_dir: str, logger: logging.Logger):
    """
    A worker function to download a single file using wget.
    This function is designed to be called by a thread pool executor.
    """
    full_output_path = os.path.join(output_dir, job['output_path'])

    # Ensure the specific subdirectory for the file exists
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

    command = [
        'wget',
        '-c',  # Resume interrupted downloads
        '--tries=3',  # Retry up to 3 times
        '--timeout=60',  # Set a 60-second timeout
        '-O', full_output_path,
        job['url']
    ]

    try:
        # Use subprocess.run for a single, blocking command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit code
        )
        if result.returncode != 0:
            logger.error(f'Failed to download {job["url"]}. Error: {result.stderr}')
    except Exception as e:
        logger.error(f'Subprocess failed for {job["url"]}: {e}')


def main():
    """
    Main function to validate paths, set permissions, and execute the download
    process inside a specified directory while streaming the output.
    """
    # 1. Get the path to the configuration file from the command line.
    args = parse_config_argument()

    # 2. Load all parameters from the specified YAML file.
    try:
        config: Dict[str, Any] = load_config(args.config_file_path)
        data_config: Dict[str, Any] = config.get('data', {})
    except Exception as e:
        # If config loading fails, we can't use a logger, so just print.
        print(f'Error loading configuration file: {e}')
        return

    # 3. Setup a logger using a path from the config (optional).
    log_dir = data_config.get('data_curation_log_dir', '../var/log/data_curation')
    os.makedirs(log_dir, exist_ok=True)
    logger = configure_multi_level_logger(log_dir=log_dir)

    # 4. Get the specific script parameters from the loaded config, validate, and
    #    make script executable
    script_path = data_config.get('zinc_downloader_script_path')
    output_dir = data_config.get('zinc_compound_save_dir')
    max_workers = config.get('max_workers", 10')

    # 4.1. Validate that 'downloader_script_path' and 'output_directory' are provided
    if not all([script_path, output_dir]):
        logger.error('Config must contain `downloader_script_path` and `output_directory`.')
        return

    # 4.2. Validate that the downloader script exists
    if not os.path.exists(script_path):
        logger.error(f'Error: Downloader script not found at `{script_path}`')
        return

    # 4.3. Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Files will be saved to: {os.path.abspath(output_dir)}')
    except OSError as e:
        logger.error(f'Could not create output directory `{output_dir}`: {e}')
        return

    # 4.4. Make the downloader script executable
    try:
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)
    except Exception as e:
        logger.error(f'Failed to set execute permissions on script: {e}')
        return

    # 5. Parse the entire downloader script into a list of jobs.
    logger.info(f'Parsing download jobs from {script_path}...')
    all_jobs = parse_wget_script(script_path, logger)
    total_jobs = len(all_jobs)
    logger.info(f'Found {total_jobs} total files to download.')

    # 6. Filter out jobs for files that already exist.
    logger.info('Checking for already downloaded files...')
    jobs_to_run = [
        job for job in all_jobs
        if not os.path.exists(os.path.join(output_dir, job['output_path']))
    ]

    skipped_count = total_jobs - len(jobs_to_run)
    if skipped_count > 0:
        logger.info(f'Skipping {skipped_count} files that already exist.')

    if not jobs_to_run:
        logger.info('All files have already been downloaded. Nothing to do.')
        return

    # 7. Use a ThreadPoolExecutor to download the remaining files concurrently.
    logger.info(f'Starting download for {len(jobs_to_run)} missing files using {max_workers} workers...')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function to pass the fixed output_dir and logger to the worker
        worker_partial = partial(download_worker, output_dir=output_dir, logger=logger)
        # Use executor.map to run the jobs
        list(executor.map(worker_partial, jobs_to_run))  # Wrap in list() to ensure all jobs complete

    logger.info('Download process finished.')


if __name__ == "__main__":
    main()