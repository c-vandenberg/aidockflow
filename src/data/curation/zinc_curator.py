import os
import shutil
import subprocess
import stat
import shlex
import glob
import gzip
import logging
from typing import Dict, List, Any

import pandas as pd
import validators
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from src.data.curation.base_curator import BaseCurator
from src.utils.file_utils import compress_and_delete_file


class ZincDatabaseCurator(BaseCurator):
    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__(config=config,logger=logger)

    def run(self):
        # 1. Download ZINC 3D druglike database using ZINC downloader
        #    script executable provided in config file.
        self._download_zinc_data()

        # 2. Concatenate all downloaded SMILES files into a single large file on disk.
        smiles_path = self._config.get(
            'zinc_concat_smiles_path',
            '../data/raw/ZINC20-3D-druglike-SMILES/raw_smiles.smi'
        )

        # 2.1. Skip concatenation if compressed SMILES file already exists
        if not os.path.exists(smiles_path + '.gzip'):
            self._concatenate_smiles_files(
                input_dir=self._config.get('zinc_raw_data_output_dir'),
                output_path=smiles_path
            )
        else:
            self._logger.info(
                f'Concatenated ZINC SMILES file exists at {smiles_path + ".gzip"}, skipping concatenation'
            )

    def _download_zinc_data(self):
        # 1. Get required parameters from the loaded config, validate, and
        #    make ZINC downloader script executable
        script_path = self._config.get('zinc_downloader_script_path')
        output_dir = self._config.get('zinc_raw_data_output_dir')
        max_workers = self._config.get('max_workers', 10)

        # 1.1. Validate configuration
        if not all([script_path, output_dir]):
            self._logger.error('Config must contain `zinc_downloader_script_path` and `zinc_raw_data_output_dir`.')
            return

        # 1.2. Validate that the downloader script exists
        if not os.path.exists(script_path):
            self._logger.error(f'Error: Downloader script not found at "{script_path}"')
            return

        # 1.3. Ensure the output directory exists
        try:
            os.makedirs(output_dir, exist_ok=True)
            self._logger.info(f'Files will be saved to: {os.path.abspath(output_dir)}')
        except OSError as e:
            self._logger.error(f'Could not create output directory "{output_dir}": {e}')
            return

        # 1.4. Make the downloader script executable
        try:
            st = os.stat(script_path)
            os.chmod(script_path, st.st_mode | stat.S_IEXEC)
        except Exception as e:
            self._logger.error(f'Failed to set execute permissions on script: {e}')
            return

        # 2. Parse the entire downloader script into a list of jobs.
        self._logger.info('Starting ZINC15 centroid library construction...')
        self._logger.info(f'Parsing download jobs from {script_path}...')
        all_jobs = self._parse_wget_script(script_path, self._logger)
        total_jobs = len(all_jobs)
        self._logger.info(f'Found {total_jobs} total files to download.')

        # 3. Filter out jobs for files that already exist.
        self._logger.info('Checking for already downloaded files...')
        jobs_to_run = [
            job for job in all_jobs
            if not os.path.exists(os.path.join(output_dir, job['output_path']))
        ]

        skipped_count = total_jobs - len(jobs_to_run)
        if skipped_count > 0:
            self._logger.info(f'Skipping {skipped_count} files that already exist.')

        if not jobs_to_run:
            self._logger.info('All files have already been downloaded. Nothing to do.')
            return

        # 4. Use a ThreadPoolExecutor to download the remaining files concurrently.
        self._logger.info(f'Starting download for {len(jobs_to_run)} missing files using {max_workers} workers...')

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function to pass the fixed output_dir and logger to the worker
            worker_partial = partial(self._download_worker, output_dir=output_dir, logger=self._logger)
            # Use executor.map to run the jobs. Wrap in list() to ensure all jobs complete
            list(executor.map(worker_partial, jobs_to_run))

    @staticmethod
    def _parse_wget_script(script_path: str, logger: logging.Logger) -> List[Dict[str, str]]:
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

                # 2. Isolate the wget command string (it comes after '&&' in the ZINC downloader script).
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
                        jobs.append({'url': url, 'output_path': output_path + '.gzip'})
                    else:
                        logger.warning(f'Skipping malformed line (invalid URL): {line.strip()}')
                except (ValueError, IndexError):
                    logger.warning(f'Could not parse line: {line.strip()}')
                    continue
        return jobs

    @staticmethod
    def _download_worker(job: Dict[str, str], output_dir: str, logger: logging.Logger):
        """
        A worker function to download a single file using wget and compress it using `gzip`.
        This function is designed to be called by a thread pool executor.
        """
        # 1. Define paths for uncompressed and gzipped files and
        #    ensure the specific subdirectory exists
        file_output_path = os.path.join(output_dir, job['output_path'])
        path_split = os.path.splitext(file_output_path)
        file_output_ext = path_split[1]
        os.makedirs(os.path.dirname(file_output_path), exist_ok=True)
        gzipped_path = file_output_path + '.gzip'

        # 2. Download the uncompressed `.smi` file using wget
        command = [
            'wget',
            '-c',  # Resume interrupted downloads
            '--tries=3',  # Retry up to 3 times
            '--timeout=60',  # Set a 60-second timeout
            '-O', file_output_path,
            job['url']
        ]

        try:
            # Use `subprocess.run` for a single, blocking command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on non-zero exit code
            )
            if result.returncode != 0:
                logger.error(f'Failed to download {job["url"]}. Error: {result.stderr}')
                return  # Stop processing this file if download fails
        except Exception as e:
            logger.error(f'Subprocess failed for {job["url"]}: {e}')

        if file_output_ext != '.gzip':
            # 4. Compress the downloaded file and delete original uncompressed file
            compress_and_delete_file(
                uncompressed_path=file_output_path,
                compressed_path=gzipped_path,
                logger=logger
            )

    def _concatenate_smiles_files(self, input_dir: str, output_path: str):
        """
        Scans a directory for .smi.gz files, decompresses them, and yields
        all SMILES strings.

        Parameters
        ----------
        input_dir : str
            The directory containing the downloaded .smi.gz tranche files.
        output_path : str
            The path to the single output file where all SMILES will be written.

        Returns
        -------
        List[str]
            A single list containing all SMILES strings from all files.
        """
        uncompressed_smiles_path = output_path
        gzipped_smiles_path = output_path + '.gzip'

        # 1. Validate ZINC raw SMILES output directory
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except OSError as e:
            self._logger.error(f'Could not create ZINC raw SMILES output directory "{output_path}": {e}')
            return

        # 2. Find all SMILES in input directory recursively
        self._logger.info(f'Scanning for .smi.gz files in {input_dir}...')
        smi_gz_files = glob.glob(os.path.join(input_dir, '**', '*.smi.gzip'), recursive=True)

        if not smi_gz_files:
            self._logger.warning("No .smi.gz files found. Please check the input directory.")
            return

        self._logger.info(f'Found {len(smi_gz_files)} ZINC SMILES files to process.')

        # 3. Open the single output file once in write mode and write all SMILES to it
        total_smiles_written = 0
        with open(uncompressed_smiles_path, 'wt') as smiles_output_file:
            # 3.1. Iterate through each `.smi` file
            for i, smi_file_path in enumerate(smi_gz_files):
                if (i + 1) % 100 == 0:
                    self._logger.info(
                        f'Processing file {i + 1}/{len(smi_gz_files)}: {os.path.basename(smi_file_path)}')
                try:
                    # 3.2. Open current `.smi` file in default mode
                    with gzip.open(smi_file_path, 'rt', encoding='utf-8') as smi_file:
                        # Skip the header line of each file
                        next(smi_file)
                        # 3.3. Process each line individually to save memory
                        for line in smi_file:
                            line = line.strip()
                            if line:
                                # The SMILES string is the first column
                                smiles = line.split()[0]
                                # 3.4. Write the SMILES string to the output file with a newline
                                smiles_output_file.write(smiles + '\n')
                                total_smiles_written += 1
                except Exception as e:
                    self._logger.error(f'Could not process SMILES file {smi_file_path}: {e}')

        # 4. Compress the SMILES file and delete original uncompressed file
        compress_and_delete_file(
            uncompressed_path=uncompressed_smiles_path,
            compressed_path=gzipped_smiles_path,
            logger=self._logger
        )

        self._logger.info(f'Concatenation complete. Written {total_smiles_written} SMILES to {gzipped_smiles_path}')