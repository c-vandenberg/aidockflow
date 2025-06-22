#!/usr/bin/env python3

import os
import argparse
from typing import List, Dict, Any

from ml_training_base import configure_logger, load_config
from src.data.curation.actives_curator import HighFidelityActivesCurator
from src.data.curation.centroid_curator import CentroidLibraryCurator


def parse_config_argument():
    """
    Parses command-line arguments.

    For script use with configuration file.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract bioactive compound SMILES strings for a given target from ChEMBL and PubChem"
                    "using a configuration file."
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        required=True,
        help="Provide path to configuration file. Required configurations are `uniprot_id` and `bioactivity_measure`"
    )

    return parser.parse_args()


def main():
    # Parse command-line arguments and configure logger
    args = parse_config_argument()

    config: Dict[str, Any] = load_config(args.config_file_path)
    data_config: Dict[str, Any] = config.get('data', {})
    os.makedirs(os.path.dirname(data_config.get('log_path', '../var/log')), exist_ok=True)
    logger = configure_logger(log_path=data_config.get('log_path'))

    # Validate all required configuration parameters
    if not all([data_config.get('uniprot_id'), data_config.get('bioactivity_measures')]):
        raise ValueError('You must provide `uniprot_id` and `bioactivity_measures` in the configuration file.')

    if not all([data_config.get("zinc_downloader_script_path"), data_config.get("zinc_raw_data_output_dir")]):
        raise ValueError(
            "You must provide 'zinc_downloader_script_path' and 'zinc_raw_data_output_dir' in the configuration file."
        )

    # --- Phase 1. Data Ingestion & Curation ---
    # 1.1. Retrieve High-Fidelity Actives
    os.makedirs(os.path.dirname(data_config.get('standardized_actives_path', '../data/processed')), exist_ok=True)
    os.makedirs(os.path.dirname(data_config.get('log_path', '../var/log')), exist_ok=True)

    actives_curator: HighFidelityActivesCurator = HighFidelityActivesCurator(config=data_config, logger=logger)
    actives_curator.run()

    # 1.2. Split High-Fidelity Actives

    # 1.3. Build ZINC15 “Druglike-Centroid Library”
    centroid_curator: CentroidLibraryCurator = CentroidLibraryCurator(config=data_config, logger=logger)
    centroid_curator.run()


if __name__ == "__main__":
    main()