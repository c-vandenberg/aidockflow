#!/usr/bin/env python3

import os
import argparse
from typing import List, Dict, Any

from ml_training_base import configure_multi_level_logger, load_config

from src.data.curation.actives_curator import HighFidelityActivesCurator
from src.data.curation.centroid_curator import CentroidLibraryCurator
from src.data.loading.actives_loader import HighFidelityActivesDataLoader


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

    # Validate all required configuration parameters
    if not all([data_config.get('uniprot_id'), data_config.get('bioactivity_measures')]):
        raise ValueError('You must provide `uniprot_id` and `bioactivity_measures` in the configuration file.')

    if not all([data_config.get("zinc_downloader_script_path"), data_config.get("zinc_raw_data_output_dir")]):
        raise ValueError(
            "You must provide 'zinc_downloader_script_path' and 'zinc_raw_data_output_dir' in the configuration file."
        )

    data_curation_log_dir = data_config.get('data_curation_log_dir', '../var/log/data_curation')
    training_log_dir = data_config.get('training_log_dir', '../var/log/training')

    os.makedirs(data_curation_log_dir, exist_ok=True)
    os.makedirs(training_log_dir, exist_ok=True)

    data_curation_logger = configure_multi_level_logger(
        name=f'{data_config.get("uniprot_id")}_data_curation',
        log_dir=data_curation_log_dir
    )
    training_logger = configure_multi_level_logger(
        name=f'{data_config.get("uniprot_id")}_training',
        log_dir=training_log_dir
    )

    # --- Phase 1. Data Ingestion & Curation ---
    # 1.1. Retrieve High-Fidelity Actives
    actives_path: str = data_config.get('standardized_actives_path', '../data/processed')
    os.makedirs(os.path.dirname(actives_path), exist_ok=True)

    actives_curator: HighFidelityActivesCurator = HighFidelityActivesCurator(
        config=data_config,
        logger=data_curation_logger
    )
    actives_curator.run()

    # 1.2. Split High-Fidelity Actives
    random_state = data_config.get('random_state', 4)
    actives_loader: HighFidelityActivesDataLoader = HighFidelityActivesDataLoader(
        test_split=data_config.get('test_split', 0.15),
        validation_split=data_config.get('validation_split', 0.15),
        logger=data_curation_logger,
        actives_path=actives_path,
        random_state=random_state
    )
    actives_loader.setup_datasets()

    # 1.3. Build ZINC15 “Druglike-Centroid Library”
    centroid_curator: CentroidLibraryCurator = CentroidLibraryCurator(config=data_config, logger=data_curation_logger)
    centroid_curator.run()


if __name__ == "__main__":
    main()