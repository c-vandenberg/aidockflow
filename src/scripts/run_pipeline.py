#!/usr/bin/env python3

import os
import argparse
from typing import List, Dict, Any

from ml_training_base import configure_multi_level_logger, load_config

from src.data.curation.actives_curator import HighFidelityActivesCurator
from src.data.curation.zinc_curator import ZincDatabaseCurator
from src.data.curation.centroid_curator import CentroidLibraryCurator
from src.data.loading.actives_loader import HighFidelityActivesDataLoader
from src.data.curation.round0_curator import Round0DatasetCurator


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
    actives_path: str = data_config.get('actives_preprocessed_path', 'data/preprocessed/standardized_actives.parquet')
    os.makedirs(os.path.dirname(actives_path), exist_ok=True)

    actives_curator: HighFidelityActivesCurator = HighFidelityActivesCurator(
        config=data_config,
        logger=data_curation_logger
    )
    actives_curator.run()

    # 1.2. Split High-Fidelity Actives and Save Train, Validation, and Test Sets
    actives_dir = os.path.dirname(actives_path)
    random_state = data_config.get('random_state', 4)
    actives_loader: HighFidelityActivesDataLoader = HighFidelityActivesDataLoader(
        test_split=data_config.get('test_split', 0.15),
        validation_split=data_config.get('validation_split', 0.15),
        logger=data_curation_logger,
        actives_path=actives_path,
        random_state=random_state
    )
    actives_loader.setup_datasets()

    actives_loader.get_train_dataset().to_parquet(
        data_config.get('actives_train_pos_path', f'{actives_dir}/train_pos.parquet')
    )
    actives_loader.get_valid_dataset().to_parquet(
        data_config.get('actives_val_pos_path', f'{actives_dir}/val_pos.parquet')
    )
    actives_loader.get_test_dataset().to_parquet(
        data_config.get('actives_test_pos_path', f'{actives_dir}/test_pos.parquet')
    )

    # 1.3. Build ZINC SMILES database
    zinc_curator: ZincDatabaseCurator = ZincDatabaseCurator(config=data_config, logger=data_curation_logger)
    zinc_curator.run()

    # 1.4. Build ZINC “Druglike-Centroid Library”
    centroid_curator: CentroidLibraryCurator = CentroidLibraryCurator(config=data_config, logger=data_curation_logger)
    centroid_curator.run()

    # 1.5. Create “round-0” Candidate Training Dataset, and Validation/Testing Datasets
    round0_curator: Round0DatasetCurator = Round0DatasetCurator(config=data_config, logger=data_curation_logger)
    round0_curator.run()

if __name__ == "__main__":
    main()