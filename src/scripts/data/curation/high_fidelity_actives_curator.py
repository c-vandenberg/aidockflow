#!/usr/bin/env python3

import os
import argparse
from typing import Dict, Any

from ml_training_base import configure_logger, load_config
from src.data.curation.actives_curator import HighFidelityActivesCurator


def parse_config_argument():
    """
    Parses command-line arguments.

    For script use with configuration file.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract bioactive compound SMILES strings for a given target from ChEMBL and PubChem "
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
    # Parse command-line arguments, validate paths and configure logger
    args = parse_config_argument()

    config: Dict[str, Any] = load_config(args.config_file_path)
    data_config: Dict[str, Any] = config.get('data', {})

    if not all([data_config.get('uniprot_id'), data_config.get('bioactivity_measures')]):
        raise ValueError('You must provide a `uniprot_id` and `bioactivity_measures` in the configuration file.')

    os.makedirs(os.path.dirname(data_config.get('standardized_actives_path', '../data/processed')), exist_ok=True)
    os.makedirs(os.path.dirname(data_config.get('log_path', '../var/log')), exist_ok=True)
    logger = configure_logger(log_path=data_config.get('log_path'))

    actives_curator: HighFidelityActivesCurator = HighFidelityActivesCurator(config=data_config, logger=logger)
    actives_curator.run()

if __name__ == "__main__":
    main()
