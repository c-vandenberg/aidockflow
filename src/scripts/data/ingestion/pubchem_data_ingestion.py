#!/usr/bin/env python3

import os
import argparse
from typing import List, Dict, Any

from ml_training_base import configure_logger, write_strings_to_file, load_config
from biochemical_data_connectors import PubChemBioactivesConnector
from biochemical_data_connectors.models import BioactiveCompound


def parse_config_argument():
    """
    Parses command-line arguments.

    For script use with configuration file.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract bioactive compound SMILES strings for a given target from PubChem "
                    "using a configuration file."
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        required=True,
        help="Provide path to configuration file. Required configurations are `uniprot_id`, `bioactivity_measure`, "
             "`bioactive_smiles_save_path`, and `log_save_path`"
    )

    return parser.parse_args()


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract bioactive compound SMILES strings for a given target from PubChem."
    )
    parser.add_argument(
        '--target_uniprot_id',
        type=str,
        required=True,
        help='Provide target as a UniProt accession (e.g., "P00533").'
    )
    parser.add_argument(
        '--compounds_smiles_output_path',
        type=str,
        required=True,
        help='Output path for bioactive compound SMILES strings.'
    )
    parser.add_argument(
        '--bioactivity_measure',
        type=str,
        required=True,
        help='The bioactivity measurement type to filter on (e.g. "IC50"). Default is "IC50".'
    )
    parser.add_argument(
        '--bioactivity_threshold',
        type=str,
        required=True,
        help='The maximum standard_value (in nM) to consider a compound bioactive. '
             'Default is 1000 (i.e. compounds with IC50 <= 1 µM).'
    )
    parser.add_argument(
        '--script_log_path',
        type=str,
        required=True,
        default='../var/log/preprocessed/pubchem/data_ingestion.log',
        help='Path log file.'
    )

    return parser.parse_args()

def main():
    # Parse command-line arguments, validate paths and configure logger
    args = parse_config_argument()

    config: Dict[str, Any] = load_config(args.config_file_path)
    data_config: Dict[str, Any] = config.get('data', {})

    if not data_config.get('bioactive_smiles_save_path'):
        raise ValueError('You must provide a `bioactive_smiles_save_path` in the configuration file.')

    if not data_config.get('log_save_path'):
        raise ValueError('You must provide a `log_save_path` in the configuration file.')


    os.makedirs(os.path.dirname(data_config.get('bioactive_smiles_save_path')), exist_ok=True)
    os.makedirs(os.path.dirname(data_config.get('log_save_path')), exist_ok=True)
    logger = configure_logger(log_path=data_config.get('log_save_path'))

    pubchem_connector = PubChemBioactivesConnector(
        bioactivity_measures=data_config.get('bioactivity_measures'),
        bioactivity_threshold=1000,
        logger=logger
    )
    pubchem_bioactives: List[BioactiveCompound] = pubchem_connector.get_bioactive_compounds(target_uniprot_id='P00533')

    write_strings_to_file(
        file_path='../data/preprocessed/uniprot_id/P00533/pubchem_bioactives',
        str_list=pubchem_bioactives,
        logger=logger,
        content_name='SMILES'
    )

if __name__ == "__main__":
    main()
