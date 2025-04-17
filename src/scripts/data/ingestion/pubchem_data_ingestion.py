#!/usr/bin/env python3

import os
import argparse
from typing import List

from data.utils.logging_utils import configure_logger
from data.utils.file_utils import write_smiles_to_file
from data.ingestion.bioactive_compounds_extraction import PubChemExtractor

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
        default='var/log/preprocessed/pubchem/data_ingestion.log',
        help='Path log file.'
    )

    return parser.parse_args()

def main():
    # Parse command-line arguments, validate paths and configure logger
    args = parse_arguments()

    os.makedirs(os.path.dirname(args.compounds_smiles_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.script_log_path), exist_ok=True)
    logger = configure_logger(log_path=args.script_log_path)

    pubchem_extractor = PubChemExtractor(bioactivity_threshold=1000, logger=logger)
    pubchem_bioactives: List[str] = pubchem_extractor.get_bioactive_compounds(target_uniprot_id='P00533')

    write_smiles_to_file(
        file_path='../data/preprocessed/uniprot_id/P00533/pubchem_bioactives',
        smiles=pubchem_bioactives,
        logger=logger
    )

if __name__ == "__main__":
    main()
