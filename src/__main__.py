import os
from typing import List

from ml_training_base.utils.logging.logging_utils import configure_logger

from data.utils.file_utils import write_smiles_to_file
from data.ingestion.bioactive_compounds_extraction import ChEMBLExtractor, PubChemExtractor

def main():
    os.makedirs('../var/log', exist_ok=True)
    logger = configure_logger('../var/log/data_ingestion.log')

    chembl_extractor = ChEMBLExtractor(bioactivity_threshold=1000, logger=logger)
    pubchem_extractor = PubChemExtractor(bioactivity_threshold=1000, logger=logger)

    chembl_bioactives: List[str] = chembl_extractor.get_bioactive_compounds(target_uniprot_id='P00533')
    pubchem_bioactives: List[str] = pubchem_extractor.get_bioactive_compounds(target_uniprot_id='P00533')

    write_smiles_to_file(
        file_path='../data/preprocessed/uniprot_id/P00533/chembl_bioactives',
        smiles=chembl_bioactives,
        logger=logger
    )

    write_smiles_to_file(
        file_path='../data/preprocessed/uniprot_id/P00533/pubchem_bioactives',
        smiles=pubchem_bioactives,
        logger=logger
    )

    print('Test run finished')

if __name__ == "__main__":
    main()